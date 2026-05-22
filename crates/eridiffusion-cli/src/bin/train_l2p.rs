//! train_l2p — L2P (T2I-L2P, Tencent Youtu) LoRA training binary.
//!
//! L2P = Z-Image-Turbo DiT body + 16×16 pixel-space patchify +
//! MicroDiffusionModel U-Net head. This trainer fine-tunes the DiT body
//! only via LoRA — the U-Net remains frozen.
//!
//! # Pipeline per step
//!
//! 1. Load cached `{pixel, cap_feats}` (from `prepare_l2p`).
//!    - `pixel`     : BF16 `[3, H, W]`,  normalized to [-1, 1]
//!    - `cap_feats` : BF16 `[1, seq, 2560]`
//! 2. Sample timestep `v ∈ (0, 1]` via LOGIT_NORMAL (matches Z-Image's
//!    flow-matching schedule and L2P's `FlowMatchScheduler "Z-Image"` preset).
//! 3. Rectified flow: `noisy = (1 - v) * clean + v * noise`,  v=sigma.
//! 4. Target = `clean - noise`. **L2P's `forward_inner` ends with
//!    `mul_scalar(-1.0)`** (sign-flip convention), so `pred ≈ -velocity ≈
//!    clean - noise`. Comparing pred against (clean - noise) gives the
//!    same sign on both sides.
//! 5. Forward: `pred = model.forward(noisy, sigma, cap_feats)` (returns
//!    BF16 `[1, 3, H, W]`).
//! 6. Loss = mean MSE in F32. Backward, AdamW (BF16 stoch-round via the
//!    AdamW family), step.
//!
//! # LoRA scope
//!
//! DiT-only — 34 attention blocks (2 noise_refiner + 2 context_refiner +
//! 30 main layers) × 5 weight keys per block = **170 LoRA modules**.
//! Per-block targets (post-translation, pre-transposed [in, out] shape):
//!   - `attention.qkv.weight`         [3840, 11520]
//!   - `attention.out.weight`         [3840, 3840]
//!   - `feed_forward.w1.weight`       [3840, 10240]
//!   - `feed_forward.w2.weight`       [10240, 3840]
//!   - `feed_forward.w3.weight`       [3840, 10240]
//!
//! U-Net `local_decoder.*` Conv2d/MaxPool2d weights are excluded from
//! the LoRA target set. Their `requires_grad` stays false (Conv2d weights
//! constructed inside `MicroDiffusionModel::new` are not autograd
//! Parameters), so they receive no gradient and are not added to the
//! optimizer — frozen by construction.

use clap::Parser;
use flame_core::diagnostics;
use flame_core::parameter::Parameter;
use flame_core::serialization::{save_file, save_tensors, SerializationFormat};
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use rand::Rng as _;

use inference_flame::lora::{LoraStack, Slot, TrainEntry};
use inference_flame::models::l2p::{weight_loader::translate_l2p_keys, L2pDiT};

// -------------------------------------------------------------------------
// Hyperparameters mirroring Z-Image preset / L2P train_run.sh
// -------------------------------------------------------------------------

const SEED: u64 = 42;

// DiT dimensions — pinned from L2pDiTConfig::default() in
// inference-flame/src/models/l2p/dit.rs.
const DIM: usize = 3840;
const QKV_OUT: usize = 3 * DIM; // 11520
const MLP_HIDDEN: usize = 10240;
const NUM_NOISE_REFINER: usize = 2;
const NUM_CONTEXT_REFINER: usize = 2;
const NUM_LAYERS: usize = 30;

#[derive(Parser)]
struct Args {
    /// L2P single-file safetensors (merged Z-Image-Turbo + L2P deltas).
    #[arg(
        long,
        default_value = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
    )]
    model: PathBuf,
    /// Directory of `prepare_l2p` outputs (one safetensors per sample).
    #[arg(long)]
    cache: PathBuf,
    /// Where to write LoRA checkpoints.
    #[arg(long)]
    output: PathBuf,
    /// Total training steps. 200 = smoke target, 1000 = real LoRA target.
    #[arg(long, default_value = "200")]
    steps: usize,
    /// Learning rate. Default from L2P's train_run.sh (`5e-5` for fine-
    /// tuning the merged DiT body).
    #[arg(long, default_value_t = 5e-5)]
    lr: f32,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    /// LoRA alpha (effective scale = alpha / rank).
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f32,
    /// Square training resolution. 512 is the smoke target. 1024² works
    /// for inference but blows the activation budget for training on
    /// 24 GB cards (per PORT_STATE.md activation estimate).
    #[arg(long, default_value_t = 512)]
    resolution: usize,
    /// Per-step gradient-clip global L2 norm.
    #[arg(long, default_value_t = 1.0)]
    clip_grad_norm: f32,
    /// Random seed (data shuffling + noise). Default matches the project
    /// convention (`SEED=42` across the codebase per CONTEXT.md).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Save LoRA every N steps. `0` disables periodic save (final-only).
    #[arg(long, default_value_t = 500)]
    save_every: usize,
    /// SQLite scalars DB for the BoardWriter (per-step loss / grad_norm).
    #[arg(long, default_value = "./l2p_train.board.db")]
    log_db: PathBuf,
    /// Number of training-time scheduler steps. Per Python
    /// `train_L2P.py:89` (`num_inference_steps: 500` in inputs_shared)
    /// L2P trains with a 500-step FLUX-shift schedule. The trainer
    /// uniform-samples an index in `[0, train_num_steps)` and reads the
    /// shift-warped sigma from that index. Audit F3+F4 fix
    /// (`MATH_AUDIT_2026-05-22.md`).
    #[arg(long, default_value_t = 500)]
    train_num_steps: usize,
    /// FlowMatch sigma shift used to warp the training schedule. Must match
    /// the inference shift (default 3.0 per L2P pipeline preset). Same
    /// formula as `build_l2p_sigma_schedule`: `shift·s/(1 + (shift-1)·s)`.
    #[arg(long, default_value_t = 3.0)]
    train_shift: f32,
    /// Optimizer family. `adamw` is the canonical project default (BF16
    /// grad → F32 moments via the default `GradDtypePolicy::CastToF32`).
    #[arg(long, default_value = "adamw")]
    optimizer: String,
}

// (build_timestep_config removed 2026-05-22 — L2P uses uniform-over-warped-schedule
//  sampling per Python `FlowMatchSFTLoss`, not LOGIT_NORMAL. See
//  build_l2p_training_sigma_table in inference_flame::sampling::l2p_sampling.)

// -------------------------------------------------------------------------
// LoRA target table — per-block weight keys + (in, out) dims
// -------------------------------------------------------------------------

/// Per-block LoRA targets, sized in pre-transposed `[in, out]` orientation.
/// These exactly match the weights placed in `L2pDiT::resident` after
/// `translate_l2p_keys` + `new_resident` pre-transpose.
fn l2p_block_targets() -> &'static [(&'static str, usize, usize)] {
    &[
        ("attention.qkv.weight", DIM, QKV_OUT),
        ("attention.out.weight", DIM, DIM),
        ("feed_forward.w1.weight", DIM, MLP_HIDDEN),
        ("feed_forward.w2.weight", MLP_HIDDEN, DIM),
        ("feed_forward.w3.weight", DIM, MLP_HIDDEN),
    ]
}

/// Enumerate every (weight_key, in_dim, out_dim) targeted by the trainer.
/// 34 blocks × 5 targets = 170 entries.
fn enumerate_lora_targets() -> Vec<(String, usize, usize)> {
    let mut out = Vec::with_capacity(170);
    for i in 0..NUM_NOISE_REFINER {
        for &(leaf, in_d, out_d) in l2p_block_targets() {
            out.push((format!("noise_refiner.{i}.{leaf}"), in_d, out_d));
        }
    }
    for i in 0..NUM_CONTEXT_REFINER {
        for &(leaf, in_d, out_d) in l2p_block_targets() {
            out.push((format!("context_refiner.{i}.{leaf}"), in_d, out_d));
        }
    }
    for i in 0..NUM_LAYERS {
        for &(leaf, in_d, out_d) in l2p_block_targets() {
            out.push((format!("layers.{i}.{leaf}"), in_d, out_d));
        }
    }
    out
}

/// Build a fresh LoRA Parameter pair for a single target.
///
/// Shape convention matches `LoraStack` training apply (`(x @ down) @ up`):
///   down: [in,   rank]  — Kaiming-ish small init (1/sqrt(rank) std)
///   up:   [rank, out ]  — zero init (canonical LoRA convention)
///
/// Both dtypes BF16 → matches the L2P resident weights and skips dtype-
/// casts in the apply matmul chain. Both `requires_grad=true` so the
/// autograd recorder picks them up.
fn make_lora_pair(
    name: &str,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    device: &Arc<flame_core::CudaDevice>,
    seed: u64,
) -> anyhow::Result<(Parameter, Parameter)> {
    // Kaiming-uniform-ish: std = 1/sqrt(rank). Matches lycoris-rs LoRA
    // init and OneTrainer (`LoRALinear::init_kaiming` default).
    let down_std = 1.0_f32 / (rank as f32).sqrt();
    let down = Tensor::randn_seeded(
        Shape::from_dims(&[in_dim, rank]),
        0.0,
        down_std,
        seed,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?
    .requires_grad_(true);
    let up = Tensor::zeros_dtype(
        Shape::from_dims(&[rank, out_dim]),
        DType::BF16,
        device.clone(),
    )?
    .requires_grad_(true);
    let _ = name; // diagnostic placeholder
    Ok((Parameter::new(down), Parameter::new(up)))
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;

    // -------------------------------------------------------------------
    // Pre-flight env warnings. We don't FORCE the env vars at runtime —
    // setting them mid-process is racy on multi-thread. We diagnose.
    // -------------------------------------------------------------------
    if std::env::var("FLAME_ALLOC_POOL").as_deref() != Ok("0") {
        eprintln!(
            "WARNING: FLAME_ALLOC_POOL is not set to 0. L2P training is known to OOM \
             without this. Recommend `FLAME_ALLOC_POOL=0 ... train_l2p ...`."
        );
    }
    if std::env::var("FLAME_AUTOGRAD_OFF").as_deref() == Ok("1") {
        eprintln!("FATAL: FLAME_AUTOGRAD_OFF=1 disables training. Unset before running.");
        std::process::exit(2);
    }
    if std::env::var("FLAME_ASSERT_GRAD_FLOW").as_deref() != Ok("1") {
        eprintln!(
            "WARNING: FLAME_ASSERT_GRAD_FLOW is not set to 1. Recommended for catching \
             dead-leaf training bugs early."
        );
    }

    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    // -------------------------------------------------------------------
    // 1. Load + translate + construct L2pDiT.
    // -------------------------------------------------------------------
    log::info!("[1/5] Loading L2P safetensors from {}...", args.model.display());
    let source = flame_core::serialization::load_file(&args.model, &device)?;
    let internal = translate_l2p_keys(source)?;
    log::info!(
        "  translated {} keys ({} after fuse + rename)",
        internal.len(),
        internal.len()
    );
    let mut model = L2pDiT::new_resident(internal, device.clone());

    // -------------------------------------------------------------------
    // 2. Build LoRA Parameters + assemble training-mode LoraStack.
    // -------------------------------------------------------------------
    log::info!(
        "[2/5] Building DiT-only LoRA: rank={} alpha={} → 170 modules",
        args.lora_rank,
        args.lora_alpha
    );
    let scale = args.lora_alpha / args.lora_rank as f32;
    let mut train_map: HashMap<String, Vec<TrainEntry>> = HashMap::new();
    let mut params: Vec<Parameter> = Vec::new();
    let mut named: Vec<(String, Parameter)> = Vec::new();
    let targets = enumerate_lora_targets();
    let n_targets = targets.len();
    for (idx, (key, in_dim, out_dim)) in targets.into_iter().enumerate() {
        // Per-target seed offset so each module's down init is distinct.
        let (down, up) =
            make_lora_pair(&key, in_dim, out_dim, args.lora_rank, &device, args.seed + idx as u64)?;
        params.push(down.clone());
        params.push(up.clone());
        // PEFT/ai-toolkit save format names.
        named.push((format!("diffusion_model.{key}.lora_A.weight"), down.clone()));
        named.push((format!("diffusion_model.{key}.lora_B.weight"), up.clone()));
        train_map.entry(key).or_default().push(TrainEntry {
            slot: Slot::Full,
            down,
            up,
            scale,
        });
    }
    if train_map.len() != n_targets {
        anyhow::bail!(
            "expected {} LoRA target keys, got {} after dedup — duplicate key?",
            n_targets,
            train_map.len(),
        );
    }
    log::info!(
        "  built {} Parameters ({} train entries × A+B pair)",
        params.len(),
        train_map.len()
    );

    let stack = Arc::new(LoraStack::new_training(train_map));
    model.set_lora(stack);

    // -------------------------------------------------------------------
    // 3. Optimizer + timestep config + BoardWriter.
    // -------------------------------------------------------------------
    let opt_kind = OptimizerKind::parse(&args.optimizer)
        .map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[3/5] Optimizer: {} lr={}", opt_kind.as_str(), args.lr);
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);

    // Training-time sigma table — uniform-over-FLUX-warped per Python L2P.
    // Audit F3+F4 fix: replaces the LOGIT_NORMAL sampling that was inherited
    // from Z-Image's preset but doesn't match what L2P actually does.
    let train_sigmas = inference_flame::sampling::l2p_sampling::
        build_l2p_training_sigma_table(args.train_num_steps, args.train_shift);
    log::info!(
        "[3/5] Training sigma table: {} steps × shift={} (FLUX-shift, matches Python FlowMatchScheduler 'Z-Image')",
        args.train_num_steps,
        args.train_shift,
    );

    let board = BoardWriter::open(
        &args.output,
        BoardWriter::new_session_id(),
        None,
    )
    .map_err(|e| log::warn!("board.db open failed: {e}"))
    .ok();
    if let Some(b) = &board {
        log::info!("[3/5] SerenityBoard writing scalars to {}", b.db_path.display());
    }
    // The `--log-db` flag is preserved for forward-compat; BoardWriter::open
    // currently picks the path under `--output`. Surface a warning if the
    // user wanted a different DB path.
    let _ = &args.log_db;

    // -------------------------------------------------------------------
    // 4. Enumerate cached samples.
    // -------------------------------------------------------------------
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache);
    }
    log::info!("[4/5] Found {} cached samples", cache_files.len());

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    // -------------------------------------------------------------------
    // 5. Training loop.
    // -------------------------------------------------------------------
    log::info!(
        "[5/5] Starting {} training steps @ resolution {}²",
        args.steps,
        args.resolution
    );
    let t_start = std::time::Instant::now();
    let mut total_loss = 0.0_f32;

    for step in 0..args.steps {
        // ── Load one cached sample ────────────────────────────────────
        let cache_idx = step % cache_files.len();
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
        let pixel = sample
            .get("pixel")
            .ok_or_else(|| anyhow::anyhow!("cache {cache_idx} missing 'pixel'"))?
            .to_dtype(DType::BF16)?;
        // pixel arrives as [3, H, W]; reshape to [1, 3, H, W].
        let pixel = {
            let d = pixel.shape().dims().to_vec();
            if d.len() != 3 {
                anyhow::bail!("pixel shape {:?} != [3, H, W]", d);
            }
            pixel.reshape(&[1, d[0], d[1], d[2]])?
        };
        let cap_feats = sample
            .get("cap_feats")
            .ok_or_else(|| anyhow::anyhow!("cache {cache_idx} missing 'cap_feats'"))?
            .to_dtype(DType::BF16)?;

        // ── Sample timestep + build noisy / target ───────────────────
        //
        // Python L2P (loss.py:6-13 with train_L2P.py:89 `num_inference_steps=500`):
        //   timestep_id = randint(0, len(scheduler.timesteps))     # uniform [0, 500)
        //   sigma       = scheduler.sigmas[timestep_id]            # FLUX-shifted
        //   noisy       = (1 - sigma) * clean + sigma * noise
        //   target      = noise - clean
        //   pred        = -DiT(noisy, timestep=sigma*1000 → forward divides by 1000)
        //   loss        = MSE(pred, target)
        //
        // Our path mirrors exactly: uniform idx → lookup shift-warped sigma →
        // pass that sigma as `v ∈ [0,1]` to L2pDiT.forward (which applies
        // `(1-v)*time_scale` internally — net effect identical to Python's
        // `timestep / 1000` pre-divide).
        let train_idx: usize = (rng.gen::<u32>() as usize) % args.train_num_steps;
        let sigma = train_sigmas[train_idx]; // shift-warped, in (0, 1]
        let v_in = sigma;

        let noise = Tensor::randn(pixel.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Rectified flow noisy: (1 - sigma) * clean + sigma * noise.
        let noisy = pixel
            .mul_scalar(1.0 - sigma)?
            .add(&noise.mul_scalar(sigma)?)?;
        // Target = (noise - clean) per Python L2P's `FlowMatchScheduler.training_target`
        // (reference/diffsynth/diffusion/flow_match.py:172-174: `target = noise - sample`).
        //
        // Python's pipeline applies the SAME negation we do: `model_fn_z_image`
        // returns `-DiT(...)`. Loss in Python: MSE(model_fn_output, training_target)
        //                                    = MSE(-v_raw, noise - clean).
        // Our pred path is identical: L2pDiT.forward returns `-v_raw` (via the
        // `mul_scalar(-1.0)` at the tail of `forward_inner`). So our target
        // must match Python's: `noise - clean`.
        //
        // (Earlier this was `clean - noise`. That inverts both sides of the MSE
        // and the model can't learn — loss saturates at ~4*var ≈ 5.5 in BF16.
        // Fix landed 2026-05-22 after a 300-step smoke confirmed the inversion.)
        let target = noise.sub(&pixel)?;

        let timestep =
            Tensor::from_vec(vec![v_in], Shape::from_dims(&[1]), device.clone())?
                .to_dtype(DType::BF16)?;

        if step == 0 {
            log::info!(
                "step 0 | pixel={:?} cap={:?} sigma={:.4} (v_in={:.4})",
                pixel.shape().dims(),
                cap_feats.shape().dims(),
                sigma,
                v_in,
            );
        }

        // ── Forward ──────────────────────────────────────────────────
        let pred = model.forward(&noisy, &timestep, &cap_feats)?;
        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "pred {:?} != target {:?}",
                pred.shape().dims(),
                target.shape().dims()
            );
        }

        // ── Loss = mean MSE in F32 ───────────────────────────────────
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let diff = pred_f32.sub(&target_f32)?;
        let loss = diff.mul(&diff)?.mean()?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        // ── Backward ─────────────────────────────────────────────────
        let mut grads = loss.backward()?;

        // Grad-flow check at step 1 (LoRA-B is zero-init so step-0
        // through `delta = down @ up` is identically zero; backward
        // through `delta * weight` produces zero gradients on down by
        // mathematical construction. Step 1 is the first step where the
        // assertion can distinguish "real bug" from "expected zero").
        if step == 1 {
            let named_refs: Vec<(&str, &Parameter)> =
                named.iter().map(|(n, p)| (n.as_str(), p)).collect();
            match diagnostics::assert_grad_flow(&grads, &named_refs) {
                Ok(report) if report.is_clean() => {
                    log::info!("[grad-flow] step 1 clean ({} params)", report.ok_count);
                }
                Ok(report) => log::warn!("{}", report.summary()),
                Err(e) => log::warn!("[grad-flow] check failed: {e}"),
            }
        }

        // ── Grad clip + assign ──────────────────────────────────────
        let grad_refs: Vec<&Tensor> = params.iter().filter_map(|p| grads.get(p.id())).collect();
        let total_norm = if grad_refs.is_empty() {
            0.0_f32
        } else {
            flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?.item()? as f32
        };
        let scale = if total_norm > args.clip_grad_norm {
            args.clip_grad_norm / total_norm
        } else {
            1.0
        };
        for p in &params {
            if let Some(g) = grads.get(p.id()) {
                let g_scaled = if scale < 1.0 {
                    g.mul_scalar(scale)?
                } else {
                    g.clone()
                };
                p.set_grad(g_scaled)?;
            }
        }

        // ── Optimizer step ──────────────────────────────────────────
        {
            let _g = AutogradContext::no_grad();
            opt.set_lr(args.lr);
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        AutogradContext::clear();

        // ── Logging ─────────────────────────────────────────────────
        eridiffusion_core::training::progress::log_step(
            "L2P-lora",
            step,
            args.steps,
            cache_files.len(),
            1,
            loss_val,
            total_norm,
            args.lr,
            t_start,
            board.as_ref(),
        );

        // ── LoRA-B nonzero-ratio diagnostic at step 1 (paired with
        //    grad-flow). After one optimizer step, LoRA-B should have
        //    moved off zero on the modules that saw gradient.
        if step == 1 {
            let mut nonzero = 0usize;
            let mut total = 0usize;
            for (name, p) in &named {
                if !name.contains(".lora_B.weight") {
                    continue;
                }
                total += 1;
                if let Ok(t) = p.tensor() {
                    // abs.sum > 0 ⇒ at least one element has moved off zero
                    // (LoRA-B is zero-init so this is the correct test).
                    let s = t
                        .to_dtype(DType::F32)
                        .and_then(|f| f.mul(&f))
                        .and_then(|f| f.sum())
                        .and_then(|f| f.item())
                        .unwrap_or(0.0);
                    if s > 0.0 {
                        nonzero += 1;
                    }
                }
            }
            log::info!(
                "[lora-B-nonzero] step 1: {}/{} modules off-zero",
                nonzero,
                total
            );
        }

        // ── Periodic save ───────────────────────────────────────────
        let step_num = step + 1;
        if args.save_every > 0
            && step_num % args.save_every == 0
            && step_num < args.steps
        {
            let path = args
                .output
                .join(format!("l2p_lora_step{step_num}.safetensors"));
            if let Err(e) = save_lora_peft(&named, &path) {
                log::warn!("[save step {step_num}] {e}");
            } else {
                log::info!("[save step {step_num}] {}", path.display());
            }
        }
    }

    // ── Final save ────────────────────────────────────────────────────
    let final_path = args
        .output
        .join(format!("l2p_lora_{}steps.safetensors", args.steps));
    save_lora_peft(&named, &final_path)?;
    let avg_loss = if args.steps > 0 {
        total_loss / args.steps as f32
    } else {
        0.0
    };
    log::info!(
        "Training complete: {} steps, avg loss = {:.4} → {}",
        args.steps,
        avg_loss,
        final_path.display()
    );
    if let Some(b) = &board {
        b.set_status("completed");
    }
    Ok(())
}

/// Save LoRA in PEFT/ai-toolkit format. Each Parameter is written under
/// its already-namespaced key (`diffusion_model.<weight_key>.lora_A.weight`
/// / `...lora_B.weight`).
fn save_lora_peft(
    named: &[(String, Parameter)],
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, p) in named {
        let t = p.tensor()?;
        tensors.insert(name.clone(), t);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    save_file(&tensors, path).map_err(|e| anyhow::anyhow!("save_file: {e}"))?;
    // Silence the unused-import warning when save_tensors path isn't used.
    let _ = save_tensors as fn(
        &HashMap<String, Tensor>,
        &std::path::Path,
        SerializationFormat,
    ) -> flame_core::Result<()>;
    Ok(())
}

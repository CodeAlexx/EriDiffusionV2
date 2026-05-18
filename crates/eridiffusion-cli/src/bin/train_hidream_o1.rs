//! train_hidream_o1 — HiDream-O1 (Qwen3-VL pixel-DiT) LoRA training (MVP).
//!
//! HiDream-O1 is a single-model pixel-level DiT — Qwen3-VL 8B text spine plus
//! three added heads (`x_embedder`, `t_embedder1`, `final_layer2`) that
//! operate on raw `PATCH_SIZE=32` RGB patches in `[-1, 1]`. There is NO VAE
//! and NO separate text encoder: `embed_tokens` runs inside the forward and
//! consumes `input_ids` directly. See
//! `EriDiffusion-v2/docs/hidream_o1_trainer_analysis.md` §1 for the full
//! refresher.
//!
//! ## Cache contract (consumed from `prepare_hidream_o1` output)
//!
//! Each `sample_NNNNNN.safetensors` carries (all F32 on disk per
//! `prepare_hidream_o1` M2 — flame-core's safetensors writer dtype-erases):
//!
//!   - `patches`      [1, L, 3072]  pixel patches in `[-1, 1]`, L=(H/32)(W/32)
//!   - `input_ids`    [1, S_text]   Qwen3-VL chat-template ids incl. boi+tms
//!   - `position_ids` [3, S_total]  3D MRoPE T/H/W stacked
//!   - `vinput_mask`  [1, S_total]  1.0 at image slots (`token_types == 1`)
//!   - `token_types`  [1, S_total]  1.0 at image slots + TMS row (cache v2)
//!   - `image_grid`   [3]           (1, H/32, W/32) — unused here, kept for parity
//!
//! Top-level `_meta.json` carries `format: "hidream-o1-v2"` which we validate.
//!
//! ## Training step (aligned to ai-toolkit `HidreamO1Model`, 2026-05-16)
//!
//! Source of truth:
//!   - `ai-toolkit/extensions_built_in/diffusion_models/hidream/hidream_o1_model.py`
//!     `add_noise` (line 48-57) and `get_loss_target` (line 517-521).
//!   - `ai-toolkit/extensions_built_in/diffusion_models/hidream/src/hidream_o1/pipeline.py`
//!     `DEFAULT_NOISE_SCALE = 8.0`, `T_EPS = 0.001`, `PATCH_SIZE = 32`.
//!
//! ```text
//!   noise_scale = 8.0          # DEFAULT_NOISE_SCALE
//!   t_idx       ~ randint(0, 999)    # ai-toolkit O1 UI default: linear
//!   t_eps       = 1e-3         # T_EPS
//!
//!   t    = (1000 - t_idx) / 1000
//!   z    ~ N(0, 1), same shape as patches
//!   z_s  = z * noise_scale                        # SCALED noise
//!   x_t  = (1 - t) * patches + t * z_s            # add_noise()
//!   target = (z_s - patches).detach()             # get_loss_target()
//!
//!   x_pred = model.forward_lora(x_t, ids, pos, mask, ...)  # model emits x0-style
//!   # Convert model x0-style output to velocity to match `target`. From
//!   # `hidream_o1_model.py:469-473`:
//!   pred  = (x_t - x_pred) / max(t, t_eps)         # = z_s - patches if perfect
//!   loss  = clamp(MSE(pred[image_rows], target), max=1.0)
//! ```
//!
//! The conversion `pred = (x_t - x_pred)/t` (in float32) reproduces ai-toolkit's
//! `get_noise_prediction` exactly, then loss compares the resulting "noise
//! prediction" to `(z_s - patches)`. This is the **only** loss form that
//! matches inference at sampling time — inference uses
//! `v = (x_pred - z)/sigma`, sign-flips for the scheduler.
//!
//! ## Diagnostics
//!
//! Set `FLAME_ASSERT_GRAD_FLOW=1` to enable per-step LoRA grad-flow assertion
//! at step 1 (mirrors train_klein / train_chroma).  Per
//! `feedback_grad_flow_default_on.md` this should be the default; we surface
//! it in the doc rather than hard-coding `std::env::set_var` so users can
//! still opt out for production runs.

use clap::Parser;
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use flame_core::parameter::Parameter;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use inference_flame::models::hidream_o1::{
    default_target_suffixes, HiDreamO1Config, HiDreamO1WeightLoader, LoraRegistry, MRopePositions,
};
use std::io::Read;
use std::path::{Path, PathBuf};

const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;
const DEFAULT_MODEL_PATH: &str = "/home/alex/HiDream-O1-Image-Dev-weights";
/// Ai-toolkit `DEFAULT_NOISE_SCALE` (pipeline.py:15). Noise is scaled by this
/// factor in both add_noise and the loss target.
const NOISE_SCALE: f32 = 8.0;
/// Ai-toolkit `T_EPS` (pipeline.py:16). Lower bound on `t` (and `1-t`) to
/// avoid the divide-by-zero in `(x_t - x_pred)/t`.
const T_EPS_AT: f32 = 1.0e-3;
/// Ai-toolkit flowmatch scheduler `shift` (hidream_o1_model.py:33-36 +
/// `set_train_timesteps`@sampler.py:161). With `use_dynamic_shifting=False`,
/// the shift mapping is `sigma_shifted = shift * u / (1 + (shift-1) * u)`.
const FLOW_SHIFT: f32 = 3.0;

#[derive(Parser)]
struct Args {
    /// TrainConfig JSON file (optional). Falls back to TrainConfig::default().
    #[arg(long)]
    config: Option<PathBuf>,
    /// Cache dir written by prepare_hidream_o1 (contains `_meta.json` +
    /// per-sample `.safetensors`).
    #[arg(long)]
    cache_dir: PathBuf,
    /// Optional max total sequence length (`vinput_mask.shape[-1]`) for 24GB
    /// runs. Overlong cached samples are skipped before training starts.
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,
    /// HiDream-O1 model dir (containing `model.safetensors.index.json` +
    /// shards + `tokenizer.json`).
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,
    #[arg(long, default_value = "3000")]
    steps: usize,
    /// Global step offset for LoRA-only resume runs. Example: resume a
    /// step-1000 LoRA with `--start-step 1000 --steps 1000` to continue cache
    /// order and save the final checkpoint as 2000 steps.
    #[arg(long, default_value_t = 0)]
    start_step: usize,
    /// LoRA rank. Ai-toolkit yaml default: 32 (`train_lora_hidream_48.yaml:26`).
    #[arg(long, default_value = "32")]
    rank: usize,
    /// LoRA alpha. Ai-toolkit yaml default: 32 (`train_lora_hidream_48.yaml:27`).
    #[arg(long, default_value = "32.0")]
    lora_alpha: f32,
    /// Learning rate. Ai-toolkit yaml default: 2e-4 (`train_lora_hidream_48.yaml:58`).
    #[arg(long, default_value = "2e-4")]
    lr: f32,
    /// Save a LoRA checkpoint every N steps (0 = end-only). Ai-toolkit yaml
    /// default: 250 (`train_lora_hidream_48.yaml:36` → `save_every: 250`).
    #[arg(long, default_value = "250")]
    save_every: usize,
    /// In-trainer sampling cadence. **Deferred to O1-M4.** Any non-zero value
    /// logs a warning and is ignored. Use `hidream_o1_infer` externally with
    /// a saved LoRA to visualize progress.
    #[arg(long, default_value = "0")]
    sample_every: usize,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    /// `weights` (default — LoRA-only safetensors) or `full` (LoRA + AdamW
    /// state + step counter).  **`full` is not yet implemented**; passing it
    /// errors out so the user isn't surprised by silently-degraded checkpoints.
    #[arg(long, default_value = "weights")]
    save_mode: String,
    /// Resume LoRA weights only.
    #[arg(long)]
    resume_lora: Option<PathBuf>,
    /// Resume LoRA + AdamW state + step counter.
    /// TODO(O1-M3.1): wire AdamW state save/load. Until then, passing this
    /// errors out (BUG-4 fix) rather than silently restarting the optimizer
    /// state, which would jolt long-resumed runs.
    #[arg(long)]
    resume_full: Option<PathBuf>,

    // ── Training-step knobs ─────────────────────────────────────────────
    /// Timestep distribution. `linear` is the current ai-toolkit HiDream-O1
    /// UI default: pick a random scheduler index from linearly-spaced
    /// timesteps 1000..1. `shift` keeps the older flowmatch shift mapping for
    /// ablation, and `uniform` keeps the continuous unshifted ablation.
    #[arg(long, default_value = "linear")]
    timestep_distribution: String,
    /// Flowmatch shift constant. Ai-toolkit default = 3.0 (per the scheduler
    /// kwargs in `hidream_o1_model.py:32-36`). Only used when
    /// `--timestep-distribution shift`.
    #[arg(long, default_value_t = FLOW_SHIFT)]
    flow_shift: f32,
    /// Optimizer. Ai-toolkit default = `adamw8bit`
    /// (`train_lora_hidream_48.yaml:57`). HiDream-O1 yaml passes only
    /// `optimizer: "adamw8bit"` and `lr: 2e-4`, no `optimizer_params`. The
    /// downstream `bitsandbytes.optim.AdamW8bit(params, lr, eps=1e-6)` call
    /// (`ai-toolkit/toolkit/optimizer.py:71`) therefore takes bitsandbytes
    /// defaults for betas=(0.9, 0.999) and weight_decay=1e-2; only `eps` is
    /// overridden (`1e-6` instead of the torch `1e-8` default).
    ///
    /// Default = `adamw` for production safety. The flame-core AdamW8bit
    /// kernel now matches bitsandbytes 0.49.2 in isolated parity tests, but
    /// HiDream-O1 still needs end-to-end optimizer parity and the G2 overfit
    /// gate is unstable with `adamw8bit` selected. Keep `adamw8bit` as an
    /// explicit opt-in while that integration work is open.
    #[arg(long, default_value = "adamw")]
    optimizer: String,
    /// AdamW β1 momentum coefficient. Default = 0.9 (bitsandbytes / torch
    /// AdamW default — ai-toolkit's HiDream-O1 yaml does not override).
    #[arg(long, default_value_t = 0.9)]
    adamw_beta1: f32,
    /// AdamW β2 second-moment coefficient. Default = 0.999 (bitsandbytes /
    /// torch AdamW default — ai-toolkit's HiDream-O1 yaml does not override).
    #[arg(long, default_value_t = 0.999)]
    adamw_beta2: f32,
    /// AdamW ε. Default = 1e-6 (ai-toolkit `optimizer.py:67,71,77,79`
    /// hard-codes this for every Adam-family path, overriding torch's 1e-8).
    #[arg(long, default_value_t = 1.0e-6)]
    adamw_eps: f32,
    /// AdamW weight decay. Default = 1e-2 (bitsandbytes AdamW8bit default;
    /// the HiDream-O1 yaml passes no `optimizer_params` so the bnb default
    /// is what runs in ai-toolkit).
    #[arg(long, default_value_t = 1.0e-2)]
    adamw_weight_decay: f32,
    /// Clamp scalar loss before backward, matching current ai-toolkit
    /// HiDream-O1 default `train.max_loss: 1.0`. Set <= 0 to disable.
    #[arg(long, default_value_t = 1.0)]
    max_loss: f32,
}

/// Distribution mode for `t` sampling. `Linear` mirrors the current
/// ai-toolkit HiDream-O1 UI default. `Shift` is retained for the older
/// flowmatch-shift ablation, and `Uniform` is retained as the continuous
/// unshifted ablation.
#[derive(Clone, Copy, Debug)]
enum TstepMode {
    Linear,
    Uniform,
    Shift,
}

fn parse_tstep_mode(s: &str) -> anyhow::Result<TstepMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "linear" => Ok(TstepMode::Linear),
        "uniform" => Ok(TstepMode::Uniform),
        "shift" | "flowmatch" => Ok(TstepMode::Shift),
        other => anyhow::bail!(
            "--timestep-distribution: expected `linear` (default), `shift`, or `uniform`, got `{other}`"
        ),
    }
}

/// Sample `t in (t_eps, 1 - t_eps)` from the configured distribution.
fn sample_t<R: rand::Rng>(rng: &mut R, mode: TstepMode, shift: f32) -> f32 {
    match mode {
        // ai-toolkit `CustomFlowMatchScheduler.set_train_timesteps(linear)`
        // creates timesteps 1000..1, then the balanced sampler draws integer
        // indices in [0, 999). This gives sigma/t in [1.0, 0.002].
        TstepMode::Linear => {
            let idx: usize = rng.gen_range(0..999);
            (1000.0 - idx as f32) / 1000.0
        }
        TstepMode::Uniform => rng.r#gen::<f32>().clamp(T_EPS_AT, 1.0 - T_EPS_AT),
        // Flow-matching shift mapping. Continuous CDF-inverse equivalent of
        // ai-toolkit `custom_flowmatch_sampler.py:161`:
        //   sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        // applied per-sample instead of vectorized over the 1000-bucket grid.
        TstepMode::Shift => {
            let u: f32 = rng.r#gen::<f32>();
            (shift * u / (1.0 + (shift - 1.0) * u)).clamp(T_EPS_AT, 1.0 - T_EPS_AT)
        }
    }
}

/// Validate the cache's `_meta.json` header.  Mirrors `prepare_hidream_o1`'s
/// emitted format string and ensures we don't accidentally consume a cache
/// produced for a different model family.
fn validate_meta(cache_dir: &std::path::Path) -> anyhow::Result<()> {
    let meta_path = cache_dir.join("_meta.json");
    let raw = std::fs::read_to_string(&meta_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", meta_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", meta_path.display()))?;
    let fmt = v
        .get("format")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow::anyhow!("{} missing `format`", meta_path.display()))?;
    // v2 (2026-05-17) added `token_types` (`token_types_bin = (raw > 0)`)
    // to fix the attention-mask TMS-row parity bug; v1 caches must be
    // re-generated by `prepare_hidream_o1`. See
    // `EriDiffusion-v2/docs/hidream_o1_g0_deep_investigation.md`.
    //
    // v3 (2026-05-18) adds multi-resolution AR-preserving rectangular
    // buckets. The per-sample tensor schema is identical to v2 (same six
    // fields, same dtypes); v3 only declares that `image_grid`
    // (H/32, W/32) may vary across samples in the cache directory. The
    // trainer's per-step `load_file` already pulls per-sample shapes, so
    // v2 caches load transparently as "v3 with one square bucket". Accept
    // either; reject anything else.
    if fmt != "hidream-o1-v2" && fmt != "hidream-o1-v3" {
        anyhow::bail!(
            "cache format mismatch: {} reports `{fmt}`, expected `hidream-o1-v2` or \
             `hidream-o1-v3`. Re-run `prepare_hidream_o1` to regenerate the cache; v1 \
             caches lack the `token_types` field added in the 2026-05-17 attention-mask \
             bug fix.",
            meta_path.display()
        );
    }
    Ok(())
}

fn safetensors_last_dim(path: &Path, tensor_name: &str) -> anyhow::Result<usize> {
    let mut file =
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| anyhow::anyhow!("read header len {}: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|e| anyhow::anyhow!("read header {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| anyhow::anyhow!("parse safetensors header {}: {e}", path.display()))?;
    let shape = v
        .get(tensor_name)
        .and_then(|x| x.get("shape"))
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow::anyhow!("{} missing shape for `{tensor_name}`", path.display()))?;
    shape
        .last()
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| anyhow::anyhow!("{} has invalid shape for `{tensor_name}`", path.display()))
}

/// Decode the on-disk F32 `position_ids: [3, S_total]` tensor into three
/// `Vec<u32>`s for `MRopePositions`.
fn decode_position_ids(pos: &Tensor) -> anyhow::Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let dims = pos.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != 3 {
        anyhow::bail!("position_ids: expected [3, S_total], got {:?}", dims);
    }
    let s_total = dims[1];
    let flat = pos.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut t = Vec::with_capacity(s_total);
    let mut h = Vec::with_capacity(s_total);
    let mut w = Vec::with_capacity(s_total);
    for i in 0..s_total {
        t.push(flat[i] as u32);
        h.push(flat[s_total + i] as u32);
        w.push(flat[2 * s_total + i] as u32);
    }
    Ok((t, h, w))
}

/// Gather rows of `x_pred [B, S_total, 3072]` where `vinput_mask[b, i] != 0`.
///
/// Cache layout per `prepare_hidream_o1`: the image slots are the **tail** of
/// the stream (`txt_seq_len..(txt_seq_len + L)`). We exploit this to skip a
/// general per-row gather kernel and just `narrow` along dim 1 for the
/// last-L rows. The MVP assumes batch=1 (which prepare currently produces).
///
/// If the cache ever interleaves image slots, replace this with a host-side
/// mask scan and `index_select`.
fn gather_image_rows(x_pred: &Tensor, vinput_mask: &Tensor) -> anyhow::Result<Tensor> {
    let xd = x_pred.shape().dims().to_vec();
    if xd.len() != 3 {
        anyhow::bail!("gather_image_rows: x_pred must be [B,S,C], got {:?}", xd);
    }
    let (b, s_total, _c) = (xd[0], xd[1], xd[2]);
    let md = vinput_mask.shape().dims().to_vec();
    if md.len() != 2 || md[0] != b || md[1] != s_total {
        anyhow::bail!(
            "gather_image_rows: vinput_mask shape {:?} != [{},{}]",
            md,
            b,
            s_total
        );
    }
    let host = vinput_mask.to_dtype(DType::F32)?.to_vec_f32()?;
    // Find the first and last non-zero index across the stream (per-batch
    // MVP: b==1). We assume contiguous tail layout (matches prep).
    let mut first: Option<usize> = None;
    let mut last: Option<usize> = None;
    for i in 0..s_total {
        if host[i] != 0.0 {
            first.get_or_insert(i);
            last = Some(i);
        }
    }
    let (first, last) = (
        first.ok_or_else(|| anyhow::anyhow!("vinput_mask has no image slots"))?,
        last.unwrap(),
    );
    let len = last - first + 1;
    // Sanity: count of 1's must equal `len` (i.e. tail is contiguous).
    let count = host.iter().filter(|&&x| x != 0.0).count();
    if count != len {
        anyhow::bail!(
            "gather_image_rows: non-contiguous image slots not yet supported \
             (got {count} non-zero, span [{first}..{}] len {len}). \
             TODO(O1-M3.1): index_select fallback.",
            last + 1
        );
    }
    Ok(x_pred.narrow(1, first, len)?)
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;
    validate_meta(&args.cache_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();
    // BUG-8 fix: set the global flame RNG seed BEFORE model load so that any
    // ephemeral random init in the loader path (Linear::new, RMSNorm::new etc.
    // before safetensors overwrite) is deterministic across runs. Previously
    // this was set just before the train loop, leaving the loader at the
    // default RNG state.
    flame_core::rng::set_seed(SEED).map_err(|e| anyhow::anyhow!("flame_core set_seed: {e}"))?;

    let mut config = if let Some(cp) = &args.config {
        TrainConfig::from_json_path(&cp.to_string_lossy())?
    } else {
        TrainConfig::default()
    };
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = args.rank as u64;
    config.lora_alpha = args.lora_alpha as f64;
    config.learning_rate = args.lr as f64;

    if args.sample_every > 0 {
        log::warn!(
            "[hidream_o1] --sample-every={} ignored: in-trainer sampling deferred to O1-M4. \
             Use `hidream_o1_infer --lora-path ...` externally between checkpoints.",
            args.sample_every
        );
    }

    let tstep_mode = parse_tstep_mode(&args.timestep_distribution)?;
    log::info!(
        "[hidream_o1] tstep_mode={:?} flow_shift={} noise_scale={} t_eps={} max_loss={}",
        tstep_mode,
        args.flow_shift,
        NOISE_SCALE,
        T_EPS_AT,
        args.max_loss,
    );

    let save_mode_full = match args.save_mode.as_str() {
        "weights" => false,
        "full" => {
            // BUG-3: fail loudly instead of silently degrading. Once
            // TODO(O1-M3.1) AdamW state save lands, switch this back to a
            // capable path. Until then, the user gets an honest error rather
            // than a checkpoint that looks `full` on disk but isn't.
            anyhow::bail!(
                "--save-mode=full is not yet implemented (TODO O1-M3.1: \
                 AdamW state + step counter). Re-run with --save-mode=weights."
            );
        }
        other => anyhow::bail!("--save-mode must be `weights` or `full`, got `{other}`"),
    };

    // ── Load model + tokenizer (the tokenizer is unused at training time but
    //    the weight loader needs the model dir laid out beside it).
    let hd_cfg = HiDreamO1Config::dev_8b();
    log::info!(
        "[hidream_o1] loading model from {} (num_layers={}, hidden={})",
        args.model_path.display(),
        hd_cfg.num_layers,
        hd_cfg.hidden_size,
    );
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader: {e}"))?;
    // The base model's parameters are all `requires_grad=false` (loader runs
    // under no_grad). The trainable surface is purely the LoRA registry.
    let mut model = loader
        .load_model(&hd_cfg, &device)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader::load_model: {e}"))?;

    // BUG-4: fail loudly on --resume-full until AdamW state restore lands.
    if args.resume_full.is_some() {
        anyhow::bail!(
            "--resume-full is not yet implemented (TODO O1-M3.1: AdamW state \
             restore). Use --resume-lora to restore LoRA weights only."
        );
    }

    // ── Build LoRA registry: 7 suffixes × 36 layers = 252 adapters for 8B.
    let mut lora = if let Some(resume) = args.resume_lora.as_ref() {
        log::info!(
            "[hidream_o1] resuming LoRA registry from {}",
            resume.display()
        );
        LoraRegistry::from_safetensors(resume, &hd_cfg, &device)
            .map_err(|e| anyhow::anyhow!("LoraRegistry::from_safetensors: {e}"))?
    } else {
        LoraRegistry::new(
            &hd_cfg,
            args.rank,
            args.lora_alpha,
            default_target_suffixes(),
            SEED,
            &device,
        )
        .map_err(|e| anyhow::anyhow!("LoraRegistry::new: {e}"))?
    };
    log::info!(
        "[hidream_o1] LoRA registry: {} adapters, rank={}, alpha={}",
        lora.len(),
        lora.rank,
        lora.alpha,
    );

    // ── Flatten registry into a Vec<Parameter> for the optimizer.
    let params: Vec<Parameter> = lora.parameters();
    log::info!("[hidream_o1] {} trainable parameters", params.len());
    if params.is_empty() {
        anyhow::bail!("LoRA registry produced no trainable parameters");
    }

    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!(
        "[hidream_o1] optimizer={} lr={} betas=({}, {}) eps={} wd={}",
        opt_kind.as_str(),
        args.lr,
        args.adamw_beta1,
        args.adamw_beta2,
        args.adamw_eps,
        args.adamw_weight_decay,
    );
    // DIVERGENCE from ai-toolkit: our default optimizer is `adamw` (full F32
    // state), NOT `adamw8bit` as the ai-toolkit yaml prescribes. flame-core's
    // AdamW8bit kernel has bnb 0.49.2 block-wise parity in isolated tests, but
    // the HiDream-O1 trainer integration is still not launch-safe: G2 overfit
    // showed late loss/grad spikes with `adamw8bit`, while plain AdamW
    // recovered. Pass `--optimizer adamw8bit` only for explicit parity probes.
    let mut opt = Optimizer::new(
        opt_kind,
        args.lr,
        args.adamw_beta1,
        args.adamw_beta2,
        args.adamw_eps,
        args.adamw_weight_decay,
    );

    // ── Index cache files.
    // BUG-6 fix: only `sample_NNNNNN.safetensors` to avoid picking up
    // companions (e.g. `features.safetensors`) or stale `*.partial` artifacts
    // from a crashed prep run.
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map_or(false, |e| e == "safetensors")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("sample_"))
                    .unwrap_or(false)
        })
        .collect();
    cache_files.sort();
    if args.max_seq_len > 0 {
        let before = cache_files.len();
        let mut kept = Vec::with_capacity(before);
        let mut skipped: Vec<(PathBuf, usize)> = Vec::new();
        for path in cache_files {
            let seq_len = safetensors_last_dim(&path, "vinput_mask")?;
            if seq_len <= args.max_seq_len {
                kept.push(path);
            } else {
                skipped.push((path, seq_len));
            }
        }
        if !skipped.is_empty() {
            let longest = skipped
                .iter()
                .max_by_key(|(_, seq_len)| *seq_len)
                .map(|(path, seq_len)| {
                    format!(
                        "{} ({seq_len})",
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("<unknown>")
                    )
                })
                .unwrap_or_else(|| "<none>".to_string());
            log::warn!(
                "[hidream_o1] --max-seq-len={} skipped {}/{} cached samples; longest skipped: {}",
                args.max_seq_len,
                skipped.len(),
                before,
                longest
            );
        }
        cache_files = kept;
    }
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("[hidream_o1] {} cached samples", cache_files.len());

    // (flame_core::rng::set_seed moved above model-load — BUG-8.)
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let board = BoardWriter::open(&args.output_dir, BoardWriter::new_session_id(), None)
        .map_err(|e| log::warn!("board.db open failed: {e}"))
        .ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;
    let mut max_loss_clamps = 0usize;
    let mut max_raw_loss = 0f32;

    let total_target_steps = args.start_step + args.steps;
    for local_step in 0..args.steps {
        let step = args.start_step + local_step;
        flame_core::debug_finite::reset();

        let cache_idx = step % cache_files.len();
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
        let path_disp = cache_files[cache_idx].display().to_string();

        let patches = sample
            .get("patches")
            .ok_or_else(|| anyhow::anyhow!("missing `patches` in {path_disp}"))?
            .to_dtype(DType::BF16)?;
        let input_ids = sample
            .get("input_ids")
            .ok_or_else(|| anyhow::anyhow!("missing `input_ids` in {path_disp}"))?
            .to_dtype(DType::I32)?;
        let position_ids = sample
            .get("position_ids")
            .ok_or_else(|| anyhow::anyhow!("missing `position_ids` in {path_disp}"))?;
        let vinput_mask = sample
            .get("vinput_mask")
            .ok_or_else(|| anyhow::anyhow!("missing `vinput_mask` in {path_disp}"))?
            .to_dtype(DType::BF16)?;
        // Cache v2 (2026-05-17): `token_types_bin = (raw > 0)`. Drives the
        // attention-mask construction so the TMS row gets full-attention,
        // matching `qwen3_vl_transformers.py:1501`.
        let token_types_bin = sample
            .get("token_types")
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing `token_types` in {path_disp} (cache v1 detected — \
                     re-run prepare_hidream_o1 to regenerate v2 cache)"
                )
            })?
            .to_dtype(DType::BF16)?;
        flame_core::debug_finite::check("g2.patches", &patches)?;
        flame_core::debug_finite::check("g2.vinput_mask", &vinput_mask)?;
        flame_core::debug_finite::check("g2.token_types", &token_types_bin)?;

        let (t_pos, h_pos, w_pos) = decode_position_ids(position_ids)?;
        let pos_view = MRopePositions {
            t: &t_pos,
            h: &h_pos,
            w: &w_pos,
        };

        // ── Sample timestep (flowmatch shift by default, see TstepMode).
        let t_scalar = sample_t(&mut rng, tstep_mode, args.flow_shift);
        // Noise ~ N(0, 1), then SCALED by NOISE_SCALE before use in both the
        // noisy input and the loss target — see ai-toolkit
        // `hidream_o1_model.py:53-56` (`scaled_noise = noise * noise_scale`)
        // and `:520-521` (`target = noise*noise_scale - latents`).
        let noise = Tensor::randn(patches.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let scaled_noise = noise.mul_scalar(NOISE_SCALE)?;
        flame_core::debug_finite::check("g2.scaled_noise", &scaled_noise)?;
        // Linear flow matching with scaled noise:
        //   x_t = (1 - t) * patches + t * (noise * noise_scale)
        let noisy = patches
            .mul_scalar(1.0 - t_scalar)?
            .add(&scaled_noise.mul_scalar(t_scalar)?)?;
        flame_core::debug_finite::check("g2.noisy", &noisy)?;
        // HiDream-O1's model expects timestep as denoising PROGRESS
        // (1=clean, 0=noisy) — inverted from the canonical convention used
        // for `noisy`. Mirror ai-toolkit `hidream_o1_model.py:439, 446`:
        //   t_pixeldit = (1.0 - timestep / 1000.0)
        // Our `t_scalar` is already in canonical [eps, 1-eps] continuous
        // (not `/1000` discrete), so the equivalent is `1.0 - t_scalar`.
        let t_pixeldit = 1.0 - t_scalar;
        let timestep = Tensor::from_vec(
            vec![t_pixeldit],
            Shape::from_dims(&[1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // Loss target = `(scaled_noise - patches).detach()` — the
        // ai-toolkit flow-matching velocity with scaled noise.
        // `hidream_o1_model.py:517-521 get_loss_target`.
        let target_full = scaled_noise.sub(&patches)?.detach()?;
        flame_core::debug_finite::check("g2.target_full", &target_full)?;

        if step == 0 {
            log::info!(
                "step 0 | patches={:?} input_ids={:?} vinput_mask={:?} t={:.4} \
                 noise_scale={} target={:?}",
                patches.shape().dims(),
                input_ids.shape().dims(),
                vinput_mask.shape().dims(),
                t_scalar,
                NOISE_SCALE,
                target_full.shape().dims(),
            );
        }

        // ── Forward with LoRA routed through every decoder layer's 7 linears.
        // TODO(O1-G2): verify autograd memory pressure with BlockOffloader
        // during 50-step overfit smoke — Skeptic Q1 (lora.rs review). If
        // saved weight Arcs in the backward tape stay pinned across 36 layers,
        // peak GPU memory could exceed the 14 GB budget at 512².
        let x_pred = model.forward_lora(
            &input_ids,
            &timestep,
            &noisy,
            &pos_view,
            &vinput_mask,
            &token_types_bin,
            None,
            Some(&lora),
        )?;
        flame_core::debug_finite::check("g2.x_pred", &x_pred)?;

        // Gather only the image rows so the loss doesn't reward fitting text
        // positions (matches inference `pipeline.py:329`).
        let x_rows = gather_image_rows(&x_pred, &vinput_mask)?;
        flame_core::debug_finite::check("g2.x_rows", &x_rows)?;
        // `target_full` is already shaped to the image rows (it's a function
        // of `patches` and `noise`, both of shape `[1, L, 3072]`). No gather
        // needed on the target side.
        if x_rows.shape().dims() != target_full.shape().dims() {
            anyhow::bail!(
                "shape mismatch: x_rows={:?} target={:?}",
                x_rows.shape().dims(),
                target_full.shape().dims()
            );
        }

        // ── Convert model x0-style output to velocity, then MSE against
        // (scaled_noise - patches). Per ai-toolkit
        // `hidream_o1_model.py:467-473`:
        //   sigma = max(t, T_EPS)
        //   pred  = (latent_model_input.float() - x0_pred.float()) / sigma
        //   return pred.to(in_dtype)   # cast back to BF16
        // SDTrainer.py:739, 806 then re-casts `pred.float()` for MSE.
        // The BF16 round-trip TRUNCATES the 1/sigma-amplified F32 difference
        // into BF16 mantissa precision. Skipping that cast leaves the full
        // F32 amplification in `pred`, blowing up the MSE (and lora_B grad)
        // when sigma is small. Mirror AIT exactly: F32 sub+div → BF16 → F32.
        // Pred has the same shape as target_full ([1, L, 3072]).
        let sigma = t_scalar.max(T_EPS_AT);
        // noisy was built from `patches` + `scaled_noise` (both BF16); gather
        // the image rows out of `noisy` too — they're already image-aligned
        // (1:1 with target_full), no narrow needed since `noisy` is shaped
        // exactly to image rows.
        let pred_f32 = noisy
            .to_dtype(DType::F32)?
            .sub(&x_rows.to_dtype(DType::F32)?)?
            .mul_scalar(1.0 / sigma)?
            // AIT parity: round-trip through in_dtype=BF16 before MSE.
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        flame_core::debug_finite::check("g2.pred_f32", &pred_f32)?;
        let target_f32 = target_full.to_dtype(DType::F32)?;
        flame_core::debug_finite::check("g2.target_f32", &target_f32)?;
        let raw_loss = pred_f32.sub(&target_f32)?.square()?.mean()?;
        flame_core::debug_finite::check("g2.raw_loss", &raw_loss)?;
        let raw_loss_val = raw_loss.to_vec()?[0];
        if !raw_loss_val.is_finite() {
            anyhow::bail!("NaN/Inf loss at step {step}: {raw_loss_val}");
        }
        max_raw_loss = max_raw_loss.max(raw_loss_val);

        let loss = if args.max_loss > 0.0 {
            if raw_loss_val > args.max_loss {
                max_loss_clamps += 1;
                if max_loss_clamps <= 10 || max_loss_clamps % 50 == 0 {
                    log::warn!(
                        "[max-loss] step {} raw loss {:.4} > {:.4}; clamping before backward",
                        step + 1,
                        raw_loss_val,
                        args.max_loss
                    );
                }
            }
            raw_loss.clamp(0.0, args.max_loss)?
        } else {
            raw_loss
        };
        flame_core::debug_finite::check("g2.loss", &loss)?;
        let loss_val = loss.to_vec()?[0];
        if !loss_val.is_finite() {
            anyhow::bail!("NaN/Inf loss at step {step}: {loss_val}");
        }
        total_loss += loss_val;

        // ── Backward.
        let grads = loss.backward()?;

        // Grad-flow diagnostic — runs at step 1 (lora_B starts zero so step 0
        // is mathematically zero-grad on the A side).
        if step == 1 && std::env::var("FLAME_ASSERT_GRAD_FLOW").ok().as_deref() == Some("1") {
            let named = lora.named_parameters();
            let named_refs: Vec<(&str, &Parameter)> =
                named.iter().map(|(n, p)| (n.as_str(), p)).collect();
            let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!(
                    "[grad-flow] step 1 clean ({} params)",
                    report.ok_count
                );
            } else {
                log::warn!("{}", report.summary());
            }
        }

        // ── Global L2 grad clip = 1.0.
        let grad_refs: Vec<&Tensor> =
            params.iter().filter_map(|p| grads.get(p.id())).collect();
        let total_norm =
            flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?.item()? as f32;
        let scale = if total_norm > CLIP_GRAD_NORM {
            CLIP_GRAD_NORM / total_norm
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

        // ── Optimizer step (constant LR for MVP; LR scheduling is a Klein-
        //     surface feature deferred to O1-M3.1).
        {
            let _g = AutogradContext::no_grad();
            opt.set_lr(args.lr);
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        // Re-sync the LoRA registry with whatever the optimizer just wrote.
        // No-op today (LoraAdapter holds the Parameter and reads via
        // `a_tensor()` / `b_tensor()`), but kept here so a future
        // optimizer-replaces-storage path stays obvious.
        let _ = &mut lora;

        AutogradContext::clear();
        flame_core::cuda_alloc_pool::clear_pool_cache();
        device.synchronize().ok();

        eridiffusion_core::training::progress::log_step(
            "HiDreamO1-lora",
            step,
            total_target_steps,
            cache_files.len(),
            1,
            loss_val,
            total_norm,
            args.lr,
            t_start,
            board.as_ref(),
        );

        // ── Periodic save.
        let step_num = step + 1;
        if args.save_every > 0
            && step_num % args.save_every == 0
            && step_num < total_target_steps
        {
            let mid_ckpt = args
                .output_dir
                .join(format!("hidream_o1_lora_step{step_num}.safetensors"));
            if let Err(e) = lora.save_safetensors(&mid_ckpt) {
                log::warn!("[save step {step_num}] failed: {e}");
            } else {
                log::info!("[save step {step_num}] {}", mid_ckpt.display());
            }
            if save_mode_full {
                // TODO(O1-M3.1): also dump AdamW state + step counter as
                // `*_full.safetensors`. Optimizer enum lacks a uniform
                // `state_dict` accessor today.
            }
        }
    }

    let avg_loss = if args.steps > 0 {
        total_loss / args.steps as f32
    } else {
        0.0
    };
    log::info!(
        "Training complete: {} steps, avg loss={:.4}, max raw loss={:.4}, max-loss clamps={}",
        total_target_steps,
        avg_loss,
        max_raw_loss,
        max_loss_clamps
    );
    if let Some(b) = &board {
        b.set_status("completed");
    }

    let final_ckpt = args
        .output_dir
        .join(format!("hidream_o1_lora_{}steps.safetensors", total_target_steps));
    if let Err(e) = lora.save_safetensors(&final_ckpt) {
        log::warn!("save_safetensors returned: {e}");
    } else {
        log::info!("Saved final LoRA to {}", final_ckpt.display());
    }
    Ok(())
}

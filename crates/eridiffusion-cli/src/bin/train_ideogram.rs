//! train_ideogram — Ideogram-4 LoRA trainer (the stage-9 capstone of the
//! parity-verified Ideogram-4 Rust vertical). Mirrors train_klein's loop on the
//! proven Ideogram pieces:
//!   - prepare_ideogram cache: {latent [1,128,h,w], text_embedding [1,L,53248],
//!     text_mask [1,L]} (stages 5-7).
//!   - IdeogramDit (stage 4/8): resident weights + per-block AutogradContext::
//!     checkpoint (bounds VRAM; connects LoRA grads), attach_block_loras (8a/8b).
//!   - flow-match predict (stage 1-3): add_noise -> packed -> MRoPE -> velocity.
//!   - loss = mean MSE in F32 (mean(), NOT mean_all() — grad-preserving).
//!   - lr via eridiffusion_core::training::levers::lr (the shared dispatch).
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin train_ideogram -- \
//!       --model <transformer.safetensors> --cache-dir <prepared> --steps 100

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use eridiffusion_core::models::ideogram;
use eridiffusion_core::models::ideogram_dit::IdeogramDit;
use eridiffusion_core::training::features::noise_modifiers;
use eridiffusion_core::training::progress::log_step_with_resume;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::training::{accumulate_parameter_grads, clip_parameter_grads};
use flame_core::gradient_clip::{GradientClipStrategy, GradientClipper};
use flame_core::{AutogradContext, DType, Shape, Tensor};

const GH_GW_CH: usize = 128;

#[derive(Parser)]
#[command(name = "train_ideogram", about = "Ideogram-4 LoRA trainer")]
struct Args {
    /// Transformer safetensors (fp8) — diffusion_pytorch_model.safetensors.
    #[arg(long)]
    model: PathBuf,
    /// prepare_ideogram cache dir (<stem>.safetensors with latent/text_embedding).
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long, default_value = "100")]
    steps: usize,
    #[arg(long, default_value = "16")]
    rank: usize,
    #[arg(long, default_value = "16.0")]
    lora_alpha: f32,
    #[arg(long, default_value = "1e-4")]
    lr: f32,
    #[arg(long, default_value = "0")]
    warmup_steps: usize,
    #[arg(long, default_value = "adamw")]
    optimizer: String,
    #[arg(long, default_value = "1.0")]
    max_grad_norm: f32,
    #[arg(long, default_value = "42")]
    seed: u64,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
}

/// velocity = -out[:, NT:].reshape(gh,gw,128).permute(0,3,1,2)
fn velocity(out: &Tensor, nt: usize, gh: usize, gw: usize) -> anyhow::Result<Tensor> {
    let nimg = gh * gw;
    Ok(out
        .narrow(1, nt, nimg)?
        .contiguous()?
        .reshape(&[1, gh, gw, GH_GW_CH])?
        .permute(&[0, 3, 1, 2])?
        .contiguous()?
        .mul_scalar(-1.0)?)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;
    let device = flame_core::global_cuda_device();

    // cache files
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("no .safetensors in cache dir {}", args.cache_dir.display());
    }
    log::info!("[1/3] {} cached samples", cache_files.len());

    log::info!("[2/3] loading Ideogram-4 transformer (resident) + LoRA rank {}", args.rank);
    let mut dit = IdeogramDit::load(&args.model.to_string_lossy(), device.clone())?;
    let mut params = dit.attach_block_loras(args.rank, args.lora_alpha)?;
    // v2 grad policy (BF16 grads stay BF16) — mirrors train_klein.
    for p in &mut params {
        p.set_grad_dtype_policy(flame_core::parameter::GradDtypePolicy::MatchParamDtype);
    }
    log::info!("    {} trainable LoRA params", params.len());

    let opt_kind = OptimizerKind::parse(&args.optimizer)
        .map_err(|e| anyhow::anyhow!("optimizer: {e}"))?;
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);
    let clipper = GradientClipper::new(GradientClipStrategy::ClipByNorm { max_norm: args.max_grad_norm });

    // base TrainConfig drives levers::lr (Constant default = warmup→flat).
    let mut cfg = eridiffusion_core::config::TrainConfig::default();
    cfg.learning_rate = args.lr as f64;

    log::info!("[3/3] training {} steps", args.steps);
    let t_start = Instant::now();
    for step in 0..args.steps {
        let cache_path = &cache_files[step % cache_files.len()];
        let sample = flame_core::serialization::load_file(cache_path, &device)?;
        let latent = sample
            .get("latent")
            .ok_or_else(|| anyhow::anyhow!("cache missing latent"))?
            .to_dtype(DType::F32)?; // [1,128,gh,gw]
        let text_emb = sample
            .get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cache missing text_embedding"))?;
        let ld = latent.shape().dims().to_vec();
        let (gh, gw) = (ld[2], ld[3]);

        // flow-match: t ~ logit-normal (sigmoid of seeded normal), as in SD3/flow.
        let u = noise_modifiers::randn_f32(Shape::from_dims(&[1]), device.clone())?
            .to_vec_f32()?[0];
        let t = 1.0 / (1.0 + (-u).exp()); // sigmoid → (0,1)
        let noise = noise_modifiers::randn_f32(latent.shape().clone(), device.clone())?;
        let noisy = ideogram::add_noise(&latent, &noise, t)?;
        let target = ideogram::flow_target(&noise, &latent)?; // noise - clean

        let packed = ideogram::build_packed_inputs(&noisy, text_emb, gh, gw, device.clone())?;
        let (cos, sin) = ideogram::build_mrope(
            &packed.position_ids,
            ideogram::HEAD_DIM,
            ideogram::MROPE_SECTION,
            ideogram::MROPE_THETA,
            device.clone(),
        )?;
        let seq = packed.x.shape().dims()[1];
        let nt = seq - gh * gw;
        let model_t = Tensor::from_vec(vec![1.0 - t], Shape::from_dims(&[1]), device.clone())?;
        let x_bf = packed.x.to_dtype(DType::BF16)?;
        let llm_bf = packed.llm_full.to_dtype(DType::BF16)?;

        let out = dit.forward(&x_bf, &llm_bf, &model_t, &packed.indicator, &cos, &sin, None, 0)?;
        let vel = velocity(&out, nt, gh, gw)?;
        let loss = vel.sub(&target)?.square()?.mean()?; // mean() preserves grad
        let loss_val = loss.to_vec_f32()?[0];

        let grads = AutogradContext::backward_v2(&loss)?;
        accumulate_parameter_grads(&params, &grads)?;
        let grad_norm = clip_parameter_grads(&params, &clipper)?;

        let cur_lr = eridiffusion_core::training::levers::lr(&cfg, args.lr, step, args.steps, args.warmup_steps);
        {
            let _g = AutogradContext::no_grad();
            opt.set_lr(cur_lr);
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        AutogradContext::clear();

        log_step_with_resume(
            "Ideogram-4", step, 0, args.steps, cache_files.len(), 1,
            loss_val, grad_norm, cur_lr, t_start, None,
        );
    }

    log::info!("Training complete: {} steps", args.steps);
    Ok(())
}

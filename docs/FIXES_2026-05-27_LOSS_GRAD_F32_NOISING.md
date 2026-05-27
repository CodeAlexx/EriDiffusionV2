# Fixes 2026-05-27: Loss/Gradient F32 Noising

## Problem

Klein and the other flow trainers set `flame_core::config::set_default_dtype(DType::BF16)` for model execution. The training noising path used `Tensor::randn`, which follows that global default dtype. Several trainers then built:

- sampled noise
- noisy model input
- supervised flow target

from BF16-rounded noise and BF16 latents before casting the loss operands back to F32.

That does not match OneTrainer. OT keeps latent/noise/noisy/target construction in F32, casts only the transformer input to the train dtype, and computes the MSE from F32 prediction and F32 target. Quantizing the target before the loss changes both the scalar loss and `dL/dpred`.

## Fix

- Added `noise_modifiers::randn_f32` and `randn_f32_seeded`, which upload F32 normal samples directly and bypass flame-core's BF16 global default.
- Updated the shared noise modifier helper so offset noise, input perturbation, and multires noise also sample in F32 before matching the caller dtype.
- Updated the main training paths to keep noising and target construction in F32, then cast only the model input to BF16:
  - Klein
  - Flux
  - Z-Image
  - Ernie
  - Chroma
  - Wan 2.2
  - SD3.5
  - Qwen-Image
  - Anima
  - ACE-Step
  - LTX2 video/audio
- Preserved the LoRA alpha export fix: exported adapter tensors now include `.alpha` metadata without making alpha a trainable parameter.

## Validation

Passed:

- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH cargo test -p eridiffusion-core --test lora_alpha_export -- --nocapture`
- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH cargo check -p eridiffusion-cli --bin train_klein --bin train_flux --bin train_zimage --bin train_ernie --bin train_chroma --bin train_wan22 --bin train_sd35 --bin train_qwenimage --bin train_anima --bin train_acestep --bin train_ltx2`
- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH cargo test --manifest-path /home/alex/EriDiffusion/flame-core/Cargo.toml --test lora_closed_loop_parity -- --nocapture`
- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH cargo test --manifest-path /home/alex/EriDiffusion/flame-core/Cargo.toml --test grad_norm_parity -- --nocapture`
- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH cargo test --manifest-path /home/alex/EriDiffusion/flame-core/Cargo.toml --test adam_torch_parity -- --nocapture`
- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH cargo build --release --bin train_klein`

GPU smoke:

```bash
RUST_LOG=info FLAME_ALLOC_POOL=0 FLAME_USE_STATIC_SLAB=0 FLAME_ASSERT_GRAD_FLOW=1 \
LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH \
./target/release/train_klein \
  --config configs/klein9b_alina.json \
  --cache-dir cache/alina_klein9b \
  --transformer /home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors \
  --steps 1 --rank 16 --lora-alpha 16 --lr 3e-5 \
  --batch-size 1 --warmup-steps 100 --offload \
  --sample-every 0 \
  --output-dir output/klein_f32target_smoke
```

Result:

- Completed 1 Klein 9B training step on GPU.
- `loss 0.9780`
- `grad_norm 0.0062`
- Wrote `output/klein_f32target_smoke/klein_lora_1steps.safetensors`.

## Notes

- This fixes the concrete OT mismatch in the loss/gradient target path. It does not claim that one step proves long-horizon 1500-2000 step Klein stability.
- Warnings seen during validation are existing workspace warnings.
- `rustfmt`/`cargo fmt` was not run.

# HiDream-O1 Trainer Status

**Last updated:** 2026-05-20
**State:** usable with scaled weights-only export; raw full-strength O1 LoRA export still needs deeper parity work.

## Current Launch Defaults

`train_hidream_o1` is the production O1 trainer in EDv2. Current defaults:

| Flag | Default | Why |
|---|---:|---|
| `--steps` | `768` | Long enough to show behavior without the old 2000-step overrun. |
| `--optimizer` | `adamw8bit` | Matches the reference O1 config; use `adamw` when testing full checkpoints. |
| `--loss-objective` | `x0` | Stable objective for the model's native clean-patch output. |
| `--export-scale` | `0.25` | Scales LoRA B tensors only when writing weights-only safetensors. |
| resident heads | on | Known-good O1 LoRAs include `x_embedder`, `t_embedder1`, and `final_layer2` heads. |
| `--max-loss` | `0.0` | Disabled; scalar clamps above the cap have zero derivative. |

Full-state saves are implemented only for `--optimizer adamw`. With any other optimizer, `--save-mode full` fails loudly instead of writing a partial checkpoint that only looks complete. Weights-only exports remain the normal path.

## O1 Model Facts

HiDream-O1 is a pixel-space image model. It has no VAE and no separate text encoder. The Qwen3-VL backbone consumes `input_ids` directly, and the image is represented as raw RGB patches:

```text
patches: [1, L, 3072]
L = (height / 32) * (width / 32)
pixel range = [-1, 1]
```

The model emits an x0-style clean-patch prediction. The reference velocity view is:

```text
x_t = (1 - sigma) * patches + sigma * (noise * 8)
target_velocity = noise * 8 - patches
pred_velocity = (x_t - x0_pred) / sigma
```

Therefore:

```text
pred_velocity - target_velocity = (patches - x0_pred) / sigma
velocity_loss = x0_error / sigma^2
```

This explains the observed low-sigma loss mountains. The spikes were not a proof of bad data or a memory leak; they are the mathematical weighting of the velocity objective.

## What Changed On 2026-05-20

- Added `--loss-objective x0|velocity`.
- Logged `hidream_o1/sigma`, `hidream_o1/loss_x0`, and `hidream_o1/loss_velocity` to SerenityBoard.
- Changed O1 LoRA training tensors to F32 and added an F32 LoRA residual path in Flame.
- Made resident-head LoRA targets the default; `--no-resident-lora` keeps the old transformer-only layout.
- Added `--export-scale`, default `0.25`, for weights-only LoRA exports.
- Weights-only safetensors metadata now says `ss_training_comment = edv2 trainer` and records `edv2.export_scale`.
- Full checkpoints keep raw in-memory weights so resume math is unchanged.

## Validation Results

Runs used `/home/alex/1/datasets/gigerver3` and HiDream-O1 Dev weights.

| Run | Result |
|---|---|
| x0, 768 steps, resident heads, F32 LoRA | Selected loss stable: avg `0.1093`, max `0.6438`, grad max `1.0359`, no clamps. Full-strength render was still flat purple/gray. |
| velocity, 256 steps, resident heads, F32 LoRA | Raw velocity spikes reproduced: max `4.0579`; diagnostic x0 stayed small: max `0.6446`. Full-strength render was flat gray. |
| same velocity LoRA, B tensors scaled by `0.25` at export | Render became clean and valid with the same Rust O1 sampler. |
| downloaded public O1 LoRA at full strength | Rendered valid with the same Rust O1 loader and sampler. |

The key artifact from the successful strength test is:

```text
/home/alex/EriDiffusion/inference-flame/output/o1_velocity_validation_20260520/lora_velocity_resident_256_Bscale025_seed42.png
```

The same unscaled checkpoint rendered flat gray:

```text
/home/alex/EriDiffusion/inference-flame/output/o1_velocity_validation_20260520/lora_velocity_resident_256_seed42.png
```

That means the O1 loader/sampler can apply a valid full-strength public LoRA, and our trained LoRA becomes usable when exported at the measured safe scale. This does not fully explain why raw EDV2 O1 weights are overpowered versus the public LoRA; it makes the trainer output usable while leaving a clear parity target.

## Memory And Speed Notes

The 768-step x0 run did not show a runaway host leak. Host RSS stayed around 15.4 GB. VRAM high-water moved from about 10.9 GB to about 12.1 GB and then plateaued.

O1 512 training measured around 3.1-3.5 s/step with flame-core block offload and boundary checkpointing. That is slower than Klein because O1 recomputes all 36 Qwen decoder blocks during backward. Retuning only the offloader window is not expected to reach Klein step time; future speed work should reduce checkpoint coverage where memory allows or add a no-recompute activation/sub-tape offload path.

## Reproduce Current Recommended Run

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2

LD_LIBRARY_PATH=/home/alex/libtorch-cu124/libtorch/lib:$LD_LIBRARY_PATH \
RUST_LOG=info FLAME_ASSERT_GRAD_FLOW=1 \
./target/release/train_hidream_o1 \
  --cache-dir cache/hidream_o1_gigerver3_512 \
  --model-path /home/alex/HiDream-O1-Image-Dev-weights \
  --output-dir output/hidream_o1_gigerver3 \
  --steps 768 \
  --rank 32 --lora-alpha 32 \
  --lr 1e-4 \
  --loss-objective x0 \
  --export-scale 0.25 \
  --save-mode weights
```

For strict reference debugging, use:

```bash
--loss-objective velocity --export-scale 0.25
```

## Remaining Risk

The practical fix is export-side scaling. The deeper unresolved issue is why raw trained EDV2 O1 deltas are too strong at full strength while a public O1 LoRA loads cleanly at full strength. Next parity work should compare raw per-layer deltas and sampler response against a known-good O1 LoRA, not rely on image quality alone.

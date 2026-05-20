# HiDream-O1 Project Status

**Updated:** 2026-05-20
**Phase:** O1 trainer launch fix landed; deeper raw-weight parity remains open.

## Scope

HiDream-O1 is the Open-1 pixel model, not HiDream-I1. It has no VAE and no external text encoder. EDV2 training and Flame inference both use the same HiDream-O1 Dev weights at:

```text
/home/alex/HiDream-O1-Image-Dev-weights
```

The production trainer is:

```text
/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_hidream_o1.rs
```

The inference path used for validation is:

```text
/home/alex/EriDiffusion/inference-flame/src/bin/hidream_o1_infer.rs
```

## Current Answer

The old O1 loss spikes came from the velocity objective:

```text
velocity_loss = x0_error / sigma^2
```

Low-sigma samples naturally create large velocity loss and grad-norm spikes even when the clean-patch x0 error is small. The trainer now logs both x0 and velocity losses so the board shows whether a spike is objective math or real x0 failure.

The current usable export path is:

```text
resident-head F32 LoRA training + weights-only export_scale=0.25
```

`--export-scale` scales B matrices only in the exported safetensors. Full checkpoints stay raw for resume.

## Validated Artifacts

| Artifact | Meaning |
|---|---|
| `output/hidream_o1_gigerver3_x0_resident_768_20260520/hidream_o1_lora_768steps.safetensors` | Stable x0 training metrics, but full-strength render was flat. |
| `output/hidream_o1_gigerver3_velocity_resident_256_20260520/hidream_o1_lora_256steps.safetensors` | Velocity spikes reproduced; unscaled full-strength render was flat. |
| `output/hidream_o1_gigerver3_velocity_resident_256_20260520/hidream_o1_lora_256steps_Bscale025.safetensors` | Manual precursor to the new `--export-scale 0.25` behavior; rendered valid. |
| `inference-flame/output/o1_velocity_validation_20260520/known_good_steampunk_gigver_prompt_seed42.png` | Public O1 LoRA rendered valid at full strength, so Flame O1 LoRA loading is not generally broken. |

## Code Decisions

- O1 LoRA tensors train in F32.
- Flame O1 LoRA residuals accept BF16 or F32 adapter tensors.
- Resident O1 heads are trained by default; `--no-resident-lora` is only for old transformer-only probes.
- Weights-only safetensors use EDV2 metadata and include `edv2.export_scale`.
- `--save-mode full` requires `--optimizer adamw`; other optimizers fail loudly.
- No in-trainer sampling yet. Render saved LoRAs with `hidream_o1_infer`.

## Flame-Core Dependency

O1 training uses the flame-core `BlockOffloader` path and `checkpoint_offload_boundary` around decoder blocks. Flame docs now record that this buys memory headroom but still recomputes the 36 Qwen decoder blocks during backward, so O1 should not be expected to match Klein speed until checkpoint coverage or activation offload changes.

## Next Work

1. Run a fresh 768-step O1 job with the new default `--export-scale 0.25`, then render the produced checkpoint directly instead of the manually scaled precursor.
2. Compare raw EDV2 O1 deltas against the public O1 LoRA per layer to explain why raw full-strength EDV2 output is overpowered.
3. Keep using x0 telemetry as the main stability signal and velocity telemetry as the parity/debug signal.

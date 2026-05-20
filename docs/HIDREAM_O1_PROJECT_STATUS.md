# HiDream-O1 Project Status

**Updated:** 2026-05-20
**Phase:** Structured-attention O1 validation in progress. The north star is valid LoRAs with speed beating, matching, or close to ai-toolkit.

## North Star / Hard Gates

Do not call HiDream-O1 training "fixed" just because the trainer runs, loss is
finite, or a short LoRA file saves. The two gates are:

1. **Valid LoRA gate**: train a real LoRA, then render a matched pair with
   Flame O1 inference: same seed, prompt, resolution, one image with no LoRA
   and one image with the trained LoRA. The LoRA image must stay clean and show
   the trigger-bound dataset style. For the active Giger run, the trigger is
   `gigver3`; do not rely on prompts that spell out Giger/biomechanical terms
   as the only proof.
2. **Speed gate**: measure EDV2 step time against the ai-toolkit O1 trainer
   reference. EDV2 does not need to copy every internal implementation detail,
   but a 3x gap is not acceptable unless the run is explicitly trading speed
   for memory through block streaming/checkpoint recompute. Record pure
   training s/step, setup time, and whether the model is resident or offloaded
   before comparing.

ai-toolkit is the behavior/config/speed reference. Exported EDV2 LoRA metadata
must still identify this trainer as `edv2 trainer` and must not write
ai-toolkit/aitoolkit strings into the weights.

Active validation run started 2026-05-20:

```text
output/hidream_o1_gigerver3_structured_800_20260520/
cache: cache/gigerver3_hidream_o1_512_mropefix
command: train_hidream_o1 --steps 800 --lr 2e-4 --save-every 200 --lora-stats-every 25
post-run required: 1024x1024 no-LoRA vs LoRA render with trigger prompt
```

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

The 800-step validation proved the spike diagnosis directly: max velocity loss was `12.0565`, but the corresponding x0 loss was only `0.008483` at sigma `0.0265`. The exported LoRA still rendered a clean image through Flame O1 inference.

## Validated Artifacts

| Artifact | Meaning |
|---|---|
| `output/hidream_o1_gigerver3_x0_resident_768_20260520/hidream_o1_lora_768steps.safetensors` | Stable x0 training metrics, but full-strength render was flat. |
| `output/hidream_o1_gigerver3_velocity_resident_256_validrun_20260520/hidream_o1_lora_256steps.safetensors` | Fresh committed-trainer run with `--export-scale 0.25`; rendered valid. |
| `output/hidream_o1_gigerver3_velocity_resident_800_rerun_20260520/hidream_o1_lora_800steps.safetensors` | 800-step committed-trainer run with `--export-scale 0.25`; rendered valid. |
| `inference-flame/output/o1_validrun_20260520/velocity_256_exportscale025_seed42.png` | Clean render from the fresh committed-trainer LoRA. |
| `inference-flame/output/o1_800_rerun_20260520/velocity_800_exportscale025_seed42.png` | Clean render from the 800-step committed-trainer LoRA. |
| `inference-flame/output/o1_velocity_validation_20260520/known_good_steampunk_gigver_prompt_seed42.png` | Public O1 LoRA rendered valid at full strength, so Flame O1 LoRA loading is not generally broken. |

Fresh 800-step validation hashes:

```text
LoRA: 2b8e9e938ff8495fe52341899bc0fb0e8e96c1595b81afbded69ce67a28e5df7
PNG:  cd0d31351db4b4dcc7048bc78ffbebef9c67e69499692ddd398fed1fd211b54e
```

The LoRA metadata has `ss_training_comment = edv2 trainer`, `edv2.export_scale = 0.25`, and no `ai-toolkit` / `aitoolkit` strings.

## Code Decisions

- O1 LoRA tensors train in F32.
- Flame O1 LoRA residuals accept BF16 or F32 adapter tensors.
- Resident O1 heads are trained by default; `--no-resident-lora` is only for old transformer-only probes.
- Weights-only safetensors use EDV2 metadata and include `edv2.export_scale`.
- `--save-mode full` requires `--optimizer adamw`; other optimizers fail loudly.
- No in-trainer sampling yet. Render saved LoRAs with `hidream_o1_infer`.

## Flame-Core Dependency

O1 training uses the flame-core `BlockOffloader` path and `checkpoint_offload_boundary` around decoder blocks. Flame docs now record that this buys memory headroom but still recomputes the 36 Qwen decoder blocks during backward, so O1 should not be expected to match Klein speed until checkpoint coverage or activation offload changes.

The observed EDV2 O1 rate is about 3.1 s/step at 512 after warmup. The 800-step run took `41:34`, with host RSS flat around 15.4 GB and VRAM plateaued at `11202 MiB`. AI-toolkit's O1 implementation uses PyTorch/HF checkpointing and, when not in `low_vram` layer-offload mode, keeps the transformer resident on GPU. A near-1 s/step AI-toolkit O1 run is therefore not apples-to-apples against EDV2's conservative BF16 block-streaming path.

Speed probe, 2026-05-20:

```text
FLAME_LOG_SDPA_BWD=1 train_hidream_o1 --steps 3 ...
108 [sdpa-bwd] bail:mask-present
```

This means every O1 decoder layer hit flame-core's masked SDPA backward fallback for the causal AR/text pass. The fallback is slower but should still be mathematically valid; it is not evidence that Giger style failed to train.

## Style Strength Notes

The 800-step Giger LoRA rendered cleanly but weakly. The speed issue above does not explain weak style application. More likely suspects:

- The validated LoRA was exported with `--export-scale 0.25`, so inference applies one-quarter of the trained delta.
- The test prompt used `gigver3` but also a long list of competing style/content tokens. That can bury a weak trigger.
- The dataset captions are mostly trigger-tagged (`gigver3` in 67/70 text files), but the captions are long and 50/70 also spell out `Giger`, `Alien`, or biomechanical terms. Trigger binding is therefore not proven by one busy prompt.
- The 800-step validation run used the velocity objective for parity/stress testing. x0 remains the cleaner production objective because velocity weights low-sigma samples harder.

A useful next debug is a no-training scale sweep: re-export or rescale the 800-step LoRA to effective strengths `0.25`, `0.5`, `1.0`, and `2.0`, then render a simple `gigver3, portrait...` prompt and the same prompt without LoRA. If full-strength shows style but artifacts, the issue is strength/stability. If full-strength still lacks style, debug captions/objective/training.

## Next Work

1. Compare raw EDV2 O1 deltas against the public O1 LoRA per layer to explain why raw full-strength EDV2 output is overpowered.
2. Add an O1 LoRA scale-sweep export path or inference runtime scale so style strength can be tested without retraining.
3. Add a high-memory O1 speed probe that reduces checkpoint coverage where VRAM allows.
4. Keep using x0 telemetry as the main stability signal and velocity telemetry as the parity/debug signal.

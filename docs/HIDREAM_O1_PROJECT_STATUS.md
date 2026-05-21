# HiDream-O1 Project Status

**Updated:** 2026-05-20
**Phase:** parity-first O1 validation. The north star is a valid LoRA, not just
finite loss or a saved safetensors file.

## Hard Gates

1. **Exact fixed-input gates first.** Compare ai-toolkit and EDV2 on the same
   cached sample, same noise, same sigma, same target, same BF16 velocity
   conversion, and same scalar loss before judging stochastic training runs.
2. **Valid LoRA gate.** Train a real LoRA, then render a matched no-LoRA and
   with-LoRA pair in Flame O1 inference using the same seed, prompt, resolution,
   and sampler. The LoRA image must stay clean and show trigger-bound dataset
   style. For the active Giger dataset, the trigger is `gigver3`; do not prove
   style by spelling out Giger/biomechanical terms in the prompt.
3. **Speed sanity.** Record step time and memory, but do not let speed obscure
   the validity gate. A fast broken LoRA is still broken.

ai-toolkit is the behavior reference for HiDream-O1. EDV2 saved LoRAs must keep
EDV2 metadata (`ss_training_comment = edv2 trainer`) and must not write
`ai-toolkit` / `aitoolkit` strings into EDV2 weights.

## Current Defaults

`train_hidream_o1` production defaults are now aligned to ai-toolkit's O1
training contract:

```text
loss objective:      velocity
LoRA surface:        ai-toolkit/public O1 surface, 257 adapters
rank/alpha:          32/32
optimizer:           AdamW8bit
export scale:        1.0
resident O1 heads:   on by default
```

The 257-adapter surface is 252 Qwen language-layer adapters plus these five
O1 heads:

```text
diffusion_model.x_embedder.proj1
diffusion_model.x_embedder.proj2
diffusion_model.t_embedder1.mlp.0
diffusion_model.t_embedder1.mlp.2
diffusion_model.final_layer2.linear
```

Use `--no-resident-lora` only for transformer-only ablations. It is not the
ai-toolkit O1 public-LoRA surface.

## Exact Parity Evidence

The new fixed training-step parity fixture is:

```text
tests/parity/hidream_o1_train_step_ref.py
crates/eridiffusion-cli/src/bin/parity_hidream_o1_train_step.rs
```

It pins the first cached Giger sample, `seed=4242`, `sigma=0.5`, scaled noise
`noise * 8`, ai-toolkit's training-style `use_flash_attn=True` forward, the
BF16-rounded velocity prediction, and scalar MSE.

Current result:

```text
ai-toolkit loss_velocity = 0.386822551
EDV2 loss_velocity       = 0.386835128
relative diff            = 3.25e-5
```

The direct `x_pred_rows` element max differs slightly more than a strict
elementwise cap because PyTorch and Flame use different BF16 attention kernels,
but the target conversion and scalar objective match tightly enough to rule out
the old data/noise/timestep/loss mismatch.

The existing G0/G1 gates still matter:

```text
G0: Python vs Flame base forward parity
G1: Flame no-LoRA forward vs Flame zero-LoRA forward self-consistency
```

## Active Validation Run

Public-style resident-head run started 2026-05-20:

```bash
LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
RUST_LOG=info \
target/release/train_hidream_o1 \
  --cache-dir cache/gigerver3_hidream_o1_512_mropefix \
  --steps 1000 \
  --save-every 500 \
  --lora-stats-every 50 \
  --output-dir output/hidream_o1_gigerver3_resident_1000_20260520 \
  --export-scale 1.0
```

Render both no-LoRA and with-LoRA at 1024x1024 using:

```text
gigver3, Male anime character centered, oni mask, glitch art, glitchcore, organic, forest druid, dark souls boss, cyber punk, hellscape, portrait, male anime character, robot, masterpiece, intricate, highly detailed, sharp, technological rings
```

Keep the user's longer prompt available for the final comparison, but the short
trigger prompt is the cleaner style-binding test.

## Stability Notes

HiDream-O1 emits clean-patch `x0`. ai-toolkit trains velocity:

```text
x_t = (1 - sigma) * patches + sigma * (noise * 8)
target_velocity = noise * 8 - patches
pred_velocity = (x_t - x0_pred) / sigma
velocity_loss = mse(pred_velocity, target_velocity)
```

Low-sigma samples amplify x0 error by `1 / sigma^2`, so velocity loss spikes
are expected. EDV2 logs both `hidream_o1/loss_x0` and
`hidream_o1/loss_velocity`; use x0 telemetry to distinguish real model failure
from velocity weighting.

## Speed Notes

O1 uses the same flame-core resident-set `BlockOffloader` and structured
prefix-causal/full attention path as the current trainer stack. The full image
token pass is now on the fast two-pass path, but O1 still wraps all 36 Qwen
decoder blocks in `checkpoint_offload_boundary`, so backward recomputes the
decoder. Speed work after validity should reduce checkpoint coverage when VRAM
allows or add a true no-recompute activation/sub-tape offload path.

## Structural Fixes Landed 2026-05-20 (Uncommitted)

Three structural correctness fixes landed in the working tree across
flame-core / inference-flame / EriDiffusion-v2. Full inventory in
[`HANDOFF_2026-05-20_HIDREAM_O1_PARITY.md`](./HANDOFF_2026-05-20_HIDREAM_O1_PARITY.md)
§"2026-05-20 Evening Update".

1. **`Op::RoPePrecomputed` now carries an explicit
   `autograd::RopeLayout` tag** (`Interleaved` / `Halfsplit`).
   Replaces the backward shape-sniffer that mis-classified HiDream-O1
   MRoPE cos `[1, S, half]` (rank-3 but Halfsplit forward) as
   Interleaved — a Q/K LoRA-B grad-direction silent corrupter.
   Forward sites in `flame-core/src/bf16_ops.rs` and trainer-level
   call sites (`EriDiffusion-v2/.../chroma.rs:2332`) now pass the
   correct tag.

2. **`fused_linear3d_native_pytorch_parity`** — new bit-exact PyTorch
   `gemm_and_bias<BF16>` mirror, ~1% perf overhead. HiDream-O1's
   `timestep_embedder` and `bottleneck_patch_embed` no-LoRA paths now
   use it. Result: `pre.t_emb` is bit-exact; `pre.patch_emb` mean_abs
   dropped from 3e-3 to 5.7e-5 (53× improvement).

3. **`AutogradContext::retain_intermediate_grads_add`** — additive
   retain-set API so probe IDs registered *during* checkpoint
   recompute are honored by the sub-tape walk. Enables the soul.md
   trap meta-pattern under gradient checkpointing.

Next-step bisect uncovered that the localized trap-probe site
("dV out of SDPA bwd cos = 0.012") is **autograd-clean in
single-layer isolation** — see
`flame-core/tests/sdpa_prefix_causal_full_grad.rs` reproducing the
exact HiDream-O1 attention chain at exact shapes with cos = 1.0 /
max_abs = 0. The end-to-end cos gap is either a parity-comparison
artifact (Python forward-hooks vs Rust TensorId capture) or a
multi-layer cascade. Structural fixes above are correctness wins
regardless and stay.

## Next Work

1. Finish the resident-head validation run kicked off 2026-05-20 and
   render no-LoRA / with-LoRA samples.
2. If LoRA validity holds, commit all three repos with the structural
   fixes from 2026-05-20.
3. If style is weak at full strength, add a fixed-input LoRA
   backward/update parity gate against ai-toolkit and re-run the
   per-layer parity sweep past `hidden_input_layer_00` end-to-end.
4. Commit only after docs, parity gates, and at least one valid sample
   pair are recorded.

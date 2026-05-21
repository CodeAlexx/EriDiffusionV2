# HiDream-O1 Trainer Status

**Last updated:** 2026-05-20
**State:** parity defaults corrected; real LoRA validity run in progress.

## Production Defaults

`train_hidream_o1` is EDV2's production HiDream-O1 trainer. Current defaults:

| Flag | Default | Reason |
|---|---:|---|
| `--steps` | `768` | Conservative CLI default; validation runs can override. |
| `--rank` / `--lora-alpha` | `32` / `32` | Matches ai-toolkit O1 config. |
| `--optimizer` | `adamw8bit` | Matches ai-toolkit O1 config. |
| `--loss-objective` | `velocity` | Matches ai-toolkit O1 `get_loss_target` / `predict_noise` path. |
| `--export-scale` | `1.0` | No hidden strength scaling in production LoRA exports. |
| resident O1 heads | on | Matches ai-toolkit/public O1 `transformer_only=false` surface. |
| `--max-loss` | `0.0` | Disabled; clamping hides real failures and gives zero derivative above cap. |

The default LoRA surface is 257 adapters: 252 Qwen language-layer adapters plus
`x_embedder`, `t_embedder1`, and `final_layer2`. `--include-resident-lora` is
kept as a compatibility no-op. `--no-resident-lora` is only a transformer-only
ablation and should not be used for public-LoRA parity.

Full-state saves are implemented only for `--optimizer adamw`. With any other
optimizer, `--save-mode full` fails loudly instead of writing a partial
checkpoint. Weights-only exports remain the normal path.

## O1 Model Facts

HiDream-O1 is a pixel-space image model. It has no VAE and no separate text
encoder. The Qwen3-VL backbone consumes `input_ids` directly, and image data is
raw RGB patches:

```text
patches: [1, L, 3072]
L = (height / 32) * (width / 32)
pixel range = [-1, 1]
```

The model emits x0-style clean-patch predictions. ai-toolkit converts that to a
velocity prediction for loss:

```text
x_t = (1 - sigma) * patches + sigma * (noise * 8)
target_velocity = noise * 8 - patches
pred_velocity = (x_t - x0_pred) / sigma
```

The BF16 round-trip matters: ai-toolkit computes the velocity prediction in F32,
casts back to the model dtype, then computes MSE in F32. EDV2 mirrors that.

## Parity Gates

Fixed-input training-step parity:

```bash
EDV2_REFERENCE_ROOT=/home/alex/ai-toolkit \
/home/alex/ai-toolkit/venv/bin/python tests/parity/hidream_o1_train_step_ref.py \
  --use-flash-attn --t-scalar 0.5 --seed 4242

LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
RUST_LOG=info \
target/release/parity_hidream_o1_train_step \
  --ref-path /tmp/hidream_o1_train_step_ref.safetensors
```

Observed scalar loss:

```text
ai-toolkit loss_velocity = 0.386822551
EDV2 loss_velocity       = 0.386835128
relative diff            = 3.25e-5
```

This gate proves the cached sample, scaled noise, timestep inversion, velocity
target, BF16 velocity conversion, and scalar loss are aligned. It does not prove
multi-step LoRA validity by itself.

## Validation Run

Current public-style validation run:

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

The dataset trigger is `gigver3`. A proper validity sample pair must render the
same prompt with no LoRA and with the trained LoRA at 1024x1024.

## Memory And Speed

O1 training uses flame-core `BlockOffloader` plus decoder
`checkpoint_offload_boundary`. This keeps VRAM within 24 GB but recomputes all
36 Qwen decoder blocks in backward. The structured two-pass attention path fixed
the old materialized-mask slowdown, but a resident-head 512 run can still sit
around `2.5-3.0 s/step` on this machine.

If ai-toolkit measures near `1 s/step`, it is likely using a more resident
PyTorch path. Future speed work should reduce checkpoint coverage when memory
allows or add a no-recompute activation/sub-tape offload path.

## Remaining Risk

The next unknown is not the scalar objective; fixed-step loss parity is already
tight. If the 1000-step run does not produce a trigger-bound style LoRA, inspect
LoRA backward/update parity and sampler LoRA scale response against ai-toolkit.

## 2026-05-20 Structural Fixes (uncommitted)

Three correctness fixes landed in the working tree across flame-core,
inference-flame, and EriDiffusion-v2. The trainer build still passes after the
swap. Full inventory: [`HANDOFF_2026-05-20_HIDREAM_O1_PARITY.md`](./HANDOFF_2026-05-20_HIDREAM_O1_PARITY.md)
§"2026-05-20 Evening Update".

- `Op::RoPePrecomputed` autograd backward now dispatches by an explicit
  `autograd::RopeLayout` tag (`Interleaved` / `Halfsplit`), not by sniffing
  the saved cos tensor's shape. Fixes the HiDream-O1 MRoPE case where rank-3
  cos `[1, S, half]` was mis-classified — that was the Q/K LoRA-B
  grad-direction corruption signature.
- `inference-flame` HiDream-O1 `timestep_embedder` and `bottleneck_patch_embed`
  no-LoRA paths now call `fused_linear3d_native_pytorch_parity`. `pre.t_emb`
  is bit-exact; `pre.patch_emb` mean_abs is now `5.7e-5` (was `3e-3`).
- `AutogradContext::retain_intermediate_grads_add` enables intermediate-grad
  probes registered *during* checkpoint recompute (the outer-tape snapshot
  already fired). Used by the soul.md trap pattern in HiDream-O1 layer 35.

`tests/parity/hidream_o1_train_step_ref.py` and
`parity_hidream_o1_train_step` (`--lora-step` mode) capture per-layer attention
intermediate gradients for parity comparison. The structural fix chain is
**bit-exact in single-layer isolation** per `flame-core/tests/sdpa_prefix_causal_full_grad.rs`
(cos = 1.0, max_abs = 0 at HiDream-O1 shapes). The end-to-end cos = 0.012 at
`v_post_repeat_kv` is therefore not single-layer corruption — likely a
parity-comparison artifact between Python forward-hook capture and Rust
TensorId capture, to be reinvestigated.

The structural fixes stay regardless; they were warranted by audit and remove
real silent-corruption hazards.

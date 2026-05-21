# HiDream-O1 LoRA Parity Handoff

Date: 2026-05-20

## Goal

Make EDV2 HiDream-O1 training produce valid LoRAs. Speed is now secondary to correctness, but the target after correctness is near ai-toolkit speed or better.

The current work is not complete. Do not call the trainer fixed until an EDV2 LoRA is trained, exported, sampled at 1024, and shown to respond to the trigger cleanly.

## Hard Constraints

- Do not touch Lance.
- Do not run `rustfmt` or `cargo fmt`.
- Do not keep accepting mismatches and moving deeper. Fix the first mismatch in the parity chain, then rerun.
- Do not add loss clamps or special low-sigma behavior unless ai-toolkit does it in its training path.
- Keep O1 LoRA surface at ai-toolkit parity unless a later parity result proves otherwise:
  - 252 adapters.
  - rank 32.
  - alpha 32.
  - A/down Kaiming init.
  - B/up zero init.
  - no resident heads.
- Exported EDV2 LoRA metadata must contain `ss_training_comment=edv2 trainer`.
- Exported EDV2 LoRA metadata must not contain `ai-toolkit` or `aitoolkit`.

## Main Repos

- Production trainer repo: `/home/alex/EriDiffusion/EriDiffusion-v2`
- Inference/model implementation repo: `/home/alex/EriDiffusion/inference-flame`
- Flame core repo: `/home/alex/EriDiffusion/flame-core`
- Reference trainer repo: `/home/alex/ai-toolkit`
- Python reference interpreter: `/home/alex/ai-toolkit/venv/bin/python`

## Current Status

The strict O1 train-step parity harness exists and builds, but parity is still failing before the decoder stack. The current first failure is predecoder assembly, specifically `hidden_input_layer_00` before transformer layer 0.

Last known strict failure:

```text
hidden_input_layer_00:
  cos ~= 0.9999965
  max_abs ~= 0.5
  mean_abs ~= 0.0014
  rel ~= 0.00495
  FAIL
```

Important: even though this looks numerically close by cosine, it is still the first mismatch and must be fixed before chasing later LoRA gradient failures.

Earlier full `--lora-step` diagnostics, when allowed to continue past earlier failures, also showed attention Q/K/V LoRA B gradient collapse. That is not the next repair target until predecoder parity passes.

## Files Added Or Changed

### EDV2

Repo: `/home/alex/EriDiffusion/EriDiffusion-v2`

Added:

- `tests/parity/hidream_o1_train_step_ref.py`
- `crates/eridiffusion-cli/src/bin/parity_hidream_o1_train_step.rs`
- this handoff file

Modified before this handoff:

- `README.md`
- `crates/eridiffusion-cli/src/bin/train_hidream_o1.rs`
- `docs/HIDREAM_O1_PROJECT_STATUS.md`
- `docs/hidream_o1_trainer_status.md`

Current EDV2 `git status --short` at handoff time:

```text
 M README.md
 M crates/eridiffusion-cli/src/bin/train_hidream_o1.rs
 M docs/HIDREAM_O1_PROJECT_STATUS.md
 M docs/hidream_o1_trainer_status.md
?? crates/eridiffusion-cli/src/bin/parity_hidream_o1_train_step.rs
?? tests/parity/__pycache__/
?? tests/parity/hidream_o1_train_step_ref.py
```

Remove `tests/parity/__pycache__/` before committing.

### inference-flame

Repo: `/home/alex/EriDiffusion/inference-flame`

Modified:

- `src/models/hidream_o1/model.rs`
- `src/models/hidream_o1/bottleneck_patch_embed.rs`
- `src/models/hidream_o1/timestep_embedder.rs`
- `src/models/hidream_o1/weight_loader.rs`
- `src/models/hidream_o1/lora.rs`

Current `git status --short` at handoff time:

```text
 M src/models/hidream_o1/bottleneck_patch_embed.rs
 M src/models/hidream_o1/lora.rs
 M src/models/hidream_o1/model.rs
 M src/models/hidream_o1/timestep_embedder.rs
 M src/models/hidream_o1/weight_loader.rs
?? acestep_output.wav
?? inference-flame/
?? output.png
?? ports/
?? tools/__pycache__/
```

The untracked files in this repo look unrelated. Do not delete them without checking ownership.

### flame-core

Repo: `/home/alex/EriDiffusion/flame-core`

Modified:

- `docs/AUTOGRAD_SPEED_GUIDE.md`

Added:

- `tests/lora_matmul_backward.rs`

Current `git status --short` at handoff time:

```text
 M docs/AUTOGRAD_SPEED_GUIDE.md
?? tests/lora_matmul_backward.rs
```

## Implemented Parity Harness

### Python Reference

Path:

```text
/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/hidream_o1_train_step_ref.py
```

Current capabilities:

- `--lora-step`
- `--dump-layers`
- dumps `x_pred_full`
- dumps `hidden_input_layer_00`
- dumps `hidden_layer_00` through `hidden_layer_35`
- dumps `hidden_final_norm`
- dumps LoRA mid-gradient tensors when `--lora-step` is used
- dumps predecoder tensors when `--dump-layers` is used:
  - `pre.text_emb`
  - `pre.t_emb`
  - `pre.text_emb_with_t`
  - `pre.patch_emb`
  - `pre.inputs_embeds`
- `_registry_key_from_lora_name` was fixed to strip `language_model.`

The predecoder dumps were added so Rust can isolate the first failure into one of:

- timestep embedding
- TMS token replacement
- patch embedding
- concatenation/input assembly

### Rust Replay

Path:

```text
/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/parity_hidream_o1_train_step.rs
```

Current capabilities:

- default O1 LoRA surface: 252 adapters, rank 32, alpha 32, no resident heads
- strict first-failure comparisons
- per-layer forward comparisons
- loss comparison
- LoRA init comparison
- mid-gradient comparison
- global grad norm comparison
- post-step LoRA weight comparison
- `--predecoder-only`
- `--per-layer-dump`

Current strict thresholds:

```text
min_cos = 0.99999
max_abs = 0.005
max_rel = 0.01
max_loss_rel = 1e-5
```

The `--predecoder-only` mode avoids loading the full model. It loads only selected predecoder weights and uses the Python-dumped `pre.text_emb` as text input, so it can focus on the currently failing area.

## Commands Already Verified

Python reference syntax:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
/home/alex/ai-toolkit/venv/bin/python -m py_compile tests/parity/hidream_o1_train_step_ref.py
```

Rust parity binary build:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
  cargo check --bin parity_hidream_o1_train_step
```

Last result:

```text
Finished `dev` profile [unoptimized + debuginfo]
```

## Reference Artifact Commands

Generate moderate-sigma predecoder/full-forward reference:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
EDV2_REFERENCE_ROOT=/home/alex/ai-toolkit \
  /home/alex/ai-toolkit/venv/bin/python tests/parity/hidream_o1_train_step_ref.py \
  --t-scalar 0.5 \
  --seed 4242 \
  --dump-layers \
  --out /tmp/hidream_o1_train_step_ref_t05_pre.safetensors \
  --meta /tmp/hidream_o1_train_step_ref_t05_pre_meta.json
```

Last known result:

```text
model loaded in 6.9s
forward done 0.69s
loss_velocity 0.386822551
wrote /tmp/hidream_o1_train_step_ref_t05_pre.safetensors
```

Run Rust predecoder-only replay:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
  RUST_LOG=info \
  target/debug/parity_hidream_o1_train_step \
  --predecoder-only \
  --ref-path /tmp/hidream_o1_train_step_ref_t05_pre.safetensors
```

This command had not been rerun after the fast selected-weight loader patch at the time this handoff was written.

Run full strict LoRA-step replay after predecoder passes:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
  RUST_LOG=info \
  target/debug/parity_hidream_o1_train_step \
  --lora-step \
  --ref-path /tmp/hidream_o1_train_step_ref_t05_pre.safetensors
```

Also run low-sigma parity after moderate sigma passes:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
EDV2_REFERENCE_ROOT=/home/alex/ai-toolkit \
  /home/alex/ai-toolkit/venv/bin/python tests/parity/hidream_o1_train_step_ref.py \
  --t-scalar 0.05 \
  --seed 4242 \
  --dump-layers \
  --lora-step \
  --out /tmp/hidream_o1_train_step_ref_t005.safetensors \
  --meta /tmp/hidream_o1_train_step_ref_t005_meta.json

LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
  RUST_LOG=info \
  target/debug/parity_hidream_o1_train_step \
  --lora-step \
  --ref-path /tmp/hidream_o1_train_step_ref_t005.safetensors
```

## Recent Code Changes In inference-flame

### `model.rs`

`scatter_tms_token` was changed from selector matmul replacement to `Tensor::where_mask` replacement.

Reason:

ai-toolkit/PyTorch uses a `torch.where` style path for the TMS token replacement. Selector matmul is algebraically similar, but it creates a different numerical path and was suspected in the first predecoder mismatch.

Current intent:

```text
build [B,S,H] mask
expand t_emb to [B,S,H]
where mask is true, use t_emb
otherwise, use text_emb
```

This compiles.

### `weight_loader.rs`

Added:

- `load_resident_weights_bf16`
- `load_selected_resident_weights_bf16`

The selected loader was rewritten to avoid loading huge unused tensors during `--predecoder-only`. It now groups requested keys by safetensor shard and reads only the requested weights.

It also now uses raw byte copy for F32 and BF16 safetensor reads, then converts to BF16 on GPU where needed. F16 still uses the slower conversion loop.

Reason:

The first selected-loader version called the broader CPU resident loader and made predecoder-only replay hang or run far too slowly.

Killed stale slow runs:

```text
561503
562225
562616
```

### `bottleneck_patch_embed.rs`

Current code uses Flame fused 3D linear helpers for `proj1` and `proj2`, with LoRA and non-LoRA paths.

This may still be wrong if `pre.patch_emb` is the first failing predecoder tensor. Do not assume this is correct until the predecoder component comparison passes.

### `timestep_embedder.rs`

Current code uses Flame fused 3D linear helpers for `mlp.0` and `mlp.2`.

Potential unresolved issue:

The sinusoidal timestep embedding is still built on CPU, then converted. If `pre.t_emb` fails first, check this path against PyTorch exactly, including dtype, device math, frequency construction, and linear input dtype.

### `lora.rs`

Metadata path was adjusted earlier so exported LoRAs report EDV2 ownership. Before any final claim, inspect actual exported safetensors metadata and verify:

```text
ss_training_comment=edv2 trainer
```

and no:

```text
ai-toolkit
aitoolkit
```

## Current First Repair Step

Run:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
  RUST_LOG=info \
  target/debug/parity_hidream_o1_train_step \
  --predecoder-only \
  --ref-path /tmp/hidream_o1_train_step_ref_t05_pre.safetensors
```

Then fix the first failed tensor in this order:

1. If `pre.t_emb` fails, repair `TimestepEmbedder`.
2. If `pre.text_emb_with_t` fails while `pre.t_emb` passes, repair TMS token replacement.
3. If `pre.patch_emb` fails, repair `BottleneckPatchEmbed` or the selected weight load for patch embed weights.
4. If `pre.inputs_embeds` fails while components pass, repair concatenation/order/shape/dtype assembly.
5. If all `pre.*` tensors pass but `hidden_input_layer_00` fails, inspect the exact tensor used by full model entry versus predecoder replay.

Do not debug decoder layer 8, layer 33, optimizer, or sampler until this passes.

## Later Parity Chain

After predecoder parity passes, rerun the full strict `--lora-step` at `t_scalar=0.5`.

Fix in this order:

1. forward tensor mismatch
2. loss mismatch
3. mid-gradient mismatch
4. global grad norm mismatch
5. post-clip grad mismatch
6. AdamW8bit/post-step LoRA weight mismatch

Only after moderate sigma passes, rerun at `t_scalar=0.05`. Low-sigma velocity loss can be large, but gradients and update must still match ai-toolkit.

## Known Later Failure From Earlier Diagnostics

Earlier full diagnostics showed attention Q/K/V LoRA B gradients collapsing while MLP and output projection LoRA gradients were much closer to reference.

Approximate old observation:

```text
q/k/v LoRA B grad cos ~= 0.01 to 0.05
MLP/o_proj grad cos ~= near parity
Rust global grad norm ~= 0.246
Reference global grad norm ~= 0.272
```

Do not treat this as the current first bug. It is a later bug after predecoder/forward/loss parity.

## Training Proof Still Required

After train-step parity passes:

1. Run 100-step EDV2 O1 LoRA training.
2. Save the LoRA.
3. Sample at 1024 without LoRA.
4. Sample at 1024 with LoRA.
5. Use the user prompt with trigger:

```text
[triggerword], Male anime character centered, oni mask, glitch art, glitchcore, organic, forest druid, dark souls boss, cyber punk, hellscape, portrait, male anime character, robot, masterpiece, intricate, highly detailed, sharp, technological rings, by james mccarthy, glowing blue lush seascape bioluminescent, by beeple and johfra bosschart, combination in the style ayami kojima, highly detailed, painting, 3 d render beeple, unreal engine render, intricate abstract, intricate artwork, by tooth wu, wlop, beeple, dan mumford. concept art, octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork. cinematic, key art, hyper realism, high detail, octane render, 8 k, iridescent accents, albedo from overlord, the library of gems, intricate abstract. intricate artwork, by tooth wu, wlop, beeple, dan mumford. concept art, octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork. cinematic, key art, hyper realism, high detail, octane render, 8 k, iridescent accents
```

6. Compare against an ai-toolkit 100-step LoRA sample with the same prompt and seed.
7. If clean and trigger responds, run 800 steps.
8. Save midpoint and final LoRA.
9. Sample midpoint and final LoRA.

Dataset requested by user:

```text
/home/alex/1/datasets/gigerver3
```

Sample prompt rule:

- Use only the trigger for style learning prompts.
- Do not include explicit Giger references in sample prompts when testing learned trigger behavior.

## Speed Notes

Before the parity stop, EDV2 O1 speed was brought down to roughly acceptable range around `1.4 s/step` in the observed run. ai-toolkit was still faster around `1 s/step`, but the current priority is valid LoRAs.

Do not optimize speed again until parity proves the LoRA update is correct.

## Output Cleanup Needed

User requested cleanup of stale HiDream output. Do this carefully after checking exact output dirs. Do not delete unrelated user artifacts.

Known generated parity artifacts under `/tmp`:

```text
/tmp/hidream_o1_train_step_ref_t05_pre.safetensors
/tmp/hidream_o1_train_step_ref_t05_pre_meta.json
```

These are disposable but useful for immediate parity replay.

## Logging Cleanup Needed

User objected to this output string:

```text
OT-style layer offload fraction=...
```

Rename it to neutral EDV2/Flame wording later. Likely source:

```text
/home/alex/EriDiffusion/flame-core/src/offload/mod.rs
```

Do not let that distract from the first parity mismatch.

## Documentation Updates Still Needed

After real fixes, update:

- EDV2 trainer docs.
- Flame API/core docs.
- Any HiDream-O1 status docs that currently imply parity or validity before it is proven.

The docs should say:

- exact parity test command used
- exact training proof command used
- sample locations
- metadata verification result
- speed result
- remaining known gaps, if any

## Commit Checklist

Before committing:

1. Remove `tests/parity/__pycache__/`.
2. Check all three repos:

```bash
git -C /home/alex/EriDiffusion/EriDiffusion-v2 status --short
git -C /home/alex/EriDiffusion/inference-flame status --short
git -C /home/alex/EriDiffusion/flame-core status --short
```

3. Run at minimum:

```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:$LD_LIBRARY_PATH \
  cargo check --bin parity_hidream_o1_train_step --bin train_hidream_o1
```

4. Do not run formatting.
5. Verify no exported LoRA metadata contains `ai-toolkit` or `aitoolkit`.
6. Only claim trainer validity after the 100-step LoRA proof and 1024 samples.

---

## 2026-05-20 Evening Update — Structural Fixes Landed, Bisect Negative

This section captures the three repos' uncommitted state at the end of
the 2026-05-20 work session. All changes are in the working tree (none
pushed yet).

### What was shipped

**flame-core**:

1. `pub enum autograd::RopeLayout { Interleaved, Halfsplit }`
   (`src/autograd.rs:177`). `Op::RoPePrecomputed` now carries this
   explicit tag instead of shape-sniffing the saved cos tensor in
   backward (`src/autograd.rs:4207-4230`). Fixes the HiDream-O1 MRoPE
   case: cos shape `[1, S, half]` (rank-3) was mis-classified as
   Interleaved by the old shape-sniffer when forward used
   `rope_halfsplit_bf16`. Three forward sites in `src/bf16_ops.rs` now
   pass the correct tag:
   - `rope_fused_bf16` → `Interleaved`
   - `rope_fused_bf16_f32pe` → `Interleaved`
   - `rope_halfsplit_bf16` → `Halfsplit`

2. `pub fn AutogradContext::retain_intermediate_grads_add(ids)`
   (`src/autograd.rs:1412`) — additive variant of
   `retain_intermediate_grads`. Required because the outer-tape retain
   snapshot fires once before the checkpoint recompute closure runs.
   `Op::Checkpoint` backward (`src/autograd.rs:3551`) and
   `Op::CheckpointOffloadBoundary` backward (`src/autograd.rs:3800`)
   re-read `RETAINED_INTERMEDIATE_GRAD_IDS` inside the sub-tape walk,
   so probe IDs added *during* recompute are honored.

3. `pub fn ops::fused_inference::fused_linear3d_native_pytorch_parity`
   (`src/ops/fused_inference.rs:479`) wrapping new
   `flame_linear3d_bf16_pytorch_parity` CUDA kernel
   (`src/cuda/fused_linear3d.cu:369`). Bit-exact PyTorch
   `at::cuda::blas::gemm_and_bias<at::BFloat16>` mirror — 1 MiB
   workspace, per-call heuristic, BIAS_POINTER set on descriptor
   *before* `cublasLtMatmulAlgoGetHeuristic` so it picks the
   bias-pointer-specialized algo. ~1% perf overhead, byte-identical
   output vs PyTorch. FFI binding at `src/cuda/ffi.rs:906`.

4. New standalone test `tests/sdpa_prefix_causal_full_grad.rs` with
   three tests reproducing the exact HiDream-O1 attention chain at
   exact shapes (B=1, H=32, S=497, D=128, prefix=263). All three
   produced cos=1.0 / max_abs=0 — proves the structural chain is
   autograd-clean in isolation.

5. Docs updated: `docs/FLAME_INDEX.md`, `docs/FLAME_KERNELS.md`,
   `docs/FLAME_CONVENTIONS.md`, `docs/FLAME_MODULES.md`,
   `docs/FLAME_DIAGNOSTICS.md` (this last is the new tier-2 doc shipped
   earlier 2026-05-20).

**inference-flame**:

1. `src/models/hidream_o1/timestep_embedder.rs` — no-LoRA path swapped
   from `fused_linear3d_native` to
   `fused_linear3d_native_pytorch_parity`. Result: `pre.t_emb` is
   bit-exact vs PyTorch.

2. `src/models/hidream_o1/bottleneck_patch_embed.rs` — same swap on
   proj1/proj2 no-LoRA paths. `pre.patch_emb` mean_abs went from 3e-3
   to 5.7e-5 (53× improvement).

3. `src/models/hidream_o1/decoder.rs` — trap probes for layer 35:
   record `v_post_repeat_kv` and `attn_out` tensor IDs into
   `super::trap`. Probes are checkpoint-recompute-aware (last-writer
   wins, so the recompute-pass IDs overwrite the first-pass IDs and
   the sub-tape retain check finds them).

4. New `src/models/hidream_o1/trap.rs` — soul.md-trap-pattern static
   registry. `arm_probes()` / `record_probe(name, id)` /
   `take_probes()` / `is_armed()` / `disarm_probes()`. Template for
   other models — documented in flame-core's
   [`FLAME_DIAGNOSTICS.md`](../../flame-core/docs/FLAME_DIAGNOSTICS.md) §6.

5. `src/models/hidream_o1/mod.rs` — declares the new `trap` module.

**EriDiffusion-v2**:

1. `tests/parity/hidream_o1_train_step_ref.py` — forward hooks on
   layer 35's `o_proj` (pre-hook captures input grad = `attn_out`)
   and `v_proj` (forward hook captures output grad = `v_proj_out`).
   `retain_grad` in both. Saved as
   `grad_probe.layers.35.{attn_out, v_proj_out}` for parity comparison.

2. `crates/eridiffusion-cli/src/bin/parity_hidream_o1_train_step.rs` —
   `--lora-step` mode arms the trap before forward, registers probe
   IDs in retain set after, dumps rearranged grads to compare against
   Python's `v_proj_out` (sum-reduce + permute + reshape to match
   Hkv·D layout).

3. `crates/eridiffusion-core/src/models/chroma.rs:2332` —
   `Op::RoPePrecomputed` record_op site updated to pass
   `layout: RopeLayout::Interleaved`.

### What the bisect did NOT find

The trap pinpointed "dV out of SDPA backward is corrupt" (cos = 0.012
at the `v_post_repeat_kv` probe). But the standalone test
`flame-core/tests/sdpa_prefix_causal_full_grad.rs` exhaustively
reproduces the exact HiDream-O1 attention chain — LoRA-fused Q/K/V
linear + reshape/permute + q_norm/k_norm + halfsplit RoPE + repeat_kv
+ structured SDPA + checkpoint + grow activation cache + retain
mechanism — at the exact shapes, and produces bit-exact gradients
(cos = 1.0, max_abs = 0).

The full chain is autograd-clean in isolation. So the cos = 0.012
observation in the real model is either:

(a) A parity-comparison artifact — the Python-side reference uses
forward hooks on `o_proj` / `v_proj` to capture grads; the Rust side
captures TensorIds from inside the decoder. The reshape / permute /
sum convention used to map between them may not be exactly correct.

(b) A multi-layer cascade interaction not reproducible at single-layer
scope.

The structural fixes today (RoPE backward layout tag, cuBLASLt
PyTorch-parity linear, checkpoint-aware retain API) are real
correctness wins regardless of whether they close the end-to-end cos
gap by themselves. Keep them. Next-step bisect should swap the Python
reference to a per-step full-grad capture instead of per-layer hook
captures.

### Status

- The 1000-step resident-LoRA training run kicked off
  `output/hidream_o1_gigerver3_resident_1000_20260520/` is the next
  validation gate. Real LoRA must render trigger-bound style at 1024².
- Strict per-tensor parity at `hidden_input_layer_00` is no longer
  the active blocker after the patch-embed bit-exact swap, but the
  per-layer parity chain past that point has not been re-run end to
  end yet.
- Nothing committed across any of the three repos. Build still passes
  (`cargo check --bin parity_hidream_o1_train_step --bin
  train_hidream_o1`).

## 2026-05-21 Update — Full-Model Parity Gate Before Training Proof

This supersedes the earlier "1000-step run is next" status above.

- Reference is `/home/alex/ai-toolkit` using
  `/home/alex/HiDream-O1-Image-Full-weights`; do not use the Dev/non-trainable
  dump for trainer parity. Metadata verified in
  `/tmp/hidream_o1_train_step_ref_meta.json`.
- Current first blocker is the decoder SDPA path, not Q/K/V projection,
  RMSNorm, MRoPE, repeat-kv, or `o_proj` applied to the same input. The
  PyTorch reference SDPA output matches CUDA FlashAttention; Flame is being
  moved toward that path inside `flame-core`.
- Flame's in-tree FA2 BF16 forward now supports head dimensions `{64, 96, 128}`
  plus a runtime causal flag. The current HiDream-focused patch uses raw
  logits, PyTorch-style `UNFUSE_FMA` `exp2(score * scale - max_scaled)`
  softmax scaling, reverse K/V tile traversal, causal masking, and HD128
  non-causal `64x32` tiling. This is still the existing Flame shared-memory
  accumulator kernel, not the full PyTorch CUTLASS/CUTE tile layout.
- Direct FA2 traps are green after fixing a stale test-reference bug where
  F32 `bmm` read a non-contiguous transposed K view as raw storage:
  `fa2_parity_naive` passes for `N={512,4096}, HD={64,128}` and
  `sdpa_ragged_sk` passes for `Sk={64,71,72,128,200}`.
- The production gate now fails honestly instead of printing PASS when only
  the LoRA sub-gate passes. Current Full-model pinned run:
  `layer00.sdpa_out` OK (`max_abs=1.953125e-3`), first failure
  `forward::layer00.attn_out`, final loss rel `~8.0e-5` vs the strict
  `1e-5` gate.
- Exact PyTorch tile parity is deferred until after the trainer gate unless
  the smoke still proves it is required. PyTorch's useful SM8x reference
  targets are HD64 non-causal `128x128`, HD96 non-causal `128x64`,
  HD128 non-causal `128x32`, and HD96/HD128 causal `64x64`.
- Do not start the 1000-step `/eri2` training proof unless
  `parity_hidream_o1_train_step` passes production parity on the Full-model
  dump. If parity fails, report the exact first failing tensor and continue
  the SDPA/FA2 trap.

# HiDream-O1 LoRA Trainer — Pre-Implementation Analysis

> Scope: feasibility + plan for a pure-Rust EDv2 `train_hidream_o1.rs` that
> mirrors the edv2-reference HiDream training PATTERN but targets our already-verified
> HiDream-**O1** Rust model (Qwen3-VL pixel-DiT), NOT HiDream-I1.
>
> No code in this doc. Analysis only.

---

## 1. HiDream-O1 architecture refresher

Sources: `/home/alex/HiDream-O1-Image/{inference.py, models/pipeline.py, models/qwen3_vl_transformers.py}`.

- **Single end-to-end model.** `Qwen3VLForConditionalGeneration` (`qwen3_vl_transformers.py:1717`) is *the* model — no separate VAE, no separate text encoder. The Qwen3-VL text spine + 3 added heads (`t_embedder1`, `x_embedder=BottleneckPatchEmbed`, `final_layer2=FinalLayer` — `qwen3_vl_transformers.py:1041-1046`) are the whole thing. edv2-reference's I1 4-encoder + VAE + separate DiT pattern does NOT apply.
- **Pixel-level, no VAE.** Operates on raw pixels patchified with `PATCH_SIZE=32` (`pipeline.py:17`). Forward eats `vinputs=[B, L, 3*32*32=3072]` and emits `x_pred=[B, S_total, 3072]` (`qwen3_vl_transformers.py:1389-1391, 1525-1526`). The caller gathers `x_pred[:, vinput_mask, :]` to recover only the image rows. **No VAE encode/decode anywhere.**
- **Forward signature** (`qwen3_vl_transformers.py:1379-1532`): `_forward_generation(input_ids, position_ids, vinputs, timestep, token_types, attention_mask?, pixel_values?, image_grid_thw?, use_flash_attn)`. Returns `Qwen3VLModelOutputWithPast(x_pred=...)`. Timestep is `t_pixeldit = 1.0 - step_t/1000.0` (`pipeline.py:344`).
- **Loss target = predicted x_0** (clean pixel patches), not noise, not velocity. `pipeline.py:349`: `v = (x_pred - z) / sigma` — velocity is *derived* from x_pred at inference time. This is the rectified-flow `x_0`-prediction parameterisation (a.k.a. data-prediction). For training, the natural loss is `MSE(x_pred, clean_pixels)`, NOT `MSE(pred, noise)`. Confirmed against pipeline forward — TBD whether O1 was trained on x_0 MSE specifically vs. velocity MSE; the inference formulation is consistent with x_0-pred and no other documentation exists. See open question in §8.
- **LoRA-targetable modules.** The 36 transformer layers each contain (`/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/decoder.rs:65-89, 113-128`): `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention), `gate_proj`, `up_proj`, `down_proj` (SwiGLU MLP). Dense, not MoE. The 3 HiDream heads (`x_embedder`, `t_embedder1`, `final_layer2`) plus `embed_tokens` and final `norm` are NOT typical LoRA targets but ARE attached to a vision tower (`self.visual` — used only when `pixel_values is not None` for ref-image conditioning, `qwen3_vl_transformers.py:1032`). For text→image LoRA, target attention + MLP linears in the 36 decoder layers, same as Klein/Qwen LoRA. **Vision tower stays frozen for T2I.**
- **Sampler / scheduler.** `FlowMatchEulerDiscreteScheduler` with `shift=3.0` for `full` 50-step (`inference.py:60, pipeline.py:84`). Dev model uses `FlashFlowMatchEulerDiscreteScheduler` with custom `DEFAULT_TIMESTEPS` list of 28 steps (`pipeline.py:25-28, 81`). Our Rust port already implements this (`inference-flame/src/models/hidream_o1/scheduler.rs`).

---

## 2. edv2-reference's HiDream training pattern (abstracted)

Sources: `/home/alex/edv2-reference/extensions_built_in/diffusion_models/hidream/hidream_model.py` (453 lines) + `/home/alex/edv2-reference/jobs/process/BaseSDTrainProcess.py:790-1450` (hook_train_loop).

Pseudo-code per step (I1, latent-space):
```
1. dataloader → batch: imgs[B,3,H,W], latents[B,16,h,w] (cached VAE encode), captions
2. timesteps idx = randint(0, 1000)                       BaseSDTrainProcess.py:1286-1297
3. noise = randn_like(latents)                             BaseSDTrainProcess.py:1301
4. noisy_latents = self.sd.add_noise(latents, noise, t)    BaseSDTrainProcess.py:1395
   (flow-match: noisy = (1-σ)·clean + σ·noise)
5. encode_prompt → PromptEmbeds (CLIP-L, CLIP-G, T5, Llama-3)   hidream_model.py:389-405
   — 4 frozen encoders, embeds cached if `cache_text_embeds: true`
6. noise_pred = transformer(hidden_states=noisy_latents, timesteps=t,
       encoder_hidden_states=text_embeds, pooled_embeds=…, img_sizes, img_ids)
   noise_pred = -noise_pred                                hidream_model.py:376-385
7. target = get_loss_target(noise=noise, batch=batch)
          = (noise - latents).detach()                     hidream_model.py:427-430
   (flow-match velocity target: noise → clean direction)
8. loss = F.mse_loss(noise_pred, target) (+SNR weight)     BaseSDTrainProcess.py:880-885
9. loss.backward() — gradient checkpointing on double_stream + single_stream blocks
10. optimizer.step() / EMA shadow update
11. save: transformer.save_pretrained(...) + lora state dict (PEFT keys, renamed `transformer.` → `diffusion_model.` for ComfyUI)
   hidream_model.py:415-449
12. sample: full pipeline.__call__ with prompt_embeds, periodically
```

`network: lora, linear: 32, linear_alpha: 32, ignore_if_contains: ["ff_i.experts","ff_i.gate"]`
in `train_lora_hidream_48.yaml:24-33`. PEFT-style LoRA on all `Linear` in `double_stream_blocks` + `single_stream_blocks` minus MoE expert FFNs.

---

## 3. Mapping edv2-reference → our O1 trainer

| edv2-reference step (I1, latent) | O1 equivalent | Rust location |
|---|---|---|
| `latents[B,16,h,w]` | `pixels[B,3,H,W]` directly. **No VAE.** | new `prepare_hidream_o1.rs` |
| `imgs` cache | resized RGB tensors `[3,H,W]` BF16, patchified to `[L, 3072]` on load | new prep |
| `add_noise(latents, noise, t)` flow-match | identical math but on pixel patches: `z = (1-σ)·clean_patches + σ·noise`. `pipeline.py:291-295` is the inference noise init (just `σ·randn`); train-side adds clean. | new helper or reuse `flame-diffusion` flow-match `add_noise` |
| 4-text-encoder `_encode_prompt` | tokenize prompt with Qwen3-VL processor → `input_ids`, `position_ids`, `token_types`, `vinput_mask` via `build_t2i_text_sample` (`pipeline.py:30-77`). **Reimplement in Rust** — we don't have the Qwen3-VL `AutoProcessor` chat-template builder yet. Caption embeds are NOT prompt-encoder embeds; they are `input_ids` consumed inline by `embed_tokens` inside the forward. | new `qwen3vl_tokenizer.rs` helper OR cache tokenized prompts |
| `transformer(...)` forward | `HiDreamO1Model::forward(...)` (`inference-flame/src/models/hidream_o1/model.rs:211`) | reuse as-is |
| target = `(noise - latents)` (velocity) | target = `(noise - clean_patches)` IF we adopt velocity-pred OR target = `clean_patches` IF we adopt x_0-pred. **The inference path uses x_0-pred** (`pipeline.py:349`). For training, x_0-pred MSE on the gen-image rows is the safe default. See §8 risk. | new helper |
| `noise_pred = -noise_pred` | symmetric sign flip — O1's `pipeline.py:374` does `model_output = -v_guided`; same convention. Apply consistently. | trainer step fn |
| MSE on full `noise_pred` | MSE on `x_pred[:, vinput_mask, :]` — must gather gen-image rows (matches inference `pipeline.py:329`). Background text positions get NO loss. | trainer step fn |
| `gradient_checkpointing=true` on dual+single blocks | flame-core checkpointing per decoder layer. Our O1 decoder already streams weights from `BlockOffloader` (`model.rs:119`) — adding activation checkpointing per layer = same pattern as `train_klein --activation-offload` | trainer flag |
| PEFT save | EDv2 PEFT-style save (already shared infra, `feedback_save_format_peft_peft.md`) | reuse |

**LoRA injection point — STRUCTURAL CONSTRAINT.** The decoder layers do NOT wrap weights in `Linear` modules — they call `fused_linear3d_native(&normed, q_w, q_b)` with raw `&Tensor` refs pulled from a `HashMap<String, Tensor>` per layer (`decoder.rs:313-350`). This is fine for inference but means **a LoRA wrapper cannot be inserted by replacing a `Linear::forward` call**. Three options:

  - **(A) Refactor decoder to use `Linear` (or `LoRALinear`).** Touches 7 call sites × 36 layers + breaks the streaming/offloader contract (the offloader serves `HashMap`, not module-owned tensors). **High effort, high risk.**
  - **(B) Add `lora_a: Option<&Tensor>, lora_b: Option<&Tensor>` params to `fused_linear3d_native`** or wrap with a `fused_linear3d_native_with_lora` that does `y = x @ W^T + (x @ A^T) @ B^T * scale + b`. **Lowest decoder churn.** Requires the trainer to also stream LoRA A/B tensors keyed by the same `layers.{i}.self_attn.q_proj.lora_A.weight` short_key so they sit next to the base weights in the per-layer `HashMap`. The `BlockOffloader` already returns an `Arc<HashMap>` — extend the loader to merge LoRA tensors into each block's HashMap at load time.
  - **(C) Train without `BlockOffloader` (load all 36 layers into GPU).** ~8 GB BF16 base + LoRA. On 24 GB this might fit at 512² for batch 1 with grad-checkpoint, but `--offload` is needed for any larger config. **Forces a 24 GB-class-only trainer.**

**Recommend (B).** Aligns with the offloader contract, keeps decoder code one extra param.

---

## 4. New trainer-side code we need

| File | Est LOC | What it does |
|---|---|---|
| `crates/eridiffusion-cli/src/bin/train_hidream_o1.rs` | ~1400–1800 | Mirror `train_klein.rs` skeleton: CLI parse, config load, weight loading, BlockOffloader wiring, LoRA bundle build, AdamW, dataloader, training loop, periodic save + sample. |
| `crates/eridiffusion-cli/src/bin/prepare_hidream_o1.rs` | ~400 | RGB-resize + bucket + patchify cache. **No VAE encode, no prompt encode** — see §4.1. |
| `crates/eridiffusion-cli/src/bin/sample_hidream_o1.rs` | ~300 (or 0 if we reuse `inference-flame/src/bin/hidream_o1_infer.rs` directly) | Inference with optional `--lora-path`. |
| `crates/eridiffusion-core/src/models/hidream_o1_lora.rs` (new) | ~200 | LoRA spec + load/save helpers in PEFT-key layout. Reuses existing `LycorisBundle` infra. |
| flame-core: extend `fused_linear3d_native` for LoRA OR a `fused_linear3d_native_with_lora` wrapper | ~80 | Per §3 option B. |
| `inference-flame/src/models/hidream_o1/decoder.rs` patch | ~30 lines added | Plumb optional LoRA tensors from `weights: &HashMap` through the 7 linear calls per layer. |
| `inference-flame/src/bin/hidream_o1_infer.rs` patch | ~40 lines | Add `--lora-path` flag, load LoRA shards, merge into `BlockOffloader`'s per-block HashMaps before serving. |

### 4.1 prepare_hidream_o1 cache schema (proposed)

```
sample_NNNNNN.safetensors:
  patches      : BF16 [L, 3072]   pre-patchified at PATCH_SIZE=32
                                  L = (H/32) * (W/32)
  image_grid   : I64  [3]         (1, H/32, W/32)
  input_ids    : I64  [S_text]    Qwen3-VL chat-template-tokenised prompt
                                  including <|tms_token|> + L <|image_pad|> placeholders
  position_ids : I64  [3, S_total]   3D MRoPE positions (precomputed)
  token_types  : I64  [S_total]      0=AR text, 1=gen image — drives attention + loss mask
  vinput_mask  : U8   [S_total]      1.0 where x_pred should be gathered

attrs:
  prompt       : str
  height, width: int
  seed         : int
```

**No prompt-encoder embeds cached** — O1 has no separate text encoder. `input_ids` ARE the text representation; `embed_tokens` runs inside the forward. This means **caption dropout** at training time becomes "swap to a pre-tokenised empty-string sample"; not a simple text-embed zero-out.

---

## 5. Config schema — `eri2_hidream_o1_lora.json`

Adapting `train_lora_hidream_48.yaml` (`/home/alex/edv2-reference/config/examples/train_lora_hidream_48.yaml`):

| Field | Apply? | Default proposed | Note |
|---|---|---|---|
| `network.type=lora, linear=32, linear_alpha=32` | yes | rank=16, alpha=16 | EDv2 convention; user can raise |
| `ignore_if_contains: ff_i.experts/gate` | **no** | n/a | O1 is dense, no MoE |
| `save.dtype=bfloat16, save_every=250, max_step_saves_to_keep=4` | yes | same | EDv2 standard |
| `datasets.resolution=[512,768,1024]` | yes | `[512, 768]` default | O1 trains/infers at multiples of 32; 1024 OK on 24 GB only with `--offload` + ckpt |
| `caption_dropout_rate=0.05` | yes | 0.05 | See §4.1 caveat — uses null-input-ids sample |
| `train.steps=3000`, `batch_size=1`, `grad_accum=1` | yes | 3000 / 1 / 1 | |
| `train_text_encoder=false` | **forced** | false | O1 spine = trainable target; no separate TE to gate. Confusingly: the Qwen3-VL spine IS the "DiT". Flag mute. |
| `gradient_checkpointing=true` | yes | true | required on 24 GB; per-layer recompute |
| `noise_scheduler=flowmatch, timestep_type=shift` | yes | flowmatch, shift=3.0 | matches inference `full` mode |
| `optimizer=adamw8bit, lr=2e-4` | **no on 8bit; yes on lr** | adamw F32 master, lr=2e-4 | `feedback_zimage_no_quantization.md` — bf16/fp32 only. Wan22 is the documented exception, not O1. |
| `ema_config.use_ema=false` | yes | false | EDv2 EMA surface available; default off |
| `dtype=bf16` | yes | bf16 | |
| `model.name_or_path` | replace | local path `/home/alex/HiDream-O1-Image-Full-weights/` | Full trainable O1 checkpoint only. The older Dev/non-trainable dump is not a valid trainer-parity reference. Single dir, not separate `extras_name_or_path` (no VAE/TE) |
| `model.quantize / quantize_te` | **no** | n/a | quant rule + no TE |
| `model_kwargs.llama_model_path` | **drop** | n/a | no Llama |
| `sample.sampler=flowmatch, width/height/prompts/sample_steps=25, guidance_scale=4` | yes | flowmatch, 1024, 25 steps, cfg 5.0 | match inference defaults (`inference.py:32, 58, 65`) |

Plus EDv2 standard flags: `--timestep-distribution logit_normal`, `--lr-scheduler`, `--warmup-steps 100`, `--save-mode full|weights`, `--resume-lora|--resume-full`, `--algo lora|locon|loha|lokr|full|oft`, `--use-autograd-v2`, `--gpu-health-monitor`, `--webhook-url`. Per TRAINERS.md §"Shared conventions".

---

## 6. Sample / save / inference compatibility

- Saved LoRA — PEFT layout with `transformer.language_model.layers.{i}.self_attn.q_proj.lora_A.weight` style keys (EDv2 convention). Inference will use the SAME key prefix the BlockOffloader strips (`model.language_model.layers.{i}.` → short_key), so LoRA short_keys end up sibling to base short_keys: `self_attn.q_proj.lora_A.weight` next to `self_attn.q_proj.weight`. Drop-in.
- `hidream_o1_infer.rs` currently has NO `--lora-path` flag (grep returned 0 hits for `lora` other than the prompt text containing "with delicate floral embroidery"). **Must add ~40 lines** to:
  1. Parse `--lora-path` (or repeat for multiple).
  2. Load LoRA safetensors, split by layer index, group by short_key.
  3. Inject into `BlockOffloader` so each `await_block(i)` HashMap also contains the LoRA tensors.
  4. Have the decoder fused-linear call recognise & apply (per §3 option B).
- The exact same code path serves training (where LoRA grads flow) and inference (where LoRA is merged in forward). No separate inference merge step needed for sampling.

---

## 7. Effort estimate

| Component | LOC | Days |
|---|---|---|
| Extend `fused_linear3d_native` for optional LoRA tape-recording | 80 | 1.5 |
| `decoder.rs` plumbing (HashMap → 7 lora pairs/layer, optional) | 30 | 0.5 |
| `BlockOffloader` LoRA-merge-on-load (loader extension) | 60 | 0.5 |
| `prepare_hidream_o1.rs` (image resize + patchify + tokenise + position_ids) | 400 | 2.0 |
| `train_hidream_o1.rs` MVP (no EMA, no LyCORIS) | 900 | 3.0 |
| `train_hidream_o1.rs` parity to Klein surface (EMA, LyCORIS, schedulers, all shared knobs) | +500 | 2.0 |
| `hidream_o1_infer.rs` `--lora-path` | 40 | 0.5 |
| `eri2_hidream_o1_lora.json` config + docs entry in TRAINERS.md | 120 | 0.5 |
| Smoke + 5-step + 1000-step verification runs | n/a | 2.0 |
| **MVP total** | **~1500** | **~7 days** |
| **Full-parity total** | **~2100** | **~12 days** |

Mirror sources for 1:1 copy:
- CLI parse, weight loader, save loop → `train_klein.rs`
- BlockOffloader pattern → `train_klein.rs` `--offload` path + `inference-flame/src/models/hidream_o1/pipeline.rs`
- LoRA bundle init / AdamW / step-0 sample → `train_klein.rs`

Novel (no analogue):
- Prep that produces tokenised input_ids + 3D MRoPE position_ids + token_types
- Loss that gathers `vinput_mask` rows before MSE (everyone else operates on dense `[B,C,H,W]`)
- `fused_linear3d_native_with_lora` autograd plumbing in flame-core

---

## 8. Top 5 risks

1. **x_0-pred vs velocity-pred — loss formulation unverified.** The inference path computes `v = (x_pred - z)/sigma` (`pipeline.py:349`) which is the canonical conversion from x_0-pred to velocity. But the model could have been TRAINED on velocity-pred and the inference code just inverts. We have zero training-code reference (edv2-reference doesn't train O1). **First-step LoRA-init=0 numerical parity vs `hidream_o1_infer.rs` cannot distinguish the two — both formulations produce identical `x_pred` at LoRA=0.** Mitigation: try x_0-pred MSE first (it's what `x_pred` literally is); if convergence is poor at 500 steps, switch to velocity-pred MSE with `target = (noise - clean) detached`.
2. **LoRA-injection structural refactor.** The `fused_linear3d_native` call sites don't take modules; weights live in a streaming HashMap. The path-of-least-resistance fix (§3 option B) requires a flame-core kernel extension that records autograd through TWO low-rank multiplies per call. Per `feedback_flame_core_bf16_fused_autograd.md` this is exactly the class of bug that has shipped corrupt LoRAs before (rope_fused, swiglu, qkv_split). **High likelihood of a "trains-but-silent-no-grad" failure mode** unless `FLAME_ASSERT_GRAD_FLOW=1` is on from day 1 and explicit grad-nonzero ratios are logged.
3. **Tokeniser parity.** `pipeline.py:30-77 build_t2i_text_sample` uses `processor.apply_chat_template`, the Qwen3-VL `boi_token` / `tms_token` machinery, `get_rope_index_fix_point` for 3D MRoPE positions, and `token_types` for attention-mask construction. We have Qwen3-VL processor in inference (`inference-flame/src/models/hidream_o1/pipeline.rs`) but not exposed as a standalone Rust tokeniser callable from a prep binary. **~200 LOC of MRoPE+token_types prep logic that must be byte-identical to the inference-time code.** Easy bug: off-by-one on tms_token position.
4. **24 GB feasibility.** edv2-reference's I1 yaml notes "~35.2 GB of vram to train" (`train_lora_hidream_48.yaml:1`). O1 is 8B (similar Llama-scale to I1's transformer), pixel-level (no VAE compression so seq_len is `H*W/1024` instead of `H*W/256`). For 512²: L=256 patches → tiny. For 1024²: L=1024 patches → manageable. For 2048²: L=4096 patches → attention becomes O(4096²)=16M per layer — likely OOM on 24 GB even with offload. **Default training resolution should be 512–768; document the cap.**
5. **No upstream training reference.** edv2-reference trains I1, not O1. No PEFT pipeline exists publicly for O1. Hyperparam choices (lr, timestep shift, noise schedule) are guesses informed by inference defaults. Validation requires user judgement on visual quality, not numerical parity with a known-good trainer.

---

## 9. Recommendation: **YES — with caveats**

Proceed with MVP. Justification:
- Architecture is dense Qwen3-VL — well-trodden LoRA target class.
- All forward-side code already verified at 1024² and 2048² inference.
- Pipeline contract is clean (single model, single forward, single output).
- Worst-case fallback (loss-target wrong) shows up as poor convergence at 500 steps, not a training crash.

**MVP scope** (drop everything optional):
- `--algo lora` only, no LyCORIS.
- AdamW F32, no EMA.
- No validation, no multi-backend, no slider.
- No periodic sample (call `hidream_o1_infer.rs` externally instead — saves ~200 LOC).
- Just: load weights via offloader, build plain `LoRALinear`-equivalent over §3 option B, dataloader → step → loss → grad → adamw step → save every 250.

**Estimated MVP**: ~1000 LOC Rust net new + ~80 LOC flame-core kernel extension + 5–7 dev-days. Acceptance gate = §10 alternative C.

If MVP shows visible likeness in 1000-step Alina/eri2 run, then port full Klein-surface in ~5 additional days.

---

## 10. Parity gate alternatives (we cannot do first-step-loss vs PyTorch)

Since edv2-reference doesn't train O1, there is no reference loss curve.

- **A. Numerical (forward-only):** Trainer's training-mode forward with LoRA-init=0 produces byte-identical `x_pred` to `hidream_o1_infer.rs`'s forward for matched `(input_ids, position_ids, vinputs, timestep, token_types)`. Tests: model wiring, embed_tokens, decoder streaming. **Does NOT test:** loss reduction direction, grad correctness, optimiser.
- **B. Visual (1000-step LoRA on /eri2):** Run with frozen seed, sample at step 0 / 250 / 500 / 1000 with same prompt. Look for monotonic style/likeness progression. Subjective but historically reliable (Klein, Z-Image, Chroma all gated this way).
- **C. Self-consistency (RECOMMENDED for MVP):**
  - C1. Forward in `train` mode with LoRA=0 == forward in inference mode (no-grad) — same `x_pred` bytewise. (= A above)
  - C2. `FLAME_ASSERT_GRAD_FLOW=1` reports ≥99% non-zero `lora_B` after step 1 — every targeted Linear receives gradient (catches the rope_fused-class silent-autograd-drop bug per `feedback_flame_core_bf16_fused_autograd.md`).
  - C3. After 50 steps on a single prompt with batch_size=1 and lr=1e-3, `MSE(x_pred, clean_patches)` on that one sample drops by ≥40% from step 1. Catches loss-target sign errors (if we picked the wrong sign on velocity vs x_0, the trainer would either flatline or diverge).
  - C4. Manual sample at step 0 + step 100 + step 500 — visible per-step change in the right direction.

**Pick C.** It is cheap, executable on day 1, catches the most common bug classes for this architecture (silent autograd, sign flip, wrong target), and does not require a reference run.

---

## Report-back summary

- **Path / lines:** `/home/alex/EriDiffusion/EriDiffusion-v2/docs/hidream_o1_trainer_analysis.md` — see `wc -l` after write.
- **Recommendation:** YES, with caveats — proceed to MVP.
- **Top single risk:** x_0-pred vs velocity-pred loss target unverified, no upstream reference to disambiguate; mitigated by self-consistency gate C3.
- **MVP LOC:** ~1000 Rust + ~80 flame-core kernel extension.
- **Parity gate pick:** **C** (self-consistency: forward-parity at LoRA=0, grad-flow assert, 50-step loss-drop on overfit sample, visual at step 500).

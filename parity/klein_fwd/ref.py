#!/usr/bin/env python3
"""Klein forward-prediction parity reference (diffusers = upstream of OneTrainer).

Feeds a fixed (x_t, timestep, text_embedding) through the diffusers
Flux2 klein transformer and dumps the velocity prediction. Our Rust
`forward_train` consumes the IDENTICAL x_t/timestep/text and we compare the
predicted velocity (cosine). This isolates our packing + RoPE ids + 32-block
DiT forward against the reference DiT on identical input.

Shared fixture: our cached training sample (latent x0 + text_embedding), so the
text representation and latent are exactly what our trainer uses.
"""
import sys, glob, json
import torch
from safetensors.torch import save_file, load_file as st_load

CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
CACHE = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_klein9b"
dev = "cuda"; dt = torch.bfloat16

# --- load one cached sample (latent x0 + text embedding) ---
sample_f = sorted(glob.glob(f"{CACHE}/*.safetensors"))[0]
s = st_load(sample_f, device="cpu")
x0 = s["latent"].to(dev, dt)              # [1,128,h,w]  (already VAE-encoded + prepped)
txt = s["text_embedding"].to(dev, dt)     # [1,512,12288]
print(f"sample={sample_f.split('/')[-1]} x0={tuple(x0.shape)} txt={tuple(txt.shape)}", flush=True)

B, C, H, W = x0.shape

# --- pin noise + timestep (sigma in (0,1)); build x_t = noise*sigma + x0*(1-sigma) ---
g = torch.Generator(device=dev).manual_seed(1234)
noise = torch.randn(x0.shape, generator=g, device=dev, dtype=dt)
sigma = 0.5
x_t = (noise * sigma + x0 * (1.0 - sigma)).to(dt)
target = (noise - x0).to(dt)              # flow target = noise - x0 (matches OneTrainer)

# --- load diffusers klein transformer from the local HF snapshot ---
from diffusers import Flux2Transformer2DModel
TF_DIR = ("/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
          "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer")
tf = Flux2Transformer2DModel.from_pretrained(TF_DIR, torch_dtype=dt)
tf = tf.to(dev)
# Gradient checkpointing so the full 9B forward+backward fits in 24GB
# (recompute activations in backward instead of storing all 32 blocks).
# Flux2 DiT has no dropout and non-affine LayerNorm, so train() is
# mathematically identical to eval() here but lets checkpointing engage.
# Gradient checkpointing to fit 24GB; weights frozen (only want dL/dx_t +
# per-block grads). Per-block grads captured via BACKWARD hooks (grad_output),
# which fire on the recomputed graph and work with checkpointing.
tf.enable_gradient_checkpointing()
tf.train()
for _p in tf.parameters():
    _p.requires_grad_(False)
print("transformer loaded; config in_channels=", tf.config.get("in_channels", "?"), flush=True)

# --- pack latents [B,C,H,W] -> [B, H*W, C] ---
def pack(l):
    b, c, h, w = l.shape
    return l.reshape(b, c, h * w).permute(0, 2, 1)
hidden = pack(x_t)

# --- img_ids: cartesian (t,h,w,l) 4-dim ---
t_ax = torch.arange(1); h_ax = torch.arange(H); w_ax = torch.arange(W); l_ax = torch.arange(1)
img_ids = torch.cartesian_prod(t_ax, h_ax, w_ax, l_ax).to(dev).unsqueeze(0).expand(B, -1, -1).float()
# txt_ids: zeros [B, n_txt, 4]
n_txt = txt.shape[1]
txt_ids = torch.zeros(B, n_txt, 4, device=dev).float()

timestep = torch.tensor([sigma], device=dev, dtype=dt).expand(B)
guidance = torch.full([B], 4.0, device=dev, dtype=torch.float32)

# Full-step gradient parity: make x_t a leaf, run forward -> MSE(vel,target) ->
# backward, and dump dL/dx_t. Compared against our forward_train backward to
# verify the COMPOSED 32-block backward (not just isolated ops).
x_t_leaf = x_t.detach().clone().float().requires_grad_(True)
hidden_g = pack(x_t_leaf.to(dt))

# Per-block probe: retain_grad on each double-block's hidden (img) output.
# n_img = H*W tokens identifies the img stream vs the txt stream.
n_img = H * W
blk_grads = {}  # i -> grad_output of the img/hidden stream
blk_acts = {}   # i -> forward output of the img/hidden stream
hooks = []
for i, blk in enumerate(tf.transformer_blocks):
    def mk(i):
        def bhook(mod, grad_input, grad_output):
            gos = grad_output if isinstance(grad_output, (tuple, list)) else (grad_output,)
            for g in gos:
                if torch.is_tensor(g) and g.dim() == 3 and g.shape[1] == n_img:
                    blk_grads[i] = g.detach().float().contiguous().cpu()
        def fhook(mod, inp, out):
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for o in outs:
                if torch.is_tensor(o) and o.dim() == 3 and o.shape[1] == n_img:
                    blk_acts[i] = o.detach().float().contiguous().cpu()
        return bhook, fhook
    bh, fh = mk(i)
    hooks.append(blk.register_full_backward_hook(bh))
    hooks.append(blk.register_forward_hook(fh))

# norm_out input = img hidden AFTER the 24 single blocks, BEFORE final layer.
# ≡ our `img_only`. Splits final-layer backward vs single-block backward.
img_only_ref = {}
def norm_out_bhook(mod, gi, go):
    for g in (gi if isinstance(gi, (tuple, list)) else (gi,)):
        if torch.is_tensor(g) and g.dim() == 3 and g.shape[1] == n_img:
            img_only_ref["grad"] = g.detach().float().contiguous().cpu()
def norm_out_fhook(mod, inp, out):
    for o in (inp if isinstance(inp, (tuple, list)) else (inp,)):
        if torch.is_tensor(o) and o.dim() == 3 and o.shape[1] == n_img:
            img_only_ref["act"] = o.detach().float().contiguous().cpu()
hooks.append(tf.norm_out.register_full_backward_hook(norm_out_bhook))
hooks.append(tf.norm_out.register_forward_hook(norm_out_fhook))

# Intra-single-block localization: hook one single block's `attn` submodule.
# attn input  = normed  (≈ our sb_normed);  attn output = fused transform (≈ our sb_out).
# attn grad_input ≈ dL/d(normed) = our sb_normed grad; grad_output ≈ our sb_out grad.
import os
SB_IDX = int(os.environ.get("KLEIN_SB_IDX", "23"))
sb_ref = {}
if SB_IDX < len(tf.single_transformer_blocks):
    sb = tf.single_transformer_blocks[SB_IDX]
    def sb_attn_b(mod, gi, go):
        for g in (go if isinstance(go, (tuple, list)) else (go,)):
            if torch.is_tensor(g) and g.dim() == 3:
                sb_ref["sb_out_grad"] = g.detach().float().contiguous().cpu()
        for g in (gi if isinstance(gi, (tuple, list)) else (gi,)):
            if torch.is_tensor(g) and g.dim() == 3:
                sb_ref["sb_normed_grad"] = g.detach().float().contiguous().cpu()
    def sb_attn_f(mod, inp, out):
        for o in (inp if isinstance(inp, (tuple, list)) else (inp,)):
            if torch.is_tensor(o) and o.dim() == 3:
                sb_ref["sb_normed_act"] = o.detach().float().contiguous().cpu(); break
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if torch.is_tensor(o) and o.dim() == 3:
                sb_ref["sb_out_act"] = o.detach().float().contiguous().cpu(); break
    hooks.append(sb.attn.register_full_backward_hook(sb_attn_b))
    hooks.append(sb.attn.register_forward_hook(sb_attn_f))
    # norm output = normed (= attn input). norm grad_output = dL/d(normed) =
    # our sb_normed grad. Captures what the attn backward-hook missed (kwarg input).
    def sb_norm_b(mod, gi, go):
        for g in (go if isinstance(go, (tuple, list)) else (go,)):
            if torch.is_tensor(g) and g.dim() == 3:
                sb_ref["sb_normed_grad"] = g.detach().float().contiguous().cpu(); break
    def sb_norm_f(mod, inp, out):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if torch.is_tensor(o) and o.dim() == 3:
                sb_ref["sb_normed_act"] = o.detach().float().contiguous().cpu(); break
    hooks.append(sb.norm.register_full_backward_hook(sb_norm_b))
    hooks.append(sb.norm.register_forward_hook(sb_norm_f))

out = tf(hidden_states=hidden_g, encoder_hidden_states=txt, timestep=timestep,
         img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, return_dict=False)[0]
vel = out[:, :H * W, :].permute(0, 2, 1).reshape(B, C, H, W)
loss = ((vel.float() - target.float()) ** 2).mean()
loss.backward()
for h in hooks: h.remove()
dLdx = x_t_leaf.grad.detach().float()
blk_ref = {f"dbl{i}_img": g for i, g in blk_grads.items()}
if "grad" in img_only_ref: blk_ref["img_only"] = img_only_ref["grad"]
print(f"captured {len(blk_ref)} per-block ref grads (incl img_only={'grad' in img_only_ref})", flush=True)
save_file(blk_ref, "block_grads_ref.safetensors")
blk_act_ref = {f"dbl{i}_img": a for i, a in blk_acts.items()}
if "act" in img_only_ref: blk_act_ref["img_only"] = img_only_ref["act"]
if "sb_out_grad" in sb_ref: blk_ref["sb_out"] = sb_ref["sb_out_grad"]
if "sb_normed_grad" in sb_ref: blk_ref["sb_normed"] = sb_ref["sb_normed_grad"]
if "sb_out_act" in sb_ref: blk_act_ref["sb_out"] = sb_ref["sb_out_act"]
if "sb_normed_act" in sb_ref: blk_act_ref["sb_normed"] = sb_ref["sb_normed_act"]
print(f"captured sb_ref keys: {sorted(sb_ref.keys())}", flush=True)
save_file(blk_ref, "block_grads_ref.safetensors")
print(f"captured {len(blk_act_ref)} per-block ref forward acts (incl img_only={'act' in img_only_ref})", flush=True)
save_file(blk_act_ref, "block_acts_ref.safetensors")
print(f"velocity pred {tuple(vel.shape)} |vel|={vel.float().norm():.4e}  "
      f"loss={loss.item():.6e}  |dL/dx_t|={dLdx.norm():.4e}", flush=True)

save_file({"x_t": x_t.contiguous().cpu(), "target": target.contiguous().cpu(),
           "text_embedding": txt.contiguous().cpu(),
           "velocity_ref": vel.detach().float().contiguous().cpu(),
           "dLdx_ref": dLdx.contiguous().cpu(),
           "timestep": torch.tensor([sigma]), "noise": noise.contiguous().cpu(),
           "x0": x0.contiguous().cpu()},
          "fwd_fixture.safetensors")
print("wrote fwd_fixture.safetensors", flush=True)

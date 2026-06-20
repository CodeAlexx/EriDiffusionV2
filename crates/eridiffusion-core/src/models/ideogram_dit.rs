//! Ideogram-4 DiT forward (inference) — Rust port of the parity-verified Mojo
//! `ideogram4_dit.mojo` (1:1 from `modeling_ideogram4.py`). Gated vs the torch
//! `velocity` by the `parity_ideogram4_predict` bin.
//!
//! 34 single-stream blocks: fused-QKV attention (per-head q/k RMSNorm + half-split
//! RoPE + flash SDPA) + swiglu MLP, both adaLN-modulated (scale = mod+1, tanh
//! gates). Conditioning: input_proj (image tokens) + llm_cond (RMSNorm→proj) +
//! t_embedding (sinusoid→MLP→silu adaln) + image-indicator embedding. Final =
//! no-affine LayerNorm * (1 + adaln_mod(silu(c))) → linear.
//!
//! VRAM: Ideogram-4 is ~10B params — too big to hold on a 24GB card. Mirroring
//! OneTrainer's `LayerOffloadConductor` residency model: conditioning weights
//! load once; each layer's weights stream to GPU just-in-time and free after
//! (here sourced per-layer from disk via `load_file_filtered` — the minimal
//! forward-only version of OT's CPU-resident + prefetch conductor). fp8 weights
//! are auto-dequanted to f32 by the loader; cast to bf16 (the compute dtype).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, FlameError, Result, Shape, Tensor};

use crate::lora::LoRALinear;

const EPS5: f32 = 1.0e-5;
const EPS6: f32 = 1.0e-6;

type WMap = HashMap<String, Tensor>;

fn cast_map(raw: WMap) -> Result<WMap> {
    let mut m = WMap::with_capacity(raw.len());
    for (k, t) in raw {
        m.insert(k, t.to_dtype(DType::BF16)?);
    }
    Ok(m)
}

fn gw<'a>(m: &'a WMap, k: &str) -> Result<&'a Tensor> {
    m.get(k)
        .ok_or_else(|| FlameError::InvalidInput(format!("ideogram: missing weight {k}")))
}

/// `x @ w^T (+ bias)` for 2D [M,C] or 3D [1,L,C] x.
fn lin(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let c = *dims.last().unwrap();
    let m: usize = dims[..dims.len() - 1].iter().product();
    let mut out = x.reshape(&[m, c])?.matmul(&w.transpose()?)?; // [M, out]
    let outc = *out.shape().dims().last().unwrap();
    let mut oshape = dims.clone();
    *oshape.last_mut().unwrap() = outc;
    out = out.reshape(&oshape)?;
    if let Some(b) = b {
        let mut bshape = vec![1usize; dims.len()];
        *bshape.last_mut().unwrap() = outc;
        out = out.add(&b.reshape(&bshape)?)?;
    }
    Ok(out)
}

pub struct IdeogramDit {
    path: String,
    cond: WMap, // non-layer (conditioning + final) weights, bf16, loaded once
    device: Arc<CudaDevice>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    hidden: usize,
    /// LoRA adapters keyed by full weight name (e.g. `layers.0.attention.qkv`).
    /// Empty = pure inference. B=0 at init → identity overlay.
    loras: std::collections::HashMap<String, LoRALinear>,
}

impl IdeogramDit {
    /// Load only the conditioning/final weights (small); layers stream per-block.
    pub fn load(path: &str, device: Arc<CudaDevice>) -> Result<Self> {
        let raw = flame_core::serialization::load_file_filtered(
            std::path::Path::new(path),
            &device,
            |k: &str| !k.starts_with("layers."),
        )?;
        Ok(Self {
            path: path.to_string(),
            cond: cast_map(raw)?,
            device,
            num_layers: 34,
            num_heads: 18,
            head_dim: 256,
            hidden: 4608,
            loras: std::collections::HashMap::new(),
        })
    }

    /// Attach LoRA (B=0 → identity at init) to each block's linears, per the OT
    /// `blocks` preset: qkv, o, feed_forward.w1/w2/w3. Returns the trainable params.
    pub fn attach_block_loras(&mut self, rank: usize, alpha: f32) -> Result<Vec<Parameter>> {
        let h = self.hidden;
        // (suffix, in, out)
        let targets: [(&str, usize, usize); 5] = [
            ("attention.qkv", h, 3 * h),
            ("attention.o", h, h),
            ("feed_forward.w1", h, 12288),
            ("feed_forward.w2", 12288, h),
            ("feed_forward.w3", h, 12288),
        ];
        let mut params = Vec::new();
        let mut seed = 0u64;
        for li in 0..self.num_layers {
            for (suf, inf, outf) in targets {
                let key = format!("layers.{li}.{suf}");
                let lora = LoRALinear::new(inf, outf, rank, alpha, self.device.clone(), seed)?;
                seed += 1;
                params.extend(lora.parameters());
                self.loras.insert(key, lora);
            }
        }
        Ok(params)
    }

    /// `lin` + optional LoRA delta for the linear identified by `key`.
    fn lin_l(&self, x: &Tensor, w: &Tensor, b: Option<&Tensor>, key: &str) -> Result<Tensor> {
        let mut out = lin(x, w, b)?;
        if let Some(lora) = self.loras.get(key) {
            out = out.add(&lora.forward_delta(x)?)?;
        }
        Ok(out)
    }

    fn load_layer(&self, li: usize) -> Result<WMap> {
        let pfx = format!("layers.{li}.");
        let raw = flame_core::serialization::load_file_filtered(
            std::path::Path::new(&self.path),
            &self.device,
            move |k: &str| k.starts_with(pfx.as_str()),
        )?;
        cast_map(raw)
    }

    /// Half-split RoPE: `roped = x*cos + rotate_half(x)*sin`.
    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims().to_vec(); // [1,L,H,Dh]
        let (l, dh) = (d[1], d[3]);
        let half = dh / 2;
        let cos4 = cos.to_dtype(DType::BF16)?.reshape(&[1, l, 1, dh])?;
        let sin4 = sin.to_dtype(DType::BF16)?.reshape(&[1, l, 1, dh])?;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        let rot = Tensor::cat(&[&x2.mul_scalar(-1.0)?, &x1], 3)?; // [-x2, x1]
        x.mul(&cos4)?.add(&rot.mul(&sin4)?)
    }

    /// Fused-QKV attention. `lw` = this layer's weights, `p` = `layers.N.attention.`
    fn attention(&self, x: &Tensor, lw: &WMap, p: &str, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims().to_vec(); // [1,L,hidden]
        let (l, hidden) = (d[1], d[2]);
        let (nh, dh) = (self.num_heads, self.head_dim);
        let qkv = self.lin_l(x, gw(lw, &format!("{p}qkv.weight"))?, None, &format!("{p}qkv"))?; // [1,L,3*hidden]
        let qkv5 = qkv.reshape(&[1, l, 3, nh, dh])?;
        let q = qkv5.narrow(2, 0, 1)?.reshape(&[1, l, nh, dh])?;
        let k = qkv5.narrow(2, 1, 1)?.reshape(&[1, l, nh, dh])?;
        let v = qkv5.narrow(2, 2, 1)?.reshape(&[1, l, nh, dh])?;
        let q = flame_core::norm::rms_norm(&q, &[dh], Some(gw(lw, &format!("{p}norm_q.weight"))?), EPS5)?;
        let k = flame_core::norm::rms_norm(&k, &[dh], Some(gw(lw, &format!("{p}norm_k.weight"))?), EPS5)?;
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;
        // [1,L,H,Dh] -> [1,H,L,Dh] (sdpa layout), bf16.
        let q = q.permute(&[0, 2, 1, 3])?.to_dtype(DType::BF16)?;
        let k = k.permute(&[0, 2, 1, 3])?.to_dtype(DType::BF16)?;
        let v = v.permute(&[0, 2, 1, 3])?.to_dtype(DType::BF16)?;
        let attn = flame_core::attention::sdpa(&q, &k, &v, None)?; // [1,H,L,Dh]
        let merged = attn.permute(&[0, 2, 1, 3])?.reshape(&[1, l, hidden])?;
        self.lin_l(&merged, gw(lw, &format!("{p}o.weight"))?, None, &format!("{p}o"))
    }

    fn block(&self, x: &Tensor, lw: &WMap, p: &str, adaln: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = self.hidden;
        let m = lin(
            adaln,
            gw(lw, &format!("{p}adaln_modulation.weight"))?,
            Some(gw(lw, &format!("{p}adaln_modulation.bias"))?),
        )?; // [1,1,4h]
        let scale_msa = m.narrow(2, 0, h)?.affine(1.0, 1.0)?; // +1
        let gate_msa = m.narrow(2, h, h)?.tanh()?;
        let scale_mlp = m.narrow(2, 2 * h, h)?.affine(1.0, 1.0)?;
        let gate_mlp = m.narrow(2, 3 * h, h)?.tanh()?;

        let an1 = flame_core::norm::rms_norm(x, &[h], Some(gw(lw, &format!("{p}attention_norm1.weight"))?), EPS5)?;
        let attn_in = an1.mul(&scale_msa)?;
        let attn_out = self.attention(&attn_in, lw, &format!("{p}attention."), cos, sin)?;
        let an2 = flame_core::norm::rms_norm(&attn_out, &[h], Some(gw(lw, &format!("{p}attention_norm2.weight"))?), EPS5)?;
        let x1 = x.add(&gate_msa.mul(&an2)?)?;

        let fn1 = flame_core::norm::rms_norm(&x1, &[h], Some(gw(lw, &format!("{p}ffn_norm1.weight"))?), EPS5)?;
        let mlp_in = fn1.mul(&scale_mlp)?;
        let gg = self.lin_l(&mlp_in, gw(lw, &format!("{p}feed_forward.w1.weight"))?, None, &format!("{p}feed_forward.w1"))?;
        let uu = self.lin_l(&mlp_in, gw(lw, &format!("{p}feed_forward.w3.weight"))?, None, &format!("{p}feed_forward.w3"))?;
        let act = gg.silu()?.mul(&uu)?; // swiglu
        let ff = self.lin_l(&act, gw(lw, &format!("{p}feed_forward.w2.weight"))?, None, &format!("{p}feed_forward.w2"))?;
        let fn2 = flame_core::norm::rms_norm(&ff, &[h], Some(gw(lw, &format!("{p}ffn_norm2.weight"))?), EPS5)?;
        x1.add(&gate_mlp.mul(&fn2)?)
    }

    /// EmbedScalar sinusoid (sin-first, prescale 1e4, /(half-1)) + F64 trig reduction.
    fn embedscalar_sinusoid(&self, t: &Tensor, dim: usize) -> Result<Tensor> {
        let tv = t.to_vec_f32()?;
        let n = tv.len();
        let half = dim / 2;
        let log_scale = (10000.0_f64).ln() / (half as f64 - 1.0);
        let two_pi = std::f64::consts::TAU;
        let mut out = vec![0f32; n * dim];
        for r in 0..n {
            let scaled = tv[r] * 10000.0;
            for dd in 0..dim {
                let i = if dd < half { dd } else { dd - half };
                let freq = (-(i as f64) * log_scale).exp() as f32;
                let angle = (scaled * freq) as f64;
                let k = (angle / two_pi + 0.5).floor();
                let reduced = angle - k * two_pi;
                out[r * dim + dd] = if dd < half { reduced.sin() as f32 } else { reduced.cos() as f32 };
            }
        }
        Tensor::from_vec(out, Shape::from_dims(&[n, dim]), self.device.clone())?.to_dtype(DType::BF16)
    }

    fn t_embedding(&self, t: &Tensor, dim: usize) -> Result<Tensor> {
        let emb = self.embedscalar_sinusoid(t, dim)?;
        let h = lin(
            &emb,
            gw(&self.cond, "t_embedding.mlp_in.weight")?,
            Some(gw(&self.cond, "t_embedding.mlp_in.bias")?),
        )?
        .silu()?;
        lin(
            &h,
            gw(&self.cond, "t_embedding.mlp_out.weight")?,
            Some(gw(&self.cond, "t_embedding.mlp_out.bias")?),
        )
    }

    /// Gather rows of `embed_image_indicator.weight` [2,hidden] by 0/1 ids → [1,L,hidden].
    fn image_indicator_embed(&self, ids: &[i64], l: usize) -> Result<Tensor> {
        let eii = gw(&self.cond, "embed_image_indicator.weight")?.to_vec_f32()?;
        let h = self.hidden;
        let mut out = vec![0f32; l * h];
        for (i, &id) in ids.iter().enumerate() {
            let src = (id as usize) * h;
            out[i * h..(i + 1) * h].copy_from_slice(&eii[src..src + h]);
        }
        Tensor::from_vec(out, Shape::from_dims(&[1, l, h]), self.device.clone())?.to_dtype(DType::BF16)
    }

    /// Full forward → velocity-trunk [1,L,128] f32.
    pub fn forward(
        &self,
        x_in: &Tensor,
        llm_in: &Tensor,
        t_in: &Tensor,
        indicator: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mut dbg: Option<&mut WMap>,
        max_layers: usize, // 0 = all 34; >0 truncates (cheap backward derisk)
    ) -> Result<Tensor> {
        let l = x_in.shape().dims()[1];
        let hidden = self.hidden;

        // masks from indicator (host): llm=3, image=2.
        let ind = indicator.to_vec_f32()?;
        let mut llm_m = vec![0f32; l];
        let mut img_m = vec![0f32; l];
        let mut img_ids = vec![0i64; l];
        for i in 0..l {
            let vi = ind[i];
            if vi > 2.5 && vi < 3.5 {
                llm_m[i] = 1.0;
            }
            if vi > 1.5 && vi < 2.5 {
                img_m[i] = 1.0;
                img_ids[i] = 1;
            }
        }
        let llm_mask = Tensor::from_vec(llm_m, Shape::from_dims(&[1, l, 1]), self.device.clone())?.to_dtype(DType::BF16)?;
        let img_mask = Tensor::from_vec(img_m, Shape::from_dims(&[1, l, 1]), self.device.clone())?.to_dtype(DType::BF16)?;

        let llm = llm_in.mul(&llm_mask)?;
        let x = x_in.mul(&img_mask)?;
        let x_proj = lin(&x, gw(&self.cond, "input_proj.weight")?, Some(gw(&self.cond, "input_proj.bias")?))?;
        if let Some(d) = dbg.as_deref_mut() {
            d.insert("input_proj_out".into(), x_proj.to_dtype(DType::F32)?);
        }
        let x = x_proj.mul(&img_mask)?;

        // t → adaln_input
        let t_cond = self.t_embedding(t_in, hidden)?.reshape(&[1, 1, hidden])?;
        let adaln = lin(&t_cond, gw(&self.cond, "adaln_proj.weight")?, Some(gw(&self.cond, "adaln_proj.bias")?))?.silu()?;

        // llm conditioning: RMSNorm(1e-6) → proj → mask
        let llm_c = llm.shape().dims()[2];
        let llm = flame_core::norm::rms_norm(&llm, &[llm_c], Some(gw(&self.cond, "llm_cond_norm.weight")?), EPS6)?;
        let llm = lin(&llm, gw(&self.cond, "llm_cond_proj.weight")?, Some(gw(&self.cond, "llm_cond_proj.bias")?))?.mul(&llm_mask)?;

        let mut h = x.add(&llm)?;
        h = h.add(&self.image_indicator_embed(&img_ids, l)?)?;
        if let Some(d) = dbg.as_deref_mut() {
            d.insert("h_pre".into(), h.to_dtype(DType::F32)?);
        }

        // 34 blocks, weights streamed per-layer (OT residency model).
        let n_layers = if max_layers == 0 { self.num_layers } else { max_layers.min(self.num_layers) };
        for li in 0..n_layers {
            let lw = self.load_layer(li)?;
            h = self.block(&h, &lw, &format!("layers.{li}."), &adaln, cos, sin)?;
            if let Some(d) = dbg.as_deref_mut() {
                match li {
                    0 => { d.insert("block0_out".into(), h.to_dtype(DType::F32)?); }
                    1 => { d.insert("block1_out".into(), h.to_dtype(DType::F32)?); }
                    8 => { d.insert("block8_out".into(), h.to_dtype(DType::F32)?); }
                    16 => { d.insert("block16_out".into(), h.to_dtype(DType::F32)?); }
                    33 => { d.insert("block33_out".into(), h.to_dtype(DType::F32)?); }
                    _ => {}
                }
            }
        }

        // final layer
        let fscale = lin(
            &adaln.silu()?,
            gw(&self.cond, "final_layer.adaln_modulation.weight")?,
            Some(gw(&self.cond, "final_layer.adaln_modulation.bias")?),
        )?
        .affine(1.0, 1.0)?;
        let hn = flame_core::layer_norm::layer_norm(&h, &[hidden], None, None, EPS6)?.mul(&fscale)?;
        let out = lin(&hn, gw(&self.cond, "final_layer.linear.weight")?, Some(gw(&self.cond, "final_layer.linear.bias")?))?;
        out.to_dtype(DType::F32)
    }
}

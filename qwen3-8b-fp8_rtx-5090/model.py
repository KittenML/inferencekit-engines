"""Qwen3-8B FP8: weight loading and the prefill / reference forward pass.

The fast single-token decode step lives in generate.py (CUDA kernels + one CUDA graph). This file
holds everything that is easier to read in PyTorch:
  - loading the checkpoint and fusing weights (q|k|v and gate|up are concatenated),
  - re-laying out FP8 weights into the "mma-tiled" order the decode GEMV streams,
  - a straightforward forward pass for many tokens (prefill), built from Triton FP8 matmuls
    and PyTorch SDPA, which doubles as the reference implementation for kernel tests.
"""
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

from kernels.prefill import fp8_linear, fp8_matmul
from kernels import prefill_fused as pff

FP8 = torch.float8_e4m3fn

# Shapes compiled into the CUDA decode kernels (kernels/decode.cu: HIDDEN, N_Q_HEADS, N_KV_HEADS, HEAD_DIM
# constants; the GEMV templates are instantiated for K = 4096 and K = 12288 only, and the SwiGLU epilogue
# splits gate|up at N/2). A checkpoint with other shapes would read out of bounds or give wrong logits,
# so loading refuses it. Layer count and vocab size are runtime parameters and may differ.
KERNEL_SHAPES = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                 "intermediate_size": 12288,
                 # the decode graph's logits buffer and argmax are sized for this vocabulary; a different
                 # size would make the lm_head kernel write past the end of the buffer
                 "vocab_size": 151936}


def check_supported(c: dict, path: str = "") -> None:
    """Raise ValueError unless config.json `c` is something this engine computes correctly: the kernels'
    hard-coded shapes, plain (unscaled) RoPE, SiLU, no attention bias, no sliding window, untied lm_head,
    and the 128x128-block FP8 W8A8 dynamic quantization recipe."""
    bad = []
    for k, want in KERNEL_SHAPES.items():
        if c.get(k) != want:
            bad.append(f"{k} = {c.get(k)!r}, the decode kernels are compiled for {want}")
    rope = c.get("rope_scaling") or {}
    rp = c.get("rope_parameters") or {}            # newer transformers config layout
    for src, d in (("rope_scaling", rope), ("rope_parameters", rp)):
        rtype = d.get("rope_type", d.get("type", "default"))
        if rtype != "default":
            bad.append(f"{src}.rope_type = {rtype!r}: only plain RoPE (rope_theta, no scaling) is implemented")
    if not (c.get("rope_theta") or rp.get("rope_theta")):
        bad.append("rope_theta missing")
    if c.get("rope_theta") and rp.get("rope_theta") and float(c["rope_theta"]) != float(rp["rope_theta"]):
        # two layouts disagree: refuse rather than guess which one the reference implementation reads
        bad.append(f"rope_theta = {c['rope_theta']!r} but rope_parameters.rope_theta = {rp['rope_theta']!r}")
    if c.get("hidden_act", "silu") != "silu":
        bad.append(f"hidden_act = {c.get('hidden_act')!r}, only silu is implemented")
    if c.get("attention_bias", False):
        bad.append("attention_bias = true is not implemented")
    if c.get("use_sliding_window", False):
        bad.append("use_sliding_window = true is not implemented")
    if c.get("tie_word_embeddings", False):
        bad.append("tie_word_embeddings = true is not supported (the lm_head must be its own bf16 tensor)")
    q = c.get("quantization_config", {})
    if not (q.get("quant_method") == "fp8" and q.get("weight_block_size") == [128, 128]
            and q.get("activation_scheme") == "dynamic"):
        bad.append(f"quantization_config = {q!r}: expected fp8, weight_block_size [128, 128], dynamic activations")
    if bad:
        raise ValueError(f"unsupported checkpoint {path or ''}: this engine is built for Qwen3-8B FP8 "
                         f"(block-128 W8A8) only.\n  - " + "\n  - ".join(bad))


@dataclass
class Config:
    hidden: int = 4096
    layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    inter: int = 12288
    vocab: int = 151936
    eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @staticmethod
    def from_json(path):
        c = json.load(open(os.path.join(path, "config.json")))
        check_supported(c, path)
        theta = c.get("rope_theta") or c.get("rope_parameters", {}).get("rope_theta")
        return Config(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                      c["num_key_value_heads"], c["head_dim"], c["intermediate_size"],
                      c["vocab_size"], c["rms_norm_eps"], float(theta))


def tile_fp8(w: torch.Tensor, kmajor: bool = False) -> torch.Tensor:
    """Row-major [N, K] fp8 -> mma-tiled layout (same bytes, different order).

    Element (n, k) with n = 8*ng + g and k = 64*kc + 16*q + 4*t + b moves to position
    [ng, kc, g, t, q, b] (group-major) or [kc, ng, g, t, q, b] (K-major). A 512-byte tile [ng, kc]
    holds 32 lanes x 16 bytes, and lane L = 4*g + t gets the bytes {4t..4t+3, 16+4t.., 32+4t..,
    48+4t..} of row g: exactly the B-fragments of two consecutive m16n8k32 mma steps (see decode.cu).
    K-major puts tile kc of every row group side by side, so all warps stream one contiguous window.
    """
    N, K = w.shape
    v = w.view(torch.uint8).view(N // 8, 8, K // 64, 4, 4, 4)            # [ng, g, kc, q, t, b]
    order = (2, 0, 1, 4, 3, 5) if kmajor else (0, 2, 1, 4, 3, 5)
    return v.permute(*order).contiguous().view(FP8).view(N, K)


def untile_fp8(wt: torch.Tensor, kmajor: bool = False, out: torch.Tensor = None) -> torch.Tensor:
    """Inverse of tile_fp8 (used by prefill, which wants row-major weights)."""
    N, K = wt.shape
    if kmajor:
        v = wt.view(torch.uint8).view(K // 64, N // 8, 8, 4, 4, 4).permute(1, 2, 0, 4, 3, 5)
    else:
        v = wt.view(torch.uint8).view(N // 8, K // 64, 8, 4, 4, 4).permute(0, 2, 1, 4, 3, 5)
    if out is None:
        return v.contiguous().view(FP8).view(N, K)
    out = out.view(torch.uint8)[: N * K].view(N // 8, 8, K // 64, 4, 4, 4)
    out.copy_(v)
    return out.view(FP8).view(N, K)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Same op order as the reference: fp32 normalize, round to bf16, multiply by weight in bf16.
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Layer:
    pass


class Qwen3FP8:
    """Weights + KV cache + a PyTorch/Triton forward for any number of tokens."""

    # Prefill implementations (start == 0): "fused" = forward_fused (4 fused Triton kernels per layer around
    # the GEMMs, bitwise identical to "ref"; used for M >= 16, shorter prompts fall back to "ref");
    # "ref" = the op-by-op PyTorch forward below. Setting any other name raises.
    PREFILL_IMPLS = ("fused", "ref")

    @property
    def prefill_impl(self) -> str:
        return self._prefill_impl

    @prefill_impl.setter
    def prefill_impl(self, name: str) -> None:
        if name not in self.PREFILL_IMPLS:
            raise ValueError(f"unknown prefill_impl {name!r}; choose one of {self.PREFILL_IMPLS}")
        self._prefill_impl = name

    def __init__(self, path: str, max_seq: int = 4096, device: str = "cuda", kmajor: bool = False,
                 rowmajor_copy: bool = True):
        self.cfg = cfg = Config.from_json(path)
        self.kmajor = kmajor
        # Prefill wants row-major weights. Keeping a second copy (+7 GB) avoids re-laying out every
        # matrix on every prefill (in a development profile that re-layout was 36% of prefill GPU time
        # at 128 tokens). prefill_copy=False instead reads the
        # tiled weights directly in the prefill GEMM (fused path) or un-tiles per matrix (ref path).
        self.rowmajor_copy = rowmajor_copy
        self.prefill_impl = "fused"
        self.last_prefill_path = None   # "fused" or "ref": what the last start == 0 forward() actually ran
        self.max_seq = max_seq
        self.device = device
        tensors = self._read_safetensors(path, device)

        def take(name):
            return tensors.pop(name)

        self.embed = take("model.embed_tokens.weight")            # [V, H] bf16
        self.norm = take("model.norm.weight")                      # [H] bf16
        self.lm_head = take("lm_head.weight").contiguous()         # [V, H] bf16
        assert self.embed.dtype == torch.bfloat16 and self.lm_head.dtype == torch.bfloat16

        self.layers = []
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            L = Layer()
            L.ln1 = take(p + "input_layernorm.weight")
            L.ln2 = take(p + "post_attention_layernorm.weight")
            L.q_norm = take(p + "self_attn.q_norm.weight")
            L.k_norm = take(p + "self_attn.k_norm.weight")

            def lin(*names):
                ws = [take(p + n + ".weight") for n in names]
                ss = [take(p + n + ".weight_scale_inv") for n in names]
                for w, s in zip(ws, ss):
                    # This checkpoint stores the block scales in bf16; the reference upcasts them to
                    # fp32 (exact), and so do we.
                    assert w.dtype == FP8 and s.dtype in (torch.float32, torch.bfloat16), (w.dtype, s.dtype)
                    assert s.shape == ((w.shape[0] + 127) // 128, (w.shape[1] + 127) // 128)
                ss = [s.float() for s in ss]
                w = torch.cat(ws, 0) if len(ws) > 1 else ws[0]
                s = torch.cat(ss, 0).contiguous() if len(ss) > 1 else ss[0].contiguous()
                return tile_fp8(w, kmajor), s, (w.contiguous() if rowmajor_copy else None)

            L.w_qkv, L.s_qkv, L.r_qkv = lin("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")  # [6144, 4096]
            L.w_o, L.s_o, L.r_o = lin("self_attn.o_proj")                                             # [4096, 4096]
            L.w_gu, L.s_gu, L.r_gu = lin("mlp.gate_proj", "mlp.up_proj")                              # [24576, 4096]
            L.w_d, L.s_d, L.r_d = lin("mlp.down_proj")                                                # [4096, 12288]
            self.layers.append(L)
            torch.cuda.empty_cache()
        assert not tensors, f"unused tensors: {list(tensors)[:5]}"

        # KV cache: [layer, kv_head, position, head_dim], bf16.
        shape = (cfg.layers, cfg.n_kv_heads, max_seq, cfg.head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)

        # RoPE tables, computed with the same ops as the reference (fp32, then rounded to bf16).
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim))
        pos = torch.arange(max_seq, device=device).float()
        freqs = pos[:, None] * inv_freq.to(device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(torch.bfloat16).contiguous()      # [max_seq, 128]
        self.sin = emb.sin().to(torch.bfloat16).contiguous()

        # Scratch for un-tiling one weight matrix at a time during prefill when there is no row-major copy.
        self._untile_buf = None if rowmajor_copy else torch.empty(2 * cfg.inter * cfg.hidden, dtype=torch.uint8,
                                                                  device=device)

    @staticmethod
    def _read_safetensors(path, device):
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        out = {}
        for fname in sorted(set(idx.values())):
            with safe_open(os.path.join(path, fname), framework="pt", device=device) as f:
                for k in f.keys():
                    out[k] = f.get_tensor(k)
        return out

    # --------------------------------------------------------------------------------------------
    # Prefill / reference forward: tokens at positions [start, start + M).
    # Either start == 0 (causal prefill of M tokens) or M == 1 (one decode step).
    # --------------------------------------------------------------------------------------------
    def _linear(self, x, w_tiled, scale, w_rowmajor=None):
        w = w_rowmajor if w_rowmajor is not None else untile_fp8(w_tiled, self.kmajor, self._untile_buf)
        return fp8_linear(x, w, scale)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        """tokens: [M] int64 on device. Returns bf16 logits of the LAST token, shape [V]."""
        cfg = self.cfg
        M = tokens.shape[0]
        assert start == 0 or M == 1
        if start == 0 and M >= 16 and self.prefill_impl == "fused" and (self.rowmajor_copy or self.kmajor):
            self.last_prefill_path = "fused"
            return self.forward_fused(tokens)
        if start == 0:
            self.last_prefill_path = "ref"
        end = start + M
        h = self.embed[tokens]                                     # [M, H]
        cos = self.cos[start:end][:, None, :]                      # [M, 1, 128]
        sin = self.sin[start:end][:, None, :]
        nq, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        for li, L in enumerate(self.layers):
            x = rms_norm(h, L.ln1, cfg.eps)
            qkv = self._linear(x, L.w_qkv, L.s_qkv, L.r_qkv)
            q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
            q = rms_norm(q.view(M, nq, hd), L.q_norm, cfg.eps)
            k = rms_norm(k.view(M, nkv, hd), L.k_norm, cfg.eps)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            self.k_cache[li, :, start:end] = k.transpose(0, 1)
            self.v_cache[li, :, start:end] = v.view(M, nkv, hd).transpose(0, 1)
            K = self.k_cache[li, :, :end][None]                    # [1, 8, end, 128]
            V = self.v_cache[li, :, :end][None]
            attn = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], K, V, is_causal=(M > 1), scale=hd ** -0.5, enable_gqa=True)
            attn = attn[0].transpose(0, 1).reshape(M, nq * hd)
            h = h + self._linear(attn.contiguous(), L.w_o, L.s_o, L.r_o)
            x = rms_norm(h, L.ln2, cfg.eps)
            gu = self._linear(x, L.w_gu, L.s_gu, L.r_gu)
            g, u = gu.split([cfg.inter, cfg.inter], dim=-1)
            a = F.silu(g) * u
            h = h + self._linear(a.contiguous(), L.w_d, L.s_d, L.r_d)
        x = rms_norm(h[-1:], self.norm, cfg.eps)
        return F.linear(x, self.lm_head)[0]

    def _mm(self, q, s, w_tiled, scale, w_rowmajor):
        # Row-major prefill copy if kept (fastest); otherwise read the decode GEMV's K-major tiles
        # directly (no extra VRAM, same bits, ~13-30% slower GEMMs).
        if w_rowmajor is not None:
            return fp8_matmul(q, s, w_rowmajor, scale)
        assert self.kmajor
        return fp8_matmul(q, s, w_tiled, scale, kmajor_tiled=True)

    @torch.no_grad()
    def forward_fused(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prefill of M tokens at positions [0, M): same result, bit for bit, as forward() (the "ref"
        path), with the per-layer elementwise work fused (kernels/prefill_fused.py):
          rmsnorm_quant   (residual add +) RMSNorm + act-quant           replaces ~10 PyTorch kernels
          qk_norm_rope_kv q/k RMSNorm + RoPE + K/V cache write            replaces ~30
          act_quant_rows  act-quant of the attention output, 8 groups per program
          swiglu_quant    SiLU(gate) * up + act-quant                     replaces 3
        The FP8 GEMMs and SDPA are the same kernels with the same inputs."""
        cfg = self.cfg
        M = tokens.shape[0]
        nq, nkv, hd, eps = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.eps
        h = self.embed[tokens]                                     # [M, H]
        h_alt = torch.empty_like(h)
        q_buf = torch.empty(M, nq, hd, dtype=torch.bfloat16, device=h.device)
        res = None
        for li, L in enumerate(self.layers):
            if res is None:
                xq, xs = pff.rmsnorm_quant(h, L.ln1, eps)
            else:                                                  # h = h + down(...), then RMSNorm
                xq, xs = pff.rmsnorm_quant(h, L.ln1, eps, res=res, h_out=h_alt)
                h, h_alt = h_alt, h
            qkv = self._mm(xq, xs, L.w_qkv, L.s_qkv, L.r_qkv)
            pff.qk_norm_rope_kv(qkv, L.q_norm, L.k_norm, self.cos, self.sin, q_buf, self.k_cache[li],
                                self.v_cache[li], eps, 0, nq, nkv)
            K = self.k_cache[li, :, :M][None]                      # [1, 8, M, 128]
            V = self.v_cache[li, :, :M][None]
            attn = F.scaled_dot_product_attention(
                q_buf.transpose(0, 1)[None], K, V, is_causal=True, scale=hd ** -0.5, enable_gqa=True)
            attn = attn[0].transpose(0, 1).reshape(M, nq * hd)
            aq, as_ = pff.act_quant_rows(attn.contiguous())
            o = self._mm(aq, as_, L.w_o, L.s_o, L.r_o)
            xq, xs = pff.rmsnorm_quant(h, L.ln2, eps, res=o, h_out=h_alt)   # h = h + o_proj(...)
            h, h_alt = h_alt, h
            gu = self._mm(xq, xs, L.w_gu, L.s_gu, L.r_gu)
            aq, as_ = pff.swiglu_quant(gu)
            res = self._mm(aq, as_, L.w_d, L.s_d, L.r_d)
        h_last = h[-1:] + res[-1:]                                 # the last residual add, last row only
        x = rms_norm(h_last, self.norm, eps)
        return F.linear(x, self.lm_head)[0]

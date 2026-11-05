"""Fused (bias add) + RMSNorm across heads + split RoPE for the attention queries and keys.

Computes, for x [B, T, C] (bf16, C = H * 2R) and every row b, what the reference computes with
    q = linear(x) + bias                      (bf16 add, rounded to bf16; bias optional)
    q = RMSNorm(C, eps, affine)(q)            (fp32: x * rsqrt(mean(x^2) + eps) * weight, rounded once to bf16)
    q = apply_split_rotary_emb(q, (cos, sin)) (fp32: out = x*cos; out1 += -sin*x2; out2 += sin*x1; rounded to bf16)
in one pass per token: the token's cos/sin (fp32, [H, R]) are read once for all B rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _norm_rope_kernel(X, BIAS, W, COS, SIN, OUT, NB, stride_b, stride_t, eps,
                      C: tl.constexpr, H: tl.constexpr, R: tl.constexpr, HAS_BIAS: tl.constexpr,
                      HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, C)
    w = tl.load(W + offs).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + offs).to(tl.float32)
    if HAS_ROPE:
        hr = tl.arange(0, H)[:, None] * R + tl.arange(0, R)[None, :]
        cs = tl.load(COS + t * (H * R) + hr)
        sn = tl.load(SIN + t * (H * R) + hr)
    for b in range(NB):
        base = b * stride_b + t * stride_t
        x = tl.load(X + base + offs).to(tl.float32)
        if HAS_BIAS:
            x = (x + bias).to(tl.bfloat16).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / C
        r = tl.math.rsqrt(ms + eps)
        y = ((x * r) * w).to(tl.bfloat16)
        if HAS_ROPE:
            y3 = tl.reshape(y.to(tl.float32), (H, 2, R))
            y3 = tl.permute(y3, (0, 2, 1))  # [H, R, 2]
            first, second = tl.split(y3)
            o1 = tl.fma(-sn, second, first * cs)
            o2 = tl.fma(sn, first, second * cs)
            o = tl.permute(tl.join(o1, o2), (0, 2, 1))  # [H, 2, R]
            y = tl.reshape(o, (C,)).to(tl.bfloat16)
        tl.store(OUT + base + offs, y)


@triton.jit
def _norm_rope_kernel_bt(X, BIAS, W, COS, SIN, OUT, NB, stride_b, stride_t, eps,
                         C: tl.constexpr, H: tl.constexpr, R: tl.constexpr, HAS_BIAS: tl.constexpr,
                         HAS_ROPE: tl.constexpr):
    # one program per (row b, token t), b fastest: the programs of one token run together, so its cos/sin
    # are read from DRAM once and from L2 for the other rows
    pid = tl.program_id(0)
    b = pid % NB
    t = pid // NB
    offs = tl.arange(0, C)
    base = b * stride_b + t * stride_t
    x = tl.load(X + base + offs).to(tl.float32)
    w = tl.load(W + offs).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + offs).to(tl.float32)
        x = (x + bias).to(tl.bfloat16).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / C
    r = tl.math.rsqrt(ms + eps)
    y = ((x * r) * w).to(tl.bfloat16)
    if HAS_ROPE:
        hr = tl.arange(0, H)[:, None] * R + tl.arange(0, R)[None, :]
        cs = tl.load(COS + t * (H * R) + hr)
        sn = tl.load(SIN + t * (H * R) + hr)
        y3 = tl.reshape(y.to(tl.float32), (H, 2, R))
        y3 = tl.permute(y3, (0, 2, 1))  # [H, R, 2]
        first, second = tl.split(y3)
        o1 = tl.fma(-sn, second, first * cs)
        o2 = tl.fma(sn, first, second * cs)
        o = tl.permute(tl.join(o1, o2), (0, 2, 1))  # [H, 2, R]
        y = tl.reshape(o, (C,)).to(tl.bfloat16)
    tl.store(OUT + base + offs, y)


CFG = {"grid": "t", "warps": 8}


@triton_op("fastltx::norm_rope", mutates_args={})
def norm_rope(x: torch.Tensor, bias: torch.Tensor | None, weight: torch.Tensor, cos: torch.Tensor | None,
              sin: torch.Tensor | None, eps: float, heads: int) -> torch.Tensor:
    B, T, C = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    has_rope = cos is not None
    R = C // heads // 2
    dummy = weight
    args = (x, bias if bias is not None else dummy, weight, cos if has_rope else dummy, sin if has_rope else dummy,
            out, B, T * C, C, eps)
    kw = dict(C=C, H=heads, R=R, HAS_BIAS=bias is not None, HAS_ROPE=has_rope)
    if CFG["grid"] == "bt":
        wrap_triton(_norm_rope_kernel_bt)[(T * B,)](*args, **kw, num_warps=CFG["warps"])
    else:
        wrap_triton(_norm_rope_kernel)[(T,)](*args, **kw, num_warps=8 if C >= 4096 else 4)
    return out


def tables(emb):
    """(cos, sin) of diffusers' split rope, [b, H, T, R] -> [T, H, R] contiguous fp32 (batch row 0)."""
    cos, sin = emb
    assert cos.ndim == 4, cos.shape
    return (cos[0].permute(1, 0, 2).contiguous(), sin[0].permute(1, 0, 2).contiguous())

"""Fused prefill kernels that reproduce the reference PyTorch ops bit for bit.

The reference prefill (model.Qwen3FP8.forward) runs every RMSNorm, RoPE, SwiGLU and residual add as
separate PyTorch kernels: ~45 kernels per layer around the 4 FP8 GEMMs. At 1024 tokens they cost
~12 ms of the ~60 ms prefill (nsys). The kernels here do the same work in 4
kernels per layer and give identical bits, because they repeat PyTorch's arithmetic exactly:

* Every intermediate PyTorch rounds to bf16 is rounded to bf16 here too, at the same point.
* No FMA contraction (`enable_fp_fusion=False`): PyTorch rounds x*x before adding it to the sum.
* RMSNorm's `mean(x^2)` uses PyTorch's reduction order (valid for >= 16 rows: with fewer rows PyTorch
  picks a wider block and a different order, so forward() uses the fused path only for M >= 16). For a contiguous row of n floats
  (ATen/native/cuda/Reduce.cuh, "vectorize along input", one warp per row, blockDim (32, 16)):
    thread t (of 32) keeps 4 partial sums a_i; it adds x[128*j + 4*t + i] to a_i for j = 0, 1, ...
    in order (one vector of 4 per step); then ((a_0 + a_1) + a_2) + a_3; then a warp shuffle tree
    with decreasing offsets: v[t] += v[t + 16], then +8, +4, +2, +1; mean = sum * (1/n).
* act-quant uses the same expressions as kernels/prefill.py (Triton `/` = div.full.f32).
* SiLU is x / (1 + expf(-x)) with libdevice expf and IEEE division (PyTorch's silu_kernel).
  Triton flushes fp32 denormals to zero (PyTorch keeps them), so for |g| < 2^-125, where silu(g) is
  zero or denormal, the value comes from a 64K-entry table made with torch's own silu. Checked on all
  65,536 bf16 gate values.
* Limit of the claim: an intermediate below 2^-126 (~1.2e-38) is flushed to zero here and kept as a
  denormal by PyTorch (x^2 in the norms for |x| < 1.1e-19, silu(g)*u underflowing). GEMM outputs here
  are 0 or far above that (FP8 products times block scales), and the end-to-end checks are memcmp-equal.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

FP8 = torch.float8_e4m3fn
NOFMA = dict(enable_fp_fusion=False)


@triton.jit
def _warp_tree(v, N: tl.constexpr):
    # v: [..., 32] partial sums of the 32 lanes -> [...]: PyTorch's shfl_down tree (offsets 16..1).
    v = tl.sum(tl.reshape(v, [N, 2, 16]), axis=1)
    v = tl.sum(tl.reshape(v, [N, 2, 8]), axis=1)
    v = tl.sum(tl.reshape(v, [N, 2, 4]), axis=1)
    v = tl.sum(tl.reshape(v, [N, 2, 2]), axis=1)
    v = tl.sum(tl.reshape(v, [N, 2, 1]), axis=1)
    return tl.reshape(v, [N])


@triton.jit
def _combine4(acc, N: tl.constexpr):
    # acc: [N, 32, 4] -> [N, 32]: ((a0 + a1) + a2) + a3. A masked sum of one value and zeros is exact.
    ii = tl.arange(0, 4)[None, None, :]
    c0 = tl.sum(tl.where(ii == 0, acc, 0.0), axis=2)
    c1 = tl.sum(tl.where(ii == 1, acc, 0.0), axis=2)
    c2 = tl.sum(tl.where(ii == 2, acc, 0.0), axis=2)
    c3 = tl.sum(tl.where(ii == 3, acc, 0.0), axis=2)
    return ((c0 + c1) + c2) + c3


@triton.jit
def _rmsnorm_quant_kernel(X, RES, HOUT, W, Q, S, eps, HAS_RES: tl.constexpr, H: tl.constexpr):
    """One program per token row of H (=4096) values.
    h = bf16(x + res) if HAS_RES (stored to HOUT) else x;  y = bf16(w * bf16(h * rsqrt(mean(h^2) + eps)));
    then act-quant of y per 128-group -> Q (fp8), S (fp32)."""
    row = tl.program_id(0).to(tl.int64)
    x_row = X + row * H
    r_row = RES + row * H
    # Pass 1: sum of squares in PyTorch's order (see module docstring).
    tt = tl.arange(0, 32)[:, None]
    ii = tl.arange(0, 4)[None, :]
    offs = tt * 4 + ii                                     # [32, 4]
    acc = tl.zeros([32, 4], dtype=tl.float32)
    for j in tl.static_range(H // 128):
        h = tl.load(x_row + j * 128 + offs)
        if HAS_RES:
            h = (h.to(tl.float32) + tl.load(r_row + j * 128 + offs).to(tl.float32)).to(tl.bfloat16)
        hf = h.to(tl.float32)
        acc = acc + hf * hf
    v = _combine4(tl.reshape(acc, [1, 32, 4]), 1)          # [1, 32]
    ssum = _warp_tree(v, 1)                                # [1]
    ms = ssum * (1.0 / H)
    ms = ms + eps
    r = tl.math.rsqrt(ms)                                  # [1]
    # Pass 2: normalise, weight, quantise, 128-groups as rows of a [H/128, 128] tile.
    g = tl.arange(0, H // 128)[:, None]
    c = tl.arange(0, 128)[None, :]
    o2 = g * 128 + c
    h = tl.load(x_row + o2)
    if HAS_RES:
        h = (h.to(tl.float32) + tl.load(r_row + o2).to(tl.float32)).to(tl.bfloat16)
        tl.store(HOUT + row * H + o2, h)
    y = (h.to(tl.float32) * tl.reshape(r, [1, 1])).to(tl.bfloat16)
    w = tl.load(W + o2)
    y = (w.to(tl.float32) * y.to(tl.float32)).to(tl.bfloat16)
    yf = y.to(tl.float32)
    s = tl.max(tl.abs(yf), axis=1) / 448.0                 # [H/128]
    q = (yf / s[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + row * H + o2, q)
    tl.store(S + row * (H // 128) + tl.arange(0, H // 128), s)


@triton.jit
def _head_rms_rope(x_ptr, w_ptr, cos_row, sin_row, out_ptr, out_head_stride, eps, NH: tl.constexpr):
    """NH heads of 128 at x_ptr (contiguous): per-head RMSNorm (PyTorch order) + weight + RoPE (rotate_half)."""
    hh = tl.arange(0, NH)[:, None, None]
    tt = tl.arange(0, 32)[None, :, None]
    ii = tl.arange(0, 4)[None, None, :]
    x = tl.load(x_ptr + hh * 128 + tt * 4 + ii).to(tl.float32)   # [NH, 32, 4]
    v = _combine4(x * x, NH)                                      # [NH, 32]
    ssum = _warp_tree(v, NH)                                      # [NH]
    ms = ssum * (1.0 / 128)
    ms = ms + eps
    r = tl.math.rsqrt(ms)[:, None]                                # [NH, 1]
    h2 = tl.arange(0, NH)[:, None]
    d = tl.arange(0, 64)[None, :]
    x1 = tl.load(x_ptr + h2 * 128 + d).to(tl.float32)
    x2 = tl.load(x_ptr + h2 * 128 + 64 + d).to(tl.float32)
    w1 = tl.load(w_ptr + d).to(tl.float32)
    w2 = tl.load(w_ptr + 64 + d).to(tl.float32)
    n1 = (w1 * (x1 * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    n2 = (w2 * (x2 * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    c1 = tl.load(cos_row + d).to(tl.float32)
    c2 = tl.load(cos_row + 64 + d).to(tl.float32)
    s1 = tl.load(sin_row + d).to(tl.float32)
    s2 = tl.load(sin_row + 64 + d).to(tl.float32)
    # q*cos + rotate_half(q)*sin, rotate_half(q) = cat(-q2, q1); each product and the sum round to bf16.
    o1 = ((n1 * c1).to(tl.bfloat16).to(tl.float32) + ((-n2) * s1).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    o2 = ((n2 * c2).to(tl.bfloat16).to(tl.float32) + (n1 * s2).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + h2 * out_head_stride + d, o1)
    tl.store(out_ptr + h2 * out_head_stride + 64 + d, o2)


@triton.jit
def _qk_norm_rope_kv_kernel(QKV, WQ, WK, COS, SIN, QOUT, KC, VC, eps, start, max_seq,
                            NQ: tl.constexpr, NKV: tl.constexpr):
    """One program per token m: q/k RMSNorm + RoPE; q -> QOUT[m] ([M, NQ, 128]); k, v -> cache[:, start+m]."""
    m = tl.program_id(0).to(tl.int64)
    pos = start + m
    W_ROW: tl.constexpr = (NQ + 2 * NKV) * 128
    base = QKV + m * W_ROW
    cos_row = COS + pos * 128
    sin_row = SIN + pos * 128
    _head_rms_rope(base, WQ, cos_row, sin_row, QOUT + m * (NQ * 128), 128, eps, NQ)
    _head_rms_rope(base + NQ * 128, WK, cos_row, sin_row, KC + pos * 128, max_seq * 128, eps, NKV)
    h2 = tl.arange(0, NKV)[:, None]
    d = tl.arange(0, 128)[None, :]
    vv = tl.load(base + (NQ + NKV) * 128 + h2 * 128 + d)
    tl.store(VC + h2 * (max_seq * 128) + pos * 128 + d, vv)


@triton.jit
def _swiglu_quant_kernel(GU, Q, S, A, LUT, I: tl.constexpr, G: tl.constexpr, STORE_A: tl.constexpr):
    """a = bf16(bf16(silu(g)) * u) for G 128-groups of one row, then act-quant -> Q, S.
    STORE_A (tests only) also writes a."""
    row = tl.program_id(0).to(tl.int64)
    gb = tl.program_id(1)
    grp = gb * G + tl.arange(0, G)[:, None]
    c = tl.arange(0, 128)[None, :]
    col = grp * 128 + c
    graw = tl.load(GU + row * (2 * I) + col)
    g = graw.to(tl.float32)
    u = tl.load(GU + row * (2 * I) + I + col).to(tl.float32)
    sg = libdevice.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16)
    bits = graw.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    den = (bits & 0x7F80) <= 0x0080                        # |g| < 2^-125: silu(g) ~ g/2 would be a
    # denormal, which Triton flushes to zero; take torch's value from the table instead.
    sg = tl.where(den, tl.load(LUT + bits, mask=den, other=0.0), sg).to(tl.float32)
    a = (sg * u).to(tl.bfloat16).to(tl.float32)
    if STORE_A:
        tl.store(A + row * I + col, a.to(tl.bfloat16))
    s = tl.max(tl.abs(a), axis=1) / 448.0
    q = (a / s[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + row * I + col, q)
    tl.store(S + row * (I // 128) + gb * G + tl.arange(0, G), s)


@triton.jit
def _act_quant_rows_kernel(X, Q, S, G: tl.constexpr):
    """Same math as prefill._act_quant_kernel, G 128-groups per program."""
    pid = tl.program_id(0).to(tl.int64)
    grp = pid * G + tl.arange(0, G)[:, None]
    offs = grp * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(X + offs).to(tl.float32)
    s = tl.max(tl.abs(x), axis=1) / 448.0
    q = (x / s[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + offs, q)
    tl.store(S + pid * G + tl.arange(0, G), s)


# ------------------------------------------------------------------------------------------------
def rmsnorm_quant(x, w, eps, res=None, h_out=None, q=None, s=None):
    M, H = x.shape
    q = torch.empty(M, H, dtype=FP8, device=x.device) if q is None else q
    s = torch.empty(M, H // 128, dtype=torch.float32, device=x.device) if s is None else s
    if res is not None:
        assert h_out is not None and h_out.data_ptr() != x.data_ptr()
    _rmsnorm_quant_kernel[(M,)](x, res if res is not None else x, h_out if h_out is not None else x, w, q, s,
                                eps, HAS_RES=res is not None, H=H, num_warps=4, **NOFMA)
    return q, s


def qk_norm_rope_kv(qkv, wq, wk, cos, sin, q_out, k_cache_l, v_cache_l, eps, start, nq, nkv):
    M = qkv.shape[0]
    _qk_norm_rope_kv_kernel[(M,)](qkv, wq, wk, cos, sin, q_out, k_cache_l, v_cache_l, eps, start,
                                  k_cache_l.shape[1], NQ=nq, NKV=nkv, num_warps=4, **NOFMA)


_silu_lut = {}


def silu_lut(device):
    """torch's silu of every bf16 bit pattern (index = bits as uint16)."""
    key = str(device)
    if key not in _silu_lut:
        allb = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
        _silu_lut[key] = torch.nn.functional.silu(allb).contiguous()
    return _silu_lut[key]


def swiglu_quant(gu, q=None, s=None, G=8, a_out=None):
    M, I2 = gu.shape
    I = I2 // 2
    q = torch.empty(M, I, dtype=FP8, device=gu.device) if q is None else q
    s = torch.empty(M, I // 128, dtype=torch.float32, device=gu.device) if s is None else s
    _swiglu_quant_kernel[(M, I // 128 // G)](gu, q, s, a_out if a_out is not None else gu, silu_lut(gu.device),
                                             I=I, G=G, STORE_A=a_out is not None, num_warps=4, **NOFMA)
    return q, s


def act_quant_rows(x, G=8):
    M, K = x.shape
    assert x.is_contiguous() and (M * K // 128) % G == 0
    q = torch.empty(M, K, dtype=FP8, device=x.device)
    s = torch.empty(M, K // 128, dtype=torch.float32, device=x.device)
    _act_quant_rows_kernel[(M * K // 128 // G,)](x, q, s, G=G, num_warps=4, **NOFMA)
    return q, s

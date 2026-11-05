"""Batch-1 NVFP4 MoE decode for vLLM 0.31.0's FLASHINFER_CUTLASS backend (Gemma 4, gelu_tanh gated).

At one token, flashinfer_cutlass_fused_moe runs expert-map building, row expansion, stride setup, two
grouped GEMMs whose tiles waste most rows, a separate activation kernel and a finalize. This path keeps the
same weights (the backend's own packed FP4 tensors and swizzled FP8 block scales, read in place), the same
input quantization (vLLM's prepare step has already quantized the input to NVFP4 with the checkpoint's
static global scale) and the same scheme for the intermediate (NVFP4 with the checkpoint's static fc2 input
scale, a2_gscale), and replaces the rest with three Triton kernels:

1. _gemm1: a gather-GEMV over the 8 chosen experts' fc1 rows. Per 8-element word, the dot product of the
   e2m1 codes is exact (integers); it is scaled by the two FP8 block scales (exact) and summed over blocks in
   fp32, then multiplied by g1_alphas and rounded to bf16, as the grouped GEMM's epilogue writes it.
2. _act_quant: as the baseline's vendor kernel computes it: GELU with erf (the vendor maps gelu_tanh to its
   erf GELU), gelu(gate) * up rounded to bf16, then NVFP4 quantization of the intermediate: per 16 values,
   sf = fp8_e4m3(a2_gscale * (amax * (1/6))), code = e2m1_rn(x * rcp(sf * rcp(a2_gscale))) in fp32.
3. _gemm2_combine: the fc2 gather-GEMV for every output row and all 8 experts, each expert's result times
   g2_alphas rounded to bf16, then summed with the routing weights in top-k order in fp32, rounded to bf16.

This is not bit-equal to the vendor path: summation order and the fc2 quantization arithmetic differ, so
intermediate FP4 codes flip (median relative L2 difference of the layer output 1.5e-2, tools/test_moe.py); the
output follows an fp64 emulation of the formula above (median 0, max 8.6e-5).
"""
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _e2m1(c):
    # value of a 4-bit e2m1 code (sign bit 3): 0, 0.5, 1, 1.5, 2, 3, 4, 6
    mag = c & 7
    v = tl.where(mag < 2, mag.to(tl.float32) * 0.5,
                 (2 + (mag & 1)).to(tl.float32) * tl.exp2(((mag >> 1) - 2).to(tl.float32)))
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _fp8(b):
    return b.to(tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _swz(m, k, KT):
    # offset of block scale (row m, block k) in the 128x4-swizzled layout of swizzle_blockscale
    r = m % 128
    return ((((m // 128) * KT + k // 4) * 32 + r % 32) * 4 + r // 32) * 4 + k % 4


@triton.jit
def _v2(c):
    # twice the value of a 4-bit e2m1 code, as an integer: 0 1 2 3 4 6 8 12, sign from bit 3
    m = c & 7
    v = tl.where(m < 4, m, tl.where(m < 6, 2 * m - 4, 4 * m - 16))
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _wdot(w, a0, a1, a2, a3):
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 t1, t2, t3, h0, h1, h2, h3, acc;
        .reg .b8 b0, b1, b2, b3;
        .reg .f16 lo, hi;
        .reg .f32 flo, fhi;
        shr.b32 t1, $1, 8;
        shr.b32 t2, $1, 16;
        shr.b32 t3, $1, 24;
        cvt.u8.u32 b0, $1;
        cvt.u8.u32 b1, t1;
        cvt.u8.u32 b2, t2;
        cvt.u8.u32 b3, t3;
        cvt.rn.f16x2.e2m1x2 h0, b0;
        cvt.rn.f16x2.e2m1x2 h1, b1;
        cvt.rn.f16x2.e2m1x2 h2, b2;
        cvt.rn.f16x2.e2m1x2 h3, b3;
        mul.rn.f16x2 acc, h0, $2;
        fma.rn.f16x2 acc, h1, $3, acc;
        fma.rn.f16x2 acc, h2, $4, acc;
        fma.rn.f16x2 acc, h3, $5, acc;
        mov.b32 {lo, hi}, acc;
        cvt.f32.f16 flo, lo;
        cvt.f32.f16 fhi, hi;
        add.f32 $0, flo, fhi;
        }""", "=f,r,r,r,r,r", [w, a0, a1, a2, a3], dtype=tl.float32, is_pure=True, pack=1)
@triton.jit
def _pair(b):
    return tl.inline_asm_elementwise(
        """{
        .reg .b8 t;
        cvt.u8.u32 t, $1;
        cvt.rn.f16x2.e2m1x2 $0, t;
        }""", "=r,r", [b], dtype=tl.int32, is_pure=True, pack=1)
@triton.jit
def _gemm1(a_ptr, as_ptr, w_ptr, ws_ptr, ids_ptr, alpha_ptr, out_ptr,
           N1: tl.constexpr, KW: tl.constexpr, KT: tl.constexpr, BR: tl.constexpr, BKW: tl.constexpr):
    # w and a viewed as int32 words of 8 e2m1 codes (low nibble first); one block scale per 2 words.
    # _wdot: the 8 code products of a word in fp16 (exact: |sum| <= 144 in steps of 0.25), then fp32.
    slot = tl.program_id(0)
    rt = tl.program_id(1)
    e = tl.load(ids_ptr + slot).to(tl.int64)
    rows = rt * BR + tl.arange(0, BR)
    acc = tl.zeros([BR], dtype=tl.float32)
    wo = tl.arange(0, BKW)
    w_base = w_ptr + e * N1 * KW
    s_base = ws_ptr + e * N1 * (KT * 4)
    for k0 in tl.static_range(0, KW, BKW):
        kw = k0 + wo
        aw = tl.load(a_ptr + kw)
        a0 = tl.broadcast_to(_pair(aw & 255)[None, :], [BR, BKW])
        a1 = tl.broadcast_to(_pair((aw >> 8) & 255)[None, :], [BR, BKW])
        a2 = tl.broadcast_to(_pair((aw >> 16) & 255)[None, :], [BR, BKW])
        a3 = tl.broadcast_to(_pair((aw >> 24) & 255)[None, :], [BR, BKW])
        ww = tl.load(w_base + rows[:, None] * KW + kw[None, :])
        s = _wdot(ww, a0, a1, a2, a3)
        j = kw // 2
        asc = _fp8(tl.load(as_ptr + (j // 4) * 512 + j % 4))
        wsc = _fp8(tl.load(s_base + _swz(rows[:, None], j[None, :], KT)))
        acc += tl.sum(s * (wsc * asc[None, :]), axis=1)
    out = (acc * tl.load(alpha_ptr + e)).to(tl.bfloat16)
    tl.store(out_ptr + slot * N1 + rows, out)


@triton.jit
def _rows_dot(a_ptr, as_ptr, w_base, s_base, rows, KW: tl.constexpr, KT: tl.constexpr, BR: tl.constexpr,
              BKW: tl.constexpr):
    # the fc1 dot products of BR rows, exactly as _gemm1 computes them
    acc = tl.zeros([BR], dtype=tl.float32)
    wo = tl.arange(0, BKW)
    for k0 in tl.static_range(0, KW, BKW):
        kw = k0 + wo
        aw = tl.load(a_ptr + kw)
        a0 = tl.broadcast_to(_pair(aw & 255)[None, :], [BR, BKW])
        a1 = tl.broadcast_to(_pair((aw >> 8) & 255)[None, :], [BR, BKW])
        a2 = tl.broadcast_to(_pair((aw >> 16) & 255)[None, :], [BR, BKW])
        a3 = tl.broadcast_to(_pair((aw >> 24) & 255)[None, :], [BR, BKW])
        ww = tl.load(w_base + rows[:, None] * KW + kw[None, :])
        s = _wdot(ww, a0, a1, a2, a3)
        j = kw // 2
        asc = _fp8(tl.load(as_ptr + (j // 4) * 512 + j % 4))
        wsc = _fp8(tl.load(s_base + _swz(rows[:, None], j[None, :], KT)))
        acc += tl.sum(s * (wsc * asc[None, :]), axis=1)
    return acc


@triton.jit
def _gemm1_act(a_ptr, as_ptr, w_ptr, ws_ptr, ids_ptr, alpha_ptr, gs_ptr, qw_ptr, qs_ptr,
               I: tl.constexpr, KW: tl.constexpr, KT: tl.constexpr, BKW: tl.constexpr):
    # _gemm1 + _act_quant for one 16-value block of the intermediate: the 16 up rows (j) and the 16 gate rows
    # (I + j) of one expert slot, rounded to bf16 as _gemm1 stores them, then the same act-quant arithmetic
    slot = tl.program_id(0)
    jt = tl.program_id(1)
    e = tl.load(ids_ptr + slot).to(tl.int64)
    N1: tl.constexpr = 2 * I
    w_base = w_ptr + e * N1 * KW
    s_base = ws_ptr + e * N1 * (KT * 4)
    j16 = jt * 16 + tl.arange(0, 16)
    alpha = tl.load(alpha_ptr + e)
    u = (_rows_dot(a_ptr, as_ptr, w_base, s_base, j16, KW, KT, 16, BKW) * alpha).to(tl.bfloat16).to(tl.float32)
    g = (_rows_dot(a_ptr, as_ptr, w_base, s_base, I + j16, KW, KT, 16, BKW) * alpha).to(tl.bfloat16).to(tl.float32)
    ge = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    x = (ge * u).to(tl.bfloat16).to(tl.float32)
    gs = tl.load(gs_ptr + e)
    vmax = tl.max(tl.abs(x), axis=0)
    sf = tl.minimum(gs * (vmax * 0.16666666666666666), 448.0).to(tl.float8e4nv).to(tl.float32)
    osc = tl.where(sf != 0.0, tl.math.div_rn(1.0, sf * tl.math.div_rn(1.0, gs)), 0.0)
    y = x * osc
    aa = tl.abs(y)
    code = ((aa > 0.25).to(tl.int32) + (aa >= 0.75).to(tl.int32) + (aa > 1.25).to(tl.int32)
            + (aa >= 1.75).to(tl.int32) + (aa > 2.5).to(tl.int32) + (aa >= 3.5).to(tl.int32)
            + (aa > 5.0).to(tl.int32))
    code = code + tl.where(y < 0, 8, 0)
    nib = tl.reshape(code, [2, 8])
    sh = tl.arange(0, 8) * 4
    words = tl.sum(nib << sh[None, :], axis=1)
    tl.store(qw_ptr + slot * (I // 8) + jt * 2 + tl.arange(0, 2), words)
    tl.store(qs_ptr + slot * (I // 16) + jt, sf)


@triton.jit
def _e2m1_rn(y):
    # round to the nearest e2m1 value, ties to even, saturating at 6 (cvt.rn.satfinite.e2m1)
    a = tl.abs(y)
    q = tl.where(a <= 0.25, 0.0,
        tl.where(a < 0.75, 0.5,
        tl.where(a <= 1.25, 1.0,
        tl.where(a < 1.75, 1.5,
        tl.where(a <= 2.5, 2.0,
        tl.where(a < 3.5, 3.0,
        tl.where(a <= 5.0, 4.0, 6.0)))))))
    return tl.where(y < 0, -q, q)


@triton.jit
def _act_quant(h_ptr, ids_ptr, gs_ptr, qw_ptr, qs_ptr, I: tl.constexpr, NB: tl.constexpr,
               GATE_FIRST: tl.constexpr):
    slot = tl.program_id(0)
    e = tl.load(ids_ptr + slot).to(tl.int64)
    jb = tl.arange(0, NB)
    ii = tl.arange(0, 16)
    m = jb < I // 16
    idx = jb[:, None] * 16 + ii[None, :]
    mm = m[:, None] & (idx < I)
    first = tl.load(h_ptr + slot * 2 * I + idx, mask=mm, other=0.0).to(tl.float32)
    second = tl.load(h_ptr + slot * 2 * I + I + idx, mask=mm, other=0.0).to(tl.float32)
    if GATE_FIRST:
        g = first
        u = second
    else:
        g = second
        u = first
    # as the baseline's FLASHINFER_CUTLASS kernel computes it (matched bit for bit on real inputs, see the
    # check mode): GELU with erf, the product rounded to bf16, then the fp32 scale chain below
    ge = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    x = (ge * u).to(tl.bfloat16).to(tl.float32)
    gs = tl.load(gs_ptr + e)
    vmax = tl.max(tl.abs(x), axis=1)
    sf = tl.minimum(gs * (vmax * 0.16666666666666666), 448.0).to(tl.float8e4nv).to(tl.float32)
    osc = tl.where(sf != 0.0, tl.math.div_rn(1.0, sf * tl.math.div_rn(1.0, gs)), 0.0)
    y = x * osc[:, None]
    a = tl.abs(y)
    # e2m1 round to nearest, ties to even, saturating (cvt.rn.satfinite.e2m1x2)
    code = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
            + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32))
    code = code + tl.where(y < 0, 8, 0)
    nib = tl.reshape(code, [NB * 2, 8])
    sh = tl.arange(0, 8) * 4
    words = tl.sum(nib << sh[None, :], axis=1)
    wo = tl.arange(0, NB * 2)
    tl.store(qw_ptr + slot * (I // 8) + wo, words, mask=wo < I // 8)
    tl.store(qs_ptr + slot * (I // 16) + jb, sf, mask=m)


@triton.jit
def _gemm2_combine(qw_ptr, qs_ptr, w_ptr, ws_ptr, ids_ptr, tw_ptr, alpha_ptr, out_ptr,
                   N2: tl.constexpr, I: tl.constexpr, KT: tl.constexpr, BR: tl.constexpr, BKW: tl.constexpr,
                   TOPK: tl.constexpr):
    rt = tl.program_id(0)
    rows = rt * BR + tl.arange(0, BR)
    KW: tl.constexpr = I // 8
    wo = tl.arange(0, BKW)
    wm = wo < KW
    j = wo // 2
    total = tl.zeros([BR], dtype=tl.float32)
    for slot in tl.static_range(TOPK):
        e = tl.load(ids_ptr + slot).to(tl.int64)
        ww = tl.load(w_ptr + e * N2 * KW + rows[:, None] * KW + wo[None, :], mask=wm[None, :], other=0)
        qw = tl.load(qw_ptr + slot * KW + wo, mask=wm, other=0)
        q0 = tl.broadcast_to(_pair(qw & 255)[None, :], [BR, BKW])
        q1 = tl.broadcast_to(_pair((qw >> 8) & 255)[None, :], [BR, BKW])
        q2 = tl.broadcast_to(_pair((qw >> 16) & 255)[None, :], [BR, BKW])
        q3 = tl.broadcast_to(_pair((qw >> 24) & 255)[None, :], [BR, BKW])
        s = _wdot(ww, q0, q1, q2, q3)
        qs = tl.load(qs_ptr + slot * (I // 16) + j, mask=wm, other=0.0)
        wsc = _fp8(tl.load(ws_ptr + e * N2 * (KT * 4) + _swz(rows[:, None], j[None, :], KT),
                           mask=wm[None, :], other=0))
        acc = tl.sum(s * (wsc * qs[None, :]), axis=1)
        y = (acc * tl.load(alpha_ptr + e)).to(tl.bfloat16).to(tl.float32)
        total += tl.load(tw_ptr + slot) * y
    tl.store(out_ptr + rows, total.to(tl.bfloat16))


_bufs = {}
GATE_FIRST = os.environ.get("G4F_MOE_GATE_FIRST", "0") == "1"
BR1, BKW1, NW1 = (int(x) for x in os.environ.get("G4F_MOE_CFG1", "16,32,4").split(","))
BR2, NW2 = (int(x) for x in os.environ.get("G4F_MOE_CFG2", "8,4").split(","))
FUSED1 = os.environ.get("G4F_MOE_FUSED1", "1") == "1" and not GATE_FIRST


def moe_decode(output, a1q, a1q_scale, w1, w1_scale, g1_alphas, a2_gscale, w2, w2_scale, g2_alphas,
               topk_weights, topk_ids):
    """output [1, H] bf16; a1q [1, H/2] uint8 (NVFP4 input), a1q_scale its swizzled FP8 block scales;
    w1 [E, 2I, H/2], w2 [E, H, I/2] packed FP4; w*_scale swizzled FP8; per-expert alphas and gscale."""
    E, N1, KB = w1.shape
    H = output.shape[-1]
    I = N1 // 2
    K = topk_ids.shape[-1]
    dev = output.device
    key = (dev, N1, H, K)
    if key not in _bufs:
        _bufs[key] = (torch.empty(K, N1, dtype=torch.bfloat16, device=dev),
                      torch.empty(K, I // 8, dtype=torch.int32, device=dev),
                      torch.empty(K, I // 16, dtype=torch.float32, device=dev))
    h, qw, qs = _bufs[key]
    ids = topk_ids.view(-1)
    tw = topk_weights.view(-1)
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    i32 = lambda t: t.view(torch.int32)  # noqa: E731
    if FUSED1 and BR1 == 16 and I % 16 == 0:
        _gemm1_act[(K, I // 16)](i32(a1q), u8(a1q_scale), i32(w1), u8(w1_scale), ids, g1_alphas, a2_gscale, qw, qs,
                                 I=I, KW=KB // 4, KT=w1_scale.shape[-1] // 4, BKW=BKW1, num_warps=NW1)
    else:
        _gemm1[(K, N1 // BR1)](i32(a1q), u8(a1q_scale), i32(w1), u8(w1_scale), ids, g1_alphas, h,
                               N1=N1, KW=KB // 4, KT=w1_scale.shape[-1] // 4, BR=BR1, BKW=BKW1, num_warps=NW1)
    if not (FUSED1 and BR1 == 16 and I % 16 == 0):
        _act_quant[(K,)](h, ids, a2_gscale, qw, qs, I=I, NB=triton.next_power_of_2(I // 16), GATE_FIRST=GATE_FIRST,
                         num_warps=4)
    _gemm2_combine[(H // BR2,)](qw, qs, i32(w2), u8(w2_scale), ids, tw, g2_alphas, output, N2=H, I=I,
                                KT=w2_scale.shape[-1] // 4, BR=BR2, BKW=triton.next_power_of_2(I // 8),
                                TOPK=K, num_warps=NW2)
    return output


_LUT = None


def _unswizzle(s, M, K):
    Mp, Kp = -(-M // 128) * 128, -(-K // 4) * 4
    t = s.reshape(-1)[: Mp * Kp].view(Mp // 128, Kp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(Mp, Kp)
    return t[:M, :K].contiguous().view(torch.float8_e4m3fn).double()


def _dec(b):
    global _LUT
    if _LUT is None or _LUT.device != b.device:
        v = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        _LUT = torch.tensor(v + [-x for x in v], dtype=torch.float64, device=b.device)
    b = b.view(torch.uint8).long()
    return torch.stack([_LUT[b & 15], _LUT[b >> 4]], -1).flatten(-2)


def moe_emulate(a1q, a1q_scale, w1, w1_scale, g1_alphas, a2_gscale, w2, w2_scale, g2_alphas, tw, ids, v=None):
    """fp64 emulation of the checkpoint's numerics (oracle for the check mode); v: variant switches."""
    v = v or {}
    E, N1, KB = w1.shape
    H, I = w2.shape[1], N1 // 2
    ids, tw = ids.reshape(-1), tw.reshape(-1)
    if v.get("alin"):
        asf = a1q_scale.view(torch.uint8).reshape(-1)[: KB // 8].view(torch.float8_e4m3fn).double()
    else:
        asf = _unswizzle(a1q_scale.view(torch.uint8), 128, KB // 8)[0]
    a = _dec(a1q.reshape(-1)) * asf.repeat_interleave(16)
    out = torch.zeros(H, dtype=torch.float64, device=a.device)
    for slot in range(ids.numel()):
        e = int(ids[slot])
        W1 = _dec(w1[e]) * _unswizzle(w1_scale[e].view(torch.uint8), N1, KB // 8).repeat_interleave(16, 1)
        h = (W1 @ a) * float(g1_alphas[e])
        if not v.get("no_r1"):
            h = h.to(torch.bfloat16).double()
        u, g = h[:I], h[I:]
        if v.get("erf"):
            ge = 0.5 * g * (1 + torch.erf(g / 2 ** 0.5))
        else:
            ge = 0.5 * g * (1 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g ** 3)))
        if v.get("gr"):
            ge = ge.to(torch.bfloat16).double()
        x = ge * u
        if v.get("xr"):
            x = x.to(torch.bfloat16).double()
        if v.get("noq"):
            W2 = _dec(w2[e]) * _unswizzle(w2_scale[e].view(torch.uint8), H, I // 16).repeat_interleave(16, 1)
            out += float(tw[slot]) * ((W2 @ x) * float(g2_alphas[e]) / float(a2_gscale[e])).to(torch.bfloat16).double()
            continue
        gs = float(a2_gscale[e])
        xb = x.view(-1, 16)
        if v.get("f32"):  # TRT-LLM's fp32 chain: SF = gs * (amax * rcp(6)); scale = rcp(float(SF) * rcp(gs))
            gs32 = torch.tensor(gs, dtype=torch.float32, device=a.device)
            vm = xb.abs().amax(1).float()
            sf = (gs32 * (vm * (1.0 / torch.tensor(6.0, device=a.device)))).clamp(max=448).to(torch.float8_e4m3fn).float()
            osc = torch.where(sf != 0, 1.0 / (sf * (1.0 / gs32)), torch.zeros_like(sf))
            y = (xb.float() * osc[:, None]).abs().clamp(max=6).double()
            sf = sf.double()
        else:
            sf = (gs * xb.abs().amax(1) / 6).clamp(max=448).float().to(torch.float8_e4m3fn).double()
            osc = torch.where(sf != 0, gs / sf, torch.zeros_like(sf))
            y = (xb * osc[:, None]).abs().clamp(max=6)
        grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64, device=a.device)
        qi = (y[..., None] - grid).abs().argmin(-1)  # nearest (ties: decided below)
        mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float64, device=a.device)
        up_even = torch.tensor([0, 1, 0, 1, 0, 1, 0], device=a.device)  # tie goes to the even code (lower if even)
        for t in range(7):
            tie = y == mid[t]
            qi = torch.where(tie, torch.full_like(qi, t + up_even[t]), qi)
        q = (grid[qi] * torch.sign(xb) * sf[:, None]).reshape(-1)
        W2 = _dec(w2[e]) * _unswizzle(w2_scale[e].view(torch.uint8), H, I // 16).repeat_interleave(16, 1)
        y2 = ((W2 @ q) * float(g2_alphas[e])).to(torch.bfloat16).double()
        out += float(tw[slot]) * y2
    return out.to(torch.bfloat16)


def install():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe as F

    orig = F.FlashInferExperts.apply
    check = os.environ.get("G4F_MOE_CHECK", "0") == "1"
    stats = {"n": 0}

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        if (hidden_states.shape[0] == 1 and self.quant_dtype == "nvfp4" and activation == MoEActivation.GELU_TANH
                and expert_map is None and not apply_router_weight_on_input and hidden_states.element_size() == 1
                and output.dtype == torch.bfloat16 and getattr(self, "w1_bias", None) is None):
            args = (hidden_states, a1q_scale, w1, self.w1_scale, self.g1_alphas, self.a2_gscale, w2, self.w2_scale,
                    self.g2_alphas, topk_weights.float(), topk_ids.to(torch.int32))
            if check:
                ref = torch.empty_like(output)
                orig(self, ref, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                     expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                     apply_router_weight_on_input)
                moe_decode(output, *args)
                emu = moe_emulate(*args, v={"xr": 1, "erf": 1, "f32": 1}).float().view(-1)
                rl = lambda x, y: ((x.float().view(-1) - y).norm() / y.norm().clamp_min(1e-30)).item()  # noqa: E731
                stats["n"] += 1
                if stats["n"] == 40:
                    def tm(f, r=50):
                        f(); torch.cuda.synchronize()
                        a_, b_ = torch.cuda.Event(True), torch.cuda.Event(True)
                        a_.record()
                        for _ in range(r):
                            f()
                        b_.record(); torch.cuda.synchronize()
                        return a_.elapsed_time(b_) / r * 1000
                    t_v = tm(lambda: orig(self, ref, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                                          global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
                                          workspace2, expert_tokens_meta, apply_router_weight_on_input))
                    t_o = tm(lambda: moe_decode(output, *args))
                    print(f"G4F_MOE_TIME vendor_us {t_v:.1f} ours_us {t_o:.1f}", flush=True)
                    from triton.testing import do_bench  # noqa: F401
                if stats["n"] <= 60:
                    vs = {"base": {}, "xr_erf": {"xr": 1, "erf": 1}, "xr_erf_f32": {"xr": 1, "erf": 1, "f32": 1},
                          "xr_erf_gr": {"xr": 1, "erf": 1, "gr": 1}, "xr_erf_f32_gr": {"xr": 1, "erf": 1, "f32": 1, "gr": 1},
                          "xr_erf_nor1": {"xr": 1, "erf": 1, "no_r1": 1}}
                    print("G4F_MOE_VARIANTS " + " ".join(
                        f"{n}={rl(ref, moe_emulate(*args, v=vv).float().view(-1)):.3e}" for n, vv in vs.items()),
                        flush=True)
                if stats["n"] <= 400:
                    print(f"G4F_MOE_CHECK ours_vs_vendor {rl(output, ref.float().view(-1)):.3e} "
                          f"ours_vs_emu {rl(output, emu):.3e} vendor_vs_emu {rl(ref, emu):.3e}", flush=True)
                return
            moe_decode(output, *args)
            return
        return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                    expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                    apply_router_weight_on_input)

    F.FlashInferExperts.apply = apply
    logger.info("g4fast: batch-1 NVFP4 MoE decode installed (gate first: %s, check: %s)", GATE_FIRST, check)

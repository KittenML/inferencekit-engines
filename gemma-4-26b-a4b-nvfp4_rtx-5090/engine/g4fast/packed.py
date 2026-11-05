"""Lossless packed BF16 weights for the batch-1 attention/MLP GEMVs (G4F_PACK=1): qkv_proj, o_proj, gate_up_proj.

Storage, per row and group of 64 values: the largest exponent (base, 1 byte); per value a sign+mantissa byte
and a 4-bit exponent delta (base - exponent). Values whose delta would be 15 or more are escapes: their slot
holds the code 15 and decodes to 0, and their raw BF16 value is kept in a per-row list (CSR). The GEMV
multiplies every decoded value and every escape by the input in fp32 and sums them into one fp32 accumulator,
so every stored BF16 weight is used exactly; only the summation order differs from a BF16 GEMV.
About 12.06 bits per weight plus 48 bits per escape (about 1e-4 of the weights).

At load (first call), every packed matrix is decoded back in torch and compared bit for bit with the stored
BF16 tensor, escapes included; a matrix that does not round-trip keeps the BF16 path. Batches of more than one
token (prefill, concurrent decode) use F.linear on the BF16 weights, as the baseline.
"""
import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
G = 64
_packs = {}
LAYERS = os.environ.get("G4F_PACK_LAYERS", "qkv_proj,o_proj,gate_up_proj").split(",")


@triton.jit
def _packed_gemv(sm_ptr, dl_ptr, base_ptr, ptr_ptr, col_ptr, val_ptr, x_ptr, o_ptr,
                 K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr, G: tl.constexpr, ME: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    r64 = rows.to(tl.int64)
    acc = tl.zeros([R, BK], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        sm = tl.load(sm_ptr + r64[:, None] * K + k[None, :]).to(tl.int32)
        dl = tl.load(dl_ptr + r64[:, None] * (K // 2) + (k // 2)[None, :]).to(tl.int32)
        d = (dl >> ((k % 2) * 4)[None, :]) & 15
        e = tl.load(base_ptr + r64[:, None] * (K // G) + (k // G)[None, :]).to(tl.int32) - d
        bits = ((sm & 0x80) << 8) | (e << 7) | (sm & 0x7F)
        w = tl.where(d == 15, 0.0, (bits << 16).to(tl.float32, bitcast=True))
        acc += w * tl.load(x_ptr + k).to(tl.float32)[None, :]
    tot = tl.sum(acc, 1)
    # escapes: bf16 x bf16 products are exact in fp32
    p0 = tl.load(ptr_ptr + rows)
    p1 = tl.load(ptr_ptr + rows + 1)
    j = tl.arange(0, ME)
    m = (p0[:, None] + j[None, :]) < p1[:, None]
    c = tl.load(col_ptr + p0[:, None] + j[None, :], mask=m, other=0)
    v = tl.load(val_ptr + p0[:, None] + j[None, :], mask=m, other=0.0).to(tl.float32)
    tot += tl.sum(v * tl.load(x_ptr + c, mask=m, other=0.0).to(tl.float32), 1)
    tl.store(o_ptr + rows, tot.to(tl.bfloat16))


def _unpack(sm, dl, base, ptr, col, val, N, K):
    k = torch.arange(K, device=sm.device)
    d = (dl.to(torch.int32)[:, k // 2] >> ((k % 2) * 4).to(torch.int32)) & 15
    e = base.to(torch.int32)[:, k // G] - d
    smi = sm.to(torch.int32)
    bits = ((smi & 0x80) << 8) | (e << 7) | (smi & 0x7F)
    bits = torch.where(d == 15, torch.zeros_like(bits), bits).to(torch.int32)
    out = (bits & 0xFFFF).to(torch.int32)
    out = (out - ((out >> 15) << 16)).to(torch.int16)  # to signed 16-bit patterns
    rows = torch.repeat_interleave(torch.arange(N, device=sm.device), (ptr[1:] - ptr[:-1]).long())
    out[rows, col.long()] = val.view(torch.int16)
    return out


def _pack(w):
    N, K = w.shape
    b = w.view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (b >> 7) & 0xFF
    base = exp.view(N, -1, G).amax(-1)
    d = base.repeat_interleave(G, -1) - exp
    esc = d >= 15
    d = torch.where(esc, 15, d)
    sm = (((b >> 8) & 0x80) | (b & 0x7F)).to(torch.uint8)
    dl = (d[:, 0::2] | (d[:, 1::2] << 4)).to(torch.uint8)
    r, c = esc.nonzero(as_tuple=True)
    cnt = torch.bincount(r, minlength=N)
    ptr = torch.zeros(N + 1, dtype=torch.int32, device=w.device)
    ptr[1:] = cnt.cumsum(0)
    me = max(2, triton.next_power_of_2(int(cnt.max()) if cnt.numel() else 1))
    p = (sm, dl, base.to(torch.uint8).contiguous(), ptr, c.to(torch.int32), w[r, c].contiguous(), me)
    # the proof: every stored value round-trips bit for bit (escapes included)
    ok = bool(torch.equal(_unpack(*p[:6], N, K), w.view(torch.int16)))
    return p if ok else None


@torch.library.custom_op("g4fast::packed_linear", mutates_args=())
def packed_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    key = (w.data_ptr(), tuple(w.shape))
    if key not in _packs or _packs[key][1] is not w:  # first call: the profiling run, before the KV cache
        _packs[key] = (_pack(w), w)  # holding w keeps its address from being reused while cached
        if _packs[key][0] is None:
            logger.warning("g4fast packed: a %s matrix does not round-trip; BF16 path kept", tuple(w.shape))
    p = _packs[key][0]
    if x.shape[0] != 1 or x.dim() != 2 or p is None:
        return F.linear(x, w)
    sm, dl, base, ptr, col, val, me = p
    N, K = w.shape
    out = torch.empty(1, N, dtype=x.dtype, device=x.device)
    _packed_gemv[(N // 8,)](sm, dl, base, ptr, col, val, x.contiguous(), out, K=K, R=8, BK=256, G=G, ME=me,
                            num_warps=4)
    return out


@packed_linear.register_fake
def _(x, w):
    return x.new_empty(x.shape[0], w.shape[0])


def install():
    from vllm.model_executor.layers import linear as L

    orig = L.UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        w = getattr(layer, "weight", None)
        if (bias is None and w is not None and w.dtype == torch.bfloat16 and w.dim() == 2
                and str(getattr(layer, "prefix", "")).rsplit(".", 1)[-1] in LAYERS
                and w.shape[0] % 8 == 0 and w.shape[1] % 256 == 0 and x.dtype == torch.bfloat16):
            return packed_linear(x, w)
        return orig(self, layer, x, bias)

    L.UnquantizedLinearMethod.apply = apply
    logger.info("g4fast: packed BF16 GEMVs installed for %s", LAYERS)

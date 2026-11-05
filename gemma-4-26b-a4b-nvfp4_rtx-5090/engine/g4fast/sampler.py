"""Exact top-k-restricted sampler for vLLM 0.31.0's V2 GPU model runner.

vLLM samples seeded requests (and any batch with a greedy or seeded request) with a full-vocabulary
Triton top-k/top-p mask followed by a full-vocabulary Gumbel-max kernel whose noise is a counter-based
hash of (seed, position, token id). That mask keeps exactly k tokens (ties at the k-th value: the lowest
token ids), then the top-p nucleus of their softmax. With top_k <= KCAP set, this path computes the same
kept set and the same noisy scores with three kernel launches and no host sync:

1. _block_top: for every block of BLK tokens (block b holds ids b, b + NB, b + 2 NB, ...), the R largest
   logits and the R-th value (the block's floor). Every token above a block's floor is a candidate.
2. _select_sample over the candidates: the exact k-th largest value t (bisection on the order-preserving
   integer key of the fp32 logit). If every block floor is < t, every token >= t is a candidate, so the
   kept set is exact: the tokens > t plus the lowest-id tokens equal to t (bisection on the id), k in all. Then top-p (keep a
   token while the mass of the kept tokens ranked before it, in (logit desc, id asc) order, is < top_p;
   softmax in fp32), the same Gumbel noise keyed by (seed, position, token id), argmax.
   Otherwise it sets a flag and writes nothing.
3. _select_sample over the whole vocabulary: returns at once unless the flag is set; then it does step 2
   on every token.

A kept token's noisy score is bit-identical to vLLM's, and the kept set is the same set, so the sampled
token is the same as vLLM's own path for the same request and seed (tools/test_sampler.py checks it).
"""
import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample import sampler as S
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable, murmur3_uniform32

logger = init_logger(__name__)

KCAP = 256  # the largest top_k served by this path
BLK = 1024  # tokens per block in the first pass
R = 16  # candidates per block
CHUNK = 4096


@triton.jit
def _key(v):
    # order-preserving int32 key of an fp32 value (+0.0 first, so -0.0 and +0.0 tie as they compare)
    b = (v + 0.0).to(tl.int32, bitcast=True)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@triton.jit
def _block_top(x_ptr, x_stride, V, cv_ptr, ci_ptr, fl_ptr, NB, R: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0)
    b = tl.program_id(1)
    offs = tl.arange(0, BLK)
    tok = b + NB * offs  # interleaved: frequent (low-id) tokens spread over all blocks
    v = tl.load(x_ptr + row * x_stride + tok, mask=tok < V, other=float("-inf")).to(tl.float32)
    m = float("-inf")
    for r in tl.static_range(R):
        m = tl.max(v, axis=0)
        i = tl.min(tl.where(v == m, offs, BLK), axis=0)
        tl.store(cv_ptr + (row * NB + b) * R + r, m)
        tl.store(ci_ptr + (row * NB + b) * R + r, b + NB * i)
        v = tl.where(offs == i, float("-inf"), v)
    tl.store(fl_ptr + row * NB + b, m)


@triton.jit
def _select_sample(src_v_ptr, src_i_ptr, src_stride, N, fl_ptr, NB, flag_ptr, kv_ptr, ki_ptr, k_ptr, p_ptr,
                   idx_map_ptr, temp_ptr, seeds_ptr, pos_ptr, out_ptr,
                   FULL: tl.constexpr, HAS_IDS: tl.constexpr, HAS_P: tl.constexpr, SINGLE: tl.constexpr,
                   CHUNK: tl.constexpr, KCAP: tl.constexpr, NBP: tl.constexpr):
    row = tl.program_id(0)
    if FULL:
        if tl.load(flag_ptr + row) == 0:
            return
    k = tl.load(k_ptr + row)
    offs = tl.arange(0, CHUNK)
    base = src_v_ptr + row * src_stride
    # the k-th largest key: the largest T with count(key >= T) >= k
    lo = tl.full((), -2147483648, tl.int64)
    hi = tl.full((), 2147483647, tl.int64)
    if SINGLE:  # N == CHUNK: keep the keys in registers
        kk1 = _key(tl.load(base + offs).to(tl.float32)).to(tl.int64)
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            cnt = tl.sum((kk1 >= mid).to(tl.int32), axis=0)
            if cnt >= k:
                lo = mid
            else:
                hi = mid - 1
    else:
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            cnt = 0
            for c in range(0, N, CHUNK):
                o = c + offs
                kk = _key(tl.load(base + o, mask=o < N, other=float("-inf")).to(tl.float32)).to(tl.int64)
                cnt += tl.sum(((kk >= mid) & (o < N)).to(tl.int32), axis=0)
            if cnt >= k:
                lo = mid
            else:
                hi = mid - 1
    T = lo
    if not FULL:
        # complete only if every block floor is below t
        fo = tl.arange(0, NBP)
        fk = _key(tl.load(fl_ptr + row * NB + fo, mask=fo < NB, other=float("-inf"))).to(tl.int64)
        bad = tl.sum(((fk >= T) & (fo < NB)).to(tl.int32), axis=0)
        tl.store(flag_ptr + row, bad)
        if bad != 0:
            return
    # compact the kept set: tokens > t, then the `need` lowest-id tokens == t
    n_gt = 0
    n_eq = 0
    for c in range(0, N, CHUNK):
        o = c + offs
        kk = _key(tl.load(base + o, mask=o < N, other=float("-inf")).to(tl.float32)).to(tl.int64)
        n_gt += tl.sum(((kk > T) & (o < N)).to(tl.int32), axis=0)
        n_eq += tl.sum(((kk == T) & (o < N)).to(tl.int32), axis=0)
    need = k - n_gt
    # the need-th smallest id among the ties: the smallest I with count(tie & id <= I) >= need
    # (all ties fit: every tie is kept, no search)
    ihi = 2147483647
    ilo = tl.where(n_eq <= need, ihi, 0)
    while ilo < ihi:
        imid = (ilo + ihi) // 2
        cnt = 0
        for c in range(0, N, CHUNK):
            o = c + offs
            inb = o < N
            kk = _key(tl.load(base + o, mask=inb, other=float("-inf")).to(tl.float32)).to(tl.int64)
            if HAS_IDS:
                ids = tl.load(src_i_ptr + row * src_stride + o, mask=inb, other=0)
            else:
                ids = o
            cnt += tl.sum(((kk == T) & inb & (ids <= imid)).to(tl.int32), axis=0)
        if cnt >= need:
            ihi = imid
        else:
            ilo = imid + 1
    kept = 0
    for c in range(0, N, CHUNK):
        o = c + offs
        inb = o < N
        v = tl.load(base + o, mask=inb, other=float("-inf")).to(tl.float32)
        if HAS_IDS:
            ids = tl.load(src_i_ptr + row * src_stride + o, mask=inb, other=0).to(tl.int32)
        else:
            ids = o.to(tl.int32)
        kk = _key(v).to(tl.int64)
        sel = inb & ((kk > T) | ((kk == T) & (ids <= ilo)))
        dst = tl.cumsum(sel.to(tl.int32), axis=0) - 1 + kept
        tl.store(kv_ptr + row * KCAP + dst, v, mask=sel)
        tl.store(ki_ptr + row * KCAP + dst, ids, mask=sel)
        kept += tl.sum(sel.to(tl.int32), axis=0)
    tl.debug_barrier()
    ko = tl.arange(0, KCAP)
    inb = ko < k
    v = tl.load(kv_ptr + row * KCAP + ko, mask=inb, other=float("-inf"))
    ids = tl.load(ki_ptr + row * KCAP + ko, mask=inb, other=2147483647)
    keep = inb
    if HAS_P:
        # rank in (logit desc, token id asc) order, as vLLM's kernel orders the nucleus
        before = ((v[None, :] > v[:, None]) | ((v[None, :] == v[:, None]) & (ids[None, :] < ids[:, None])))
        before = before & inb[None, :]
        p = tl.load(p_ptr + row)
        vmax = tl.max(tl.where(keep, v, float("-inf")), axis=0)
        e = tl.where(keep, tl.exp(v - vmax), 0.0)
        probs = e / tl.sum(e, axis=0)
        excl = tl.sum(tl.where(before, probs[None, :], 0.0), axis=1)
        keep = keep & (excl < p)
    req = tl.load(idx_map_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req)
        pos = tl.load(pos_ptr + row)
        u = murmur3_uniform32(seed, pos, ids)
        score = tl.where(keep, v + (-tl.log(-_log1p_neg_stable(u))), float("-inf"))
    else:
        score = tl.where(keep, v, float("-inf"))
    best = tl.max(score, axis=0)
    # ties on the score go to the lowest token id, as the full-vocabulary block argmax does
    tok = tl.min(tl.where((score == best) & keep, ids, 2147483647), axis=0)
    tl.store(out_ptr + row, tok.to(tl.int64))


_bufs = {}


def fast_sample(logits, top_k, top_p, expanded_idx_mapping, temperature, seeds, pos):
    """Sampled token ids [B] (int64), the same tokens as vLLM's seeded top-k/top-p path."""
    B, V = logits.shape
    NB = triton.cdiv(V, BLK)
    key = (B, V, logits.device)
    if key not in _bufs:
        d = logits.device
        _bufs[key] = (torch.empty(B, NB * R, device=d), torch.empty(B, NB * R, dtype=torch.int32, device=d),
                      torch.empty(B, NB, device=d), torch.empty(B, dtype=torch.int32, device=d),
                      torch.empty(B, KCAP, device=d), torch.empty(B, KCAP, dtype=torch.int32, device=d))
    cv, ci, fl, flag, kv, ki = _bufs[key]
    out = torch.empty(B, dtype=torch.int64, device=logits.device)
    p = top_p if top_p is not None else temperature
    _block_top[(B, NB)](logits, logits.stride(0), V, cv, ci, fl, NB, R=R, BLK=BLK, num_warps=4)
    common = dict(HAS_P=top_p is not None, CHUNK=CHUNK, KCAP=KCAP, NBP=triton.next_power_of_2(NB))
    rest = (flag, kv, ki, top_k, p, expanded_idx_mapping, temperature, seeds, pos, out)
    _select_sample[(B,)](cv, ci, NB * R, NB * R, fl, NB, *rest, FULL=False, HAS_IDS=True, SINGLE=NB * R == CHUNK, num_warps=8,
                         **common)
    _select_sample[(B,)](logits, logits, logits.stride(0), V, fl, NB, *rest, FULL=True, HAS_IDS=False, SINGLE=False,
                         num_warps=8, **common)
    return out


def install():
    orig = S.Sampler._sample_random

    def _sample_random(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p,
                       use_fused_sampler):
        if (not use_fused_sampler and top_k is not None and not self.use_fp64_gumbel
                and not self.return_sampling_mask and self.logprobs_mode not in S.PROCESSED_LOGPROBS_MODES
                and processed_logits.shape[0] == len(idx_mapping_np)
                and int(np.max(self.sampling_states.top_k.np[idx_mapping_np])) <= KCAP):
            out = fast_sample(processed_logits, top_k, top_p, expanded_idx_mapping,
                              self.sampling_states.temperature.gpu, self.sampling_states.seeds.gpu, pos)
            return out, processed_logits
        return orig(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p,
                    use_fused_sampler)

    S.Sampler._sample_random = _sample_random
    logger.info("g4fast: top-k-restricted exact sampler installed")

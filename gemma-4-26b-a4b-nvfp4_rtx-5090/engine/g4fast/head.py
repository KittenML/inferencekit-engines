"""Exact bound-pruned output layer for Gemma 4 at batch 1 (vLLM 0.31.0, V2 GPU model runner).

The output layer (262,144 x 2,816 BF16, 1.48 GB) is the largest read of a decode step. With top_k = k set
(and nothing else that can raise a logit), only rows that can reach the top k matter. This module:

* computes every logit with one Triton GEMV (_head_rows), for the full head and for rescored rows alike:
  the same compiled kernel and per-row reduction, so a rescored row is bit-identical to the full head's;
* keeps an int8 proxy of the head (per row and group of 64: scale s, codes q; per row: E = max |w - q s|
  rounded up, A = max |w|), built once at load from the stored BF16 weights;
* per token: proxy logits p (0.78 GB read) and a rigorous bound |z - p| <= B with
  B = (E + 2 gamma (A + E)) * ||x||_1 * 1.01 + |p| 2^-20 (gamma = n 2^-24 / (1 - n 2^-24), n = H + H/64 + 1),
  so L = p - B <= z <= U = p + B for the fp32 logit z of the full head; L and U are rounded outward to bf16
  and passed through the same soft-cap ops as the logits (f, monotone non-decreasing), then
  T = the k-th largest of max-per-64-row-block f(L) (a lower bound of the k-th largest f(z)); the candidates
  are the rows with f(U) >= T; at least k rows have f(z) >= T, so every pruned row is strictly below the
  k-th largest logit and can never be kept by top-k; candidates are rescored exactly, the other rows are
  set to -inf. More than CAP candidates: the full head runs (device-side switch, no host sync).

The pruned path runs only for batch 1, top_k <= 256 (greedy: k = 1; with min_tokens, k plus the stop ids), no logprobs, no penalties, logit bias, allowed or bad
words, min_p, structured output or thinking budget (G4F_HEAD=1). Everything else uses the full head.
"""
import os
import threading

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

KCAP = 256
CAP = 16384
G = 64
R = 8
BK = 256
_st = threading.local()
FALLBACK_LOOP = os.environ.get("G4F_HEAD_FBLOOP", "1") == "1"
PR, PGPI, PNW = (int(v) for v in os.environ.get("G4F_HEAD_PROXY", "16,4,4").split(","))


@triton.jit
def _rows_logits(x_ptr, b, x_stride, w_ptr, rows, m, H: tl.constexpr, R: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros([R, BK], dtype=tl.float32)
    for k0 in tl.static_range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + b * x_stride + k).to(tl.float32)
        w = tl.load(w_ptr + rows.to(tl.int64)[:, None] * H + k[None, :], mask=m[:, None], other=0.0)
        acc += w.to(tl.float32) * xv[None, :]
    return tl.sum(acc, axis=1)


@triton.jit
def _head_rows(x_ptr, x_stride, w_ptr, rows_ptr, cnt_ptr, out_ptr, out_stride, V,
               H: tl.constexpr, R: tl.constexpr, BK: tl.constexpr, CAP: tl.constexpr, MODE: tl.constexpr):
    # MODE 0: rows pid*R.. of the full head (batch row = program_id(1)); MODE 1: candidate list entries
    # pid*R..; MODE 2: the full head only if the candidate count overflowed CAP; MODE 3: as MODE 2 with a
    # fixed grid looping over the row tiles (the same per-row arithmetic, through _rows_logits)
    pid = tl.program_id(0)
    b = tl.program_id(1)
    if MODE == 3:
        if tl.load(cnt_ptr) <= CAP:
            return
        for t in range(pid, tl.cdiv(V, R), tl.num_programs(0)):
            rows = t * R + tl.arange(0, R)
            m = rows < V
            z = _rows_logits(x_ptr, b, x_stride, w_ptr, rows, m, H, R, BK)
            tl.store(out_ptr + b * out_stride + rows, z.to(tl.bfloat16), mask=m)
        return
    if MODE == 1:
        cnt = tl.minimum(tl.load(cnt_ptr), CAP)
        if pid * R >= cnt:
            return
        ri = pid * R + tl.arange(0, R)
        m = ri < cnt
        rows = tl.load(rows_ptr + ri, mask=m, other=0)
    else:
        if MODE == 2:
            if tl.load(cnt_ptr) <= CAP:
                return
        rows = pid * R + tl.arange(0, R)
        m = rows < V
    z = _rows_logits(x_ptr, b, x_stride, w_ptr, rows, m, H, R, BK)
    tl.store(out_ptr + b * out_stride + rows, z.to(tl.bfloat16), mask=m)


@triton.jit
def _bf16_bits(v, UP: tl.constexpr):
    # top 16 bits of an fp32 value, rounded toward +inf (UP) or -inf
    bits = v.to(tl.int32, bitcast=True)
    neg = v < 0
    away = (bits + 0xFFFF) >> 16
    trunc = bits >> 16
    if UP:
        r = tl.where(neg, trunc, away)
    else:
        r = tl.where(neg, away, trunc)
    return (r & 0xFFFF).to(tl.int16)


@triton.jit
def _proxy(x_ptr, q_ptr, s_ptr, ea_ptr, lo_ptr, hi_ptr, V, gamma,
           H: tl.constexpr, NG: tl.constexpr, R: tl.constexpr, GS: tl.constexpr, GPI: tl.constexpr):
    # GPI groups of GS codes per iteration: a [R, GPI * GS] int8 tile, summed per group, times its scale
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    m = rows < V
    r64 = rows.to(tl.int64)
    p = tl.zeros([R], dtype=tl.float32)
    n1 = 0.0
    BKq: tl.constexpr = GPI * GS
    for k0 in tl.static_range(0, H, BKq):
        k = k0 + tl.arange(0, BKq)
        xv = tl.load(x_ptr + k).to(tl.float32)
        n1 += tl.sum(tl.abs(xv), axis=0)
        qv = tl.load(q_ptr + r64[:, None] * H + k[None, :], mask=m[:, None], other=0).to(tl.float32)
        part = tl.sum(tl.reshape(qv * xv[None, :], [R, GPI, GS]), axis=2)
        gi = k0 // GS + tl.arange(0, GPI)
        sc = tl.load(s_ptr + r64[:, None] * NG + gi[None, :], mask=m[:, None], other=0.0)
        p += tl.sum(part * sc, axis=1)
    E = tl.load(ea_ptr + r64 * 2, mask=m, other=0.0)
    A = tl.load(ea_ptr + r64 * 2 + 1, mask=m, other=0.0)
    B = (E + 2.0 * gamma * (A + E)) * n1 * 1.01 + tl.abs(p) * 9.5367431640625e-07 + 1e-30
    tl.store(lo_ptr + rows, _bf16_bits(p - B, False), mask=m)
    tl.store(hi_ptr + rows, _bf16_bits(p + B, True), mask=m)


@triton.jit
def _select(uf_ptr, t_ptr, rows_ptr, cnt_ptr, mask_ptr, V, CAP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    u = tl.load(uf_ptr + offs, mask=m, other=float("-inf")).to(tl.float32)
    t = tl.load(t_ptr).to(tl.float32)
    c = (u >= t) & m
    tl.store(mask_ptr + offs, c.to(tl.int8), mask=m)
    n = tl.sum(c.to(tl.int32), axis=0)
    base = tl.atomic_add(cnt_ptr, n)
    pos = base + tl.cumsum(c.to(tl.int32), axis=0) - 1
    tl.store(rows_ptr + pos, offs, mask=c & (pos < CAP))


class _Head:
    def __init__(self, w):
        V, H = w.shape
        self.V, self.H, self.NG = V, H, H // G
        n = H + H // G + 1
        self.gamma = n * 2.0 ** -24 / (1 - n * 2.0 ** -24)
        dev = w.device
        self.q = torch.empty(V, H, dtype=torch.int8, device=dev)
        self.s = torch.empty(V, self.NG, dtype=torch.float32, device=dev)
        self.ea = torch.empty(V, 2, dtype=torch.float32, device=dev)
        inf = torch.tensor(float("inf"), device=dev)
        for r0 in range(0, V, 4096):
            wd = w[r0:r0 + 4096].double().view(-1, self.NG, G)
            a = wd.abs().amax(-1)
            s = (a / 127).float().clamp_min(torch.finfo(torch.float32).tiny)
            q = torch.round(wd / s.double()[..., None]).clamp(-127, 127)
            e = (wd - q * s.double()[..., None]).abs().amax(-1).amax(-1)
            ef = e.float()
            ef = torch.where(ef.double() < e, torch.nextafter(ef, inf), ef)
            self.q[r0:r0 + 4096] = q.view(-1, H).to(torch.int8)
            self.s[r0:r0 + 4096] = s
            self.ea[r0:r0 + 4096, 0] = ef
            self.ea[r0:r0 + 4096, 1] = a.amax(-1).float()  # exact: max |w| of BF16 values
        self.lo = torch.empty(V, dtype=torch.bfloat16, device=dev)
        self.hi = torch.empty(V, dtype=torch.bfloat16, device=dev)
        self.rows = torch.empty(CAP, dtype=torch.int32, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=dev)
        self.mask = torch.empty(V, dtype=torch.int8, device=dev)
        logger.info("g4fast head: int8 proxy built (%.0f MB)", (self.q.numel() + self.s.numel() * 4) / 1e6)


_heads = {}


def full_logits(w, h):
    """Raw (pre-soft-cap) BF16 logits [B, V] of hidden states h [B, H] through w [V, H], with _head_rows."""
    V, H = w.shape
    h = h.contiguous()
    out = torch.empty(h.shape[0], V, dtype=torch.bfloat16, device=h.device)
    if h.shape[0]:
        _head_rows[(triton.cdiv(V, R), h.shape[0])](h, h.stride(0), w, out, out, out, out.stride(0), V,
                                                    H=H, R=R, BK=BK, CAP=CAP, MODE=0, num_warps=4)
    return out


def softcap(x, c):
    # the same ops, in the same dtype, as LogitsProcessor.forward
    x = x / c
    x = torch.tanh(x)
    return x * c


def pruned_logits(w, h, k, c):
    """Soft-capped BF16 logits [1, V]: candidates exact, the others -inf (see the module docstring)."""
    hd = _heads.get(id(w))
    if hd is None:
        hd = _heads[id(w)] = _Head(w)
    V, H = w.shape
    x = h.reshape(-1).contiguous()
    _proxy[(triton.cdiv(V, PR),)](x, hd.q, hd.s, hd.ea, hd.lo.view(torch.int16), hd.hi.view(torch.int16), V, hd.gamma, H=H, NG=hd.NG, R=PR, GS=G,
                                  GPI=PGPI, num_warps=PNW)
    lf = softcap(hd.lo, c)
    uf = softcap(hd.hi, c)
    t = lf.view(-1, 64).amax(1).topk(k).values[-1:]
    hd.cnt.zero_()
    _select[(triton.cdiv(V, 4096),)](uf, t, hd.rows, hd.cnt, hd.mask, V, CAP=CAP, BLOCK=4096, num_warps=8)
    out = torch.empty(1, V, dtype=torch.bfloat16, device=h.device)
    _head_rows[(CAP // R, 1)](x, 0, w, hd.rows, hd.cnt, out, V, V, H=H, R=R, BK=BK, CAP=CAP, MODE=1, num_warps=4)
    if FALLBACK_LOOP:
        _head_rows[(1024, 1)](x, 0, w, hd.rows, hd.cnt, out, V, V, H=H, R=R, BK=BK, CAP=CAP, MODE=3, num_warps=4)
    else:
        _head_rows[(triton.cdiv(V, R), 1)](x, 0, w, hd.rows, hd.cnt, out, V, V, H=H, R=R, BK=BK, CAP=CAP, MODE=2,
                                           num_warps=4)
    out = softcap(out, c)
    return out.masked_fill_(hd.mask.view(1, -1) == 0, float("-inf")), hd


def _k(sp):
    # greedy (temperature 0): the argmax, k = 1; else the request's top_k. While min_tokens is in force vLLM
    # sets the stop tokens to -inf after the head, so up to len(all_stop_token_ids) rows of the top k can be
    # removed: keep k + that many, and at least k rows remain at or above the threshold after the masking.
    k = 1 if sp.temperature is not None and sp.temperature < 1e-5 else (sp.top_k or 0)
    if k > 0 and (sp.min_tokens or 0) > 0:
        k += len(getattr(sp, "all_stop_token_ids", None) or ()) + len(sp.stop_token_ids or ())
    return k


def _ok(sp):
    tk = _k(sp)
    return bool(
        0 < tk <= KCAP and sp.logprobs is None and not sp.logit_bias
        and not sp.allowed_token_ids and not getattr(sp, "bad_words", None)
        and not getattr(sp, "_bad_words_token_ids", None)
        and sp.presence_penalty == 0.0 and sp.frequency_penalty == 0.0 and sp.repetition_penalty == 1.0
        and sp.min_p == 0.0 and getattr(sp, "structured_outputs", None) is None
        and getattr(sp, "thinking_token_budget", None) is None and not getattr(sp, "logits_processors", None))


def install():
    from vllm.model_executor.models import gemma4 as G4
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu.sample import sampler as S

    check = os.environ.get("G4F_HEAD_CHECK", "0") == "1"
    stats = {"n": 0, "cand": [], "bad": 0, "fallback": 0}

    orig_add = S.Sampler.add_request

    def add_request(self, req_idx, sampling_params):
        if not hasattr(self, "_g4f_ok"):
            self._g4f_ok = np.zeros(len(self.needs_logits_processing), dtype=bool)
            self._g4f_k = np.zeros(len(self.needs_logits_processing), dtype=np.int64)
        self._g4f_ok[req_idx] = _ok(sampling_params)
        self._g4f_k[req_idx] = _k(sampling_params)
        return orig_add(self, req_idx, sampling_params)

    S.Sampler.add_request = add_request

    orig_sample = MR.GPUModelRunner.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        k = 0
        sm = self.sampler
        if (sm is not None and hasattr(sm, "_g4f_ok") and self.batch_sharder is None and grammar_output is None
                and input_batch.num_reqs == 1 and input_batch.num_draft_tokens == 0
                and not getattr(self.model, "_suppress_token_ids", None)  # suppression after the head
                and len(sm.logits_processors) == 3):  # the built-in bias, penalties and bad-words states only
            idx = input_batch.idx_mapping_np
            if bool(np.all(sm._g4f_ok[idx])):
                k = int(np.max(sm._g4f_k[idx]))
        _st.k = k
        try:
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            _st.k = 0

    MR.GPUModelRunner.sample = sample

    def compute_logits(self, hidden_states):
        lp = self.logits_processor
        w = self.lm_head.weight
        c = lp.soft_cap
        if (lp.scale != 1.0 or c is None or getattr(self.lm_head, "tp_size", 1) != 1 or w.dtype != torch.bfloat16
                or lp.logits_as_input or w.shape[0] != lp.org_vocab_size):
            return lp(self.lm_head, hidden_states)
        if id(w) not in _heads:  # build the proxy at the first call (the profiling run, before the KV cache)
            _heads[id(w)] = _Head(w)
        k = getattr(_st, "k", 0)
        if k and hidden_states.shape[0] == 1:
            out, hd = pruned_logits(w, hidden_states, k, c)
            if check:
                full = softcap(full_logits(w, hidden_states), c)
                cm = hd.mask.view(1, -1) != 0
                same = bool(torch.equal(full[cm], out[cm]))
                kth = full.float().topk(k).values[0, -1]
                need = (full.float() >= kth).view(-1)
                miss = int((need & ~cm.view(-1)).sum())
                n = int(hd.cnt.item())
                stats["n"] += 1
                stats["cand"].append(n)
                stats["bad"] += (not same) + (miss > 0)
                stats["fallback"] += n > CAP
                if stats["n"] % 64 == 1 or not same or miss:
                    cs = sorted(stats["cand"])
                    print(f"G4F_HEAD_CHECK n={stats['n']} cand={n} median={cs[len(cs) // 2]} max={cs[-1]} "
                          f"rows_equal={same} missed_topk={miss} bad_total={stats['bad']} "
                          f"fallbacks={stats['fallback']}", flush=True)
            return out
        return softcap(full_logits(w, hidden_states), c)

    G4.Gemma4ForCausalLM.compute_logits = compute_logits
    logger.info("g4fast: Triton output layer + exact bound-pruned head installed (check: %s)", check)

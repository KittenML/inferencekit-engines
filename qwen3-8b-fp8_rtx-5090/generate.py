"""Prefill + CUDA-graph decode loop.

One decode step = embed -> 36 x [qkv GEMV, attention (incl. q/k norm, RoPE, KV write), o_proj GEMV,
gate|up GEMV, down GEMV] -> lm_head GEMV + argmax -> finalize. The whole step, including the
argmax and the bookkeeping (next token, position), runs on the GPU and is captured once into a
CUDA graph. The host just replays the graph gen_len times without waiting: the next step reads
its input token and position from device memory, so the CPU never sits in the loop.
"""
import hashlib
import os
import subprocess

import torch

from kernels import ops
from model import Qwen3FP8

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_FILES = ("model.py", "generate.py", "kernels/__init__.py", "kernels/decode.cu", "kernels/prefill.py",
              "kernels/prefill_fused.py")


def source_identity() -> dict:
    """Which engine code is running: its directory, `git describe` if it is in a git checkout, and an md5
    over the files that decide numerics and speed (_SRC_FILES), so copies outside git can be matched too."""
    h = hashlib.md5()
    for rel in _SRC_FILES:
        h.update(rel.encode())
        with open(os.path.join(_HERE, rel), "rb") as f:
            h.update(f.read())
    try:
        desc = subprocess.run(["git", "-C", _HERE, "describe", "--always", "--dirty", "--tags"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        desc = None
    return {"dir": _HERE, "git_describe": desc, "src_md5": h.hexdigest()}


class Engine:
    def __init__(self, model_path: str, max_seq: int = 4096, lm_head_ctas: int = 0, pdl: bool = True,
                 kmajor: bool = True, prefill_graphs: bool = True, prefill_copy: bool = True):
        # prefill_copy=False drops the 7 GB row-major weight copy; prefill then reads the decode
        # weights' tiled layout (same bits, slower prefill GEMMs).
        self.m = Qwen3FP8(model_path, max_seq=max_seq, kmajor=kmajor, rowmajor_copy=prefill_copy)
        self.k = ops()
        # Programmatic dependent launch: each kernel may start (and prefetch its weights) while the
        # previous one is finishing. Applied when the kernels are launched, i.e. at graph capture.
        self.pdl = pdl
        self.k.set_pdl(pdl)
        cfg = self.m.cfg
        dev = "cuda"
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.max_seq = max_seq
        # Device-side step state.
        self.token = torch.zeros(1, dtype=torch.int32, device=dev)        # input token of the step
        self.state = torch.zeros(2, dtype=torch.int32, device=dev)        # [pos, step]
        self.out_tokens = torch.zeros(max_seq, dtype=torch.int32, device=dev)
        self.best = torch.zeros(1, dtype=torch.int64, device=dev)         # packed argmax
        # Activations (all static addresses, so the graph can be replayed).
        self.resid = torch.zeros(cfg.hidden, **bf)
        self.qkv = torch.zeros((cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim, **bf)
        self.attn = torch.zeros(cfg.n_heads * cfg.head_dim, **bf)
        self.h = torch.zeros(cfg.inter, **bf)
        self.logits = torch.zeros(cfg.vocab, **bf)
        max_splits = (max_seq + 31) // 32  # enough for the smallest attention chunk (32 keys)
        self.o_part = torch.zeros(cfg.n_heads, max_splits, cfg.head_dim, dtype=torch.float32, device=dev)
        self.ml_part = torch.zeros(cfg.n_heads, max_splits, 2, dtype=torch.float32, device=dev)
        self.counters = torch.zeros(cfg.n_kv_heads, dtype=torch.int32, device=dev)
        # GEMV launch/tuning knobs, per kind (qkv, o_proj, gate_up, down). All tuned end to end with
        # interleaved A/B runs of the whole decode step:
        #   depth: 128-wide K blocks each warp keeps in flight (register pipeline);
        #   warps: warps per CTA;
        #   pre:   pipeline slots filled BEFORE waiting on the previous kernel (the rest right after);
        #   trig:  K block at which this kernel lets the next one launch (PDL), so the next kernel's
        #          weight prefetch lands in this kernel's drain window instead of competing earlier.
        # o_proj follows attention, which leaves DRAM mostly idle: attention triggers o_proj right away
        # (attn_late = 0) and o_proj prefetches its whole 8-deep pipeline before waiting.
        self.depth = [4, 8, 4, 8]
        self.warps = [4, 4, 4, 4]
        self.pre = [2, 8, 2, 2]
        self.trig = [24, 20, 30, 88]   # out of 32, 32, 32, 96 K blocks
        self.attn_late = 0
        self.attn_chunk = 64        # keys per attention CTA (32, 64 or 128)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.lm_head_ctas = lm_head_ctas or sms
        self.graph = None
        self.prefill_graphs = prefill_graphs  # capture prefill per prompt length (see prefill())
        self.prefill_graph_slots = 4
        self._prefill_graphs, self._prefill_seen = {}, {}

    def settings(self) -> dict:
        """The knobs in effect (for benchmark/provenance records)."""
        m = self.m
        return {"pdl": self.pdl, "decode_graph": True, "prefill_graphs": self.prefill_graphs,
                "prefill_graph_slots": self.prefill_graph_slots, "prefill_graph_lengths": sorted(self._prefill_graphs),
                "prefill_impl": m.prefill_impl, "prefill_copy": m.rowmajor_copy, "kmajor": m.kmajor,
                "depth": list(self.depth), "warps": list(self.warps), "pre": list(self.pre), "trig": list(self.trig),
                "attn_late": self.attn_late, "attn_chunk": self.attn_chunk, "lm_head_ctas": self.lm_head_ctas,
                "max_seq": self.max_seq}

    # ---------------------------------------------------------------------------------------------
    def decode_step(self):
        """Launch one decode step (eager). Also what gets captured into the graph."""
        m, k, cfg = self.m, self.k, self.m.cfg
        k.embed(self.token, m.embed, self.resid)
        for li, L in enumerate(m.layers):
            k.fp8_gemv(0, L.w_qkv, L.s_qkv, self.resid, L.ln1, self.qkv, L.w_qkv.shape[0], cfg.eps, self.depth[0], self.warps[0],
                       self.pre[0], self.trig[0], int(m.kmajor))
            k.attention(self.qkv, L.q_norm, L.k_norm, m.cos, m.sin, self.state, m.k_cache[li], m.v_cache[li],
                        self.o_part, self.ml_part, self.counters, self.attn, cfg.head_dim ** -0.5, cfg.eps,
                        self.attn_late, self.attn_chunk)
            k.fp8_gemv(1, L.w_o, L.s_o, self.attn, None, self.resid, L.w_o.shape[0], cfg.eps, self.depth[1], self.warps[1],
                       self.pre[1], self.trig[1], int(m.kmajor))
            k.fp8_gemv(2, L.w_gu, L.s_gu, self.resid, L.ln2, self.h, L.w_gu.shape[0], cfg.eps, self.depth[2], self.warps[2],
                       self.pre[2], self.trig[2], int(m.kmajor))
            k.fp8_gemv(3, L.w_d, L.s_d, self.h, None, self.resid, L.w_d.shape[0], cfg.eps, self.depth[3], self.warps[3],
                       self.pre[3], self.trig[3], int(m.kmajor))
        k.lm_head_argmax(m.lm_head, self.resid, m.norm, self.logits, self.best, cfg.eps, self.lm_head_ctas)
        k.finalize(self.best, self.token, self.out_tokens, self.state)

    def capture(self):
        """Record one decode step into a CUDA graph. Must be called with a valid state (after prefill)."""
        # The warm-up steps re-compute (identically) the KV entry at the current position, so only
        # the token and the position need restoring afterwards.
        token, state = self.token.clone(), self.state.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # warm up eagerly on a side stream before capture
                self.token.copy_(token); self.state.copy_(state)
                self.decode_step()
        torch.cuda.current_stream().wait_stream(s)
        self.token.copy_(token); self.state.copy_(state)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.decode_step()
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------------------
    def _prefill_body(self, ids: torch.Tensor) -> torch.Tensor:
        P = ids.shape[0]
        logits = self.m.forward(ids, 0)
        first = logits.float().argmax().to(torch.int32)
        self.token.copy_(first.view(1))
        self.state[0].fill_(P)     # position of the token the next step processes
        self.state[1].fill_(0)     # number of tokens written to out_tokens (fill_: graph-capturable)
        self.best.zero_()
        return first

    @torch.no_grad()
    def prefill(self, prompt_ids) -> torch.Tensor:
        """Run the prompt, fill the KV cache, and set up the device state for decoding.
        Returns the first generated token as a device tensor (no host sync).

        Eager prefill is hundreds of small launches (377 kernels with the fused path at M >= 16, 2068 with
        the op-by-op path; nsys at 1024 tokens), so short prompts are bound by host launch overhead.
        With prefill_graphs on, prefill is CUDA-graphed PER EXACT PROMPT LENGTH: the first time a length
        is seen it runs eagerly, the second time it is captured, later calls replay it (same kernels, same
        numerics, no host overhead). At most prefill_graph_slots lengths get a graph; any other length
        stays eager. So repeated lengths get the fast TTFT, and a new length pays eager prefill
        (bench.py reports both: ttft_ms and ttft_first_seen_ms)."""
        ids = torch.as_tensor(prompt_ids, dtype=torch.int64, device="cuda")
        P = ids.shape[0]
        assert 0 < P < self.max_seq
        if not self.prefill_graphs:
            return self._prefill_body(ids)
        entry = self._prefill_graphs.get(P)
        if entry is None:
            self._prefill_seen[P] = self._prefill_seen.get(P, 0) + 1
            if self._prefill_seen[P] < 2 or len(self._prefill_graphs) >= self.prefill_graph_slots:
                return self._prefill_body(ids)   # first sighting (also compiles Triton kernels): eager
            ids_buf = ids.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                first = self._prefill_body(ids_buf)
            entry = self._prefill_graphs[P] = (g, ids_buf, first)
        g, ids_buf, first = entry
        ids_buf.copy_(ids)
        g.replay()
        return first

    @torch.no_grad()
    def generate(self, prompt_ids, gen_len: int, use_graph: bool = True):
        """Greedy generation of exactly gen_len tokens (no early stop). Returns a list of ids."""
        assert len(prompt_ids) + gen_len <= self.max_seq
        first = self.prefill(prompt_ids)
        if gen_len > 1:
            if use_graph and self.graph is None:
                self.capture()
            for _ in range(gen_len - 1):
                if use_graph:
                    self.graph.replay()
                else:
                    self.decode_step()
        out = [int(first.item())]
        out += self.out_tokens[: gen_len - 1].tolist()
        return out

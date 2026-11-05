"""EngineModel.stream (the real-engine adapter) driven by a list-backed fake torch and a fake Engine.

Checks the bookkeeping that a GPU run would otherwise be the first to exercise: step 0 is the prefill
token, step s >= 1 reads out_tokens[s - 1] after decode replay s - 1, exactly max_tokens - 1 replays
run, at most `lookahead` steps are in flight, logprobs reach the host, and the prefill path is labelled.
"""
from __future__ import annotations

import importlib.util
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ENGINE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qwen3_fp8_serve_adapter", ENGINE_DIR / "serve.py")
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)


class FT:
    """A 1-D tensor view over a Python list (slices share storage, like torch views)."""

    def __init__(self, store, off=0, n=None):
        self.store, self.off = store, off
        self.n = len(store) - off if n is None else n

    def vals(self):
        return self.store[self.off:self.off + self.n]

    def __getitem__(self, k):
        if isinstance(k, slice):
            start, stop, _ = k.indices(self.n)
            return FT(self.store, self.off + start, stop - start)
        return self.store[self.off + k]

    def copy_(self, src, non_blocking=False):
        v = src.vals()
        assert len(v) == self.n, (len(v), self.n)
        self.store[self.off:self.off + self.n] = v
        return self

    def view(self, *shape):
        assert shape == (1,) and self.n == 1
        return self

    def float(self):
        return FT([float(x) for x in self.vals()])

    def max(self):
        return FT([max(self.vals())])

    def __sub__(self, other):
        return FT([a - b for a, b in zip(self.vals(), other.vals())])


class FakeTorch:
    int32, float32 = "int32", "float32"
    syncs = 0

    class cuda:
        class Stream:
            def wait_event(self, ev):
                assert ev.recorded

        class Event:
            recorded = False

            def record(self, stream=None):
                self.recorded = True

            def synchronize(self):
                assert self.recorded

        @staticmethod
        @contextmanager
        def stream(s):
            yield

        @staticmethod
        def synchronize():
            FakeTorch.syncs += 1

    @staticmethod
    def zeros(n, dtype=None, pin_memory=False, device=None):
        return FT([0] * n)

    @staticmethod
    def logsumexp(t, dim):
        return FT([math.log(sum(math.exp(x) for x in t.vals()))])

    @staticmethod
    def set_grad_enabled(flag):
        assert flag is False


V = 10


def logits_for(tok):
    row = [0.0] * V
    row[tok] = 2.0
    return row


LP = 2.0 - math.log(math.exp(2.0) + (V - 1))   # log-softmax of the argmax in logits_for()


class FakeEngine:
    """Engine's interface as serve.py uses it, with its prefill-graph policy (graph on the 2nd sighting)."""

    def __init__(self, script, max_seq=64):
        self.script, self.max_seq = script, max_seq
        self.out_tokens = FT([-1] * max_seq)
        self.logits = FT([0.0] * V)
        self.m = SimpleNamespace(cfg=SimpleNamespace(vocab=V), last_prefill_path=None, forward=self._forward)
        self._prefill_graphs, self._prefill_seen = {}, {}
        self.prefill_graph_slots = 4
        self.graph = SimpleNamespace(replay=self._replay)
        self.replays = self.step = 0

    def _forward(self, tokens, start):
        return FT(logits_for(self.script[0]))

    def prefill(self, prompt_ids):
        P = len(prompt_ids)
        replay = P in self._prefill_graphs
        self.m.forward(prompt_ids, 0)   # also runs inside a replayed graph (the hook's ops are captured)
        if not replay:
            self.m.last_prefill_path = "fused" if P >= 16 else "ref"
            self._prefill_seen[P] = self._prefill_seen.get(P, 0) + 1
            if self._prefill_seen[P] >= 2:
                self._prefill_graphs[P] = "graph"
        else:
            self.m.last_prefill_path = "stale"   # a replay runs no Python: whatever ran last stays here
        self.step = 0
        return FT([self.script[0]])

    def _replay(self):
        self.replays += 1
        tok = self.script[(self.step + 1) % len(self.script)]
        self.logits.copy_(FT(logits_for(tok)))
        self.out_tokens[self.step:self.step + 1].copy_(FT([tok]))
        self.step += 1


def adapter(script, lookahead=3, logprobs=True):
    eng = FakeEngine(script)
    em = serve.EngineModel.__new__(serve.EngineModel)
    em._setup(FakeTorch, eng, lookahead, logprobs, device="cpu")
    return em, eng


def test_stream_tokens_logprobs_and_replay_count():
    script = [5, 6, 7, 8, 9, 1, 3]
    em, eng = adapter(script)
    info = {}
    out = list(em.stream(list(range(20)), 6, True, info))
    assert [t for t, _ in out] == script[:6]
    assert all(abs(lp - LP) < 1e-9 for _, lp in out)
    assert eng.replays == 5                      # max_tokens - 1 decode replays; token 0 is the prefill's
    assert info == {"prefill": "eager", "prefill_impl": "fused"}


def test_prefill_path_labels_follow_the_engine_policy():
    em, eng = adapter([4, 5])
    labels = []
    for _ in range(3):
        info = {}
        list(em.stream([1] * 8, 2, False, info))
        labels.append((info["prefill"], info["prefill_impl"]))
    assert labels == [("eager", "ref"), ("graph_capture", "ref"), ("graph_replay", "ref")]


def test_single_token_and_no_logprobs():
    em, eng = adapter([4, 5])
    assert list(em.stream([1, 2], 1, False, {})) == [(4, None)]
    assert eng.replays == 0


def test_early_close_stops_launching_and_drains():
    em, eng = adapter(list(range(1, 9)), lookahead=3)
    before = FakeTorch.syncs
    g = em.stream([1, 2, 3], 40, False, {})
    assert [next(g)[0], next(g)[0]] == [1, 2]
    g.close()
    # steps 0..4 launched (2 collected + 3 in flight); step 0 is the prefill, so 4 replays
    assert eng.replays == 4
    assert FakeTorch.syncs == before + 1

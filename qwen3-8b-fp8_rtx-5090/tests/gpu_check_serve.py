"""GPU check for serve.py (not collected by pytest): run on the GPU host before trusting serve numbers.

  python tests/gpu_check_serve.py --model PATH [--prompt-len 128] [--gen-len 256] [--reps 5] [--http]

1. Correctness: the token ids serve.py streams (EngineModel.stream) equal Engine.generate's for the same
   prompt; again after a stream that was closed early (stop token / client gone); with logprobs on (the
   default; --no-logprobs skips the prefill hook) the tokens are unchanged and every logprob is finite
   and <= 0. With the hook on, both generate and stream TTFT include its log-softmax.
2. Overhead, interleaved per round, medians printed as JSON lines on stdout:
     generate  bench.py's method: (gen_len - 1) / (t(gen_len) - t(1)), host-timed Engine.generate
     stream    EngineModel.stream in-process: (gen_len - 1) / (t_last_token - t_first_token), TTFT
     http      (--http) the HTTP server in this process, a client in a separate process timing SSE chunks
   stream vs generate isolates the per-step event + side-stream copy; http adds HTTP/SSE/detokenization.
Exit code 1 if a correctness check fails. Logs on stderr. Take a GPU lock (flock) when sharing the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIENT = r"""
import http.client, json, sys, time
port, body = int(sys.argv[1]), sys.argv[2]
c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
t0 = time.perf_counter()
c.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
r = c.getresponse()
assert r.status == 200, r.read()
times, ids = [], []
while True:
    line = r.readline()
    if not line:
        break
    if line.startswith(b"data: {"):
        ev = json.loads(line[6:])
        if ev["choices"]:
            times.append(time.perf_counter() - t0)
            ids += ev["choices"][0]["token_ids"]
    elif line.startswith(b"data: [DONE]"):
        break
print(json.dumps({"times": times, "ids": ids}))
"""


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--http", action="store_true", help="also time the HTTP path (loads the tokenizer)")
    ap.add_argument("--no-logprobs", action="store_true",
                    help="start without the prefill logprob hook (TTFT then matches a server without --logprobs)")
    a = ap.parse_args()
    sys.path.insert(0, ENGINE_DIR)
    import serve
    tokenizer = serve.HFTokenizer(a.model) if a.http else None   # before the engine import (sys.path)
    import torch
    from bench import synthetic_prompt, timed_generate

    args = SimpleNamespace(model=a.model, max_seq=4096, no_pdl=False, no_prefill_copy=False, no_prefill_graphs=False,
                           lookahead=a.lookahead, logprobs=not a.no_logprobs, warmup_lengths=[a.prompt_len])
    em = serve.EngineModel(args)
    eng = em.engine
    prompt = synthetic_prompt(a.prompt_len)
    ok = True

    def stream(n, logprobs=False, stop_after=None):
        t0 = time.perf_counter()
        times, toks, lps = [], [], []
        g = em.stream(prompt, n, logprobs, {})
        for t, lp in g:
            times.append(time.perf_counter() - t0)
            toks.append(t)
            lps.append(lp)
            if stop_after and len(toks) >= stop_after:
                g.close()
                break
        return times, toks, lps

    ref = eng.generate(prompt, a.gen_len)
    _, toks, _ = stream(a.gen_len)
    checks = {"stream_equals_generate": toks == ref}
    stream(a.gen_len, stop_after=7)
    _, toks2, _ = stream(a.gen_len)
    checks["equal_after_early_close"] = toks2 == ref
    lps = []
    if em.logprobs_supported:
        _, toks3, lps = stream(a.gen_len, logprobs=True)
        checks["logprobs_keep_tokens"] = toks3 == ref
        checks["logprobs_finite_nonpositive"] = all(lp is not None and math.isfinite(lp) and lp <= 1e-6 for lp in lps)
    ok = all(checks.values())
    print(json.dumps({"kind": "correctness", **checks, "prompt_len": a.prompt_len, "gen_len": a.gen_len,
                      "first_logprobs": [round(x, 4) for x in lps[:4]]}), flush=True)

    srv = None
    if a.http:
        import threading
        srv = serve.make_server(em, tokenizer, port=0, served_model_name="check", eos_ids=(),
                                vocab_size=em.vocab_size)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        body = json.dumps({"prompt": prompt, "max_tokens": a.gen_len, "temperature": 0, "ignore_eos": True,
                           "stream": True})

    rows = []
    for r in range(a.reps + 1):   # round 0 is a warm-up
        t1, _ = timed_generate(eng, prompt, 1)
        tn, _ = timed_generate(eng, prompt, a.gen_len)
        times, _, _ = stream(a.gen_len)
        row = {"generate_decode_tok_s": (a.gen_len - 1) / (tn - t1), "generate_ttft_ms": t1 * 1e3,
               "stream_decode_tok_s": (a.gen_len - 1) / (times[-1] - times[0]), "stream_ttft_ms": times[0] * 1e3}
        if srv is not None:
            out = subprocess.run([sys.executable, "-I", "-c", CLIENT, str(srv.server_address[1]), body],
                                 capture_output=True, text=True, timeout=600)
            res = json.loads(out.stdout)
            ht = res["times"]
            row.update({"http_decode_tok_s": (a.gen_len - 1) / (ht[-1] - ht[0]), "http_ttft_ms": ht[0] * 1e3,
                        "http_tokens_equal": res["ids"] == ref})
            ok = ok and row["http_tokens_equal"]
        log(f"[check] round {r}: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
        if r > 0:
            rows.append(row)
    summary = {"kind": "overhead", "claim_class": "observed", "n": len(rows), "lookahead": a.lookahead,
               "settings": eng.settings(), "gpu": torch.cuda.get_device_name(0)}
    for k in rows[0]:
        if isinstance(rows[0][k], float):
            summary[k + "_median"] = round(statistics.median(x[k] for x in rows), 3)
    summary["stream_vs_generate_decode_ratio"] = round(summary["stream_decode_tok_s_median"] /
                                                       summary["generate_decode_tok_s_median"], 4)
    print(json.dumps(summary), flush=True)
    if srv is not None:
        srv.shutdown()
        srv.scheduler.stop()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

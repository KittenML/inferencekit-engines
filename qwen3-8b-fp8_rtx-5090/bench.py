"""Batch-1 decode benchmark.

Harness interface (JSON lines on stdout, logs on stderr):
  python bench.py --model PATH --prompt-ids-file FILE --gen-len 256 --reps 5 --warmup 2
stdout:
  1. one {"kind": "meta", ...} line once the engine is loaded: versions (torch and its CUDA, triton, GPU,
     driver, nvcc used for the extension), engine identity (directory, git describe, source md5) and the
     settings in effect (PDL, prefill graphs, prefill implementation, prefill weight copy, GEMV knobs, ...).
     The same information goes to stderr.
  2. one line per measured rep: {"rep", "ttft_ms", "ttft_first_seen_ms", "decode_tok_s", "tokens"}
     (tokens: the first 64 generated ids of rep 0, else null).

Timing (greedy, no early stop, host wall clock, each run ends with a device sync):
  ttft_ms             a 1-token generation (prefill + first token) for a prompt length whose prefill is
                      already captured as a CUDA graph. Engine.prefill graphs prefill per EXACT prompt length
                      (eager on the 1st sighting of a length, captured on the 2nd, replayed afterwards; at
                      most 4 lengths). So this is the TTFT of a REPEATED prompt length.
  ttft_first_seen_ms  the same request with prefill run eagerly: the TTFT of a prompt length that has no
                      captured graph (kernels already compiled; a cold process also pays Triton compilation,
                      and the 2nd sighting of a length pays the graph capture: see meta "ttft_second_sighting_ms").
  decode_tok_s        (gen_len - 1) / (t_gen_len - t_1), both runs with the graph-replayed prefill.
With --no-prefill-graphs every prefill is eager, so ttft_ms is the first-seen number too; decode_tok_s then
gets noisier, because eager prefill at short prompts is host-bound and its jitter (several ms) enters
t_gen_len - t_1. ttft_first_seen_ms itself varies more between reps than ttft_ms for the same reason.
Every graph capture (the decode graph and this length's prefill graph) happens during loading, before any
timed run, so no rep includes capture time, also with --warmup 0. Warm-up reps are run and discarded.

--summary prints one line with medians instead of the per-rep lines (after the meta line).
--loop-seconds S is a profiling mode: after loading, generate(prompt, gen_len) runs back to back for S
seconds, one {"kind": "loop", ...} line per run, so a profiler started with a delay (nsys --delay
--duration) records steady-state decoding. The context is the prompt (--prompt-len / --prompt-ids-file).
Without --prompt-ids-file a fixed synthetic prompt of --prompt-len tokens is used.
This script never takes a GPU lock itself; wrap it (e.g. flock) when sharing a GPU.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from generate import Engine, source_identity  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def emit(row):
    print(json.dumps(row), flush=True)


def synthetic_prompt(n):
    # Deterministic, ordinary-looking token ids (no special tokens); only the length matters for speed.
    return [(i * 7919 + 13) % 150000 + 100 for i in range(n)]


def timed_generate(engine, prompt, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks = engine.generate(prompt, n)  # returns host ints, so it ends with a device sync
    return time.perf_counter() - t0, toks


def timed_first_seen(engine, prompt):
    """1-token generation with the prefill run eagerly, as for a prompt length without a captured graph."""
    saved = engine.prefill_graphs
    engine.prefill_graphs = False
    try:
        return timed_generate(engine, prompt, 1)
    finally:
        engine.prefill_graphs = saved


def _cmd_out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def versions():
    v = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0), "sm": "%d.%d" % torch.cuda.get_device_capability(0)}
    try:
        import triton
        v["triton"] = triton.__version__
    except ImportError:
        v["triton"] = None
    drv = _cmd_out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"])
    v["driver"] = drv.splitlines()[0] if drv else None
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = _cmd_out([os.path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc", "--version"])
    v["nvcc"] = nvcc.splitlines()[-1] if nvcc else None   # e.g. "Build cuda_13.0.r13.0/..."
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-ids-file")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--loop-seconds", type=float, default=0,
                    help="profiling mode: decode back to back for this many seconds after loading")
    ap.add_argument("--no-pdl", action="store_true", help="disable programmatic dependent launch (A/B)")
    ap.add_argument("--no-prefill-copy", action="store_true",
                    help="no row-major prefill weight copy (-6.4 GiB VRAM, slower prefill, same outputs)")
    ap.add_argument("--no-prefill-graphs", action="store_true",
                    help="never CUDA-graph the prefill: every TTFT is the first-seen (eager) one")
    args = ap.parse_args()
    if args.gen_len < 2:
        ap.error("--gen-len must be >= 2 (decode_tok_s is measured over gen_len - 1 tokens)")
    if args.warmup < 0 or args.reps < 0:
        ap.error("--warmup and --reps must be >= 0")

    prompt = json.load(open(args.prompt_ids_file)) if args.prompt_ids_file else synthetic_prompt(args.prompt_len)
    assert isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)
    log(f"[bench] prompt_len={len(prompt)} gen_len={args.gen_len}")

    t = time.perf_counter()
    engine = Engine(args.model, max_seq=args.max_seq, pdl=not args.no_pdl, prefill_copy=not args.no_prefill_copy,
                    prefill_graphs=not args.no_prefill_graphs)
    load_s = time.perf_counter() - t
    # Untimed: 1st sighting of this length (eager prefill, compiles the Triton kernels) + decode-graph capture.
    t = time.perf_counter()
    engine.generate(prompt, 2)
    first_call_s = time.perf_counter() - t
    # 2nd sighting: captures this length's prefill graph (no-op with --no-prefill-graphs). Timed for the
    # record, never part of a rep.
    second_s, _ = timed_generate(engine, prompt, 1)

    meta = {"kind": "meta", "engine": "qwen3-fp8-decode", "versions": versions(), "source": source_identity(),
            "settings": engine.settings(),
            "run": {"prompt_len": len(prompt), "prompt": args.prompt_ids_file or f"synthetic({args.prompt_len})",
                    "gen_len": args.gen_len, "reps": args.reps, "warmup": args.warmup,
                    "loop_seconds": args.loop_seconds, "summary": args.summary},
            "load_s": round(load_s, 2), "first_call_s": round(first_call_s, 3),
            "ttft_second_sighting_ms": round(second_s * 1e3, 3),
            "timing": {"ttft_ms": "repeated prompt length: prefill CUDA graph replayed"
                                  if not args.no_prefill_graphs else "eager prefill (prefill graphs off)",
                       "ttft_first_seen_ms": "eager prefill, kernels compiled (a length with no captured graph)",
                       "decode_tok_s": "(gen_len - 1) / (t_gen_len - t_1)"}}
    log(f"[bench] {json.dumps(meta)}")
    emit(meta)

    if args.loop_seconds:
        t_end = time.perf_counter() + args.loop_seconds
        while time.perf_counter() < t_end:
            dt, toks = timed_generate(engine, prompt, args.gen_len)
            assert len(toks) == args.gen_len
            emit({"kind": "loop", "t": round(time.time(), 3), "seconds": round(dt, 4),
                  "decode_tok_s_incl_prefill": round(args.gen_len / dt, 3)})
        return

    rows = []
    for i in range(args.warmup + args.reps):
        t1, _ = timed_generate(engine, prompt, 1)
        tn, toks = timed_generate(engine, prompt, args.gen_len)
        te, _ = timed_first_seen(engine, prompt)
        assert len(toks) == args.gen_len
        rep = i - args.warmup
        row = {"rep": rep, "ttft_ms": t1 * 1e3, "ttft_first_seen_ms": te * 1e3,
               "decode_tok_s": (args.gen_len - 1) / (tn - t1), "tokens": toks[:64] if rep == 0 else None}
        log(f"[bench] {'warmup' if rep < 0 else 'rep'} {i}: ttft {row['ttft_ms']:.2f} ms "
            f"(first-seen length {row['ttft_first_seen_ms']:.2f} ms), decode {row['decode_tok_s']:.2f} tok/s")
        if rep >= 0:
            rows.append((row, toks))
            if not args.summary:
                emit(row)

    if args.summary and rows:
        emit({
            "decode_tok_s": statistics.median(r["decode_tok_s"] for r, _ in rows),
            "ttft_ms": statistics.median(r["ttft_ms"] for r, _ in rows),
            "ttft_first_seen_ms": statistics.median(r["ttft_first_seen_ms"] for r, _ in rows),
            "prompt_len": len(prompt), "gen_len": args.gen_len,
            "tokens": rows[0][1], "versions": meta["versions"], "settings": meta["settings"],
        })


if __name__ == "__main__":
    main()

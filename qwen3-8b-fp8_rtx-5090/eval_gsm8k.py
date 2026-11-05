"""Greedy generation that stops at EOS, for task evaluations (e.g. GSM8K) on pre-tokenized prompts.

  python eval_gsm8k.py --model PATH --inputs prompts.jsonl --out outputs.jsonl [--max-new 512]

Input: one JSON object per line with "prompt_ids" (and an "idx" that is copied to the output).
Output: {"idx", "output_ids", "finish": "stop" | "length"}; output_ids end with the EOS token when one
was produced. Prompt formatting (chat template) and grading are left to the caller.

Engine.generate always produces exactly gen_len tokens. Here the decode graph is replayed in chunks of
--chunk steps and the device-side token buffer is checked for EOS after each chunk. Steps after EOS only
write KV-cache entries beyond the answer, so the tokens up to EOS are the same as one long generation.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from generate import Engine  # noqa: E402

EOS = (151645, 151643)  # Qwen3: <|im_end|>, <|endoftext|>


@torch.no_grad()
def generate_until(eng: Engine, prompt, max_new: int, stop=EOS, chunk: int = 16):
    assert len(prompt) + max_new + chunk <= eng.max_seq
    out = [int(eng.prefill(prompt).item())]
    if out[0] in stop:
        return out, "stop"
    if eng.graph is None:
        eng.capture()
    while len(out) < max_new:
        n = min(chunk, max_new - len(out))
        for _ in range(n):
            eng.graph.replay()
        # replay j (0-based, since prefill) writes out_tokens[j]; len(out) - 1 replays came before
        for t in eng.out_tokens[len(out) - 1: len(out) - 1 + n].tolist():
            out.append(t)
            if t in stop:
                return out, "stop"
    return out, "length"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args()
    t = time.time()
    eng = Engine(a.model, max_seq=4096)
    print(f"[eval] loaded {time.time() - t:.0f}s", file=sys.stderr, flush=True)
    with open(a.out, "w") as f:
        for r in map(json.loads, open(a.inputs)):
            ids, finish = generate_until(eng, r["prompt_ids"], a.max_new)
            f.write(json.dumps({"idx": r.get("idx"), "engine": "ours", "output_ids": ids, "finish": finish}) + "\n")
            f.flush()
    print(f"[eval] done {time.time() - t:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

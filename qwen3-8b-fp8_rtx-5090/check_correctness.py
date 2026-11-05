"""Greedy-output correctness vs a Hugging Face transformers reference.

Step 1 (reference, separate process):
  python check_correctness.py hf --model PATH --out ref.json [--attn sdpa|eager] [--gen-len 256]
  Runs transformers greedy generation (no early stop) on a fixed prompt set and records the
  tokens plus the top-8 logits of every step.
Step 2 (this engine):
  python check_correctness.py engine --model PATH --ref ref.json
  - free-running: exact match of the generated tokens, first divergence position;
  - teacher-forced: feed the reference tokens and check that our argmax equals the reference's
    next token at every step (isolates per-step agreement from divergence cascades);
    at each disagreement, print the reference's top-1 vs top-2 logit gap (near-ties expected).
Step 2b: python check_correctness.py compare --ref a.json --other b.json  (two references, e.g.
  sdpa vs eager attention in transformers, to measure the noise floor of equivalent implementations)
Prefill bitwise identity (memcmp), for changes to the prefill path:
  python check_correctness.py prefill-dump --model PATH --out DIR [--impl fused|ref] [--lengths 128,512,1024,2048]
  python check_correctness.py prefill-compare --a DIR_A --b DIR_B
  For each prompt length: last-position logits and the KV cache [:, :, :len] after an eager prefill, plus
  an in-process check that the prefill CUDA-graph replay gives the same first token and KV bytes.
  Each dump records its provenance (requested impl, the forward path that actually ran, prefill weight
  copy, engine source md5). prefill-compare prints both sides' provenance, fails on missing files, exits
  non-zero unless everything is identical, and says so when both sides ran the same code and path (then
  the result only shows run-to-run determinism, not equivalence of two implementations).
  python check_correctness.py prefill-selftest --model PATH
  Checks the checker: "fused" vs "ref" must run different forward paths and still be identical, a
  deliberately perturbed implementation (RMSNorm rounded once instead of twice) must be reported as
  NOT IDENTICAL, and an unknown --impl name must be rejected.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARAGRAPHS = [
    "The history of mathematics spans thousands of years, from counting marks on bones to modern abstract algebra. "
    "Early civilizations in Mesopotamia and Egypt developed arithmetic and geometry for trade, astronomy and land surveying.",
    "In computer architecture, the memory hierarchy trades capacity for speed: registers, caches, main memory and storage "
    "each sit at a different point on that curve, and good programs are written to respect it.",
    "A river delta forms where a river meets a slower body of water and drops the sediment it carries. Over centuries the "
    "channels shift, splitting and rejoining, and the land grows outward into the sea.",
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly red and blue light, which is "
    "why leaves look green, and the captured energy is stored in sugars built from carbon dioxide and water.",
]

PROMPTS = {
    "short_story": "Once upon a time, in a small village at the edge of a forest, there lived a clockmaker who",
    "code": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "facts": "Here are five interesting facts about the planet Mars:\n1.",
    "explain": "Explain in simple terms why the sky is blue during the day and red at sunset.",
    "math": "Question: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nAnswer:",
    # ~1000-token prompt: repeated paragraphs followed by a question.
    "long_context": " ".join(PARAGRAPHS * 11) + "\n\nSummarize the four topics above in one sentence each:\n",
}


def cmd_hf(args):
    # This directory contains a package named `kernels`, which would shadow the `kernels` package
    # transformers uses to fetch its FP8 kernel. The reference does not need our code: drop it.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="cuda",
                                                 attn_implementation=args.attn)
    model.eval()
    gc = GenerationConfig(do_sample=False, max_new_tokens=args.gen_len, eos_token_id=None,
                          pad_token_id=tok.pad_token_id or 0, output_scores=False, output_logits=True,
                          return_dict_in_generate=True)
    out = {"model": args.model, "attn": args.attn, "gen_len": args.gen_len,
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
           "prompts": {}}
    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            g = model.generate(ids, generation_config=gc)
        gen = g.sequences[0, ids.shape[1]:].tolist()
        assert len(gen) == args.gen_len, len(gen)
        top = []
        for lg in g.logits:  # one [1, V] tensor per generated token (raw logits, before processors)
            v, i = lg[0].float().topk(8)
            top.append({"ids": i.tolist(), "vals": v.tolist()})
        out["prompts"][name] = {"prompt_ids": ids[0].tolist(), "tokens": gen, "top": top}
        print(f"[hf] {name}: prompt {ids.shape[1]} tok -> {tok.decode(gen[:24])!r}", file=sys.stderr)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}", file=sys.stderr)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def cmd_engine(args):
    import torch
    from generate import Engine, source_identity

    ref = json.load(open(args.ref))
    eng = Engine(args.model, max_seq=4096)
    print(f"[engine] path={args.path}", file=sys.stderr)
    report = {"ref": args.ref, "path": args.path, "source": source_identity(), "settings": eng.settings(),
              "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(0)},
              "prompts": {}}
    tot_exact = tot_tf = tot_steps = 0
    for name, r in ref["prompts"].items():
        prompt, rtoks, top = r["prompt_ids"], r["tokens"], r["top"]
        n = len(rtoks)
        logit_hist = torch.empty(n, eng.logits.numel(), dtype=torch.bfloat16, device="cuda")
        if args.path == "graph":
            # Free-running greedy generation.
            ours = eng.generate(prompt, n)
            div = first_divergence(ours, rtoks)
            # Teacher-forced: the input of step i is the reference token i; compare our argmax with ref token i+1.
            first = eng.prefill(prompt)
            tf_pred = [int(first.item())]
            forced = torch.tensor(rtoks, dtype=torch.int32, device="cuda")
            for i in range(n - 1):
                eng.token.copy_(forced[i:i + 1])
                eng.graph.replay()
                logit_hist[i + 1].copy_(eng.logits)
            tf_pred += eng.out_tokens[: n - 1].tolist()
        else:
            # Reference path: PyTorch/Triton forward one token at a time (SDPA attention, cuBLAS lm_head).
            P = len(prompt)
            with torch.no_grad():
                lg = eng.m.forward(torch.tensor(prompt, device="cuda"), 0)
                tf_pred = [int(lg.float().argmax())]
                for i in range(n - 1):
                    lg = eng.m.forward(torch.tensor([rtoks[i]], device="cuda"), P + i)
                    logit_hist[i + 1].copy_(lg)
                    tf_pred.append(int(lg.float().argmax()))
            div = None if tf_pred == rtoks else -1  # free-running not run on this path
        agree = [int(a == b) for a, b in zip(tf_pred, rtoks)]
        mism = [i for i, a in enumerate(agree) if not a]
        details = []
        for i in mism[:10]:
            vals = top[i]["vals"]
            details.append({"step": i, "ref": rtoks[i], "ours": tf_pred[i], "ref_top1_minus_top2": vals[0] - vals[1]})
        # Logit error on the reference's top-8 tokens (teacher-forced steps 1..n-1).
        ids = torch.tensor([t["ids"] for t in top[1:]], device="cuda")
        rv = torch.tensor([t["vals"] for t in top[1:]], device="cuda")
        ov = logit_hist[1:].float().gather(1, ids)
        err = (ov - rv).abs()
        report["prompts"][name] = {
            "prompt_len": len(prompt), "gen_len": n, "exact_match": div is None, "first_divergence": div,
            "teacher_forced_agreement": sum(agree) / n, "tf_mismatches": details,
            "top8_logit_abs_err_max": err.max().item(), "top8_logit_abs_err_mean": err.mean().item(),
        }
        tot_exact += int(div is None)
        tot_tf += sum(agree)
        tot_steps += n
        print(f"[engine] {name:13s} prompt {len(prompt):5d}  exact {div is None!s:5s}  first div {div!s:>4s}  "
              f"teacher-forced {sum(agree)}/{n}  top8 |dlogit| max {err.max().item():.3f} mean {err.mean().item():.4f}"
              + (f"  mismatch gaps {[round(d['ref_top1_minus_top2'], 3) for d in details]}" if details else ""))
    report["summary"] = {"exact_match_prompts": f"{tot_exact}/{len(ref['prompts'])}",
                         "teacher_forced_agreement": tot_tf / tot_steps}
    print(json.dumps(report["summary"]))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


def cmd_compare(args):
    a, b = json.load(open(args.ref)), json.load(open(args.other))
    for name in a["prompts"]:
        ta, tb = a["prompts"][name]["tokens"], b["prompts"][name]["tokens"]
        div = first_divergence(ta, tb)
        print(f"{name:13s} exact {div is None!s:5s} first divergence {div}")


def prefill_dump(eng, prompts, out_dir, impl=None, label=None):
    """Dump last-position logits + KV cache after prefill for each prompt (see the module docstring).
    impl: model.Qwen3FP8.prefill_impl to use (validated by the model; None = keep the engine default)."""
    import torch
    from generate import source_identity

    m = eng.m
    if impl is not None:
        m.prefill_impl = impl            # raises ValueError for an unknown name
    os.makedirs(out_dir, exist_ok=True)
    src = source_identity()
    for name, p in prompts.items():
        n = len(p)
        ids = torch.tensor(p, device="cuda")
        m.k_cache.zero_(); m.v_cache.zero_()
        with torch.no_grad():
            logits = m.forward(ids, 0).clone()
        path = m.last_prefill_path
        kc, vc = m.k_cache[:, :, :n].clone(), m.v_cache[:, :, :n].clone()
        # The engine's prefill (eager, capture, replay) must give the same token and KV bytes.
        eng._prefill_graphs.clear(); eng._prefill_seen.clear()
        toks = []
        for _ in range(3):
            m.k_cache.zero_(); m.v_cache.zero_()
            toks.append(int(eng.prefill(p)))
            same = torch.equal(kc, m.k_cache[:, :, :n]) and torch.equal(vc, m.v_cache[:, :, :n])
            assert same, f"{name}: engine prefill KV differs from forward()"
        assert toks == [int(logits.float().argmax())] * 3, (name, toks)
        eng._prefill_graphs.clear(); eng._prefill_seen.clear()
        meta = {"impl": m.prefill_impl, "path": path, "prefill_copy": m.rowmajor_copy, "label": label,
                "src_md5": src["src_md5"], "git_describe": src["git_describe"], "dir": src["dir"]}
        torch.save({"logits": logits.cpu(), "k": kc.cpu(), "v": vc.cpu(), "first": toks[0], "meta": meta},
                   os.path.join(out_dir, f"{name}.pt"))
        print(f"[prefill-dump] {name}: len {n} path {path} first token {toks[0]}  eager/capture/replay KV identical",
              file=sys.stderr)


def prefill_compare(dir_a, dir_b, out=print):
    """memcmp two prefill dumps. Returns (identical, same_provenance)."""
    import torch
    fa, fb = sorted(os.listdir(dir_a)), sorted(os.listdir(dir_b))
    ok = bool(fa) and fa == fb
    if fa != fb:
        out(f"FILE SETS DIFFER: only in a: {sorted(set(fa) - set(fb))}  only in b: {sorted(set(fb) - set(fa))}")
    same_prov = True
    for f in sorted(set(fa) & set(fb)):
        a = torch.load(os.path.join(dir_a, f), weights_only=False)
        b = torch.load(os.path.join(dir_b, f), weights_only=False)
        ma, mb = a.get("meta"), b.get("meta")
        desc = lambda md: ("no provenance (old dump)" if md is None else
                           f"impl {md['impl']} path {md['path']} copy {md['prefill_copy']} src {md['src_md5'][:8]}"
                           + (f" [{md['label']}]" if md.get("label") else ""))
        keys = ("path", "prefill_copy", "src_md5", "label")
        same_prov &= ma is not None and mb is not None and all(ma.get(k) == mb.get(k) for k in keys)
        res = []
        for key in ("logits", "k", "v"):
            x, y = a[key], b[key]
            if x.shape != y.shape:
                res.append(f"{key} SHAPE {tuple(x.shape)} vs {tuple(y.shape)}")
                ok = False
                continue
            xi, yi = x.view(torch.int16), y.view(torch.int16)          # memcmp of the bf16 bits
            eq = torch.equal(xi, yi)
            res.append(f"{key} {'identical' if eq else f'DIFF in {int((xi != yi).sum())}/{x.numel()}'}")
            ok &= eq
        ok &= a["first"] == b["first"]
        out(f"{f:14s} " + "  ".join(res) + f"  first token {a['first']} vs {b['first']}")
        out(f"{'':14s} a: {desc(ma)}   b: {desc(mb)}")
    if ok:
        out("ALL BITWISE IDENTICAL" + ("  (NOTE: both sides ran the same code and the same forward path, so this "
                                       "only shows run-to-run determinism)" if same_prov else ""))
    else:
        out("NOT IDENTICAL")
    return ok, same_prov


def _dump_prompts(lengths, ref=None):
    from bench import synthetic_prompt
    prompts = {f"syn{n}": synthetic_prompt(int(n)) for n in lengths.split(",")}
    if ref:
        r = json.load(open(ref))["prompts"]["long_context"]["prompt_ids"]
        prompts[f"text{len(r)}"] = r
    return prompts


def cmd_prefill_dump(args):
    from generate import Engine
    eng = Engine(args.model, max_seq=4096, prefill_copy=not args.no_prefill_copy)
    prefill_dump(eng, _dump_prompts(args.lengths, args.ref), args.out, impl=args.impl, label=args.label)


def cmd_prefill_compare(args):
    ok, _ = prefill_compare(args.a, args.b)
    sys.exit(0 if ok else 1)


def cmd_prefill_selftest(args):
    """The checker must (1) switch implementations for real, (2) find a deliberately different one,
    (3) reject unknown implementation names."""
    import shutil
    import tempfile

    import torch

    import model as md
    from generate import Engine

    eng = Engine(args.model, max_seq=4096)
    prompts = _dump_prompts(args.lengths)
    tmp = tempfile.mkdtemp(prefix="prefill_selftest_")
    lines = []
    say = lambda s: (lines.append(s), print(s, flush=True))
    passed = True
    try:
        d_fused, d_ref, d_bad = (os.path.join(tmp, x) for x in ("fused", "ref", "perturbed"))
        prefill_dump(eng, prompts, d_fused, impl="fused")
        prefill_dump(eng, prompts, d_ref, impl="ref")

        # (1) fused vs ref: different forward paths (recorded per dump), identical bits.
        say("== fused vs ref (expected: different paths for M >= 16, ALL BITWISE IDENTICAL)")
        ok, same_prov = prefill_compare(d_fused, d_ref, say)
        paths = {f: (torch.load(os.path.join(d_fused, f), weights_only=False)["meta"]["path"],
                     torch.load(os.path.join(d_ref, f), weights_only=False)["meta"]["path"]) for f in os.listdir(d_fused)}
        want = {f: ("fused" if len(prompts[f[:-3]]) >= 16 else "ref", "ref") for f in paths}
        switched = paths == want and any(a != b for a, b in paths.values())
        say(f"paths actually run (fused dump, ref dump): {paths}  -> switching works: {switched}")
        passed &= ok and switched and not same_prov

        # (2) A deliberately different implementation: RMSNorm with the weight multiply done in fp32 and
        # one final rounding, instead of rounding to bf16 first (a typical 1-ulp-class change).
        def rms_norm_perturbed(x, w, eps):
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            return (w.float() * xf).to(x.dtype)
        orig = md.rms_norm
        md.rms_norm = rms_norm_perturbed
        try:
            prefill_dump(eng, prompts, d_bad, impl="ref", label="ref+perturbed-rmsnorm")
        finally:
            md.rms_norm = orig
        say("== ref vs perturbed ref (expected: NOT IDENTICAL)")
        ok_bad, _ = prefill_compare(d_ref, d_bad, say)
        say(f"perturbed implementation detected: {not ok_bad}")
        passed &= not ok_bad

        # (3) Unknown names must fail loudly instead of silently running the default path.
        try:
            eng.m.prefill_impl = "fused_v2"
            rejected = False
        except ValueError as e:
            rejected = True
            say(f"unknown impl rejected: {e}")
        passed &= rejected
        eng.m.prefill_impl = "fused"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say("PREFILL SELFTEST PASSED" if passed else "PREFILL SELFTEST FAILED")
    sys.exit(0 if passed else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf")
    h.add_argument("--model", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--attn", default="sdpa")
    h.add_argument("--gen-len", type=int, default=256)
    e = sub.add_parser("engine")
    e.add_argument("--model", required=True)
    e.add_argument("--ref", required=True)
    e.add_argument("--out")
    e.add_argument("--path", choices=["graph", "ref"], default="graph",
                   help="graph: the CUDA-graph decode engine; ref: the PyTorch/Triton reference forward")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--other", required=True)
    d = sub.add_parser("prefill-dump")
    d.add_argument("--model", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--impl", choices=["fused", "ref"],
                   help="model.Qwen3FP8.prefill_impl to use (default: the engine default, fused)")
    d.add_argument("--label", help="free-text tag stored in the dump's provenance")
    d.add_argument("--lengths", default="128,512,1024,2048")
    d.add_argument("--no-prefill-copy", action="store_true", help="prefill reads the tiled decode weights")
    d.add_argument("--ref", help="HF reference JSON: also dump its long_context prompt (real text)")
    pc = sub.add_parser("prefill-compare")
    pc.add_argument("--a", required=True)
    pc.add_argument("--b", required=True)
    st = sub.add_parser("prefill-selftest")
    st.add_argument("--model", required=True)
    st.add_argument("--lengths", default="12,128,1024",
                    help="synthetic prompt lengths (below 16 the fused impl falls back to ref)")
    args = ap.parse_args()
    {"hf": cmd_hf, "engine": cmd_engine, "compare": cmd_compare, "prefill-dump": cmd_prefill_dump,
     "prefill-compare": cmd_prefill_compare, "prefill-selftest": cmd_prefill_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()

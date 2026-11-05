# Gemma 4 NVFP4: faster single-user decode on RTX 5090

## Result

Gemma generates text about **33% faster** with the delivered vLLM plugin: **201.4 → 267.4 tokens per second** against tuned vLLM 0.31.0 on one RTX 5090. Quality checks against the tuned baseline and a set of math questions passed. Additional checks of sampled-token distribution and repeatability were inconclusive for both engines; generated responses are not guaranteed to be identical.

| What matters | Result |
|---|---|
| Model and hardware | `nvidia/Gemma-4-26B-A4B-NVFP4`, one RTX 5090 (32 GB) |
| Sampled decode | Tuned vLLM 201.4 → plugin 267.4 tokens/s, measured from the client |
| Improvement | 1.328× [95% CI 1.327–1.329]; conservative bound 1.327× |
| Greedy decode | 207.0 → 271.5 tokens/s; 1.312× [1.311–1.312] |
| Quality | No added loss versus tuned NVFP4 within the judged margins; GSM8K subset PASS; sampler proof UNPROVEN |
| Scope and target | One request at a time, four prompt-length buckets, 256 output tokens; the development target of 1.5× was missed |

[Timing and quality measurements](evidence/measurements.json) · [Protocol and qualifiers](#performance-and-target)

## Run it

Use Linux x86_64 with an RTX 5090, driver 580 or newer, a C compiler, `curl`, at least 32 GB of system RAM and about 50 GB free disk. The installer supplies the matching CUDA compiler. The model needs no Hugging Face token; see [requirements](README.md#requirements).

```bash
git clone https://github.com/kittenml/inferencekit-engines
cd inferencekit-engines/gemma-4-26b-a4b-nvfp4_rtx-5090
./install.sh
./serve.sh
```

In a second terminal in the same directory:

```bash
./test.sh
```

The test sends a chat request and prints the response. The OpenAI-compatible endpoint is `http://127.0.0.1:8000/v1/chat/completions`, with model name `gemma-4-26b-a4b-nvfp4`. The first launch downloads weights and compiles kernels; allow 30–40 minutes plus download time.

## Files and outputs

| File | Purpose |
|---|---|
| [Installer](install.sh), [server launcher](serve.sh), [example request](test.sh) | Pinned setup and runnable entry points |
| [Plugin source](engine/g4fast/) | Sampler, expert kernels, output layer and packed-weight kernels |
| [Kernel check runner](tools/run.sh) | Lists the shipped GPU checks and runs a selected check |
| [Measurement excerpt](evidence/measurements.json) | Sanitized final timing, quality, configuration and source hashes |

Saved responses and an interactive review page are not included in this public package. `test.sh` produces a fresh response locally.

## Details

### What changed

The four fast paths target one-token decode and fall back to vLLM outside their supported cases. Their source and numerical qualifications are described in [the implementation overview](README.md#how-it-was-made).

- **Sampling:** [blockwise candidate selection](engine/g4fast/sampler.py) reduces full-vocabulary masking and seeded-noise work, with a completeness check and fallback.
- **Experts:** [batch-1 MoE kernels](engine/g4fast/moe.py) combine expert dispatch, NVFP4 matrix-vector operations and routing. They retain checkpoint quantization; reduction and rounding differ from the vendor kernels.
- **Output layer:** [bounded vocabulary pruning](engine/g4fast/head.py) uses an int8 proxy only to eliminate rows that cannot enter the top k, then recomputes retained logits with the full-head arithmetic.
- **Dense weights:** [BF16 packing](engine/g4fast/packed.py) reconstructs stored weights exactly while reducing reads. Load-time round-trip checks protect the representation; accumulation order can differ from the baseline.

Development profiles guided the reduction of expert-dispatch work and weight traffic. Those profiles are not distributed here, and individual kernel gains are not treated as the final request-level speedup.

### Performance and target

The final comparison used vLLM 0.31.0 on both sides, model revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, and the selected `compile_sizes: [1]` setting. Resolved settings include BF16 dense computation, `modelopt_fp4` quantization, FP8 E4M3 KV cache, an 8192-token context limit and disabled prefix caching. The plugin enables all four fast paths in [serve.sh](serve.sh).

There were four balanced interleaved rounds per sampling mode, with a fresh engine process per turn, 64 prompts per round and 256 output tokens per request. The sampled pass used temperature 1.0, top-k 64, top-p 0.95 and seed 1234; thinking was disabled. Compilation caches were warmed before timing.

| Claim | Workload / bucket | Baseline | Candidate | Ratio [CI95] | Quality | Mechanism | Gates | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Sampled decode timing | Single user; four length buckets | 201.4 tokens/s | 267.4 tokens/s | 1.328 [1.327, 1.329] | Baseline-relative PASS; sampled proof UNPROVEN | Four fast paths | PASS | Timing PASS; combined sampled proof UNPROVEN | `ev_5700ccecd632`; [excerpt](evidence/measurements.json) |
| Greedy decode timing | Same prompts and output length | 207.0 tokens/s | 271.5 tokens/s | 1.312 [1.311, 1.312] | Baseline-relative PASS | Same configuration | PASS | PASS for measured timing | `ev_d77f1a84a5b0`; [excerpt](evidence/measurements.json) |
| Development target ≥1.5× | Sampled decode | Tuned vLLM | Plugin | Upper bound 1.329 < 1.5 | As above | Same configuration | PASS | FAIL | Same sampled record |

The per-bucket medians are **observed**. Headline tokens/s are **derived** as their geometric mean; the speedup is the geometric mean of candidate/baseline bucket ratios. The recorded CI resamples paired rounds 10,000 times with seed 0. It captures this session's variation, not variation across machines.

The baseline was selected after tuning vLLM and SGLang. The [baseline-selection table](README.md#tested-baselines) records those screening results separately; they are not the denominator of the final comparison.

### Quality and correctness

The [quality excerpt](evidence/measurements.json) retains the matching final-run score and its scope:

- **Baseline-relative quality: PASS**, on 64 prompts scored against the BF16 reference, using a repeat of tuned NVFP4 as the floor. Candidate/reference top-1 agreement was 94.09% versus the floor's 94.29%; KL was 0.06615 versus 0.06775 nats/token. The judged margins were 0.02 agreement, 0.02 KL and 0.05 reference NLL (`ev_5291360b6df1`).
- **GSM8K subset: PASS** under the recorded nonregression rule: 341/350 correct versus 339/350, greedy at concurrency 1. Both sides had two answers truncated at 1024 tokens. This final rerun checked the first 350 of 1319 test problems, not the full test split.
- **Request gates: PASS** for caching, token count, output length, hidden probes and rare tokens. `unique_prompts` and `spec_exact` were NOT RUN; this engine does not use speculative decoding.
- **Sampled distribution: UNPROVEN.** The first-token check found no informative prompts because returned log-probability supports were unstable. Repeated-seed output checks were also UNPROVEN for both engines. The combined scorer verdict remains UNPROVEN despite passing timing and baseline-relative quality checks.

Both NVFP4 engines fail the separate informational comparison against the BF16 model's own repeat floor. The passing claim concerns additional loss relative to tuned NVFP4, rather than equivalence to BF16 or bitwise identity.

### What did not work and where this applies

Development rejected slower MoE implementations, lower-precision output-layer proxies whose bounds retained too many rows, and additional fusions that did not improve complete requests. Packing the down projection also lost the benefit of an existing fused operation. These discarded experiments are not included in the reported timing.

The gain applies to single-stream decode on RTX 5090 with vLLM 0.31.0. Prefill and batches above one token use the baseline paths. Concurrency scaling, other GPUs and multimodal use are NOT RUN for this result. Requests with log-probabilities, penalties, logit bias, min-p or structured output use the full output layer; top-k pruning supports k up to 256.

### Reproduce and inspect

With the server running, the README's client command measures the delivered engine:

```bash
uvx --from 'inferencekit-cli==0.1.0' inferencekit bench serve --base-url http://127.0.0.1:8000 \
  --model gemma-4-26b-a4b-nvfp4 --max-tokens 256 --rounds 4 \
  --extra-body '{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1234}'
```

Run the same workload against tuned vLLM with matching launch arguments and the plugin disabled. A single-server run is a local measurement; reproducing the published comparison also requires balanced fresh-process rounds and the same prompt set. The final measurement harness was version 0.2.1; the sealed inputs and complete run directory are not shipped. Stop the server before running the GPU checks listed by `./tools/run.sh`.

### Time and cost

The [published development accounting](README.md#how-it-was-made) records Claude Opus 5.5 at high effort in Claude Code with inferencekit: 19 hours of agent time, 630M tokens including cache reads, and $199 at API list price. These are development totals, not per-request inference costs. Complete GPU rental time and cost are not retained in this public package.

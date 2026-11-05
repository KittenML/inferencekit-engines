# Qwen3-8B on RTX 5090: 1.22× faster single-user decode than vLLM with FlashInfer

## Result

The optimized engine generates text at **186.3 tokens per second**, compared with **153.0** for vLLM with FlashInfer attention on the same RTX 5090. It had the fastest decode in the tested vLLM, SGLang and llama.cpp panel. The engine and the Hugging Face FP8 reference both answered **93 of the first 100 GSM8K questions correctly**; their generated text is not identical.

| What matters | Result |
|---|---|
| Model and hardware | Qwen3-8B, FP8; one RTX 5090 with 32 GB |
| Generation speed | **153.0 → 186.3 tokens/s**, vLLM with FlashInfer → this engine |
| Improvement | **1.22×** single-user decode speed |
| Quality | 93/100 GSM8K, matching the FP8 reference's score on this subset |
| Scope | Greedy generation, one request at a time, 128-token prompt and 256 generated tokens; in-process timers |
| Main limit | The gain is in generation after the first token; time to first token is slower than vLLM at the tested 1024-token prompt |

A token is a unit of text, often part of a word. These measurements cover the engine's generation loop, not HTTP request latency. [Timing and quality measurements](evidence/measurements.json).

## Run it

Check the [hardware and software requirements](README.md#requirements). From the repository root:

```bash
cd qwen3-8b-fp8_rtx-5090
./install.sh
./serve.sh
```

In another terminal, from the same engine folder:

```bash
./test.sh
```

The test waits for the server and prints a generated reply. Its elapsed time includes the prompt and HTTP request, so it is not the decode-only benchmark above. The server exposes an OpenAI-compatible API on `http://127.0.0.1:8000`; requests must set `temperature` to `0`.

## Files and outputs

| What you need | Where to find it |
|---|---|
| Install, serve and try a request | [Installer](install.sh), [launcher](serve.sh), [example request](test.sh) |
| Engine implementation | [Model loader](model.py), [generation loop](generate.py), [CUDA and Triton kernels](kernels/) |
| Timing and quality results | [Sanitized measurement extract](evidence/measurements.json) |
| Benchmark and checks | [Decode benchmark](bench.py), [kernel checks](test_kernels.py), [reference checks](check_correctness.py), [GSM8K evaluator](eval_gsm8k.py) |
| Technical design and experiments | [Design notes](DESIGN.md) |

The package prints responses through the example request; it does not bundle a saved response gallery.

## Details

### What changed

Claude Opus 5.5 used inferencekit to build a custom engine around the memory traffic of one-request-at-a-time decoding:

1. **Read FP8 weights efficiently.** The loader arranges weights in the order consumed by custom matrix-vector kernels, avoiding conversions and intermediate staging during generation.
2. **Queue a whole generation step together.** A CUDA graph covers the decode step and token bookkeeping, reducing repeated host work between steps.
3. **Overlap independent weight reads.** Programmatic dependent launch starts the next kernel's weight prefetch while the preceding kernel finishes; dependent activations are read only after the wait.
4. **Fuse attention work.** Query/key normalization, rotary position encoding, cache writes and attention reduction share a kernel instead of separate launches and intermediate transfers.
5. **Combine normalization and activation work with projections.** RMS normalization and FP8 quantization run at kernel entry; residual additions and SwiGLU run with the corresponding outputs.
6. **Use a separate prefill path.** Fused Triton operations handle the initial prompt, while the decode kernels specialize in generating one new token at a time.

The [implementation details](DESIGN.md#decode-path) explain the kernel choices. These changes form one measured engine; their individual gains should not be added together.

### Performance and baseline

| Configuration | Decode, tokens/s ↑ | This engine / baseline |
|---|---:|---:|
| This engine, FP8 | **186.3** | — |
| vLLM 0.31.0, FP8, FlashInfer attention | 153.0 | **1.22×** |
| vLLM 0.31.0, FP8, default settings | 149.6 | 1.25× |
| SGLang 0.5.21, FP8 | 142.0 | 1.31× |
| llama.cpp, Q8_0 | 168.3 | 1.11× |

Observed throughput comes from ten interleaved rounds on the same RTX 5090, with a fresh engine process for each run. This engine, vLLM and SGLang used in-process Python timers; llama.cpp used `llama-bench`. The 1024-token panel has the same decode ranking. SGLang used static memory fraction 0.8 with radix caching disabled and was not tuned. llama.cpp changes weight format and is a separate speed/quality comparison.

The headline rounds the recorded ratio of median throughputs, **1.218×**, against FlashInfer. The observed round ratios range from **1.202× to 1.236×**; that range is not a confidence interval. The [measurement extract](evidence/measurements.json) preserves per-engine throughput confidence intervals and round medians; the retained summary does not supply a paired speedup interval.

| Claim | Workload | Baseline | Candidate | Ratio [CI95] | Quality | Mechanism | Gates | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Decode speed | 128-token prompt; 256 generated tokens; one request | FlashInfer vLLM: 153.0 tokens/s | 186.3 tokens/s | 1.218; paired CI unavailable | FP8-reference subset below; FlashInfer not separately scored | Custom decode kernels and CUDA graph | Not supplied by the retained panel summary | Observed timing; full paired-CI proof UNPROVEN | [Timing and quality extract, with source hashes](evidence/measurements.json) |

At a repeated 1024-token prompt length, this engine's measured time to first token was **43.0 ms**, versus **39.6 ms** for vLLM with FlashInfer and **36.7 ms** for the other vLLM configuration. Prefill graph replay depends on the process having seen that prompt length before.

### Quality and correctness

| Configuration | Correct answers, first 100 GSM8K questions |
|---|---:|
| This engine, FP8 | 93/100 |
| Hugging Face FP8 reference | 93/100 |
| vLLM, original attention configuration | 92/100 |
| SGLang, FP8 | 92/100 |
| llama.cpp, Q8_0 | 96/100 |

This was greedy decoding with thinking disabled and a 512-token limit. The FlashInfer configuration was measured for speed but was not separately scored in this quality study. llama.cpp scored higher on the subset; the small sample does not establish a general accuracy ranking or equivalence on other tasks.

On the separate 26-prompt check, the decode engine's next-token agreement with the reference was **98.86%**, versus **98.99%** when the reference's own attention implementation was changed. These are observed, tie-inclusive agreement rates. They do not mean generated responses are token-identical. The [quality extract](evidence/measurements.json) contains counts, accuracy intervals and the two agreement rates; [design notes](DESIGN.md#correctness) describe the kernel checks.

### What did not work and where this applies

Deeper weight-prefetch pipelines and a different distribution of work across multiprocessors did not improve decode in the recorded experiments. Larger prefill tiles, extra pipeline stages and split-K variants were not retained. The [experiment history](DESIGN.md#decode-experiments-1024-token-prompt) records the individual trials.

This engine supports the pinned checkpoint and greedy decoding at batch one. It does not establish results for other models, hardware or concurrent serving. It uses no speculative decoding. Attention and output-layer rounding can change later tokens, and the long-prompt first-token result remains behind vLLM.

### Reproduce and inspect

Stop the server before running the included decode benchmark from the engine folder:

```bash
. .venv/engine.env
.venv/bin/python bench.py --model "$MODEL_DIR" --gen-len 256 --summary
.venv/bin/python bench.py --model "$MODEL_DIR" --prompt-len 1024 --gen-len 256 --summary
```

These commands measure this engine; they do not rerun the entire competitor panel. The [installer](install.sh) pins the checkpoint and software. The [measurement extract](evidence/measurements.json) retains the recorded engine settings, summary-source hashes and round-level throughput without private paths, prompts or logs.

### Time and cost

The published estimate is **about 10 hours, 240 million agent tokens and $125 at API list price** for Claude Opus 5.5 at xhigh effort. This is the same estimate as the [engine README](README.md#how-it-was-made), not a new accounting measurement or a hardware-rental total.

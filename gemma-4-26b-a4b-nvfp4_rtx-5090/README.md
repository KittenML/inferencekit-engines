# Gemma 4 26B-A4B NVFP4 on one RTX 5090

A vLLM plugin with exact fast paths for `nvidia/Gemma-4-26B-A4B-NVFP4`: 1.33× faster single-user decode than a tuned vLLM 0.31.0 setup on the same GPU, at the same accuracy.

## Requirements

- One NVIDIA RTX 5090 (32 GB). The kernels target Blackwell consumer GPUs (sm_120) and batch size 1.
- Linux x86_64, NVIDIA driver 580 or newer. No CUDA toolkit needed: `install.sh` puts a CUDA compiler that matches vLLM's PyTorch into `.venv/cuda`, and `serve.sh` uses it for the kernels FlashInfer compiles on first use.
- A C compiler (`build-essential` on Ubuntu) and `curl`. No root needed: `install.sh` installs [uv](https://docs.astral.sh/uv/) into `~/.local/bin` if it is missing.
- About 50 GB free disk: about 10 GB for `.venv`, about 17 GB for the model in the Hugging Face cache, plus compile caches.
- At least 32 GB of RAM. The first start compiles CUDA kernels, about 8 GB of RAM per job; `serve.sh` runs one job per 10 GB available (set `MAX_JOBS` to override).
- Hugging Face: no token needed. Model `nvidia/Gemma-4-26B-A4B-NVFP4`, pinned revision `a19cfe0`; the chat template comes from `google/gemma-4-26B-A4B-it` and is checked by sha256.

## Run

```bash
git clone https://github.com/kittenml/inferencekit-engines && cd inferencekit-engines/gemma-4-26b-a4b-nvfp4_rtx-5090
./install.sh   # builds ./.venv here: vLLM 0.31.0 + the g4fast plugin + the chat template
./serve.sh     # OpenAI-compatible server on 127.0.0.1:8000 (PORT=... to change)
./test.sh      # in a second terminal: sends one chat request and prints the reply
```

The first `./serve.sh` downloads the model and compiles CUDA kernels; allow 30-40 minutes, plus the download. Later starts take about 5 minutes. `./test.sh` waits for the server.
The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`) with the model name `gemma-4-26b-a4b-nvfp4`.

## Results

[Run report and measurement details](REPORT.md).

| Single-user decode, one request at a time | tuned vLLM 0.31.0 | This engine | Speedup |
|---|---|---|---|
| Sampled (temperature 1.0, top-k 64, top-p 0.95) | 201.4 tokens/s | 267.4 tokens/s | **1.33×** |
| Greedy | 207.0 tokens/s | 271.5 tokens/s | 1.31× |

- Both servers ran on the same RTX 5090, one at a time, in 4 interleaved rounds, with the same prompts and 256 output tokens per request. Tokens/s is the geometric mean over four prompt lengths, from very short to long.
- Tuned vLLM is stock vLLM 0.31.0 with the same `vllm serve` arguments as `serve.sh`, without the `G4F_*` fast paths.
- The baseline was selected after tuning both vLLM 0.31.0 and SGLang 0.5.21; the selected batch-1 compilation setting was carried into the final comparison.
- Your speedup depends on the machine: on a host with an older CPU we measured 1.24×.
- Quality: matched the tuned baseline within the quality limits and passed the 350-question GSM8K check; sampling and repeatability checks were inconclusive ([details](REPORT.md#quality-and-correctness)).

**Reproduce:** with `./serve.sh` running, time it from a second terminal with inferencekit's client, then do the same with tuned vLLM on the same port:

```bash
uvx --from 'inferencekit-cli==0.1.1' inferencekit bench serve --base-url http://127.0.0.1:8000 \
  --model gemma-4-26b-a4b-nvfp4 --max-tokens 256 --rounds 4 \
  --extra-body '{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1234}'
```

### Tested baselines

| Engine / configuration | Decode, tokens/s ↑ | Result |
|---|---:|---|
| **vLLM, `compile_sizes: [1]`** | **195.3** | **Selected and validated** |
| vLLM, selected config + graph partition | 196.4 | Gain below the 2% selection threshold |
| vLLM recipe | 180.1 | Slower |
| SGLang, torch.compile | 157.6 | Slower |
| SGLang, BF16 KV cache | 157.2 | Slower |
| SGLang recipe | 152.6 | Slower |
| TensorRT-LLM | Not measured | Server did not become ready |
| LMDeploy | Not measured | Model unsupported |
| llama.cpp | Not measured | GGUF conversion timed out |

Baseline tuning: one client-timed round of 16 medium prompts. The 1.33× result above uses the selected vLLM configuration in the final comparison, not these tuning measurements.

## How it was made

Built by Claude Opus 5.5 (effort high) in Claude Code with the inferencekit plugin: 19 h of agent time, 630M tokens (mostly cache reads), $199 at API list price. The agent added four fast paths to vLLM for one-token decode, targeting expert dispatch, vocabulary processing and weight traffic:

- **Exact top-k/top-p sampling.** The [sampler](engine/g4fast/sampler.py) selects candidates from blocks of the 262k vocabulary, then applies the same tie-breaking and seeded Gumbel noise as vLLM. It checks that the candidate set contains every token that could survive top-k and falls back to the full vocabulary if that cannot be established. This avoids full-vocabulary masking and noise generation on the fast path without changing the sampled token for the same request and seed.

- **MoE kernels specialized for batch 1.** The [expert path](engine/g4fast/moe.py) replaces expert-map setup, row expansion and underfilled grouped matrix multiplies with three Triton kernels: selected-expert matrix-vector multiplication, activation plus NVFP4 quantization, and the output projection combined with routing weights. It reads the existing packed expert weights and keeps the checkpoint's activation quantization scheme. Reduction and quantization rounding differ from the vendor implementation, so this part is not bit-identical to it.

- **An output layer that skips provably irrelevant rows.** The [head](engine/g4fast/head.py) uses a smaller int8 proxy and conservative error bounds to identify vocabulary rows that cannot reach the top k. Surviving rows are recomputed with the full-head kernel's arithmetic; an oversized candidate set triggers the full head. The proxy reduces weight reads without substituting approximate scores for the retained logits.

- **Lossless BF16 weight packing.** The [packed matrix-vector kernels](engine/g4fast/packed.py) store attention and gate/up weights using sign/mantissa bytes and exponent deltas, at about 12 bits per weight plus rare exceptions. They reconstruct the original BF16 values while consuming them, reducing memory traffic without changing the stored weights. Packing is checked by a bit-for-bit round trip at load time; accumulation order can still differ from the baseline matrix multiply.

All four are switched on in `serve.sh` (`G4F_SAMPLER`, `G4F_MOE`, `G4F_HEAD`, `G4F_PACK`). Each fast path covers the case it was built for and falls back to vLLM's own code otherwise. `tools/` holds the exactness checks: `tools/run.sh` lists them and runs one (they need the GPU, so stop the server first).

## Limits

- Tuned for one user at a time. Batches above one token (prefill, concurrent requests) use vLLM's own paths, so the speedup applies to single-stream decode only.
- The pruned output layer applies to top-k sampling (k up to 256) and greedy decoding. Requests with logprobs, penalties, logit bias, min-p or structured output use the full output layer.
- Context is limited to 8192 tokens, text only (`--language-model-only`), thinking off by default.
- `serve.sh` sets `VLLM_DISABLE_COMPILE_CACHE=1`, so every start compiles again (several minutes).
- Only tested on RTX 5090 with vLLM 0.31.0.

## License

Apache-2.0, as the rest of this repository. The model and chat template have their own licenses on Hugging Face.

# Qwen3-8B on RTX 5090: 1.22× faster single-user decode than vLLM with FlashInfer

A single-user decode engine for `Qwen/Qwen3-8B-FP8` with its own CUDA kernels: fastest decode in our tested vLLM, SGLang and llama.cpp panel, at 186 tokens/s—**1.22× versus vLLM 0.31.0 with FlashInfer attention** on the same GPU, with the same GSM8K score as the Hugging Face reference.

[Full report: results, optimizations and evidence](REPORT.md).

## Requirements
- One NVIDIA RTX 5090 (32 GB; the server uses about 17 GB). Linux x86_64, tested on Ubuntu 22.04.
- NVIDIA driver 580 or newer (CUDA 13). A CUDA 13 toolkit is used if one is installed (`CUDA_HOME` or `/usr/local/cuda-13*`); if not, `install.sh` puts NVIDIA's CUDA 13.0 compiler from PyPI into `.venv`.
- A C++ compiler (`g++`), `curl`, and about 25 GB of free disk (weights about 10 GB, Python env about 6 GB, download cache).
- No root: `install.sh` installs `uv` into `~/.local/bin` if it is missing.
- No Hugging Face login: the weights are public. They are pinned to revision `220b46e3`.

## Run
```bash
git clone https://github.com/kittenml/inferencekit-engines && cd inferencekit-engines/qwen3-8b-fp8_rtx-5090
./install.sh   # Python env, CUDA 13 compiler if needed, about 10 GB of weights, then a kernel build of a few minutes
./serve.sh     # OpenAI-compatible API on http://127.0.0.1:8000 (PORT and HOST to change)
./test.sh      # in a second terminal: waits until the server is ready, then prints one reply
```
Most of the install time is downloading PyTorch and the weights. `serve.sh` passes extra arguments to `serve.py` (`--help` lists them). The API has `/v1/chat/completions` and `/v1/completions`, streaming or not.

To run the GPU-free server tests: `uv pip install --python .venv/bin/python pytest`, then `.venv/bin/python -m pytest tests`. If `uv` is not found, it is in `~/.local/bin`.

## Results

Fastest decode in the tested panel, including vLLM, SGLang and llama.cpp:

| Single-user decode, greedy, 256 tokens | tokens/s | This engine / baseline |
|---|---|---|
| This engine (FP8) | **186.3** | — |
| vLLM 0.31.0, FP8, default settings | 149.6 | **1.25×** |
| vLLM 0.31.0, FP8, FlashInfer attention | 153.0 | **1.22×** |
| SGLang 0.5.21, FP8 | 142.0 | **1.31×** |
| llama.cpp, Q8_0† | 168.3 | **1.11×** |

† The llama.cpp comparison changes weight format (FP8 versus Q8_0); quality scores are reported below.

- All engines ran on the same RTX 5090, one request at a time, in 10 interleaved rounds. The table uses a 128-token prompt; the 1024-token panel has the same decode ranking. Timers are in-process: this engine uses `bench.py`, vLLM and SGLang use their offline APIs, and llama.cpp uses `llama-bench`.
- SGLang used default inference settings with static memory fraction 0.8 and radix caching disabled for the benchmark; it was not tuned.
- Quality: on the first 100 GSM8K questions, this engine and the Hugging Face FP8 reference scored 93/100; default vLLM and SGLang scored 92/100; llama.cpp Q8_0 scored 96/100.

**Reproduce:** stop the server, load the installed environment with `. .venv/engine.env`, then run `.venv/bin/python bench.py --model "$MODEL_DIR" --gen-len 256 --summary` (a 128-token prompt by default; `--prompt-len 1024` for the longer one). It prints the median decode tokens/s.

## How it was made

Claude Opus 5.5, effort xhigh, used inferencekit in an interactive Claude Code session: about 10 h, about 240M tokens, about $125 at API list price. The agent developed a custom CUDA decode path around the memory traffic of batch-1 inference, then used profiles and in-kernel timing to refine fusion and launch scheduling:

- **FP8 matrix-vector kernels and a matching weight layout.** The [loader](model.py) joins Q/K/V and gate/up matrices and rearranges FP8 weights into the tiled order consumed by the tensor-core instructions. Each 16-byte load supplies two instructions directly, avoiding weight conversions, shuffles and shared-memory staging. Dynamic activation scaling remains per 128 values, matching the checkpoint's FP8 recipe.

- **A complete decode step captured in one CUDA graph.** [Graph execution](generate.py) covers all 183 kernels per token, including argmax and token/position bookkeeping. Those inputs remain in device memory, so the host can queue subsequent steps without reading a token back before each replay.

- **Overlapped weight prefetch between kernels.** Programmatic dependent launch lets the next kernel load immutable weights while its predecessor finishes. The [CUDA kernels](kernels/decode.cu) wait before reading dependent activations; their prefetch depth and launch-trigger positions are tuned to keep memory requests flowing across kernel boundaries.

- **One fused attention kernel per layer.** Query/key normalization, RoPE, the KV-cache write, split-KV attention and the merge of partial results run together. This removes separate launches and intermediate transfers between those operations. Its attention rounding differs from the Hugging Face path, so the full engine does not promise identical generated tokens.

- **Normalization and activations folded into matrix-vector kernels.** RMS normalization and FP8 input quantization run in the projection prologue; residual additions and SwiGLU are incorporated into the relevant output paths. The FP8 projection kernels are checked against a Triton copy of the reference arithmetic; that agreement does not extend to every operation in the engine.

- **A separate fused prefill path.** [Four Triton kernels](kernels/prefill_fused.py) per layer combine normalization, quantization, rotary embeddings, KV writes and activation work. Smaller matrix-multiply tiles improve coverage of the tested prompt shapes. The fused path preserves the reference rounding sequence and retains the unfused path where its shape assumptions do not apply; the recorded prefill checks matched logits and KV-cache bytes.

[Design and experiment history](DESIGN.md) covers the kernel choices, rejected changes and remaining bottlenecks. The decode result above does not imply faster time to first token: this engine remains slower than vLLM at the tested 1024-token prompt.

## Limits
- Batch 1: one request runs at a time; others wait in a queue. Not built for many users.
- No speculative decoding.
- Greedy only: `temperature` must be sent as 0. Any parameter that would change the output (sampling, stop strings, penalties, tools) gets HTTP 400 naming it.
- This checkpoint only: other models, or other shapes, are refused at load. Tuned for the RTX 5090; other GPUs are untested.
- Context up to 4096 tokens by default (`--max-seq`). Fast first-token times need a prompt length seen twice before: prefill CUDA graphs are kept for up to 4 exact lengths.
- Time to first token at 1024-token prompts is slower than vLLM's.
- Outputs are not token-for-token identical to Hugging Face: attention rounds slightly differently, so long generations diverge at near-ties.

## License
Apache-2.0, like the rest of this repository. The model weights are not included and have their own license; see the model page on Hugging Face.

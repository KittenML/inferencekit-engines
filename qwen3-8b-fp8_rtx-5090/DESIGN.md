# Design notes

## Result

This single-user decode engine for `Qwen/Qwen3-8B-FP8` reached **186.3 tokens/s** on RTX 5090 in the final
10-round measurement at a 128-token prompt, using in-process timers. On the first 100 GSM8K questions,
the engine and Hugging Face reference both scored **93/100**, but generated tokens are not identical.
Time to first token still trails vLLM at the tested
1024-token prompt. See the [measured comparison and its scope](README.md#results).

## Run it

Use the [installation and serving commands](README.md#run), or the [decode benchmark command](README.md#results)
to reproduce the reported measurement.

For GPU-free server tests: `uv pip install --python .venv/bin/python pytest`, then `.venv/bin/python -m pytest tests`.

## Files and outputs

| File | Purpose |
|---|---|
| [Model loader](model.py) and [generation loop](generate.py) | Weight layout and CUDA graph execution |
| [Kernels](kernels/) | CUDA decode and Triton prefill implementations |
| [Benchmark](bench.py) | Prints decode timing and tokens/s |
| [Kernel checks](test_kernels.py), [reference checks](check_correctness.py) and [server tests](tests/) | Numerical and serving checks described below |

## Details

The six files that decide speed and numerics (`model.py`, `generate.py`, `kernels/*`) are the code that was
benchmarked, with comment-only edits since.

### Memory-bandwidth ceiling

At batch 1 every weight byte is read once per token, so decode speed is set by memory bandwidth.

- Bytes per token: 6.95 GB of FP8 linear weights, 1.25 GB of bf16 output layer (`lm_head`), plus 147,456 B of
  KV cache per context token. About 8.23 GB per token at a 256-token context.
- Measured read bandwidth 1,680 GB/s (datasheet 1,792), so the ceiling is about 204 tok/s at prompt 128 and
  201 tok/s at prompt 1024. The engine reaches 91-92% of it.

### Checkpoint format

FP8 (e4m3) linear weights with bf16 block scales per 128×128 block; activations are quantized to FP8 at run
time per 128-wide group (W8A8, "dynamic"). Matching the reference therefore means doing exactly that, not
dequantizing weights to bf16. The reference is transformers 5.19 `FP8Linear`, which runs the
`kernels-community/finegrained-fp8` Triton kernel on this GPU.

### Decode path

- `model.py` loads the weights, joins q|k|v and gate|up, upcasts the block scales to fp32 (exact) and re-lays
  out each FP8 matrix into the tiled, K-major order that the decode kernel streams. A row-major copy is kept
  for prefill (`--no-prefill-copy` drops it: 6.4 GiB less VRAM, slower prefill, same bits).
- `kernels/decode.cu`, per layer: qkv GEMV (RMSNorm and FP8 quantization in its prologue) → attention (q/k
  RMSNorm, RoPE, KV write, split-KV, in-kernel merge) → o_proj GEMV (+ residual) → gate|up GEMV (SwiGLU in
  its epilogue) → down GEMV (+ residual). Then the `lm_head` GEMV with argmax.
- The FP8 GEMV uses the tensor-core instruction `mma.sync.m16n8k32...e4m3` as a dot-product engine (the same
  instruction and scale order as the reference kernel). One 16-byte load per lane is exactly that lane's
  operands for two instructions: no conversions, shuffles or shared memory for weights.
- `generate.py` captures one CUDA graph per decode step (183 kernels). Token and position live in GPU
  memory, so the host can queue several steps ahead.
- `serve.py` replays the graph up to 4 steps ahead of the host and copies each token id to pinned memory on a
  side stream; the HTTP layer is the Python standard library.

### Decode experiments (1024-token prompt)

| Step | Result |
|---|---|
| v1: tensor-core FP8 GEMV, whole step in one CUDA graph | 156.5 tok/s; nsys: only 1.4% idle between kernels, so not launch-bound |
| Programmatic dependent launch (prefetch weights before waiting on the previous kernel) | +1.4% |
| Refuted: "not enough bytes in flight per warp": pipeline depth 4 → 8 → 16 | no change, gate/up slower |
| Refuted: per-SM load imbalance | no gain |
| Shorter per-kernel prologue (found with in-kernel timestamps) + fused attention (13.9 → 8.8 µs per layer) | 158.8 → 172.4 tok/s |
| Launch timing: when the next kernel starts and how much it prefetches | +1.8%, then +0.8% |
| K-major weight tiles; per-kernel trigger tuning from an in-graph timeline | +0.6%, then +1.3% |
| Refuted: staging the input before the post-wait prefetch | −0.6% (DRAM idles) |
| Final | 185.2 tok/s at a 1024-token prompt, 92% of the ceiling (the README's 186.3 is the final 10-round measurement at a 128-token prompt) |

### Prefill experiments

For time to first token, nsys showed about 2,068 kernels for a 1024-token prefill, most of them small
elementwise ops, and ncu showed the FP8 GEMMs under-filled (DRAM 7-9%). Four fused Triton kernels per layer
(RMSNorm+quant, q/k norm+RoPE+KV write, SwiGLU+quant, quant) cut that to 377 kernels, and 128×64 / 64×64
tiles fill the SMs better: 1.42x faster, with logits and the whole KV cache byte-identical to before.
Refuted there: larger tiles, more pipeline stages, 32×64 tiles (25-40% slower), split-K.

### Correctness

- `test_kernels.py`: the four FP8 GEMVs are bit-for-bit equal to a Triton copy of the reference recipe
  (`kernels/prefill.py`) in 7 launch variants; the `lm_head` differs by about 1 ulp on 0.014-0.03% of logits.
- `check_correctness.py`: the reference path (Triton prefill, PyTorch SDPA, cuBLAS `lm_head`) reproduces
  Hugging Face's FP8 greedy output exactly. The fast decode graph agrees with it on 97.98% of teacher-forced
  steps on 6 development prompts, against 97.92% for Hugging Face's own eager-attention path. The likely
  cause of the difference is attention rounding (this engine keeps softmax probabilities in fp32).
- On a 26-prompt study: 98.86% vs 98.99% for the Hugging Face A/A control; GSM8K-100 93 vs 93.
- `tests/`: GPU-free server tests. `tests/gpu_check_serve.py` checks on a GPU that the
  streamed token ids equal `Engine.generate`'s.

### Remaining work

1. Exact token-for-token match with Hugging Face: port FlashAttention-2's split-KV rounding into attention.
2. The `lm_head` lost about 5% of its in-step bandwidth after the launch-timing change; not explained.
3. About 1-3 µs per kernel boundary: a persistent kernel per layer would remove it.
4. Time to first token at long prompts (behind vLLM at 1024 tokens): pad prompts to length buckets so
   prefill graphs are reused; a persistent GEMM for the tail wave.
5. An FP8 KV cache would cut long-context bytes but changes numerics.

### Environment

RTX 5090: sm_120, 170 SMs, 96 MB L2, 32 GB GDDR7. PyTorch 2.14.1 (CUDA 13.0), transformers 5.19.0,
Triton 3.8.0. The CUDA extension must be built with a
CUDA 13 nvcc (PyTorch refuses a different major version) and C++20. This folder has a package named
`kernels`, like the Hub `kernels` package that the transformers reference needs, so `check_correctness.py hf`
removes this folder from `sys.path` first.

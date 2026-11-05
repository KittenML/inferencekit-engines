# LTX-2.5 text-to-video on one RTX PRO 6000

A clip server for Lightricks' LTX-2.5 text-to-video model: 1.25× faster than ComfyUI per clip on the same GPU, with the same model, the same BF16 weights and the same 30 steps.

## Requirements

- One NVIDIA RTX PRO 6000 Blackwell (96 GB); the whole model stays on the GPU (about 76 GB in use).
- Linux x86-64, NVIDIA driver 580 or newer. No CUDA toolkit needed: PyTorch brings its own CUDA 13.2 libraries.
- `gcc` (Triton uses it at start-up), `curl`, and about 100 GB of free disk: 66 GB of weights in the Hugging Face
  cache, about 10 GB of Python packages, plus kernel caches. No root needed: `install.sh` installs `uv` into `~/.local/bin` if it is missing.
- A Hugging Face account with access to the model, which is gated (see Run).

## Run

First open [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) while logged in to Hugging Face and accept the licence. Then:

```bash
git clone https://github.com/kittenml/inferencekit-engines && cd inferencekit-engines/ltx-2.5_rtx-pro-6000
export HF_TOKEN=<your Hugging Face token>   # a token that can read the model
./install.sh   # Python packages into ./.venv, then 66 GB of model weights
./serve.sh     # clip server on 127.0.0.1:8000 (PORT and HOST to change); ready when it prints "engine ready"
./test.sh      # in a second terminal: one 5-second clip with sound, saved to ./out.mp4
```

Without `HF_TOKEN`, `install.sh` installs the packages, then stops and asks you to log in: run `.venv/bin/hf auth login`, then `./install.sh` again. It is safe to run again: finished steps are skipped and an interrupted download resumes.
API: `GET /health`, `POST /v1/clips` (request as in `test.sh`; the MP4 comes back as base64 in `video_b64`).

## Results

[Run report and measurement details](REPORT.md).

| Engine, one clip at a time | Seconds per clip | Speedup |
|---|---|---|
| ComfyUI (BF16) | 116.2 | 1.00× |
| This engine | 93.2 | **1.25×** |

- ComfyUI had the lowest clip time in the baseline comparison against the Diffusers wrapper and the official LTX pipeline. The 1.25× above is measured against ComfyUI in the final comparison.
- A clip: 768 x 512, 121 frames at 24 fps (5 s) with audio, 30 steps, the Diffusers LTX-2 default guidance.
- Both engines ran on the same RTX PRO 6000, one at a time, in 6 interleaved rounds, timing whole requests from the client. Quality checks
  passed: every denoising step against a BF16 Diffusers reference, and the final clips by PSNR.

**Reproduce:** with `./serve.sh` running, write the request from `test.sh` as one line of `clips.jsonl` (add an `"id"`, drop the `"seed"`), then time it from a second terminal:

```bash
uvx --from 'inferencekit-cli==0.1.1' inferencekit bench clip --base-url http://127.0.0.1:8000 \
  --clips clips.jsonl --rounds 6
```

For ComfyUI, run the same command against its server with `--endpoint-kind comfyui --workflow <your LTX-2.5 workflow JSON>`.

### Tested baselines

| Implementation | Text-to-video, s/clip ↓ | Image-to-video, s/clip ↓ |
|---|---:|---:|
| **ComfyUI — selected** | **92.67** | **98.73** |
| Diffusers wrapper | 112.14 | 119.11 |
| Official LTX pipeline | 132.12 | 134.18 |

Baseline selection: three rounds of four timed clips per engine at 768 × 512, 121 frames and 30 steps with audio. ComfyUI's BF16 text-encoder setting was rechecked at 92.81 s/clip; the 1.25× headline uses the final six-round comparison above.

## How it was made

Claude Opus 5.5 (effort high) in Claude Code with the inferencekit plugin: 3.6 h of agent time, 47M tokens, $18 at API list price.

The agent optimized the repeated transformer work inside each denoising step, keeping the BF16 checkpoint and all 30 steps. The shipped configuration combines six changes:

- **Compiled transformer blocks.** `torch.compile` specializes the block function for the fixed shapes and guidance variants, with Inductor's coordinate-descent tuning enabled. Precision-cast emulation retains the eager block's BF16 rounding points in compiled elementwise chains.

- **cuDNN attention.** The attention wrapper selects cuDNN's scaled dot-product attention backend for the tested workload instead of relying on PyTorch's automatic backend choice. This changes the implementation of attention, while retaining its inputs, head layout and guidance structure.

- **GELU fused into the feed-forward matrix multiply.** The first feed-forward projection applies GELU in the cuBLASLt epilogue, directly on its FP32 accumulator. This removes a separate activation pass over the intermediate tensor, but changes the rounding sequence from two BF16 roundings to one.

- **Separate bias addition for other linear layers.** The linear wrapper runs a plain matrix multiply followed by bias addition, allowing a different GEMM kernel from the combined bias-epilogue path. This is a deliberate execution choice for the tested shapes; the additional BF16 rounding is part of the candidate's numerical differences.

- **Fused query/key normalization and position encoding.** A [Triton kernel](engine/fastltx/qknorm.py) combines optional bias addition, RMS normalization and split rotary position embeddings in one pass per token. It reuses the position tables across batch rows and avoids separate intermediate passes for normalization and rotation.

- **Reuse before guidance branches diverge.** The spatio-temporal guidance pass shares the conditional pass's computation through blocks 0–27, then branches at block 28 where its attention first differs. The shipped `reference` grouping preserves the reference's matrix-multiply batch shapes while skipping that repeated prefix; it does not omit a guidance pass or a denoising step.

[Transformer implementation](engine/fastltx/model.py) · [Enabled settings](serve.sh). The combination is evaluated by the step and clip checks above; final clips are not bit-identical to the reference.

## Limits

- Text to video only, tested at 768 x 512 and 121 frames on the RTX PRO 6000 Blackwell Workstation Edition. One clip at a time.
- First start: a few minutes (loading the weights and compiling kernels); later starts about 30 s (cache in `./.cache`).
- Without a CUDA toolkit, a `CUDA_HOME` warning at start-up is expected.
- Clips are not bit-identical to the Diffusers reference; they passed the registered step and final-clip checks.

## License

This folder is distributed under the LTX-2.x Community License Agreement ([LICENSE](LICENSE)), not the Apache-2.0
licence of the rest of this repository. Parts of `engine/fastltx/` follow Hugging Face diffusers (Apache-2.0),
with modifications; see [NOTICE](NOTICE). The model weights are not included; you accept their licence (the same
LTX-2.x licence) on Hugging Face.

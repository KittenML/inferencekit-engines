# LTX-2.5: shorter video requests on RTX PRO 6000

## Result

The delivered text-to-video server reduces complete clip latency from **116.2 seconds with ComfyUI to 93.2 seconds** on one RTX PRO 6000: **1.25×** the clips per unit time, using the same BF16 model and 30 denoising steps.
The selected candidate passes the step audit, request gates and final-clip gross fidelity checks. Clips are not bit-identical to the Diffusers reference.

| What matters | Result |
|---|---|
| Model and hardware | LTX-2.5 full BF16 model, one RTX PRO 6000 Blackwell (96 GB) |
| Performance | ComfyUI 116.2 → fastltx 93.2 seconds per complete request |
| Uncertainty | Six interleaved rounds; per-engine latency intervals retained, candidate/ComfyUI paired ratio interval not included in the retained result |
| Quality | Denoising-step audit PASS; request gates PASS; gross PSNR/SSIM gate PASS |
| Scope | One text-to-video request at a time, 768 × 512, 121 frames at 24 fps, audio, 30 steps |

[Timing and quality measurements](evidence/measurements.json) · [Protocol and qualifiers](#performance-and-target)

## Run it

Use Linux x86_64 with an RTX PRO 6000 Blackwell, driver 580 or newer, `gcc`, `curl` and about 100 GB free disk. Accept the gated model's licence as described in [the README](README.md#run), then make a readable Hugging Face token available as `HF_TOKEN`, or use `.venv/bin/hf auth login` when prompted.

```bash
git clone https://github.com/kittenml/inferencekit-engines
cd inferencekit-engines/ltx-2.5_rtx-pro-6000
./install.sh
./serve.sh
```

In a second terminal in the same directory:

```bash
./test.sh
```

The test generates a clip with sound and writes `./out.mp4`. The server exposes `GET /health` and `POST /v1/clips` on `http://127.0.0.1:8000`; the clip response contains `video_b64`. First launch compiles kernels and takes a few minutes.

## Files and outputs

| File | Purpose |
|---|---|
| [Installer](install.sh), [server launcher](serve.sh), [example request](test.sh) | Pinned setup, launch and local MP4 generation |
| [Transformer](engine/fastltx/model.py), [pipeline](engine/fastltx/pipeline.py) | Optimized forward pass and complete clip pipeline |
| [Normalization/position kernel](engine/fastltx/qknorm.py), [step recording](engine/fastltx/stepdump.py) | Fused operation and audit support |
| [Measurement excerpt](evidence/measurements.json) | Sanitized timing, quality, settings and source hashes |
| [Licence](LICENSE), [notices](NOTICE) | Distribution and upstream terms |

Benchmark clips and an interactive review page are not included in this public package. `test.sh` produces a fresh playable output locally.

## Details

### What changed

The shipped configuration concentrates on repeated transformer work while retaining the full model, guidance passes and denoising schedule. [serve.sh](serve.sh) enables the following combination:

- **Compiled blocks:** `torch.compile` specializes fixed-shape transformer operations, with coordinate-descent tuning and precision-cast emulation to retain eager BF16 rounding points in fused elementwise work.
- **cuDNN attention:** the transformer explicitly selects the cuDNN scaled dot-product attention backend for the tested shapes.
- **GELU epilogue:** the feed-forward projection applies GELU inside the matrix-multiply epilogue, avoiding a separate activation pass. This changes the rounding sequence.
- **Separate bias addition:** selected linear layers use a plain matrix multiply followed by bias addition, permitting a different GEMM kernel and an additional BF16 rounding.
- **Fused query/key work:** the [Triton kernel](engine/fastltx/qknorm.py) combines bias, normalization and rotary position encoding to reduce intermediate passes.
- **Guidance-prefix reuse:** the spatio-temporal guidance pass reuses the conditional computation before their first differing block, then runs its remaining blocks. The shipped `reference` grouping retains the reference's matrix-multiply batch shapes.

The fixed negative prompt's encoding is reused by both engines. Development profiles identified matrix multiplication, attention and repeated elementwise work as optimization targets; the profiles are not included here. Complete-request measurements establish the combined result, not a sum of isolated kernel timings.

### Performance and target

The candidate loads `Lightricks/LTX-2.5-Diffusers` revision `da0b690960026eab794b8ceaf841233cf7064478`, using `transformer_full` in BF16. The shipped dependencies pin Diffusers commit `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941` and PyTorch 2.13.0. The selected baseline is ComfyUI commit `3541246a9c997f4b8c32f9f88f3c985d02021c79`, classic caching, BF16 text encoder.

The final run used six balanced interleaved rounds, six timed clips per engine per round, a fresh process for each turn and new shared seeds each round. A third candidate participated in the balanced schedule; the table reports the delivered candidate and the selected baseline. The client timer includes the whole request through receipt of the finished clip. Kernel caches were warmed before timing.

| Claim | Workload / bucket | Baseline | Candidate | Ratio [CI95] | Quality | Mechanism | Gates | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Complete clip latency | Text-to-video; 768 × 512, 121 frames, 30 steps, audio | 116.2 s | 93.2 s | 1.25; paired ratio CI not retained | Step and gross clip checks PASS | Compiled transformer and fused operations | PASS | Recorded checks PASS; no stored paired-ratio verdict for this pair | `ev_e3a81353aaf9`; [excerpt](evidence/measurements.json) |

Latency is **observed** as the median over rounds of each round's mean clip time. Its per-engine 95% intervals are **116.047–116.322 seconds** for ComfyUI and **92.839–93.285 seconds** for the candidate. The **derived** speedup is `116202.2544 / 93234.7540 = 1.2463×`, rounded to 1.25× in the README. The retained aggregate's ratio intervals use the third candidate as their denominator, so they are not reused for this pair. No conservative paired ratio bound is asserted.

The measured ratio exceeds the development target of 1.2×; a paired-interval proof of that threshold is not supplied here. ComfyUI was chosen after comparison with the Diffusers wrapper and official pipeline; [baseline-selection measurements](README.md#tested-baselines) are separate from this final run.

### Quality and correctness

The [quality excerpt](evidence/measurements.json) records the selected candidate's results:

- **Step audit: PASS**, for all 120 video/audio step outputs in two audited clips. The rule was error ≤ twice the alternative-reference error floor + 0.001; the largest error/floor ratio was 1.499 (`ev_cdcc4dbf45ab`). This checks every step of those clips, not every generated clip.
- **Request gates: PASS**, for requested steps, guidance, decoded dimensions/frame count, seed and prompt sensitivity, and cache/replay behavior. The ComfyUI baseline also passed these gates.
- **Final-clip gross fidelity: PASS**, on 36 clips: mean PSNR 25.24 dB [23.84, 26.59] and SSIM 0.8343 [0.8098, 0.8580] (`ev_72062538e45d`).

The final-clip check used the reference's alternative attention backend as its floor and the registered gross-regression rule; its margins and loss intervals are in the linked excerpt. Outputs are not bit-identical. Decoded audio fidelity was not compared; audio latents were included in the step audit.

ComfyUI draws its initial noise differently, so matching-reference pixel and step fidelity were not required of that baseline. The selected candidate's passing verdict is separate from other candidates in the same run.

### What did not work and where this applies

Development rejected eager transformer execution and GEMM autotuning variants that did not beat the selected kernels. Batching all guidance rows together was faster in screening but left less numerical margin in the step checks; the shipped build uses reference grouping. Those screens do not replace the final comparison.

This result covers text-to-video at the measured size and duration, one request at a time, on the RTX PRO 6000 Blackwell Workstation Edition. Image-to-video, different shapes, multi-GPU execution and concurrency scaling are NOT RUN for the delivered result. BF16 weights remain resident on the GPU; final clips are not bit-identical to Diffusers.

### Reproduce and inspect

With the server running, write the request from [test.sh](test.sh) as one JSON line in `clips.jsonl`, add an `id` and remove `seed`. Then run the README's timing command:

```bash
uvx --from 'inferencekit-cli==0.1.1' inferencekit bench clip --base-url http://127.0.0.1:8000 \
  --clips clips.jsonl --rounds 6
```

For ComfyUI, use the same command against its server with `--endpoint-kind comfyui --workflow <your LTX-2.5 workflow JSON>`. This measures a local clip set; the published comparison also requires matching settings, balanced fresh-process rounds and the original request set. The final measurement harness was version 0.3.0; the sealed clip set, complete quality harness and raw run directory are not shipped.

### Time and cost

The [published development accounting](README.md#how-it-was-made) records Claude Opus 5.5 at high effort in Claude Code with inferencekit: 3.6 hours of agent time, 47M tokens and $18 at API list price. These are development totals, not per-clip inference costs. Complete GPU rental time and cost are not retained in this public package.

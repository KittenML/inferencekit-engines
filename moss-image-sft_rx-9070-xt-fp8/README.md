# MOSS-Image-SFT FP8 on RX 9070 XT

A speed-first experiment to explore how far inference speed can go when quality is secondary. MOSS-Image-SFT reached **2.46× faster** image generation than the tested tuned vLLM-Omni baseline on a 16 GB RX 9070 XT: **60.52 → 24.60 s/image**, paired 95% CI **2.453–2.465×** (conservative lower bound **2.453×**), over four balanced timing rounds.

<sub>‡ Lossy: online FP8 with 24 steps vs BF16 with 50 steps; same model and architecture. [Output differences and quality review](evidence/QUALITY.md).</sub>

[Run report](REPORT.md).

```sh
./install.sh
MOSS_RUN_DEADLINE="$(date -u -d '+1 day' +%Y-%m-%dT%H:%M:%SZ)" ./serve.sh
# In another terminal after readiness:
./test.sh
```

Requests must set `steps: 24`, `guidance_scale: 3.0`, and `quant: "fp8-scale-1.03125"` for this profile; examples are in `images-24.jsonl`. Requires RX 9070 XT 16 GB, the tested ROCm/WSL environment, Ubuntu 24.04, Python 3.12, git, patch and ample disk for weights and dependencies; the measured host had about 62 GiB RAM. The measured adapter and source patches are preserved; standalone installation and launch have only CPU validation. [Provenance](PROVENANCE.md).

## Tested baselines

| Implementation / configuration | Mean seconds/image ↓ |
|---|---:|
| **vLLM-Omni: regional compilation, 42 resident layers, unpinned offload — selected** | **56.92** |
| Official MOSS pipeline: CPU text encoder, compiled blocks and cast preservation | 58.33 |
| SGLang Diffusion: prefetch 2, pinned text weights | 74.45 |
| ComfyUI: compiled blocks, compatibility variant with CPU text encoder | 75.00 |

Best measured configuration per implementation in the 16-configuration baseline panel: BF16, 1024 × 1024, 50 Euler steps, six measured requests. The final balanced comparison measured 60.52 → 24.60 s/image.

Separate ComfyUI 25-step recipes measured 52.60 and 42.76 s/image, but changed sampling/cache settings and failed resource-monitor checks; they are outside the 50-step panel.

## How it was made

GPT-6.1 Terra (high) used inferencekit to integrate online FP8 into the native vLLM-Omni MOSS pipeline and select a faster generation profile for the 16 GB RX 9070 XT. The work changes both execution precision and the number of denoising steps; the reported result is the accepted speed/quality trade-off of that combination. The main changes were:

- **Route FP8 to the main transformer projections.** The [model wiring patch](online-fp8-wiring.patch) passes the quantization configuration into QKV, attention output, merged feed-forward up/gate and down projections: four projections in each of 42 blocks, or 168 in total. It loads the original BF16 checkpoint and quantizes those weights online. Text encoding, text-refinement blocks, modulation layers and the VAE stay outside this FP8 routing.

- **Preserve the activation contract at FP8 layer boundaries.** The original forward path chose input dtype from the stored weight dtype. With FP8 weights, that would prematurely cast activations to FP8. The patch instead supplies BF16 inputs using each layer's original dtype, leaving dynamic activation quantization to the FP8 linear implementation, including after attention and the feed-forward activation.

- **Tune the scale before quantizing weights.** The [scale patch](prequant-scale.patch) multiplies the weight quantization scale by 1.03125 before generating FP8 weight values for the selected projections. This changes their representable range and rounding; it is not an output multiplier or a lossless packing scheme. The [loading integration](engine/adapter/scale_audit.py) tags the intended layers before processing their weights and checks the applied scales.

- **Respect text-encoder offload during construction.** Online quantization initializes the pipeline under an accelerator default device. The model wiring patch constructs the text encoder on the device selected by the existing offload configuration, avoiding a temporary placement of its BF16 weights in accelerator memory before moving them to the CPU. This supports the existing memory plan rather than adding a separate claimed speedup.

- **Reduce the denoising work per image.** The selected request profile uses 24 steps at guidance scale 3.0 instead of the baseline's 50 steps. Combined with FP8, this changes image outputs and quality. One requested-text check failed and the fresh-seed FP8 holdout was not run; the profile was selected with those limitations as an accepted trade-off, as recorded in the [quality review](evidence/QUALITY.md).

[Source patches and validation scope](PROVENANCE.md) · [Campaign accounting](evidence/ACCOUNTING.json).

Apache-2.0; model and dependencies retain their own licenses.

# Drama3 on Apple M4 with 16 GiB

The newest qualified runnable profile is **[MAIN5/Tower5](profiles/main5-tower5/README.md)**. It measured **1.111556× speedup**, 95% CI **[1.058612×, 1.156197×]**, against the selected stock MLX-Audio configuration using identical quantized weights. Output-preservation checks passed for all 144 fresh pairs, matching PCM audio, chunks, codes and completion. Independent verification also covers all 192 timing responses and both native histories. Four timing round pairs remain underpowered. [Comparison and baseline scope](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json).

Follow the **[MAIN5 build, guarded serving and smoke commands](profiles/main5-tower5/README.md)**. Its exact portable build and HTTP/WAV smoke passed, including a new dependency installation and conversion from cached pinned inputs. [Reproduction scope](profiles/main5-tower5/ACTUAL_REPRODUCTION.json) records storage reuse and the cached-input limitation.

The baseline was selected from a four-round comparison of supported MLX-Audio memory settings on the same artifact; the top two settings were statistically tied. Both engines still failed every-request realtime playback in this qualified comparison.

[Run report](REPORT.md).

## Tested baselines

| Engine / configuration | Real-time factor ↓ | Result |
|---|---:|---|
| **MLX-Audio MAIN5/Tower5, 11 GiB wired memory** | **0.994208** | **Selected baseline** |
| MLX-Audio MAIN5/Tower5, 12 GB wired memory | 1.002140 | Statistically tied with selected setting |
| MLX-Audio, device-maximum wired memory | Not measured | Failed startup |
| audio.cpp / Metal, published GGUF | 6.995–8.034 | Four expert-cache profiles |
| audio.cpp / Metal, mixed-precision repackage | 2.674–3.051 | Fastest tested placement profile |
| speech.cpp / Metal, Q4_K | Not measured | Stopped on sustained swap growth |
| Drama3 llama.cpp fork / Mara | Not measured | Memory allocation failure or sustained swap growth |

Real-time factor is generation time divided by audio duration; lower is faster. MLX values are medians of four round means; native-engine screens used different quantizations and inputs. The [1.112× comparison](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json) uses identical weights against the selected MLX-Audio baseline.

## How it was made

GPT-6.1 Sol (high) used inferencekit to optimize speech generation within the M4's 16 GiB of unified memory. The broader preparation produced a mixed-precision MAIN5/Tower5 model and a fixed-reference serving path. The final speed comparison holds those weights, the reference/input cache and the FP16/refold codec setup equal between the two engines; its optimized arm adds three runtime changes:

- **Remove a second sort from sampling.** On the pinned MLX 0.32.4 Metal backend, the initial `argpartition` already returns the ordered values needed for the top-k prefix. The [sampler change](profiles/main5-tower5/source/measured/sampler_sort.py) removes the subsequent sort and its two gathers while keeping temperature scaling, top-p arithmetic and native categorical sampling. This relies on that backend's behavior for finite logits and negative-infinity masks with at least one finite entry, not on a general guarantee that `argpartition` sorts.

- **Compile the single-token gated-delta blocks.** The [GDN wrapper](profiles/main5-tower5/source/measured/fixed_gdn.py) compiles each of the 30 recurrent blocks for the fixed decode shape, passing convolution and recurrent state explicitly into and out of the compiled function. It retains the native block's operations and advances its cache after each token. Prefill and the ten full-attention blocks keep their native paths, concentrating compilation on the repeated single-token work.

- **Release encoder-only codec weights after reference preparation.** Once the fixed reference has been encoded and its cache hook installed, the [residency change](profiles/main5-tower5/source/measured/codec_encoder_release.py) removes encoder-only modules and quantizer input projections that generation no longer needs. Decoder and folded tensors are retained. This reduces resident codec state for the tested fixed-reference workflow; it does not provide an encoder fallback for arbitrary new reference audio.

The [profile comparison](profiles/main5-tower5/README.md) documents the shared model preparation and baseline settings. The reported speedup belongs to the combined runtime changes on the same quantized artifact, not to quantization versus the original model.

## Available profiles

| Profile | Recorded comparison | Entry |
|---|---|---|
| MAIN5/Tower5 | Current qualified 1.111556×, same artifact | [Build and guarded serve](profiles/main5-tower5/README.md) |
| MAIN6/Tower5 | Historical qualified 1.112132×, different artifact | [Profile guide](profiles/main6-tower5/README.md) |

Root `install.sh` prepares the parent model and environment. Use the linked MAIN5 guide to build and serve the current profile. Profile results are separate comparisons and must not be multiplied or compared as if they used one model artifact.

## Evidence, usage and limits

The current profile preserves all 40 MAIN layers, 256 experts and trained top-eight routing. It uses five-bit MAIN/Tower projections, mixed two-/three-bit experts, BF16 output heads and four-bit text embeddings. The sampler keeps temperature 1, top-p 0.9, top-k 4097 and the original EOS/frame limits. A 24-frame first chunk and 0.32-second interval remain. The exact model and runtime hashes are in the profile evidence.

The source/reference/model license terms and existing failed or unresolved studies remain recorded. Output preservation retains existing content/speaker errors. This fixed synthetic-reference experiment does not establish arbitrary-voice quality, RAM-only operation or performance on other chips. MLX allocation, physical footprint and whole-host swap are distinct measurements.

[Earlier profiles and experiments](https://github.com/haaziq-sl/inferencekit-engines/tree/50948559f6b058121c1f8fdd09a6319f64810b5e/drama-3_apple-m4-16gb) remain in Git history. Adapter/tools are Apache-2.0; pinned upstream code retains MIT attribution, and weights, conversions and the reference remain subject to the [Fish Audio Community License](licenses/FISH_AUDIO_COMMUNITY_LICENSE.txt). Built with Fish Audio; unofficial and not endorsed by Fish Audio.

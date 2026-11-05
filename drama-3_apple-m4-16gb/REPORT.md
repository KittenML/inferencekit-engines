# Drama3: faster speech generation on Apple M4

## Result

Speech generation was about **1.11× faster** with the selected **MAIN5/Tower5** profile than with tuned stock MLX-Audio using identical quantized weights. The paired 95% confidence interval is **[1.06, 1.16]**, with a conservative lower bound of **1.06×**. Fresh output-preservation checks passed; every-request realtime playback did not. [Qualified comparison](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json).

| What matters | Result |
|---|---|
| Model and hardware | Drama3 MAIN5/Tower5, mixed-precision weights, Apple M4 with 16 GiB unified memory |
| Performance | Stock MLX-Audio **1.0335 → 0.9298 real-time factor**, client-timed; lower is better |
| Improvement | About **1.11× [1.06, 1.16]**; four timing round pairs remain underpowered |
| Quality | **PASS** for observed preservation of PCM audio, chunks, codes and completion across 144 fresh pairs |
| Scope and missed target | Fixed synthetic reference; same-artifact comparison; **1.5× target not met**; both arms failed every-request realtime criteria |

Real-time factor is generation time divided by generated audio duration. An aggregate value below one does not establish uninterrupted playback for every request. Timing values and their derivation are recorded in [Performance and target](#performance-and-target).

## Run it

Use native arm64 macOS on the tested memory class. From this model directory, install the pinned parent environment and frozen client:

```sh
./install.sh
./install-benchmark-tooling.sh
```

Complete the [protected Tower5 input build](profiles/main6-tower5/README.md#build-from-the-installed-default), then the [MAIN5 build](profiles/main5-tower5/README.md#prepare-and-build). These are required preparation steps; the parent installation alone does not build the selected profile. The guides specify the free-disk floors and fresh output directories.

After those builds, this command launches the selected guarded runtime and makes its fixed example HTTP request:

```sh
profile_python="$PWD/.venv/bin/python"
profile_source="$PWD/profiles/main5-tower5/source"
profile_work="$PWD/main5-profile"

"$profile_python" -B "$PWD/profiles/main5-tower5/serve_guarded.py" \
  --profile-source "$profile_source" --runtime "$profile_work/runtime" \
  --model-dir "$profile_work/model-dc860" --python "$profile_python" \
  --client-python "$PWD/.benchmark-venv/bin/python" \
  --guard-runtime "$profile_work/guard-runtime" \
  --receipt-dir "$profile_work/smoke-evidence" \
  --timeout-seconds 14400 --smoke
```

The generated example is `main5-profile/smoke-evidence/client/request-0000.wav`, with PCM and receipts alongside it. Omit `--smoke` and select a fresh receipt directory to serve on `127.0.0.1:8765` until the caller deadline. Stop one invocation before starting another.

The exact portable rebuild and HTTP/WAV smoke passed, including a separate new dependency installation and conversion from cached pinned inputs. New network downloads and reproduction on another host were not tested. [Reproduction scope](profiles/main5-tower5/ACTUAL_REPRODUCTION.json).

## Files and outputs

| File or location | Contents |
|---|---|
| [Current profile guide](profiles/main5-tower5/README.md) | Complete build, guarded serving and smoke instructions |
| [Measured runtime source](profiles/main5-tower5/source/measured/) | Sampler, GDN, codec and reference handling |
| [Qualified comparison](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json) | Current timing qualification, preservation and baseline scope |
| [Reproduction receipt](profiles/main5-tower5/ACTUAL_REPRODUCTION.json) | Artifact identity, installation, build and smoke scope |
| [Synthetic reference WAV](profiles/main5-tower5/source/measured/conditioning-cache-v2/reference.wav) | Included conditioning input, not generated benchmark output |
| `main5-profile/smoke-evidence/client/` | Locally generated WAV, PCM and request records after the smoke |

The current comparison's full generated-audio collection and review page are not bundled here. The included reference WAV is not a substitute for those outputs. The [earlier MAIN6 evidence](proof/paired-v57/README.md) belongs to a different artifact and is not used for this report's timing or quality result.

## Details

### What changed

GPT-6.1 Sol (high) used inferencekit to prepare and optimize this profile. Both comparison arms share the converted weights, overlay loader, fixed-reference/input cache, audit instrumentation and FP16/refold codec. The measured runtime combines three changes:

- [Sampler sort removal](profiles/main5-tower5/source/measured/sampler_sort.py) removes a second sort and two gathers. It depends on the pinned MLX Metal backend's observed `argpartition` ordering for the admitted logits; temperature, top-p arithmetic and native categorical sampling remain.
- [Single-token GDN compilation](profiles/main5-tower5/source/measured/fixed_gdn.py) compiles the 30 recurrent blocks with explicit convolution and recurrent state. Prefill and the ten full-attention blocks retain their native paths.
- [Codec encoder release](profiles/main5-tower5/source/measured/codec_encoder_release.py) removes encoder-only modules after preparing the fixed reference, retaining decoder and folded tensors. Arbitrary new-reference encoding is outside this workflow.

These sources explain the combined runtime change. This report makes no separate speedup claim for any one component, and the public package does not include a current device profile that apportions the gain among them.

### Performance and target

The pinned runtime is MLX **0.32.4** and MLX-Audio commit `76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d`, as recorded in the [dependency lock](requirements.runtime.lock). Both arms use **11 GiB wired memory and zero cache**, the same MAIN5/Tower5 artifact and the same client/streaming workload.

The finite baseline panel selected that stock setting after four balanced rounds; its comparison with the 12 GB setting included a statistical tie. The supported maximum setting failed startup. The [tested-baseline table](README.md#tested-baselines) also records native-engine screens, which used other quantizations and inputs and are not same-artifact comparisons.

| Claim | Workload / bucket | Baseline | Candidate | Ratio [CI95] | Quality | Mechanism | Gates | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Qualified same-artifact runtime gain | Fixed-reference speech, single request; 192 responses, 96 matched pairs | Stock MLX-Audio, RTF 1.033498 | MAIN5/Tower5 optimized runtime, RTF 0.929775 | 1.111556× [1.058612, 1.156197] | Fresh preservation PASS | Combined sampler/GDN/encoder changes | Model identity, native replay, completion and worker cleanup PASS | PASS, narrow qualified scope; underpowered | [Current qualification](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json) |
| At least 1.5× runtime speedup | Same comparison | Same stock arm | Same candidate | Upper CI bound 1.156197× | Same preservation result | Same changes | Same checks | FAIL | [Target result](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json) |
| Every-request realtime playback | Same streaming workload | Failed | Failed | Not a ratio claim | Preservation does not establish realtime playback | Complete streaming behavior | Realtime criteria FAIL in both arms | FAIL | [Streaming result](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json) |

**Observed:** the frozen comparison's median round-mean RTF values are **1.0334978438012987** stock and **0.9297754375516589** optimized. Its evidence id is `ev_acf276892c33261b022e524258f8154cf5650bb5d4dc66c6f85e55df722b55f2`; the current qualification confirms the independent timing verification and unchanged frozen statistics. Absolute values are from that final comparison, not the earlier baseline-selection panel.

**Derived:** speedup = `1.0334978438012987 / 0.9297754375516589 = 1.1115564060530227`. The frozen tool bootstrapped four paired round indices with 10,000 resamples and seed zero. Its candidate/stock ratio CI `[0.8649046439658106, 0.9446335511244521]` gives the speedup CI by reciprocal endpoints. The published qualification retains `powered=false`.

### Quality and correctness

The [current preservation record](profiles/main5-tower5/QUALIFIED_COMPARISON_V65.json) reports **144/144 fresh pairs PASS** across **288 responses**, with matching PCM, chunks, generated codes and completion. Independent native replay passed **144 requests and segments per arm**. Every recorded request completed with EOS; there were no matching completion failures.

Independent verification also covers all **192 timing responses**, with **128 native prompt/history segments per arm**, and model identity across timing and preservation runs. These are observed counts from the qualification and [profile comparison scope](profiles/main5-tower5/README.md#comparison-and-baseline).

This establishes preservation against the same quantized baseline on the checked corpus. Existing content and speaker errors remain preserved.

### What did not work and where this applies

The baseline search was finite; it does not establish a globally best engine. Four timing pairs limit precision even though the reported interval excludes one.

The measured profile retains all 40 MAIN layers, 256 experts and trained top-eight routing. Five-bit MAIN/Tower projections, mixed two-/three-bit experts, BF16 heads and four-bit embeddings are shared by both arms. The [profile policy](profiles/main5-tower5/source/model-metadata/MAIN5_TOWER5_POLICY_V62.json) and [generation recipe](profiles/main5-tower5/source/measured/selected_recipe.json) pin the configuration.

RAM-only operation, other chips, arbitrary reference changes and concurrent-serving capacity are not established by this comparison. MLX allocation, process footprint and whole-system swap are different measurements.

### Reproduce and inspect

Use the current build and smoke commands above. The [registered client](profiles/main5-tower5/source/measured/registered_client.py) and [runtime manifest](profiles/main5-tower5/source/measured/RUNTIME_MANIFEST.json) preserve the client and serving source identities; the full original timing plan is not shipped as a new benchmark command here.

The [packaging provenance](profiles/main5-tower5/PACKAGING_PROVENANCE.json) records retained runtime bytes and CPU packaging checks. The reproduction receipt verifies the measured artifact revision `dc860f699a91fa5e30e32698aeb1ab4d5608eff03496c5d7d11bbdcfb3c895ce`. A smoke demonstrates execution and cleanup, not a repeated performance or quality study.

### Time and cost

The [accounting snapshot](ACCOUNTING_V65.json) records about **70 elapsed hours**, **2.07 billion tokens** and **$475 standard-rate API list-price equivalent** for direct Drama3 work through the frozen October 26 checkpoint. Time and tokens are observed; the cost is derived from recorded usage and pricing. Cached input is included in the token total.

Elapsed time includes waits; active time and actual bills were not measured. Shared coordination, tooling, reference validation, hardware and later activity are outside this snapshot; the linked accounting retains the full breakdown.

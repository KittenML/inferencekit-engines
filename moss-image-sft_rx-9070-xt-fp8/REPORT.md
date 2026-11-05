# MOSS-Image-SFT: selected FP8 trade-off on RX 9070 XT

## Result

The selected online FP8 profile reduces decoded-image generation from **60.52 to 24.60 seconds/image** against the tested tuned vLLM-Omni BF16 baseline: **2.46×**, paired 95% CI **[2.453, 2.465]**, with a conservative lower bound of **2.453×**. It combines FP8 with fewer denoising steps and was selected as an accepted lossy trade-off. [Timing comparison](evidence/comparison.json) · [Selection](evidence/selection.json).

| What matters | Result |
|---|---|
| Model and hardware | MOSS-Image-SFT, one RX 9070 XT with 16 GB device memory |
| Performance | Tuned vLLM-Omni BF16/50 steps **60.52 → 24.60 s/image**, client-timed through decoded image output |
| Improvement | **2.46× [2.453, 2.465]** across four balanced timing pairs |
| Quality | Primary requested-text check **FAIL**: 15/16 candidate images passed versus 16/16 baseline; accepted trade-off |
| Scope | Online FP8 scale 1.03125, 24 steps, 1024 × 1024, single request; fresh-seed FP8 holdout **NOT RUN** |

This is a **cross-precision format and work trade-off**, with numerics changed. It does not isolate an engine-only gain or establish unchanged quality. The [quality review and original images](evidence/QUALITY.md) retain the visible text defect.

## Run it

Use the tested ROCm/WSL environment, Ubuntu 24.04, Python 3.12, git and patch, with ample disk for dependencies and weights. The measured system had about 62 GiB host RAM. Run from this model directory:

```sh
./install.sh
MOSS_RUN_DEADLINE="$(date -u -d '+1 day' +%Y-%m-%dT%H:%M:%SZ)" ./serve.sh
```

In another terminal after readiness, send the included example request:

```sh
./test.sh
```

It writes `runs/moss-test.png` and `runs/moss-test-receipt.json`. The [example client](test.sh) checks PNG decoding, dimensions, effective settings and a worker receipt hash. The server uses `127.0.0.1:8000` by default.

Requests must use `steps: 24`, `guidance_scale: 3.0` and `quant: "fp8-scale-1.03125"`; [request fixtures](images-24.jsonl) include the remaining pinned settings. The launcher requires a finite UTC deadline and verifies the source, model and configuration before execution.

See [provenance and validation scope](PROVENANCE.md) for source identities and the packaged execution checks.

## Files and outputs

| File or location | Contents |
|---|---|
| [Side-by-side comparison](evidence/comparison.html) | Original baseline and candidate poster from the failed text case |
| [Baseline PNG](evidence/baseline-poster.png), [FP8 PNG](evidence/fp8-poster.png) | Unretouched saved outputs |
| [Timing evidence](evidence/comparison.json) | Paired comparison, timer scope and source evidence identities |
| [Quality review](evidence/QUALITY.md), [sealed review summary](evidence/recognizability-summary.json) | Review criteria, counts and retained failure |
| [Selection record](evidence/selection.json) | Accepted lossy profile and exact image hashes |
| [Source provenance](PROVENANCE.md), [adapter inventory](engine-manifest.json) | Pinned source and validation boundaries |
| `runs/moss-test.png` | Locally generated example after a successful packaged request |

The public comparison page contains the failure pair; it is not a full primary or holdout gallery.

## Details

### What changed

GPT-6.1 Terra (high) used inferencekit to integrate online FP8 into the native vLLM-Omni pipeline and choose the faster request profile. The architecture and source checkpoint remain the same; numerical execution and work per image change.

- [Model wiring](online-fp8-wiring.patch) routes online FP8 into QKV, attention output, merged feed-forward up/gate and down projections: four projections in each of 42 blocks, **168 total**. The text encoder, text refinement, modulation layers and VAE stay outside that routing.
- The same patch preserves BF16 inputs at FP8 layer boundaries, leaving dynamic activation quantization to the linear implementation. It also respects the existing text-encoder offload device during construction.
- [Scale adjustment](prequant-scale.patch) multiplies the weight quantization scale by **1.03125 before quantizing** the selected weights. It changes representable range and rounding; the [scale audit](engine/adapter/scale_audit.py) checks the intended layers and applied values.
- The selected request uses **24 rather than 50 Euler steps**, at guidance scale **3.0**. Fewer steps and FP8 jointly produce the reported result; their separate speed contributions were not isolated.

These mechanisms are supported by the preserved implementation and configuration. GPU-kernel attribution and a separate custom-kernel speedup were not established.

### Performance and target

The baseline is the tested tuned vLLM-Omni configuration with regional compilation, 42 resident layers and unpinned offload. It was selected from the [baseline panel](README.md#tested-baselines); that panel's six-request means are separate from the final balanced comparison.

The [source provenance](PROVENANCE.md#source-and-model-identities) pins vLLM-Omni commit `7cbc776c16a39f73978ae5ce400729d94e0fda35` and MOSS-Image-SFT revision `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`. Both arms use the same four image tasks and paired seed policy, at **1024 × 1024** and batch one.

| Claim | Workload / bucket | Baseline | Candidate | Ratio [CI95] | Quality | Mechanism | Gates | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Selected cross-precision format/work trade-off | Four prompts × four paired seeds; 1024²; single request | BF16, 50 steps; 60.516 s/image | FP8 scale 1.03125, 24 steps; 24.597 s/image | 2.460× [2.453, 2.465] | Primary text FAIL; accepted lossy selection | FP8 and fewer steps combined | Timing complete; full lifecycle/reliability UNPROVEN | FAIL under original quality rule; selected with that loss | [Timing](evidence/comparison.json), [quality](evidence/recognizability-summary.json), [selection](evidence/selection.json) |
| Fresh-seed FP8 quality | Independent holdout | Not evaluated | Not evaluated | Not a timing claim | NOT RUN | Same selected profile | Primary failure prevented holdout | NOT RUN | [Review scope](evidence/QUALITY.md#review-scope) |
| Standalone packaged execution | Clean install and GPU smoke | Not applicable | Preserved measured adapter and patches | Not a timing claim | No new model output from CPU checks | Pinned installer and launcher | CPU checks PASS; packaged GPU execution NOT RUN | NOT RUN | [Package checks](evidence/package-checks.json), [provenance](PROVENANCE.md) |

**Observed:** the comparison records median round-level request means of **60,515.553948625005 ms** baseline and **24,596.500428 ms** candidate. Timing includes complete image response handling and decoding. The evidence id is `ev_45f3ff6fb70a0d8fe356aa1c94d47a46535663483abf894aa77693ebc1674cc4` in the [timing record](evidence/comparison.json).

**Derived:** `60515.553948625005 / 24596.500428 = 2.460331872242106`. The paired bootstrap used four interleaved round indices, 10,000 resamples and seed zero. Inverting the candidate/baseline ratio CI `[0.4056227792624249, 0.40759418664266034]` gives **[2.453420663913209, 2.4653447762928327]**.

The timing exceeds the campaign's 1.5× objective. The selected profile retains the original quality verdict shown above.

### Quality and correctness

The [blinded primary review](evidence/recognizability-summary.json) found **15/16 candidate images PASS**, including **7/8 requested-text cases**, versus **16/16 baseline images PASS**. A requested final letter is largely hidden by the subject in one candidate poster. This is a semantic text-layout defect, not a numerical-tolerance finding.

The review covers four prompt types and four seeds, assessed by one AI reviewer. There was no human review panel or population quality confidence interval. The fresh-seed FP8 holdout remains **NOT RUN** after the primary failure. [Selection and review limitations](evidence/selection.json).

The [original image pair](evidence/comparison.html) makes the retained defect inspectable. General quality equivalence is not established.

### What did not work and where this applies

Earlier BF16 and other FP8 experiments belong to different step counts or numerical settings; their quality results are not transferred to this profile.

The timing record leaves fresh-process lifecycle and repeated-run reliability **UNPROVEN**. Complete detached-process containment, exact GPU-kernel dispatch and general memory headroom are not established. Image decoding and receipt checks do not prove those properties.

Results apply to the measured single-request RX 9070 XT workload. Other devices, resolutions, concurrency levels and deployment environments were not established by this comparison. No supported global-best implementation claim follows from the finite baseline panel.

### Reproduce and inspect

The [source reconstruction and installer checks](evidence/package-checks.json) passed on CPU. To inspect the bundled source snapshot without running the model:

```sh
python3 verify_snapshot.py
```

Use the install, launch and example request above for the selected profile. Preserve the resulting configuration, worker/source receipts, output hashes and deadline records. A successful example is a functional check; reproducing the timing requires balanced paired rounds and a separate quality review.

### Time and cost

The [campaign accounting snapshot](evidence/ACCOUNTING.json) records about **19.6 elapsed hours**, **1.80 billion tokens** and **$206 standard-rate API list-price equivalent**, including setup, failed attempts, preparation, coordination and optimization. Time and tokens are observed; the cost is derived from the recorded usage and pricing. Cached input is included in the token total.

Elapsed time includes waits; active time and actual bills were not measured. The linked accounting retains the hardware allocation estimates and their unreconciled billing scope.

# Provenance and validation scope

## Result

The development engine completed four balanced timing pairs at FP8/24 steps: **2.460× [2.453, 2.465]** versus the tuned BF16/50-step baseline. The user selected this fastest lossy profile on October 29 with an accepted quality tradeoff. The requested-text check remains **FAIL** and the fresh-seed FP8 holdout remains **NOT RUN**. See the [quality review](evidence/QUALITY.md) and [current selection](evidence/selection.json).

The bundled adapter and source patches match the measured configuration. A complete standalone clean installation, packaged GPU smoke test and native worker launch remain **NOT RUN**. CPU source and installer checks do not establish those; reproduction still needs worker/source/configuration receipts.

## Run it

Follow the [package setup and serving commands](README.md). The package smoke request uses 24 steps.

## Files and outputs

| File | Purpose |
|---|---|
| [Adapter inventory](engine-manifest.json) | Pins the isolated evaluation adapter in `engine/` |
| [Model wiring patch](online-fp8-wiring.patch) | Connects online FP8 to the native MOSS pipeline |
| [Original FP8 implementation](fp8-upstream.py), [scale patch](prequant-scale.patch), [patched implementation](fp8-overlay.py) | Preserve the reviewed source and its isolated change |
| [Snapshot verifier](verify_snapshot.py) | Checks bundled bytes and patch application on a temporary copy, on CPU only |
| [Package checks](evidence/package-checks.json) | CPU verification results |

## Details

### Source and model identities

The `engine/` inventory is the exact isolated evaluation adapter from `workload/fp8-scale-integration`, pinned by `engine-manifest.json` SHA-256 `7f33a94b55db495bbd0ff02443b95b0172d9ec752475a4966340c1f8269146c3`. Its source manifest specifies vLLM-Omni commit `7cbc776c16a39f73978ae5ce400729d94e0fda35`. The model is `OpenMOSS-Team/MOSS-Image-SFT` at revision `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`; a full model manifest and actual runtime model binding are required before serving.

The candidate vLLM-Omni checkout requires `online-fp8-wiring.patch`, SHA-256 `93f315bc8e6cc822c4c09fa9824cb6671172afdaec47d502beb999da4bbcf170`; the candidate source manifest pins the modified files. Offline application to the exact pinned upstream native-source files passed and reproduced every expected candidate hash.

### FP8 change

The candidate uses online per-tensor FP8 with a 1.03125 scale multiplier for 168 selected transformer projections, with unmodified BF16 checkpoint weights; numerics change. The SHA-256s for `fp8-upstream.py`, `prequant-scale.patch` and `fp8-overlay.py` are `ce504bea5fb8d7b9683eb4df0d88a850aeec8d802af63b61bfd011a24dda71b7`, `95aee4ac3b9aca81327777619f93119e18c299a70c9e4e69fa074561102a78ac` and `8bcc9a37d448ed271bc5994b1211876a16d082fa1f3fdc84ea95d99465a6776d`, respectively.

### Installer and validation scope

The installer downloads the copied pinned baseline runtime manifests, applies the candidate wiring patch to the isolated checkout, and copies the wheel-pinned vLLM package into an isolated overlay. It verifies the installed package against the pinned vLLM wheel and verifies the overlay against that wheel with the FP8 source replaced; it refuses an existing runtime. The launcher checks the snapshot, resolved model/config and overlay, and uses an offline clean environment. This implementation has CPU fixture tests and a verified reconstruction of the pinned native-source patch. The baseline installer and launcher sources were copied, then adapted; no shared runtime is modified.

Historical draft metadata is preserved in [the initial proposal](evidence/history/initial-proposal.json). Campaign time, tokens, standard-rate equivalent API cost and hardware estimates are in [campaign accounting](evidence/ACCOUNTING.json).

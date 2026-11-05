# MAIN5/Tower5 on Apple M4, 16 GiB

## Result

This is the newest qualified model profile. It measured **1.111556× speedup**, 95% CI **[1.058612×, 1.156197×]**, against the selected stock MLX-Audio configuration on identical MAIN5/Tower5 weights. Output-preservation checks passed for all 144 fresh pairs, matching PCM audio, chunks, codes and completion; inherited model errors remain visible. Four round pairs remain underpowered, and both engines failed the every-request realtime criteria. See [the qualified comparison](QUALIFIED_COMPARISON_V65.json).

The portable rebuild and one-request HTTP/WAV smoke passed before this packaging cleanup. A separate fresh dependency installation and original-source conversion also passed using cached pinned inputs; a new network download and a different host were not tested. See [reproduction scope](ACTUAL_REPRODUCTION.json).

## Run it

### Prepare and build

First run the parent installation and build the protected [MAIN6/Tower5 input](../main6-tower5/README.md). The MAIN5 builder reads that model and the complete original checkpoint without changing them. It adds about 922 MiB when unchanged parent files can be hardlinked.

Run from `drama-3_apple-m4-16gb`, using fresh output directories:

```sh
profile_python="$PWD/.venv/bin/python"
profile_source="$PWD/profiles/main5-tower5/source"
profile_work="$PWD/main5-profile"
profile_tower5="$PWD/tower5-profile/model9343"
profile_original="$PWD/models/original-fp8"

"$profile_python" -B "$profile_source/profile_setup.py" \
  --output "$profile_work" --python "$profile_python" \
  --tower5-model "$profile_tower5" --original-model "$profile_original" \
  --timeout-seconds 14400

"$profile_python" -B "$profile_source/prepare_guard_tools.py" \
  --output "$profile_work/guard-runtime" --timeout-seconds 14400

"$profile_python" -B "$profile_source/build_guarded.py" \
  --python "$profile_python" --guard-runtime "$profile_work/guard-runtime" \
  --base-model "$profile_tower5" --original-model "$profile_original" \
  --output-model "$profile_work/model-dc860" \
  --receipt-dir "$profile_work/build-evidence" --timeout-seconds 14400
```

The builder requires at least 40,265,318,400 bytes free before building, with a 32 GiB free-disk floor during the build.

### Serve or smoke-test

Install the frozen client once for the smoke, then run the guarded entry:

```sh
./install-benchmark-tooling.sh
"$profile_python" -B "$PWD/profiles/main5-tower5/serve_guarded.py" \
  --profile-source "$profile_source" --runtime "$profile_work/runtime" \
  --model-dir "$profile_work/model-dc860" --python "$profile_python" \
  --client-python "$PWD/.benchmark-venv/bin/python" \
  --guard-runtime "$profile_work/guard-runtime" \
  --receipt-dir "$profile_work/smoke-evidence" --timeout-seconds 14400 --smoke
```

Omit `--smoke` and use a fresh receipt directory to serve at `127.0.0.1:8765` until the finite caller deadline. This profile supports the included synthetic reference. Keep at least 38,923,141,120 bytes free before loading the model and stop one invocation before starting another.

## Files and outputs

| Location | Contents |
|---|---|
| `main5-profile/model-dc860` | Built model; model payloads are not included in the repository |
| `main5-profile/build-evidence` | Build receipts |
| `main5-profile/smoke-evidence` | Fixed public WAV/PCM output and identity/cleanup receipts |
| [Measured source](source/measured/) | Unchanged serving code, reference assets and smoke client |

## Details

### Comparison and baseline

Here “stock” means pinned MLX-Audio native model operators with the same custom converted weights, 984-tensor overlay loader, fixed-reference/input cache, audit instrumentation, FP16/refold codec and measurement client. Both arms use 11 GiB wired memory and cache 0. The optimized arm alone removes the redundant sampler sort, compiles the 30 single-token GDN blocks and releases the codec encoder after fixed-reference preparation. Stock retains its encoder and native sampler/GDN implementation.

Independent timing covered 192 responses with 128 native prompt/history segments per arm. The stock panel tested three supported memory settings, admitted two, and selected 11 GiB/cache 0 after four balanced rounds; its comparison with 12 GB includes a statistical tie. See the [quality and baseline scope](QUALIFIED_COMPARISON_V65.json) for the checks covered by this comparison.

### Build and reproduction scope

The builder quantizes the original BF16 MAIN matrices directly, preserves the 234 Tower5 arrays and every expert/head/embedding/codec choice, and reproduces the exact registered artifact. It streams one matrix at a time under the tested pressure/ownership/deadline guards and restores process limits. Its expected model revision is `dc860f699a91fa5e30e32698aeb1ab4d5608eff03496c5d7d11bbdcfb3c895ce`. Child thread environment variables are set to two; this is not an OS CPU-core affinity claim.

The separate installation used a new managed Python/venv and cached pinned checkpoint, codec and tool inputs. Every fresh shard was fully verified before optional storage sharing.

The [original V65 benchmark package](https://github.com/haaziq-sl/inferencekit-engines/tree/50948559f6b058121c1f8fdd09a6319f64810b5e/drama-3_apple-m4-16gb/profiles/main5-tower5) remains in Git history. Adapter Apache-2.0, upstream MIT and Fish Audio model-license notices remain with the source.

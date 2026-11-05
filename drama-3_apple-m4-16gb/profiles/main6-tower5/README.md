# Optional MAIN6/Tower5 profile

## Result

On the tested M4 with 16 GiB memory, the optimized engine measured **1.112132× speedup**, 95% CI **[1.069841×, 1.152939×]**, against the strongest stock configuration in the tested finite panel. Both arms received the same Tower5 weights. Output-preservation checks passed for all 144 fresh paired outputs against this baseline. Four timing rounds give limited statistical power; this is not a claim about every available engine. Both arms failed the full realtime criteria.

The portable rebuild and guarded HTTP/WAV smoke passed on the Mac before this packaging cleanup, using existing admitted inputs. The complete download/install/build/smoke sequence has not been run as one fresh installation. See [actual reproduction evidence](ACTUAL_REPRODUCTION.json).

## Run it

### Build from the installed default

Use native arm64 macOS, the pinned environment created by the parent `install.sh`, and its complete original and converted model directories. The overlay adds about 626 MiB when unchanged model files can be hardlinked; copying across filesystems needs additional space. The builder retains a 32 GiB free-disk floor for macOS paging and work files.

Run from `drama-3_apple-m4-16gb` after `./install.sh`:

```sh
profile_python="$PWD/.venv/bin/python"
profile_source="$PWD/profiles/main6-tower5/source"
profile_work="$PWD/tower5-profile"
profile_base="$PWD/models/e2-mse-e3last8-d6-t6-bf16heads-emb4-g128"
profile_original="$PWD/models/original-fp8"

"$profile_python" "$profile_source/profile_setup.py" \
  --output "$profile_work" --python "$profile_python" \
  --base-model "$profile_base" --original-model "$profile_original" \
  --timeout-seconds 14400

"$profile_python" "$profile_source/prepare_guard_tools.py" \
  --output "$profile_work/guard-runtime" --timeout-seconds 14400

"$profile_python" "$profile_source/build_guarded.py" \
  --python "$profile_python" --guard-runtime "$profile_work/guard-runtime" \
  --base-model "$profile_base" --original-model "$profile_original" \
  --output-model "$profile_work/model9343" \
  --receipt-dir "$profile_work/build-evidence" --timeout-seconds 14400
```

Use a fresh work directory for each attempt. The timeout is an operator-controlled limit for that invocation. The original models are read-only inputs.

### Serve and smoke-test

Using the variables from the build commands above, start the guarded service:

```sh
"$profile_python" "$PWD/profiles/main6-tower5/serve_guarded.py" \
  --profile-source "$profile_source" --runtime "$profile_work/runtime" \
  --model-dir "$profile_work/model9343" --python "$profile_python" \
  --guard-runtime "$profile_work/guard-runtime" \
  --receipt-dir "$profile_work/serve-evidence" --timeout-seconds 14400
```

The service binds `127.0.0.1:8765`, supports the included synthetic reference, and retains process, pressure and cleanup receipts. Keep at least 38,923,141,120 bytes free before loading the full model. Stop it before running another invocation.

For the fixed one-request smoke, install the frozen measurement client with the parent script, then use a fresh receipt directory:

```sh
./install-benchmark-tooling.sh
"$profile_python" "$PWD/profiles/main6-tower5/serve_guarded.py" \
  --profile-source "$profile_source" --runtime "$profile_work/runtime" \
  --model-dir "$profile_work/model9343" --python "$profile_python" \
  --client-python "$PWD/.benchmark-venv/bin/python" \
  --guard-runtime "$profile_work/guard-runtime" \
  --receipt-dir "$profile_work/smoke-evidence" --timeout-seconds 14400 --smoke
```

The smoke checks completion and model/runtime identity, then closes its processes. It adds no new speed measurement.

## Files and outputs

| Location | Contents |
|---|---|
| `tower5-profile/model9343` | Built model |
| `tower5-profile/build-evidence` | Build progress and cleanup receipts |
| `tower5-profile/serve-evidence` | Service process, pressure and cleanup receipts |
| `tower5-profile/smoke-evidence` | Smoke receipts, `client/request-0000.wav` and PCM |
| [Measured source](source/measured/) | Serving code and assets used in the completed comparison |

## Details

### Model and build

This profile keeps MAIN at six bits and rebuilds the 78 Tower projections at five bits. It uses the same sampler and fixed GDN implementation as the default engine. The model architecture, active experts, generation settings and synthetic reference stay the same.

The builder verifies every protected input and builds one Tower matrix at a time under process and memory guards. The final artifact must match revision `93432d8a6c97faba88309880c3fd2d1b8356bde679dc20f9e0daa57282eb904c`.

### Evidence and reproduction scope

The rebuild matched all 234 arrays and the complete measured artifact; the fixed smoke request reached EOS. The recorded run used separately admitted existing model and client environments. The parent installation, client installer and overlay build have separate validation histories.

The repository's [V57 evidence](../../proof/paired-v57/README.md) contains the timing and preservation tables, frozen statistics, baseline selection and native replay summaries.

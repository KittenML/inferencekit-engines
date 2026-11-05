#!/usr/bin/env bash
set -euo pipefail
: "${OMNI_SOURCE:?set OMNI_SOURCE to pinned vLLM-Omni checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the pinned model directory}"
: "${RUN_DIR:?set RUN_DIR to a new empty path}"
root="$(cd "$(dirname "$0")" && pwd)"
python3 "$root/adapter/make_config.py" --fp8-overlay-root "${FP8_OVERLAY_ROOT:?set isolated vLLM overlay root}" --source-root "$OMNI_SOURCE" --model-dir "$MODEL_DIR" --run-directory "$RUN_DIR" --transport-file "$root/transport/moss_amd/server.py" --mode lightweight --fixed-negative --deadline-utc "${MOSS_RUN_DEADLINE:?set UTC deadline}" --compile-policy regional --resident-layers 42 --pin-memory false --pipeline-profiler false

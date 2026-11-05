#!/usr/bin/env bash
# Starts the OpenAI-compatible server on http://127.0.0.1:8000 (PORT and HOST change it).
# It loads the weights, compiles and warms up the kernels and captures the decode CUDA graph before it
# listens, which takes about 1-2 minutes; it is ready when it prints "listening on".
# Extra arguments go to serve.py (python serve.py --help lists them).
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] && [ -f .venv/engine.env ] || { echo "ERROR: run ./install.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
. .venv/engine.env
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: no weights in $MODEL_DIR; run ./install.sh." >&2; exit 1; }
exec .venv/bin/python serve.py --model "$MODEL_DIR" --served-model-name Qwen/Qwen3-8B-FP8 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"

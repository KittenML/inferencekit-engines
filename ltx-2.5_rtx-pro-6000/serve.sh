#!/bin/sh
# Start the clip server on 127.0.0.1:8000 (set PORT to change the port, HOST to listen elsewhere).
# The first start compiles the model's kernels (a few minutes; cached in ./.cache, so later starts take about 30 s).
# The server is ready when it prints "engine ready" and GET /health answers {"status": "ok"}.
set -eu
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "[serve] ERROR: .venv not found. Run ./install.sh first." >&2; exit 1; }
[ -f .venv/ltx25_diffusers_snapshot ] || { echo "[serve] ERROR: model weights not downloaded. Run ./install.sh first." >&2; exit 1; }
exec env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 DIFFUSERS_NO_ADVISORY_WARNINGS=1 \
  PYTHONPATH=engine FASTLTX_COMPILE=1 FASTLTX_ATTN=cudnn FASTLTX_GELU_EPI=1 FASTLTX_CDT=1 FASTLTX_SPLIT_BIAS=1 \
  FASTLTX_QK_FUSED=1 FASTLTX_GROUPING=reference TORCHINDUCTOR_CACHE_DIR=.cache/inductor \
  .venv/bin/python -m fastltx --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"

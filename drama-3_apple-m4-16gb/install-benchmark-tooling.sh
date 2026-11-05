#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
cd "$(dirname "$0")"
WHEEL="${1:-benchmark/vendor/inferencekit_cli-0.1.0-py3-none-any.whl}"
.venv/bin/python - "$WHEEL" <<'PY'
import hashlib,sys
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()!='7eebd59f181b6070a946335421bb7fd7fe6d5ff7d5ea8878835eee64d9d9b176':raise SystemExit('Frozen benchmark wheel checksum mismatch')
PY
UV_BIN="$PWD/.tools/uv-aarch64-apple-darwin/uv"
export UV_CACHE_DIR="$PWD/.cache/uv"
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ ! -x .benchmark-venv/bin/python ]; then "$UV_BIN" venv --python .venv/bin/python .benchmark-venv; fi
"$UV_BIN" pip sync --python .benchmark-venv/bin/python benchmark/requirements.client.lock
"$UV_BIN" pip install --python .benchmark-venv/bin/python --no-deps "$WHEEL"

#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG:?set CONFIG to generated configuration.json}"
: "${MOSS_RUN_DEADLINE:?set UTC deadline}"
root="$(cd "$(dirname "$0")" && pwd)"
exec python "$root/adapter/moss_omni_adapter.py" --configuration "$CONFIG" --transport-root "$root/transport" --port "${PORT:-8000}"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 source/local_tts.py sample vieneu-cpp "$@"

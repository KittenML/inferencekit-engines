#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v adb >/dev/null; then
    echo 'Install Android platform-tools and put adb on PATH.' >&2
    exit 1
fi
adb get-state >/dev/null
exec python3 source/local_tts.py stage kokoro-opencl

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-full}" in
    full) plan=smoke-baseline ;;
    adaptive) plan=smoke-adaptive ;;
    *) echo 'Usage: ./test.sh [full|adaptive] [new-output-directory]' >&2; exit 2 ;;
esac
output="${2:-build/output/$plan}"
bash latency/scripts/capture-phone.sh latency/build/s25-study.apk "$output" 5 "$plan" --installed
python3 - "$output" <<'PY'
from pathlib import Path
import sys
import wave

root = Path(sys.argv[1])
files = sorted((root / "cells").rglob("*.wav"))
if not files:
    raise SystemExit("No WAV audio was exported")
for file in files:
    with wave.open(str(file), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or not any(wav.readframes(wav.getnframes())):
            raise SystemExit(f"Expected non-silent mono PCM16 audio in {file}")
    print(file)
PY

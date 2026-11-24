#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${1:-} == playback ]]; then
    shift
    apk=$(realpath "${1:?Supply the signed AudioTrack benchmark APK}")
    output=$(realpath -m "${2:?Supply a new output directory}")
    exec bash latency/reproduce.sh adb "$apk" "$output" "${3:-26}"
fi
phase=${1:-kokoro-sustained}
if [[ ! $phase =~ ^[a-z0-9][a-z0-9-]*$ || ! -f source/$phase.json ]]; then
    echo 'Choose a registered phase listed in REPORT.md.' >&2
    exit 2
fi
result=results/"$phase"
adb get-state >/dev/null
archive=build/"$phase".zip
if [[ ! -f $archive ]]; then
    bash source/build_farm_package.sh "$phase"
fi
payload=$(mktemp -d "$PWD/build/bench.XXXXXXXX")
trap 'rm -rf "$payload"' EXIT
unzip -q "$archive" 'payload/*' 'run-plan.json' -d "$payload"
plan_flag() {
    python3 - "$payload/run-plan.json" "$1" <<'PY'
import json
import sys

with open(sys.argv[1]) as plan:
    sys.exit(not json.load(plan).get(sys.argv[2]))
PY
}
if plan_flag local_reproduction; then
    unzip -q "$archive" 'local-reproduction/*' -d "$payload"
fi
python3 source/run_device.py --scenario "$payload/run-plan.json" --payload "$payload/payload" \
    --output "$result/speech-capture"
if plan_flag quality_only; then
    exit 0
fi
python3 source/import_capture.py "$result/speech-capture" "$result/analysis"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
capture="${1:?capture directory}"
baseline="${2:?baseline label}"
candidate="${3:?candidate label}"
run=runs/vieneu-v3-turbo-galaxy-s25-20261120
label=$(basename "$capture")
summary="results/${label}-summary.json"
quality="results/${label}-quality.json"
python3 scripts/summarize.py "$capture" --baseline "$baseline" --candidate "$candidate" > "$summary"
"${QUALITY_PYTHON:-python3}" scripts/score_quality.py "$capture" --baseline "$baseline" --candidate "$candidate" > "$quality"
if [[ $label == confirmation ]]; then
    python3 scripts/verify_capture.py "$capture" > "results/${label}-checks.json"
fi
python3 scripts/export_review.py "$capture" "$quality" "$run"
INFERENCEKIT_RUNS_DIR="$PWD/$run/evidence" inferencekit review build "$run" --reference "$baseline"

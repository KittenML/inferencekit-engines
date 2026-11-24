#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash latency/scripts/capture-phone.sh latency/build/s25-study.apk "${1:-results/confirmation}" "${2:-26}"

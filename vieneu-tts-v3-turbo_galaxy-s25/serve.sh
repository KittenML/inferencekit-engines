#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
apk=latency/build/s25-study.apk
[[ -s "$apk" ]] || { echo 'Run ./install.sh first.' >&2; exit 2; }
model=$(adb shell getprop ro.product.model | tr -d '\r')
[[ "$model" == SM-S931* ]] || { echo 'Connect a base Galaxy S25 with USB debugging.' >&2; exit 2; }
adb install -r "$apk"

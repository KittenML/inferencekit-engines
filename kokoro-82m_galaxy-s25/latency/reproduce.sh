#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
    build)
        bash source/build.sh "${2:-final}"
        ;;
    adb)
        apk="${2:?Supply the signed study APK}"
        output="${3:?Supply a new artifact directory}"
        minutes="${4:-26}"
        [[ "$minutes" =~ ^[1-9][0-9]?$ ]] || { echo "Supply a valid timeout in minutes" >&2; exit 2; }
        if [[ -e "$output" ]]; then
            echo "Use a new artifact directory" >&2
            exit 2
        fi
        model=$(adb shell getprop ro.product.model | tr -d '\r')
        if [[ "$model" != SM-S931* ]]; then
            echo "Expected a connected base Galaxy S25" >&2
            exit 2
        fi
        package=org.kittenml.s25study
        if adb shell pm path "$package" | grep -q '^package:'; then
            echo "The study is installed. Use app data with no prior study rounds." >&2
            exit 2
        fi
        mkdir -p "$output/results"
        adb install "$apk"
        if adb shell dumpsys package "$package" | grep -Fq 'android.permission.RECORD_AUDIO'; then
            adb shell pm grant "$package" android.permission.RECORD_AUDIO
        fi
        adb shell am start -n "$package/.StudyActivity"
        deadline=$((SECONDS + minutes*60))
        status=''
        while ((SECONDS < deadline)); do
            sleep 25
            status=$(adb shell run-as "$package" cat files/status 2>/dev/null | tr -d '\r' || true)
            printf '%s\n' "$status"
            case "$status" in DONE|ERROR*) break ;; esac
        done
        python3 - "$output/device-metadata.json" "$status" <<'PYCODE'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({"status": sys.argv[2]}) + "\n")
PYCODE
        adb logcat -d -v threadtime |
            grep -E 'S25Release|Created OpenCL device|Initializing OpenCL-based API|Falling back to CPU|OpenCL initialization failed' |
            tail -350 > "$output/backend-placement.txt" || true
        case "$status" in
            DONE|ERROR*)
                adb exec-out run-as "$package" cat files/study.zip > "$output/study.zip"
                unzip -q "$output/study.zip" -d "$output/results"
                ;;
            *) echo "The benchmark is still running or could not report status" >&2; exit 1 ;;
        esac
        [[ "$status" == DONE ]]
        ;;
    *)
        echo "Usage: bash reproduce.sh build [final|smoke|compute|acoustic|recovery|ui|acoustic-ui|seeded|matched-chunks|precision-matrix|release-panel] | adb <apk> <new-output-directory> [minutes]" >&2
        exit 2
        ;;
esac

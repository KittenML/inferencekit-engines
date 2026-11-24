#!/usr/bin/env bash
set -euo pipefail

apk=${1:?Supply the signed VieNeu playback benchmark APK}
output=${2:?Supply a new output directory}
minutes=${3:-26}
package=org.inferencekit.s25study
remote=files/study/latest

[[ "$minutes" =~ ^[1-9][0-9]?$ ]] || { echo 'Supply a timeout in minutes from 1 to 99.' >&2; exit 2; }
[[ -s "$apk" ]] || { echo 'Build the signed playback APK first.' >&2; exit 2; }
model=$(adb shell getprop ro.product.model | tr -d '\r')
[[ "$model" == SM-S931* ]] || { echo 'Connect a base Galaxy S25 with USB debugging.' >&2; exit 2; }
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
capture_id=$(python3 -c 'import uuid; print(uuid.uuid4())')
adb install -r "$apk"
adb shell am start -W -S -n "$package/.MainActivity" --es plan_name confirmation --es capture_id "$capture_id"

deadline=$((SECONDS + minutes * 60))
while ((SECONDS < deadline)); do
    if adb exec-out run-as "$package" cat "$remote/index.json" > "$output/index.partial.json" 2>/dev/null; then
        if state=$(python3 - "$output/index.partial.json" "$capture_id" <<'PY'
import json
import sys
from pathlib import Path

index = json.loads(Path(sys.argv[1]).read_text())
if index.get("capture_id") == sys.argv[2]:
    if index.get("error"):
        print("error")
    elif index.get("complete"):
        print("complete")
PY
        ); then
            case "$state" in
                complete)
                    mv "$output/index.partial.json" "$output/index.json"
                    adb exec-out run-as "$package" cat "$remote/bundle.zip" > "$output/bundle.zip"
                    unzip -q "$output/bundle.zip" -d "$output"
                    exec python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["failed_cells"] != 0)' "$output/index.json"
                    ;;
                error)
                    mv "$output/index.partial.json" "$output/index.json"
                    echo "Playback benchmark failed. See $output/index.json." >&2
                    exit 1
                    ;;
            esac
        fi
    fi
    sleep 5
done
echo "Playback benchmark timed out. See $output/index.partial.json for the last capture state." >&2
exit 1

#!/bin/sh
# Send one text-to-video request to the running server (./serve.sh) and save the clip to ./out.mp4.
# Waits up to 30 minutes for the server to finish starting. PORT and HOST as for serve.sh.
set -eu
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "[test] ERROR: .venv not found. Run ./install.sh first." >&2; exit 1; }
URL="http://${HOST:-127.0.0.1}:${PORT:-8000}" .venv/bin/python - <<'PY'
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

url = os.environ["URL"]
start = time.time()
waiting = hinted = False
while True:
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as r:
            if json.load(r).get("status") == "ok":
                break
    except urllib.error.HTTPError as e:
        if e.code != 503:  # 503 means the server is still loading
            sys.exit(f"[test] ERROR: /health answered {e.code}")
    except OSError:
        if time.time() - start > 60 and not hinted:
            print(f"[test] nothing is listening on {url}; is ./serve.sh running?", flush=True)
            hinted = True
    if time.time() - start > 1800:
        sys.exit("[test] ERROR: the server did not become ready within 30 minutes")
    if not waiting:
        print("[test] waiting for the server to finish starting...", flush=True)
        waiting = True
    time.sleep(10)

req = {"task": "t2v", "prompt": "A red fox trots through fresh snow in a quiet pine forest at sunrise, "
       "breath steaming; soft wind and the crunch of paws in the snow.",
       "width": 768, "height": 512, "num_frames": 121, "fps": 24, "steps": 30, "guidance_scale": 3.0, "seed": 42}
print("[test] generating one clip (768x512, 121 frames, 24 fps, with audio)...", flush=True)
t = time.time()
r = urllib.request.Request(url + "/v1/clips", data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(r, timeout=3600) as resp:
        out = json.load(resp)
except urllib.error.HTTPError as e:
    sys.exit(f"[test] ERROR: {e.code} {e.read().decode(errors='replace')}")
with open("out.mp4", "wb") as f:
    f.write(base64.b64decode(out["video_b64"]))
print(f"[test] saved out.mp4 ({os.path.getsize('out.mp4') / 1e6:.1f} MB) in {time.time() - t:.1f} s")
print("[test] stage times (ms):", json.dumps(out.get("stage_ms", {})))
PY

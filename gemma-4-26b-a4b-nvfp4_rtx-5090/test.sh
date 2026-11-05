#!/usr/bin/env bash
# Send one chat request to the server started by ./serve.sh and print the reply.
# Waits up to WAIT_S seconds (default 3600) for the server to come up. PORT and HOST as for serve.sh.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "test.sh: run ./install.sh first." >&2; exit 1; }
exec .venv/bin/python - "http://${HOST:-127.0.0.1}:${PORT:-8000}" "${WAIT_S:-3600}" <<'EOF'
import json, sys, time, urllib.request, urllib.error
base, wait_s = sys.argv[1], float(sys.argv[2])
deadline = time.time() + wait_s
said = False
while True:
    try:
        urllib.request.urlopen(base + "/v1/models", timeout=5).read()
        break
    except (urllib.error.URLError, OSError):
        if time.time() > deadline:
            sys.exit(f"test.sh: no server at {base} after {wait_s:.0f} s. Is ./serve.sh running?")
        if not said:
            print(f"waiting for the server at {base} (the first start can take 30 minutes or more) ...", flush=True)
            said = True
        time.sleep(5)
body = {"model": "gemma-4-26b-a4b-nvfp4", "max_tokens": 128, "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        "messages": [{"role": "user", "content": "In two sentences, why is the sky blue?"}]}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t = time.time()
out = json.load(urllib.request.urlopen(req, timeout=300))
print(out["choices"][0]["message"]["content"].strip())
u = out.get("usage", {})
print(f"\n({u.get('completion_tokens', '?')} tokens in {time.time() - t:.2f} s)")
EOF

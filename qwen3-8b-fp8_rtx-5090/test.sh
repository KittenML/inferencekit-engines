#!/usr/bin/env bash
# Sends one chat request to the server that ./serve.sh started and prints the reply.
# Waits up to 10 minutes for the server to be ready. PORT and HOST as for serve.sh.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "ERROR: run ./install.sh first." >&2; exit 1; }
exec .venv/bin/python - "${HOST:-127.0.0.1}" "${PORT:-8000}" <<'PY'
import json, sys, time, urllib.error, urllib.request
base = f"http://{sys.argv[1]}:{sys.argv[2]}"
deadline = time.time() + 600
while True:
    try:
        urllib.request.urlopen(base + "/v1/models", timeout=5).read()
        break
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        if time.time() > deadline:
            sys.exit(f"ERROR: no server at {base} after 10 minutes. Is ./serve.sh running? Check its output.")
        time.sleep(3)
body = {"model": "Qwen/Qwen3-8B-FP8", "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t = time.perf_counter()
try:
    out = json.load(urllib.request.urlopen(req, timeout=300))
except urllib.error.HTTPError as e:
    sys.exit(f"ERROR: HTTP {e.code}: {e.read().decode(errors='replace')}")
dt = time.perf_counter() - t
n = out["usage"]["completion_tokens"]
print(out["choices"][0]["message"]["content"].strip())
print(f"\n[{n} tokens in {dt:.2f} s, {n / dt:.0f} tokens/s including the prompt; finish_reason: "
      f"{out['choices'][0]['finish_reason']}]")
PY

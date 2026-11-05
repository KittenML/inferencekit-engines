#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$root/runs"
"$root/runtime/venv/bin/python" - "${PORT:-8000}" "${MOSS_TEST_OUTPUT:-$root/runs/moss-test.png}" "${MOSS_TEST_RECEIPT:-$root/runs/moss-test-receipt.json}" "${MOSS_TEST_RESPONSE:-}" <<'PY'
import base64, hashlib, io, json, pathlib, re, sys, urllib.request
from PIL import Image

body = {'model':'OpenMOSS-Team/MOSS-Image-SFT','model_revision':'f972bd4b4b9e3852133cc5da954d87bc32cd8fb9','dtype':'bf16','quant':'fp8-scale-1.03125','prompt':'A red bicycle leaning against a brick wall, in a realistic street photograph.','width':1024,'height':1024,'steps':24,'guidance_scale':3.0,'seed':42,'scheduler':'euler','flow_shift':5.0,'negative_prompt':'','uncond_skip_layers':[22],'max_sequence_length':4096}
req = urllib.request.Request('http://127.0.0.1:' + sys.argv[1] + '/v1/images/generations', json.dumps(body).encode(), {'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=1800) as response:
    response_bytes = response.read()
data = json.loads(response_bytes)
if data.get('format') != 'png' or data.get('effective', {}).get('steps_run') != body['steps']:
    raise SystemExit('unexpected output format or effective step count')
effective = data['effective']
for key in ('model', 'model_revision', 'dtype', 'quant', 'width', 'height', 'guidance_scale', 'seed', 'scheduler', 'flow_shift', 'negative_prompt', 'uncond_skip_layers', 'max_sequence_length'):
    if effective.get(key) != body[key]:
        raise SystemExit('unexpected effective field: ' + key)
execution = data.get('execution', {})
receipt_sha = execution.get('native_worker_receipt_sha256')
if not isinstance(receipt_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', receipt_sha):
    raise SystemExit('missing native worker receipt hash')
image = base64.b64decode(data['image_b64'], validate=True)
with Image.open(io.BytesIO(image)) as decoded:
    if decoded.format != 'PNG' or decoded.size != (body['width'], body['height']):
        raise SystemExit('unexpected image format or dimensions')
    decoded.verify()
with Image.open(io.BytesIO(image)) as decoded:
    decoded.load()
    rgba_sha = hashlib.sha256(decoded.convert('RGBA').tobytes()).hexdigest()
path = pathlib.Path(sys.argv[2])
path.write_bytes(image)
receipt = {'format': 'png', 'width': body['width'], 'height': body['height'], 'steps': body['steps'],
           'prompt_sha256': hashlib.sha256(body['prompt'].encode()).hexdigest(), 'effective': effective,
           'response_sha256': hashlib.sha256(response_bytes).hexdigest(),
           'image_sha256': hashlib.sha256(image).hexdigest(), 'decoded_rgba_sha256': rgba_sha,
           'native_worker_receipt_sha256': receipt_sha}
pathlib.Path(sys.argv[3]).write_text(json.dumps(receipt, indent=2) + '\n')
if sys.argv[4]:
    pathlib.Path(sys.argv[4]).write_bytes(response_bytes)
print(json.dumps(receipt, sort_keys=True))
PY

#!/usr/bin/env python3
"""CPU-only verification of the source snapshot; never installs or loads a model."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
pins = {
    'online-fp8-wiring.patch': '93f315bc8e6cc822c4c09fa9824cb6671172afdaec47d502beb999da4bbcf170',
    'engine-manifest.json': '7f33a94b55db495bbd0ff02443b95b0172d9ec752475a4966340c1f8269146c3',
    'fp8-upstream.py': 'ce504bea5fb8d7b9683eb4df0d88a850aeec8d802af63b61bfd011a24dda71b7',
    'prequant-scale.patch': '95aee4ac3b9aca81327777619f93119e18c299a70c9e4e69fa074561102a78ac',
    'fp8-overlay.py': '8bcc9a37d448ed271bc5994b1211876a16d082fa1f3fdc84ea95d99465a6776d',
}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in pins.items():
    path = root / name
    if path.is_symlink() or digest(path) != expected:
        raise SystemExit('snapshot mismatch: ' + name)
manifest = json.loads((root / 'engine-manifest.json').read_text())['files']
engine = root / 'engine'
actual = {p.relative_to(engine).as_posix() for p in engine.rglob('*') if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts and p.suffix != '.pyc'}
if actual != set(manifest):
    raise SystemExit('engine inventory differs')
for name, expected in manifest.items():
    path = engine / name
    for parent in path.parents:
        if parent == engine:
            break
        if parent.is_symlink():
            raise SystemExit('symlink parent: ' + name)
    if path.is_symlink() or digest(path) != expected:
        raise SystemExit('engine file differs: ' + name)
with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp) / 'vllm/model_executor/layers/quantization/online/fp8.py'
    target.parent.mkdir(parents=True)
    target.write_bytes((root / 'fp8-upstream.py').read_bytes())
    subprocess.run(['patch', '-s', '-p1', '-i', str(root / 'prequant-scale.patch')], cwd=tmp, check=True)
    if digest(target) != pins['fp8-overlay.py']:
        raise SystemExit('patch output differs')
print('PASS: source snapshot and patch; standalone installation and GPU execution NOT RUN by this check; measured results in evidence/')

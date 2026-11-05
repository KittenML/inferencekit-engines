#!/usr/bin/env python3
"""Build isolated candidate on an already freshly installed pinned runtime."""
import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
PATCH = '93f315bc8e6cc822c4c09fa9824cb6671172afdaec47d502beb999da4bbcf170'
BASE = 'ce504bea5fb8d7b9683eb4df0d88a850aeec8d802af63b61bfd011a24dda71b7'
SCALE = '8bcc9a37d448ed271bc5994b1211876a16d082fa1f3fdc84ea95d99465a6776d'
FP = Path('vllm/model_executor/layers/quantization/online/fp8.py')

def tree(root):
    out = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError('symlink in isolated source: ' + str(p))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            out[p.relative_to(root).as_posix()] = sha(p)
    return out

WHEEL = '733504dae65ee4cfa1b9ac8cf771116eec0f81264d4e0561b4ea255b6ba2190b'
def wheel_inventory(runtime, patched=False):
    matches = [p for p in (runtime/'wheelhouse').glob('vllm-*.whl') if not p.is_symlink() and sha(p)==WHEEL]
    if len(matches)!=1: raise ValueError('pinned vLLM wheel missing or ambiguous')
    out = {}
    with zipfile.ZipFile(matches[0]) as z:
        for entry in z.infolist():
            name = entry.filename
            if not name.startswith('vllm/') or entry.is_dir(): continue
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts or name in out: raise ValueError('invalid wheel entry')
            with z.open(entry) as f: out[name] = hashlib.file_digest(f, 'sha256').hexdigest()
    if not out or out.get(FP.as_posix()) != BASE: raise ValueError('wheel source mismatch')
    if patched: out[FP.as_posix()] = SCALE
    return out

def build(runtime, source_package):
    runtime = runtime.resolve(strict=True); source_package = source_package.resolve(strict=True)
    source = runtime / 'vllm-omni'; overlay = runtime / 'fp8-overlay'; receipt = runtime / 'fp8-install-complete.json'
    if overlay.exists() or overlay.is_symlink() or receipt.exists() or receipt.is_symlink(): raise ValueError('candidate already exists')
    base_receipt = runtime / 'install-complete.json'
    if base_receipt.is_symlink() or not base_receipt.is_file(): raise ValueError('base install incomplete')
    manifest = json.loads((ROOT/'engine/adapter/SOURCE_MANIFEST.json').read_text())
    if sha(ROOT/'online-fp8-wiring.patch') != PATCH or manifest['candidate_patch_sha256'] != PATCH: raise ValueError('wiring patch mismatch')
    if sha(source_package/FP.relative_to('vllm')) != BASE: raise ValueError('upstream FP8 mismatch')
    for row in manifest['files']:
        p = source / row['path']; expected = row.get('baseline_sha256', row['sha256'])
        if p.is_symlink() or not p.is_file() or sha(p) != expected: raise ValueError('baseline source mismatch: '+row['path'])
    if {'vllm/'+k:v for k,v in tree(source_package).items()} != wheel_inventory(runtime): raise ValueError('installed vLLM differs from pinned wheel')
    env = {'PATH':'/usr/bin:/bin','HOME':str(runtime),'LANG':'C.UTF-8','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
    subprocess.run(['git','apply','--check',str(ROOT/'online-fp8-wiring.patch')],cwd=source,env=env,check=True)
    subprocess.run(['git','apply',str(ROOT/'online-fp8-wiring.patch')],cwd=source,env=env,check=True)
    for row in manifest['files']:
        if sha(source/row['path']) != row['sha256']: raise ValueError('candidate source mismatch: '+row['path'])
    overlay.mkdir(mode=0o700); shutil.copytree(source_package, overlay/'vllm', symlinks=False, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=overlay/FP
    if sha(target) != BASE or sha(ROOT/'fp8-overlay.py') != SCALE: raise ValueError('overlay mismatch')
    target.write_bytes((ROOT/'fp8-overlay.py').read_bytes())
    if tree(overlay) != wheel_inventory(runtime, patched=True): raise ValueError('overlay differs from pinned wheel and patch')
    result = {'status':'CPU_INSTALLED_NOT_RUN','source_manifest_sha256':sha(ROOT/'engine/adapter/SOURCE_MANIFEST.json'),'engine_manifest_sha256':sha(ROOT/'engine-manifest.json'),'base_receipt_sha256':sha(base_receipt),'patch_sha256':PATCH,'overlay_tree':tree(overlay),'overlay_fp8_sha256':SCALE}
    tmp=receipt.with_suffix('.partial');tmp.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');tmp.replace(receipt)

def main():
    runtime=ROOT/'runtime';py=runtime/'venv/bin/python'
    sites=subprocess.check_output([str(py),'-I','-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True,env={'PATH':'/usr/bin:/bin','HOME':str(runtime),'LANG':'C.UTF-8'}).strip()
    pkg=Path(sites)/'vllm'
    if runtime.resolve() not in pkg.resolve().parents: raise ValueError('vLLM package is outside managed runtime')
    build(runtime,pkg)
if __name__ == '__main__': main()

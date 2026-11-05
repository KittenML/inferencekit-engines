#!/usr/bin/env python3
"""Offline tests of wheel/overlay integrity and clean launcher controls."""
import hashlib, importlib.util, subprocess, sys, tempfile, zipfile
from pathlib import Path
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate',p/'fp8_install.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
h=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as d:
 r=Path(d);(r/'wheelhouse').mkdir(); w=r/'wheelhouse/vllm-test.whl'
 with zipfile.ZipFile(w,'w') as z:
  z.writestr(m.FP.as_posix(),b'base');z.writestr('vllm/__init__.py',b'init')
 m.WHEEL=h(w.read_bytes());m.BASE=h(b'base');m.SCALE=h(b'scaled')
 assert m.wheel_inventory(r)[m.FP.as_posix()]==h(b'base')
 overlay=r/'overlay';(overlay/m.FP).parent.mkdir(parents=True);(overlay/m.FP).write_bytes(b'scaled');(overlay/'vllm/__init__.py').write_bytes(b'init')
 assert m.tree(overlay)==m.wheel_inventory(r,patched=True)
 (overlay/m.FP).write_bytes(b'drift');assert m.tree(overlay)!=m.wheel_inventory(r,patched=True)
 (overlay/'vllm/alias').symlink_to('missing')
 try:m.tree(overlay);raise AssertionError('symlink was accepted')
 except ValueError:pass
 w.write_bytes(w.read_bytes()+b'changed')
 try:m.wheel_inventory(r);raise AssertionError('wheel drift accepted')
 except ValueError:pass
with tempfile.TemporaryDirectory() as d:
 top=Path(d);root=top/'package';root.mkdir();m.ROOT=root
 runtime=top/'runtime';source=runtime/'vllm-omni';(source/'vllm_omni').mkdir(parents=True)
 original=b'original\n';changed=b'candidate\n';(source/'vllm_omni/item.py').write_bytes(original)
 patch=b'--- a/vllm_omni/item.py\n+++ b/vllm_omni/item.py\n@@ -1 +1 @@\n-original\n+candidate\n'
 (root/'online-fp8-wiring.patch').write_bytes(patch);m.PATCH=h(patch)
 (root/'engine/adapter').mkdir(parents=True);(root/'engine/adapter/SOURCE_MANIFEST.json').write_text(__import__('json').dumps({'candidate_patch_sha256':m.PATCH,'files':[{'path':'vllm_omni/item.py','baseline_sha256':h(original),'sha256':h(changed)}]}))
 (root/'engine-manifest.json').write_text('{}');(root/'fp8-overlay.py').write_bytes(b'scaled');(runtime/'install-complete.json').write_text('{}')
 (runtime/'wheelhouse').mkdir();w=runtime/'wheelhouse/vllm-test.whl'
 with zipfile.ZipFile(w,'w') as z:z.writestr(m.FP.as_posix(),b'base');z.writestr('vllm/__init__.py',b'init')
 m.WHEEL=h(w.read_bytes());m.BASE=h(b'base');m.SCALE=h(b'scaled')
 package=runtime/'site/vllm';(package/m.FP.relative_to('vllm')).parent.mkdir(parents=True);(package/m.FP.relative_to('vllm')).write_bytes(b'base');(package/'__init__.py').write_bytes(b'init')
 m.build(runtime,package);assert (source/'vllm_omni/item.py').read_bytes()==changed
 assert (runtime/'fp8-overlay'/m.FP).read_bytes()==b'scaled'
 assert __import__('json').loads((runtime/'fp8-install-complete.json').read_text())['status']=='CPU_INSTALLED_NOT_RUN'
 try:m.build(runtime,package);raise AssertionError('existing candidate accepted')
 except ValueError:pass
result=subprocess.run([sys.executable,str(p/'verify_candidate_install.py'),'--configuration','/nonexistent-moss-test-config'],capture_output=True,text=True)
assert result.returncode != 0 and '/nonexistent-moss-test-config' in result.stderr and "No such file or directory: '--configuration'" not in result.stderr
s=(p/'serve.sh').read_text()
assert 'env -i' in s and 'PYTHONPATH=$FP8_OVERLAY_ROOT' in s and 'verify_candidate_install.py' in s and 'FP8_OVERLAY_ROOT ||' not in s
print('PASS: pinned wheel and overlay checks, drift and symlink rejection, launch controls')

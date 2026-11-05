#!/usr/bin/env python3
"""Install exact historical baseline dependencies; no GPU inference."""
import hashlib,json,os,subprocess,sys,urllib.request,urllib.parse
from pathlib import Path
from artifact_selection import select_verified
from verify_install import MANIFEST_SHA256, MODEL_REVISION, SOURCE_COMMIT, sha
p=Path(__file__).resolve().parent; root=p/'runtime'
try:
 root.mkdir()
except FileExistsError:
 raise SystemExit('runtime already exists; preserve it and move it aside before a fresh install')
wh=root/'wheelhouse';wh.mkdir()
manifest=json.loads((p/'python-artifacts.json').read_text())
if manifest['source_commit'] != SOURCE_COMMIT or sha(p/'model-artifacts.json') != MANIFEST_SHA256:
 raise ValueError('package source or model manifest pin differs')
env=os.environ.copy();env.pop('PYTHONPATH', None);env.update(MAX_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',VLLM_TARGET_DEVICE='rocm',VLLM_OMNI_TARGET_DEVICE='rocm',VLLM_ROCM_USE_AITER='0',ROCPROFILER_REGISTER_ENABLED='0',PIP_DISABLE_PIP_VERSION_CHECK='1',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
def run(a):
 print('RUN',a[0:4],flush=True);subprocess.run(a,env=env,check=True)
verified=[]
for i,item in enumerate(manifest['artifacts']):
 url=item['url'];name=Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name;target=wh/name
 if not target.exists() or sha(target)!=item['sha256']:
  tmp=target.with_suffix(target.suffix+'.partial')
  with urllib.request.urlopen(url,timeout=180) as r,tmp.open('wb') as w:
   while b:=r.read(1024*1024):w.write(b)
  if sha(tmp)!=item['sha256']:raise ValueError('artifact hash mismatch '+name)
  tmp.replace(target)
 verified.append(target)
 print('verified',i+1,'/',len(manifest['artifacts']),name,flush=True)
if not (root/'debs-complete').exists():
 for fn,dest,base in [('rocm-debs.json','rocm','https://repo.radeon.com/rocm/apt/7.2.4/'),('v3-runtime-debs.json','runtime-libs','https://archive.ubuntu.com/ubuntu/'),('v4-profiler-runtime-debs.json','profiler-sdk','https://repo.radeon.com/rocm/apt/7.2.4/'),('v5-profiler-runtime-debs.json','aqlprofile','https://repo.radeon.com/rocm/apt/7.2.4/')]:
  for item in json.loads((p/fn).read_text())['packages']:
   rel=item.get('filename',item.get('Filename'));digest=item.get('sha256',item.get('SHA256'));target=root/'debs'/Path(rel).name;target.parent.mkdir(exist_ok=True)
   if not target.exists() or sha(target)!=digest:
    with urllib.request.urlopen(base+rel,timeout=180) as r,target.open('wb') as w:
     while b:=r.read(1024*1024):w.write(b)
   if sha(target)!=digest:raise ValueError('deb hash mismatch')
   destdir=root/dest;destdir.mkdir(exist_ok=True);run(['dpkg-deb','--extract',str(target),str(destdir)])
 (root/'debs-complete').touch()
env['ROCM_PATH']=str(root/'rocm/opt/rocm-7.2.4');env['HIP_PATH']=env['ROCM_PATH'];env['PATH']=env['ROCM_PATH']+'/bin:'+env['PATH'];env['LD_LIBRARY_PATH']='/opt/rocm-host/lib:'+env['ROCM_PATH']+'/lib:'+env.get('LD_LIBRARY_PATH','')+':'+':'.join(str(root/x) for x in ['profiler-sdk/opt/rocm-7.2.4/lib','aqlprofile/opt/rocm-7.2.4/lib','runtime-libs/usr/lib/x86_64-linux-gnu'])
venv=root/'venv';bootstrap=root/'virtualenv.pyz'
if not bootstrap.exists() or sha(bootstrap)!='3965fd7b3f140e5cb3890f44787ed5803c9fcc88b3065d39a1bb603bb5f9a01a':
 with urllib.request.urlopen('https://github.com/pypa/virtualenv/releases/download/21.12.1/virtualenv.pyz',timeout=180) as r,bootstrap.open('wb') as w:
  while b:=r.read(1024*1024):w.write(b)
if sha(bootstrap)!='3965fd7b3f140e5cb3890f44787ed5803c9fcc88b3065d39a1bb603bb5f9a01a':raise ValueError('virtualenv bootstrap hash mismatch')
run([sys.executable,str(bootstrap),'--clear','--no-download','--no-periodic-update',str(venv)]);py=str(venv/'bin/python')
wheels,sdists=select_verified(verified)
run([py,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)])
for sdist in sdists:run([py,'-m','pip','install','--no-index','--no-deps','--no-build-isolation',str(sdist)])
source=root/'vllm-omni'
if not source.exists():run(['git','clone','--filter=blob:none','--no-checkout',manifest['source_repository'],str(source)])
run(['git','-C',str(source),'checkout','--detach',manifest['source_commit']]);run([py,'-m','pip','install','--no-index','--no-deps','--no-build-isolation','-e',str(source)]);run([py,'-m','pip','check'])
for item in json.loads((p/'model-artifacts.json').read_text()):
 target=root/'model'/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists() or target.stat().st_size!=item['size'] or sha(target)!=item['sha256']:
  tmp=target.with_suffix(target.suffix+'.partial')
  with urllib.request.urlopen(item['url'],timeout=180) as r,tmp.open('wb') as w:
   while b:=r.read(1024*1024):w.write(b)
  if tmp.stat().st_size!=item['size'] or sha(tmp)!=item['sha256']:raise ValueError('model hash mismatch '+item['path'])
  tmp.replace(target)
 print('verified model',item['path'],flush=True)
receipt=root/'install-complete.json'
temporary=root/'install-complete.json.partial'
temporary.write_text(json.dumps({'source_commit':manifest['source_commit'],'model_revision':MODEL_REVISION,'model_manifest_sha256':sha(p/'model-artifacts.json'),'inference':'not run'},indent=2)+'\n')
temporary.replace(receipt)

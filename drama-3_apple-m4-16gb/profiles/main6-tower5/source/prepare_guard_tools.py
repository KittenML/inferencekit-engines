"""Future native-only bootstrap of tiny C probes; never imports ML or models."""
import argparse,os,platform,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--timeout-seconds',type=float,required=True);a=p.parse_args()
    require(platform.system()=='Darwin' and platform.machine()=='arm64','Native arm64 macOS required')
    end=deadline(seconds=a.timeout_seconds);verify_packet();out=explicit(a.output);require(not out.exists(),'Fresh guard tool directory required');disk(out.parent);out.mkdir()
    commands=[];started=time.monotonic()
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):env[key]='2'
    for name in ('memory_guard.py','memory_watch.py','memory_probe.c','process_identity.c'):shutil.copy2(HERE/'guard-source'/name,out/name)
    try:
        for args in [('/usr/bin/clang','-O2','-dynamiclib','process_identity.c','-lproc','-o','process_identity.dylib'),
                     ('/usr/bin/clang','-O2','memory_probe.c','-lproc','-o','memory_probe')]:
            before_deadline(end);remaining=(datetime.datetime.fromisoformat(end)-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            with (out/(args[-1]+'.compile.log')).open('xb') as log:
                child=subprocess.Popen(args,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                try:code=child.wait(timeout=remaining)
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,15)
                        try:child.wait(timeout=2)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait(timeout=2)
                    raise
            commands.append({'argv':list(args),'returncode':code});require(code==0,'Native guard compile failed')
        files={name:sha(out/name) for name in ('memory_guard.py','memory_watch.py','memory_probe.c','process_identity.c','process_identity.dylib','memory_probe')}
        atomic(out/'GUARD_TOOLS_ACTUAL.json',{'status':'ACTUAL_NATIVE_DARWIN_GUARD_TOOLS_BUILT','files':files,'commands':commands,'compiler_sha256':sha('/usr/bin/clang'),'elapsed_seconds':time.monotonic()-started,'model_calls':0})
    except BaseException as error:
        atomic(out/'FAILED.json',{'status':'FAILED_RETAINED','error_type':type(error).__name__,'error':str(error),'commands':commands});raise
if __name__=='__main__':main()

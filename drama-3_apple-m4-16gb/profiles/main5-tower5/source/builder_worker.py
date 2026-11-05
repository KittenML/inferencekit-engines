"""Guard-released numerical child; no model discovery or model forwards."""
import argparse,importlib.metadata,os,platform,resource,signal,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_common import *

def main():
    p=argparse.ArgumentParser()
    for name in ('base-model','original-model','output-model','guard-runtime','owner-manifest','release','receipt','deadline'):p.add_argument('--'+name,required=True)
    p.add_argument('--guard-control',choices=('normal','hold'))
    a=p.parse_args();require(platform.system()=='Darwin','Native Darwin required')
    runtime,_=guard_tools(a.guard_runtime);sys.path.insert(0,str(runtime));from memory_guard import KernelIdentity
    kernel=KernelIdentity();owner=load(a.owner_manifest);owner_identity=tuple(owner['kernel_identity'])
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:(_ for _ in ()).throw(SystemExit(128+n)))
    while not Path(a.release).exists():
        require(kernel.read(owner['pid'])==owner_identity,'Owner disappeared before release');before_deadline(a.deadline);time.sleep(.025)
    gate=load(a.release);require(tuple(gate['child_identity'])==kernel.read(os.getpid()),'Child PID birth mismatch')
    require(tuple(gate['owner_identity'])==owner_identity,'Owner release identity mismatch')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):require(os.environ.get(key)=='2','Thread bound missing')
    def check():
        require(kernel.read(owner['pid'])==owner_identity,'Owner identity changed')
        require(kernel.read(gate['watcher_pid'])==tuple(gate['watcher_identity']),'Pressure watcher stopped')
        require(kernel.read(gate['deadline_pid'])==tuple(gate['deadline_identity']),'Independent deadline watcher stopped')
        before_deadline(a.deadline);disk(Path(a.output_model).parent)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=2**30,'Builder RSS exceeds1GiB')
    result={'status':'INCOMPLETE','model_forwards':0}
    try:
        check();source_sha=verify_packet()
        if a.guard_control:
            payload=bytearray(1024*1024);end=time.monotonic()+1
            while a.guard_control=='hold' or time.monotonic()<end:
                check();time.sleep(.1)
            result.update(status='PASS_GUARD_CONTROL_NOT_NUMERICAL',control_mode=a.guard_control,
                          packaging_source_seal_sha256=source_sha,model_revision=None,
                          numerical_calls=0,allocated_control_bytes=len(payload))
            return
        import mlx.core as mx
        import numpy as np
        import main5_builder
        result=main5_builder.build(a.base_model,a.original_model,a.output_model,mx=mx,np=np,check=check,
            report=lambda value:atomic(Path(a.receipt).with_name('BUILD_PROGRESS.json'),value))
        check();require(verify_packet()==source_sha,'Packaging source changed during build')
        result.update(packaging_source_seal_sha256=source_sha,actual_runtime_versions={name:importlib.metadata.version(name) for name in ('mlx','numpy')},actual_child_identity=kernel.read(os.getpid()),fresh_install_proof=None)
    except BaseException as error:
        result.update(status='FAILED_RETAINED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        result.update(maxrss_kernel_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(a.receipt,result)
if __name__=='__main__':main()

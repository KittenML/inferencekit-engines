"""Independent task-only deadline/orphan cleanup; exact native PID births."""
import argparse,datetime,json,os,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--guard-runtime',required=True);p.add_argument('--owner-manifest',required=True);p.add_argument('--registry',required=True);p.add_argument('--ready',required=True);p.add_argument('--deadline',required=True);p.add_argument('--pressure-file',required=True);p.add_argument('--pressure-ready',required=True);a=p.parse_args()
    runtime,_=guard_tools(a.guard_runtime);sys.path.insert(0,str(runtime));from memory_guard import KernelIdentity,VerifiedTask
    m=load(a.owner_manifest);owner=VerifiedTask(m['pid'],a.owner_manifest);kernel=KernelIdentity()
    atomic(a.ready,{'pid':os.getpid(),'kernel_identity':kernel.read(os.getpid()),'owner_identity':owner.identity,'deadline':a.deadline})
    while kernel.read(owner.pid)==owner.identity and datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(a.deadline):
        try:pressure_fresh(a.pressure_file,load(a.pressure_ready))
        except (OSError,ValueError,KeyError):break
        time.sleep(.25)
    rows=load(a.registry)['children']
    def live(row):return row['pid']!=os.getpid() and kernel.read(row['pid'])==tuple(row['kernel_identity']) and row['kernel_identity'][1]==os.getuid()
    def send(row,sig):
        if live(row):
            try:require(os.getpgid(row['pid'])==row['pid'],'Unowned process group');os.killpg(row['pid'],sig)
            except ProcessLookupError:pass
    active=[r for r in rows if live(r)]
    for row in active:send(row,signal.SIGTERM)
    owner.send(signal.SIGTERM);until=time.monotonic()+3
    while time.monotonic()<until and any(live(r) for r in active):time.sleep(.025)
    for row in active:send(row,signal.SIGKILL)
    if kernel.read(owner.pid)==owner.identity:owner.send(signal.SIGKILL)
    atomic(Path(a.ready).with_name('DEADLINE_CLEANUP.json'),{'expired_or_orphaned':True,'remaining_verified_children':[r for r in active if live(r)]})
if __name__=='__main__':main()

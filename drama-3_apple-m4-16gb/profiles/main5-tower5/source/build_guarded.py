"""Foreground portable Tower5 supervisor; no private predecessor or fixed date.

Original pressure policy/watch source and tested Popen cleanup are reused.
"""
import argparse,fcntl,os,platform,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_common import *

def main():
    p=argparse.ArgumentParser()
    for name in ('python','guard-runtime','base-model','original-model','output-model','receipt-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--guard-control',choices=('normal','hold'))
    until=p.add_mutually_exclusive_group(required=True);until.add_argument('--deadline-utc');until.add_argument('--timeout-seconds',type=float)
    a=p.parse_args();require(platform.system()=='Darwin' and platform.machine()=='arm64','Native arm64 macOS required')
    end=deadline(a.deadline_utc,a.timeout_seconds);source_sha=verify_packet()
    python=selected_python(a.python)
    runtime,_=guard_tools(a.guard_runtime);sys.path.insert(0,str(runtime))
    sys.path.insert(1,str(HERE/'guard-source'))
    from memory_guard import KernelIdentity,VerifiedTask,POLICY,startup_preflight
    from cleanup_owned import cleanup_owned_children
    from owned_observation import observe_owned,cleanup_with_exit_race
    base=explicit(a.base_model,exists=True);original=explicit(a.original_model,exists=True)
    require(base.is_dir() and original.is_dir() and base!=original,'Separate explicit input directories required')
    output=separate(a.output_model,base,original);work=separate(a.receipt_dir,base,original,output)
    require(output.parent.is_dir() and work.parent.is_dir(),'Output/receipt parents must exist')
    require(not output.is_relative_to(work) and not work.is_relative_to(output),'Separate model and receipt roots required')
    require(not output.is_relative_to(HERE) and not work.is_relative_to(HERE),'Generated artifacts must be outside sealed package')
    disk(output.parent);disk(work.parent)
    if not a.guard_control:require(os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize>=40265318400,'Unchanged40.265GB prebuild floor')
    work.mkdir();os.chdir(work)
    kernel=KernelIdentity();identity=kernel.read(os.getpid());require(identity and identity[1]==os.getuid(),'Native owner identity unavailable')
    locks=[];children=[];identities={};peak=0;last=None;report={'status':'INCOMPLETE','packaging_source_seal_sha256':source_sha,'deadline_utc':end,'model_forwards':0,'fresh_install_proof':None,'owner_identity':identity}
    def record(event,**fields):
        with (work/'EVENTS.jsonl').open('a') as f:f.write(json.dumps({'event':event,'time_ns':time.time_ns(),**fields},allow_nan=False)+'\n')
    def groups(pgid):return [v for v in subprocess.check_output(['ps','-axo','pid=,uid=,pgid='],text=True,timeout=.5).splitlines() if v.split()[-1]==str(pgid)]
    def cleanup():cleanup_with_exit_race(children,cleanup_owned_children,kernel.read,groups,record)
    def interrupted(n,_):raise SystemExit(128+n)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):env[key]='2'
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
    registry=work/'children-registry.json';atomic(registry,{'children':[]})
    def spawn(args,label):
        with (work/(label+'.log')).open('xb') as log:child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=work,env=env,start_new_session=True)
        children.append((child,label));birth=kernel.read(child.pid)
        require(birth and birth[1]==os.getuid(),'Child kernel identity unavailable')
        identities[label]=birth;atomic(registry,{'children':[{'pid':c.pid,'kernel_identity':identities[name],'label':name} for c,name in children]})
        record('owned_child_started',label=label,identity=birth,command=args);return child
    def ready(path,child):
        start=time.monotonic()
        while not path.exists():
            require(child.poll() is None,'Guard helper exited before readiness');before_deadline(end)
            require(time.monotonic()-start<10,'Guard readiness unavailable');time.sleep(.025)
        return load(path)
    try:
        for suffix in ('-invocation',''):
            fd=os.open('/tmp/inferencekit-drama3-m4-'+str(os.getuid())+suffix+'.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);locks.append(fd)
            require(os.fstat(fd).st_uid==os.getuid(),'UID lock ownership');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight=startup_preflight();report['pressure_preflight']=preflight
        command=shlex.split(subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True,timeout=.5).strip())
        manifest=work/'OWNER.json';atomic(manifest,{'pid':os.getpid(),'kernel_identity':identity,'command':command,'deadline_utc':end,'policy':POLICY})
        verified=VerifiedTask(os.getpid(),manifest);require(verified.identity==identity,'Owner verification changed')
        watcher=spawn([str(python),'-B',str(runtime/'memory_watch.py'),'--pid',str(os.getpid()),'--output',str(work/'PRESSURE.jsonl'),'--interval','.5','--pressure-guard-manifest',str(manifest),'--startup-ready',str(work/'pressure-ready.json')],'pressure')
        state=ready(work/'pressure-ready.json',watcher);require(state.get('ready') and state['policy']==POLICY,'Pressure policy not ready')
        require(all(state['task_proof'][key]==verified.proof[key] for key in ('pid','uid','birth_sec','birth_usec')),'Pressure owner task mismatch')
        deadline_child=spawn([str(python),'-B',str(HERE/'portable_deadline_watch.py'),'--guard-runtime',str(runtime),'--owner-manifest',str(manifest),'--registry',str(registry),'--ready',str(work/'deadline-ready.json'),'--deadline',end,'--pressure-file',str(work/'PRESSURE.jsonl'),'--pressure-ready',str(work/'pressure-ready.json')],'deadline')
        d=ready(work/'deadline-ready.json',deadline_child);require(tuple(d['owner_identity'])==identity and tuple(d['kernel_identity'])==identities['deadline'],'Deadline watcher identity')
        cmd=[str(python),'-B',str(HERE/'builder_worker.py'),'--base-model',str(base),'--original-model',str(original),'--output-model',str(output),'--guard-runtime',str(runtime),'--owner-manifest',str(manifest),'--release',str(work/'release.json'),'--receipt',str(work/'BUILDER_ACTUAL.json'),'--deadline',end]
        if a.guard_control:cmd.extend(['--guard-control',a.guard_control])
        worker=spawn(cmd,'builder')
        atomic(work/'release.json',{'owner_identity':identity,'child_identity':identities['builder'],'watcher_pid':watcher.pid,'watcher_identity':identities['pressure'],'deadline_pid':deadline_child.pid,'deadline_identity':identities['deadline']})
        while worker.poll() is None:
            before_deadline(end);disk(output.parent);disk(work)
            pressure_fresh(work/'PRESSURE.jsonl',state)
            require(kernel.read(watcher.pid)==identities['pressure'] and kernel.read(deadline_child.pid)==identities['deadline'],'Resource/deadline helper stopped')
            probe=subprocess.run([str(runtime/'memory_probe'),str(worker.pid)],capture_output=True,text=True,timeout=1)
            observation=observe_owned(worker,identities['builder'],kernel.read,probe,record)
            if observation['terminal']:break
            row=json.loads(observation['probe']);require(row['ok'],'Native builder memory probe failed')
            peak=max(peak,row['resident_bytes'],row['physical_footprint_bytes']);require(peak<=2**30,'Builder kernel footprint/RSS exceeds1GiB')
            now=time.monotonic();record('builder_memory',observation=row,gap_s=None if last is None else now-last);last=now
            time.sleep(.25)
        require(worker.wait()==0,'Builder failed; log/partial model retained')
        actual=load(work/'BUILDER_ACTUAL.json')
        if a.guard_control:
            require(actual['status']=='PASS_GUARD_CONTROL_NOT_NUMERICAL' and actual['model_revision'] is None and not output.exists(),'Guard control cannot admit a model')
        else:
            require(actual['status']=='PASS' and actual['model_revision']=='dc860f699a91fa5e30e32698aeb1ab4d5608eff03496c5d7d11bbdcfb3c895ce','Builder artifact not admitted')
        require(verify_packet()==source_sha,'Portable source changed during invocation')
        report.update(status='COMPLETE_GUARD_CONTROL_NO_NUMERICS' if a.guard_control else 'COMPLETE_EXACT_MAIN5_PROTECTED_TOWER5_BUILD_NO_MODEL_FORWARD',actual_builder_receipt_sha256=sha(work/'BUILDER_ACTUAL.json'),actual_builder_identity=identities['builder'],actual_builder_returncode=worker.returncode,source_evidence_scope='New portable reproduction/control, not old measured installation or runtime proof')
    except BaseException as error:
        report.update(status='FAILED_RETAINED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        try:
            cleanup()
            require(all(c.poll() is not None for c,_ in children),'Owned direct child did not reap')
            for c,label in children:require(kernel.read(c.pid)!=identities.get(label) and not groups(c.pid),'Owned child/group not closed')
            report['cleanup_completed']=True
        except BaseException as error:
            report.update(status='FAILED_CLEANUP',cleanup_error=str(error));raise
        finally:
            report.update(sampled_builder_peak_bytes=peak,child_exit_codes={name:c.returncode for c,name in children},actual_owner_end_ns=time.time_ns())
            atomic(work/'INVOCATION_ACTUAL.json',report)
            for fd in locks:os.close(fd)
if __name__=='__main__':main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""External macOS observations; actual intervals and probe deadlines are retained."""
import argparse
import json
import os
import platform
import signal
import time
from pathlib import Path
from memory_guard import (POLICY, PressureGuard, VerifiedTask, collect,
                          collect_essential, parse_essential, startup_preflight)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--interval',type=float,default=1)
    p.add_argument('--native-probe',default=str(Path(__file__).with_name('memory_probe')))
    p.add_argument('--pressure-guard-manifest')
    p.add_argument('--startup-ready')
    args=p.parse_args()
    if platform.system()!='Darwin' or args.interval<.25:p.error('macOS and interval at least0.25s required')
    if args.pressure_guard_manifest and not args.startup_ready:p.error('Guarded startup requires a readiness handshake')
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    guard=task=None
    term_sent=None
    if args.pressure_guard_manifest:
        try:
            preflight=startup_preflight()
            task=VerifiedTask(args.pid,args.pressure_guard_manifest)
            guard=PressureGuard(preflight['physical_memory_bytes'])
            guard.observe(preflight['observations'])
            atomic_json(args.startup_ready,{'ready':True,'time_ns':time.time_ns(),'task_proof':task.proof,
                                           'policy':POLICY,'preflight':preflight})
        except Exception as error:
            atomic_json(args.startup_ready,{'ready':False,'time_ns':time.time_ns(),
                                           'error_type':type(error).__name__,'error':str(error)})
            return
    previous_start=None
    last_thermal=0
    while running:
        if task is not None:
            if task.identity_reader(args.pid)!=task.identity:break
        else:
            try:os.kill(args.pid,0)
            except (ProcessLookupError,PermissionError):break
        row=collect_essential()
        sample_start=row['sample_monotonic_ns']/1e9
        row.update(pid=args.pid,requested_interval_s=args.interval,
                   observed_start_interval_s=sample_start-previous_start if previous_start is not None else None)
        previous_start=sample_start
        _,errors=parse_essential(row)
        row['essential_telemetry_valid']=not errors
        unhealthy=bool(errors)
        if guard is not None:
            state=guard.observe(row)
            row['guard_evaluated_monotonic_ns']=time.monotonic_ns()
            row['pressure_guard']={'policy':POLICY,**state}
            unhealthy=unhealthy or bool(state['reasons'])
            if state['triggered'] and term_sent is None:
                sent=task.send(signal.SIGTERM)
                row['pressure_guard']['action']='TERM' if sent else 'task absent or kernel identity changed'
                if sent:term_sent=time.monotonic()
            elif term_sent is not None and time.monotonic()-term_sent>=POLICY['term_to_kill_grace_s']:
                sent=task.send(signal.SIGKILL)
                row['pressure_guard']['action']='KILL' if sent else 'task absent or kernel identity changed'
        if unhealthy or term_sent is not None:
            row['optional_diagnostics']={'status':'NOT RUN','reason':'essential telemetry unhealthy or task stopping'}
            for key in ('ps_rss_kib_vsz_kib_elapsed_state','native_proc_rusage','thermal'):
                row[key]={'status':'NOT RUN','reason':'essential telemetry evaluated first; optional diagnostic skipped'}
        else:
            timeout=POLICY['optional_probe_timeout_s']
            row['ps_rss_kib_vsz_kib_elapsed_state'],ps_s=collect(['ps','-o','rss=,vsz=,etime=,stat=','-p',str(args.pid)],timeout)
            durations={'ps':ps_s}
            if Path(args.native_probe).is_file():
                native,durations['native_rusage']=collect([args.native_probe,str(args.pid)],timeout)
                if isinstance(native,str):
                    try:native=json.loads(native)
                    except ValueError:native={'status':'UNPROVEN','error':'invalid native JSON'}
                row['native_proc_rusage']=native
            else:row['native_proc_rusage']={'status':'NOT RUN','reason':'native helper absent'}
            if time.monotonic()-last_thermal>=5:
                row['thermal'],durations['thermal']=collect(['pmset','-g','therm'],timeout)
                row['thermal_observed_time_ns']=time.time_ns();last_thermal=time.monotonic()
            else:row['thermal']={'status':'NOT RUN','reason':'thermal observation not due'}
            row['optional_probe_durations_s']=durations
        row['sample_finished_monotonic_ns']=time.monotonic_ns()
        row['sample_duration_s']=row['sample_finished_monotonic_ns']/1e9-sample_start
        with target.open('a') as output:output.write(json.dumps(row)+'\n')
        time.sleep(max(0,args.interval-(time.monotonic()-sample_start)))


if __name__=='__main__':main()

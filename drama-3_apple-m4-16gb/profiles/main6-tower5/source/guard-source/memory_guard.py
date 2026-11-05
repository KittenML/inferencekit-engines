# SPDX-License-Identifier: Apache-2.0
"""Time-bounded telemetry and task-only pressure intervention for native trials."""
import concurrent.futures
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

POLICY = {'version': 2, 'critical_free_percent': 2, 'critical_wired_compressed_fraction': .92,
          'max_swap_growth_bytes': 8*2**30, 'minimum_disk_available_bytes': 32*2**30,
          'minimum_critical_samples': 3, 'critical_persistence_s': 5,
          'maximum_telemetry_blind_s': 5, 'essential_probe_timeout_s': .5,
          'optional_probe_timeout_s': .2, 'ownership_probe_timeout_s': .5,
          'term_to_kill_grace_s': 5}


def collect(argv, timeout_s):
    start=time.monotonic()
    try:
        value=subprocess.check_output(argv,stderr=subprocess.DEVNULL,timeout=timeout_s,text=True).strip()
    except (OSError,subprocess.SubprocessError) as error:
        value={'status':'UNPROVEN','error_type':type(error).__name__,'timeout_s':timeout_s}
    return value,time.monotonic()-start


def collect_essential():
    row={'time_ns':time.time_ns(),'sample_monotonic_ns':time.monotonic_ns()}
    commands={'vm_stat':['vm_stat'],'swap':['sysctl','-n','vm.swapusage'],
              'pressure':['memory_pressure','-Q']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={name:pool.submit(collect,argv,POLICY['essential_probe_timeout_s']) for name,argv in commands.items()}
        durations={}
        for name,future in futures.items():row[name],durations[name]=future.result()
    try:
        disk=os.statvfs('.')
        row['task_disk_available_bytes']=disk.f_bavail*disk.f_frsize
    except OSError:row['task_disk_available_bytes']=None
    row['essential_probe_durations_s']=durations
    row['essential_finished_monotonic_ns']=time.monotonic_ns()
    return row


def parse_essential(row):
    values,errors={},[]
    for key,pattern in [('page_bytes',r'page size of (\d+) bytes'),('wired_pages',r'Pages wired down:\s+(\d+)\.'),
                        ('compressed_pages',r'Pages occupied by compressor:\s+(\d+)\.')]:
        raw=row.get('vm_stat');match=re.search(pattern,raw) if isinstance(raw,str) else None
        if match:values[key]=int(match[1])
        else:errors.append(key)
    raw=row.get('pressure');match=re.search(r'memory free percentage:\s+(\d+)%',raw) if isinstance(raw,str) else None
    if match and 0<=int(match[1])<=100:values['free_percent']=int(match[1])
    else:errors.append('free_percent')
    raw=row.get('swap');match=re.search(r'used\s*=\s*([0-9.]+)([MGT])',raw) if isinstance(raw,str) else None
    if match:
        try:
            used=float(match[1])*1024**{'M':2,'G':3,'T':4}[match[2]]
            if not math.isfinite(used):raise ValueError('nonfinite swap')
            values['swap_bytes']=used
        except ValueError:errors.append('swap_bytes')
    else:errors.append('swap_bytes')
    disk=row.get('task_disk_available_bytes')
    if type(disk) is int and disk>=0:values['disk_available_bytes']=disk
    else:errors.append('disk_available_bytes')
    return values,errors


class PressureGuard:
    def __init__(self,physical_bytes):
        if type(physical_bytes) is not int or physical_bytes<=0:raise ValueError('Observed physical memory is required')
        self.physical_bytes=physical_bytes
        self.initial_swap=None
        self.critical_since=None
        self.blind_since=None
        self.consecutive=0

    def observe(self,row,now_s=None):
        now=time.monotonic() if now_s is None else now_s
        values,errors=parse_essential(row)
        reasons=[]
        if errors:
            if self.blind_since is None:self.blind_since=now
            reasons.append('essential_telemetry_missing_or_unparseable')
        else:
            self.blind_since=None
            if self.initial_swap is None:self.initial_swap=values['swap_bytes']
            if values['swap_bytes']-self.initial_swap>POLICY['max_swap_growth_bytes']:reasons.append('swap_growth_exceeded')
            unreclaimable=(values['wired_pages']+values['compressed_pages'])*values['page_bytes']
            if values['free_percent']<=POLICY['critical_free_percent'] and unreclaimable>self.physical_bytes*POLICY['critical_wired_compressed_fraction']:
                reasons.append('critical_unreclaimable_pressure')
            if values['disk_available_bytes']<POLICY['minimum_disk_available_bytes']:reasons.append('task_filesystem_space_exceeded')
            if reasons:
                self.consecutive+=1
                if self.critical_since is None:self.critical_since=now
            else:self.consecutive=0;self.critical_since=None
        blind=now-self.blind_since if self.blind_since is not None else 0
        critical=now-self.critical_since if self.critical_since is not None else 0
        triggered=(blind>=POLICY['maximum_telemetry_blind_s'] or
                   self.consecutive>=POLICY['minimum_critical_samples'] and critical>=POLICY['critical_persistence_s'])
        return {'telemetry_valid':not errors,'telemetry_errors':errors,'values':values,
                'triggered':triggered,'consecutive_critical_observations':self.consecutive,
                'telemetry_blind_s':blind,'critical_state_s':critical,'reasons':reasons,
                'initial_swap_bytes':self.initial_swap}


class KernelIdentity:
    def __init__(self):
        self.library=ctypes.CDLL(str(Path(__file__).with_name('process_identity.dylib')))
        self.function=self.library.ik_process_identity
        self.function.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.function.restype=ctypes.c_int

    def read(self,pid):
        values=(ctypes.c_uint64*4)()
        if not self.function(pid,values):return None
        return tuple(int(v) for v in values)


class VerifiedTask:
    def __init__(self,pid,manifest_path,identity_reader=None):
        if type(pid) is not int or pid<=1:raise ValueError('Invalid task PID')
        self.pid=pid
        self.identity_reader=identity_reader or KernelIdentity().read
        first=self.identity_reader(pid)
        if not first or first[0]!=pid or first[1]!=os.getuid():raise ValueError('Kernel task identity is not owned')
        manifest=json.loads(Path(manifest_path).read_text())
        if not isinstance(manifest,dict) or manifest.get('pid')!=pid:raise ValueError('Task manifest PID mismatch')
        timeout=POLICY['ownership_probe_timeout_s']
        raw=subprocess.check_output(['ps','-ww','-p',str(pid),'-o','uid=,args='],text=True,timeout=timeout).strip().split(None,1)
        if len(raw)!=2 or int(raw[0])!=os.getuid() or shlex.split(raw[1])!=manifest.get('command'):raise ValueError('Task argv or UID mismatch')
        raw=subprocess.check_output(['lsof','-a','-p',str(pid),'-d','cwd','-Fn'],text=True,timeout=timeout)
        if not any(Path(line[1:]).resolve()==Path.cwd().resolve() for line in raw.splitlines() if line.startswith('n')):raise ValueError('Task cwd mismatch')
        if self.identity_reader(pid)!=first:raise ValueError('Task identity changed during verification')
        self.identity=first
        self.proof={'pid':pid,'uid':first[1],'birth_sec':first[2],'birth_usec':first[3],
                    'source':'proc_pidinfo(PROC_PIDTBSDINFO)','argv_sha256':hashlib.sha256(json.dumps(manifest['command'],separators=(',',':')).encode()).hexdigest(),
                    'task_cwd_verified':True,'startup_argv_verified':True}

    def send(self,requested_signal):
        if self.identity_reader(self.pid)!=self.identity:return False
        try:os.kill(self.pid,requested_signal);return True
        except OSError:return False


def startup_preflight():
    physical,elapsed=collect(['sysctl','-n','hw.memsize'],POLICY['essential_probe_timeout_s'])
    if not isinstance(physical,str) or not physical.isdigit():raise RuntimeError('Physical memory telemetry unavailable')
    identity=KernelIdentity().read(os.getpid())
    if not identity or identity[1]!=os.getuid():raise RuntimeError('Native process-identity probe unavailable')
    row=collect_essential();state=PressureGuard(int(physical)).observe(row)
    if not state['telemetry_valid']:raise RuntimeError('Essential pressure telemetry unavailable: '+','.join(state['telemetry_errors']))
    if state['reasons']:raise RuntimeError('Host already exceeds physical-pressure guard policy')
    return {'physical_memory_bytes':int(physical),'physical_probe_s':elapsed,'observations':row,'state':state,'policy':POLICY}

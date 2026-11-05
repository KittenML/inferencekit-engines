"""Guard the optional exact Tower5 runtime; --smoke runs one fixed public request.

Wrapper/reproduction identity is separate from the untouched measured runtime.
No fresh-install, comparative speed, absolute quality or listening claim follows.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import wave

SOURCE = 'fc511f01bfd0f5d6b1b384ee8cea783cc636b22454e348feecff01e11577862a'
MODEL = '93432d8a6c97faba88309880c3fd2d1b8356bde679dc20f9e0daa57282eb904c'
RUNTIME = 'bafe948ad8395d981a83269af5d5f16f1e991635c5378da95e054279c8cb83cc'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_info(info):
    resolved = info['resolved']
    assert info['ready'] is True
    assert resolved['model_revision'] == MODEL
    assert resolved['runtime_recipe_sha256'] == RUNTIME
    defaults = resolved['generation_defaults']
    assert (defaults['temperature'], defaults['top_p'], defaults['top_k'], defaults['max_tokens']) == (1.0, .9, 4097, 1024)
    return resolved


def validate_response(rows, actual, request_id='screen_short', seed=17):
    assert len(rows) == 1
    row = rows[0]
    assert row['ok'] is True and row['id'] == actual['id'] == request_id
    assert row['seed'] == actual['seed'] == seed and row['request_index'] == actual['request_index'] == 0
    assert actual['generation_status'] == 'EOS_OBSERVED' and actual['source_receipt_unique'] is True
    assert actual['transport'] == row and actual['pcm_sha256'] == row['pcm_sha256']
    assert row['sample_rate'] == 44100 and row['channels'] == 1 and row['sample_count'] > 0
    assert row['pcm_bytes'] == 2 * row['sample_count']
    event = actual['generation_observation']
    assert event['runtime_recipe_sha256'] == RUNTIME and event['all_segments_eos'] is True and event['termination_frame_counts_match'] is True
    assert row['effective']['model_revision'] == MODEL
    return row


def validate_restoration(implementation, events):
    stopped = [row for row in events if row['event'] == 'process_stopped']
    assert len(stopped) == 1 and stopped[0]['per_process_limits_restored']
    helpers = implementation['final_helpers']
    assert len(helpers) == 1
    assert all(row['method_hooks_restored'] and row['parameter_objects_shapes_dtypes_unchanged']
               and row['all_decode_invocations_accounted'] and row['trace_counts_valid'] for row in helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile-source','runtime','model-dir','python','guard-runtime','receipt-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--client-python')
    parser.add_argument('--smoke', action='store_true')
    until = parser.add_mutually_exclusive_group(required=True)
    until.add_argument('--deadline-utc')
    until.add_argument('--timeout-seconds', type=float)
    a = parser.parse_args()
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    source = Path(a.profile_source).expanduser()
    assert source.is_absolute() and '..' not in source.parts and not source.is_symlink()
    source = source.resolve(strict=True)
    assert file_sha(source/'SOURCE_SEAL.json') == SOURCE
    sealed = json.loads((source/'SOURCE_SEAL.json').read_text())
    for name in ('portable_common.py','portable_deadline_watch.py'):
        assert file_sha(source/name) == sealed['files'][name]['sha256']
    sys.path.insert(0, str(source))
    from portable_common import (atomic, before_deadline, deadline, disk, explicit,
        guard_tools, load, pressure_fresh, require, selected_python, separate,
        verify_measured, verify_packet)
    assert verify_packet() == SOURCE
    runtime = explicit(a.runtime, exists=True)
    verify_measured(runtime)
    model = explicit(a.model_dir, exists=True)
    assert json.loads((model/'origin.json').read_text())['revision'] == MODEL
    assert file_sha(model/'origin.json') == 'a47ca84ca752d73d07b311ba575e0b8a2817a8751cc50238198f0c7f1cda46ce'
    python = selected_python(a.python)
    client_python = selected_python(a.client_python or a.python)
    native, tools = guard_tools(a.guard_runtime)
    work = separate(a.receipt_dir, source, runtime, model, native)
    assert work.parent.is_dir()
    end = deadline(a.deadline_utc, a.timeout_seconds)
    disk(work.parent)
    # Every full-model load gets the same original cold-load floor.
    assert os.statvfs(work.parent).f_bavail * os.statvfs(work.parent).f_frsize >= 38923141120
    work.mkdir()
    os.chdir(work)
    sys.path.insert(0, str(native))
    sys.path.insert(1, str(source/'guard-source'))
    from memory_guard import KernelIdentity, VerifiedTask, POLICY, startup_preflight
    from cleanup_owned import cleanup_owned_children
    from owned_observation import observe_owned, cleanup_with_exit_race
    kernel = KernelIdentity()
    identity = kernel.read(os.getpid())
    assert identity and identity[1] == os.getuid()
    children, identities, locks, peaks = [], {}, [], {}
    report = {'status':'INCOMPLETE', 'adapter_sha256':file_sha(__file__),
        'packaging_source_seal_sha256':SOURCE, 'measured_runtime_manifest_sha256':file_sha(runtime/'RUNTIME_MANIFEST.json'),
        'model_origin_sha256':file_sha(model/'origin.json'), 'expected_model_revision':MODEL,
        'expected_runtime_recipe_sha256':RUNTIME, 'deadline_utc':end,
        'scope':'Guarded runtime/smoke only; no fresh install, speed, quality, listening or reliability promotion.',
        'smoke_requested':a.smoke, 'owner_identity':identity}

    def record(event, **fields):
        with (work/'EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps({'event':event,'time_ns':time.time_ns(),**fields},allow_nan=False)+'\n')

    def groups(pgid):
        return [row for row in subprocess.check_output(['ps','-axo','pid=,uid=,pgid='],text=True,timeout=.5).splitlines()
                if row.split()[-1] == str(pgid)]

    def cleanup():
        cleanup_with_exit_race(children,cleanup_owned_children,kernel.read,groups,record)

    def interrupted(number, frame):
        raise SystemExit(128+number)

    for number in (signal.SIGTERM,signal.SIGINT):
        signal.signal(number,interrupted)
    env = {key:os.environ[key] for key in ('PATH','HOME','TMPDIR','LANG','LC_ALL') if key in os.environ}
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        env[key] = '2'
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
    registry = work/'children-registry.json'
    atomic(registry,{'children':[]})

    def spawn(command, label, cwd):
        with (work/(label+'.log')).open('xb') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=cwd,env=env,start_new_session=True)
        children.append((child,label))
        birth = kernel.read(child.pid)
        assert birth and birth[1] == os.getuid()
        identities[label] = birth
        atomic(registry,{'children':[{'pid':item.pid,'kernel_identity':identities[name],'label':name} for item,name in children]})
        record('owned_child_started',label=label,identity=birth,command=command,cwd=str(cwd))
        return child

    def ready(path, child):
        began = time.monotonic()
        while not path.exists():
            assert child.poll() is None, 'Guard helper exited before readiness'
            before_deadline(end)
            assert time.monotonic()-began < 10, 'Guard readiness unavailable'
            time.sleep(.025)
        return load(path)

    def check():
        before_deadline(end)
        disk(work)
        pressure_fresh(work/'PRESSURE.jsonl',pressure_state)
        assert kernel.read(watcher.pid) == identities['pressure']
        assert kernel.read(deadline_child.pid) == identities['deadline']

    def observe(child, label, ceiling):
        probe = subprocess.run([str(native/'memory_probe'),str(child.pid)],capture_output=True,text=True,timeout=1)
        observation = observe_owned(child,identities[label],kernel.read,probe,record)
        if observation['terminal']:
            return False
        row = json.loads(observation['probe'])
        assert row['ok']
        peaks[label] = max(peaks.get(label,0),row['resident_bytes'],row['physical_footprint_bytes'])
        assert peaks[label] <= ceiling
        record('sampled_child_memory',label=label,observation=row)
        return True

    try:
        for suffix in ('-invocation',''):
            fd = os.open('/tmp/inferencekit-drama3-m4-'+str(os.getuid())+suffix+'.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            locks.append(fd)
            assert os.fstat(fd).st_uid == os.getuid()
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with socket.socket() as port:
            port.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            port.bind(('127.0.0.1',8765))
        preflight = startup_preflight()
        report['pressure_preflight'] = preflight
        command = shlex.split(subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True,timeout=.5).strip())
        owner = work/'OWNER.json'
        atomic(owner,{'pid':os.getpid(),'kernel_identity':identity,'command':command,'deadline_utc':end,'policy':POLICY})
        verified = VerifiedTask(os.getpid(),owner)
        assert verified.identity == identity
        watcher = spawn([str(python),'-B',str(native/'memory_watch.py'),'--pid',str(os.getpid()),'--output',str(work/'PRESSURE.jsonl'),
                         '--interval','.5','--pressure-guard-manifest',str(owner),'--startup-ready',str(work/'pressure-ready.json')], 'pressure',work)
        pressure_state = ready(work/'pressure-ready.json',watcher)
        assert pressure_state.get('ready') and pressure_state['policy'] == POLICY
        assert all(pressure_state['task_proof'][key] == verified.proof[key] for key in ('pid','uid','birth_sec','birth_usec'))
        deadline_child = spawn([str(python),'-B',str(source/'portable_deadline_watch.py'),'--guard-runtime',str(native),
            '--owner-manifest',str(owner),'--registry',str(registry),'--ready',str(work/'deadline-ready.json'),
            '--deadline',end,'--pressure-file',str(work/'PRESSURE.jsonl'),'--pressure-ready',str(work/'pressure-ready.json')], 'deadline',work)
        state = ready(work/'deadline-ready.json',deadline_child)
        assert tuple(state['owner_identity']) == identity and tuple(state['kernel_identity']) == identities['deadline']
        check()
        server_command = [str(python),'-B',str(runtime/'serve_selected.py'),'--model-dir',str(model),
            '--implementation-receipt',str(work/'IMPLEMENTATION.json'),'--telemetry',str(work/'RUNTIME.jsonl'),
            '--startup-warmup-receipt',str(work/'WARMUP.json'),'--port','8765']
        server = spawn(server_command,'server',runtime)
        while True:
            check()
            assert observe(server,'server',preflight['physical_memory_bytes']), 'Server exited before API admission'
            try:
                with urllib.request.urlopen('http://127.0.0.1:8765/info',timeout=1) as response:
                    info = json.load(response)
            except (OSError,urllib.error.URLError,TimeoutError):
                time.sleep(.25)
                continue
            validate_info(info)
            atomic(work/'INFO.json',info)
            break
        warmup = load(work/'WARMUP.json')
        assert warmup['source_EOS'] and warmup['attempts'] == 1 and warmup['sample_count'] > 0
        report['actual_resolved'] = info['resolved']
        record('actual_API_admission',model_revision=MODEL,runtime_recipe_sha256=RUNTIME,server_identity=identities['server'])
        print(json.dumps({'event':'guarded_API_ready','pid':server.pid,'model_revision':MODEL,'runtime_recipe_sha256':RUNTIME}),flush=True)
        if a.smoke:
            fixture_source = runtime/'fixtures/encoder-release-public8-v14-bench.jsonl'
            fixture = next(row for line in fixture_source.read_text().splitlines() if (row:=json.loads(line))['id']=='screen_short')
            assert fixture['text'] == 'The copper latch clicked twice before the door opened.'
            fixture['reference_audio'] = str(runtime/'conditioning-cache-v2/reference.wav')
            assert file_sha(fixture['reference_audio']) == 'e00d65ba8fa036139d361c0a2133a20231e68f0c722952da7b76c96e5897a970'
            fixture_path, plan_path = work/'SMOKE_FIXTURE.jsonl', work/'SMOKE_PLAN.jsonl'
            fixture_path.write_text(json.dumps(fixture)+'\n')
            plan = {'request_index':0,'id':'screen_short','text_sha256':hashlib.sha256(fixture['text'].encode()).hexdigest(),'seed':17}
            plan_path.write_text(json.dumps(plan)+'\n')
            atomic(work/'SMOKE_PREDECLARATION.json',{'source_fixture_sha256':file_sha(fixture_source),'fixture_sha256':file_sha(fixture_path),
                'plan_sha256':file_sha(plan_path),'request':plan,'settings':info['resolved']['generation_defaults'],
                'scope':'One predeclared public smoke; no timing/quality/CI promotion.'})
            assert file_sha(runtime/'registered_client.py') == '450a95b355d7343136dbd67c1047713d8c3d92eef8a4e6eccd2b122641a9dd0c'
            client = spawn([str(client_python),'-B',str(runtime/'registered_client.py'),
                '--fixtures',str(fixture_path),'--fixtures-sha256',file_sha(fixture_path),
                '--plan',str(plan_path),'--plan-sha256',file_sha(plan_path),
                '--output',str(work/'client'),'--server-telemetry',str(work/'RUNTIME.jsonl'),
                '--expected-runtime-sha256',RUNTIME,'--timeout-s','900','--post-request-quiescence-s','0'], 'client',runtime)
            while client.poll() is None:
                check()
                assert observe(server,'server',preflight['physical_memory_bytes'])
                observe(client,'client',2**30)
                time.sleep(.25)
            assert client.wait() == 0, 'Client failed; retain every output'
            rows = [json.loads(line) for line in (work/'client/requests.jsonl').read_text().splitlines()]
            actual = load(work/'client/request-0000.json')
            validate_response(rows,actual)
            pcm = work/'client/request-0000.pcm'
            wav = work/'client/request-0000.wav'
            assert file_sha(pcm) == rows[0]['pcm_sha256'] and pcm.stat().st_size == rows[0]['pcm_bytes']
            with wave.open(str(wav),'rb') as audio:
                assert audio.getparams()[:3] == (1,2,44100)
                assert audio.getnframes() == rows[0]['sample_count']
                assert hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest() == rows[0]['pcm_sha256']
            report['smoke'] = {'requests':1,'id':'screen_short','seed':17,'generation_status':actual['generation_status'],
                'pcm_sha256':file_sha(pcm),'wav_sha256':file_sha(wav),'sample_count':rows[0]['sample_count'],
                'quality':'UNPROVEN','performance':'SMOKE_ONLY_NOT_COMPARATIVE'}
        else:
            while server.poll() is None:
                check()
                observe(server,'server',preflight['physical_memory_bytes'])
                time.sleep(.25)
        report['status'] = 'SMOKE_REQUEST_COMPLETE_CLEANUP_PENDING' if a.smoke else 'SERVING_STOPPED_CLEANUP_PENDING'
    except BaseException as error:
        report.update(status='FAILED_RETAINED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        try:
            cleanup()
            assert all(child.poll() is not None for child,label in children)
            assert all(kernel.read(child.pid) != identities[label] and not groups(child.pid) for child,label in children)
            report['cleanup_completed'] = True
            if 'server' in identities:
                events = [json.loads(line) for line in (work/'RUNTIME.jsonl').read_text().splitlines()]
                implementation = load(work/'IMPLEMENTATION.json')
                validate_restoration(implementation,events)
                report.update(per_process_limits_restored=True,implementation_receipt_sha256=file_sha(work/'IMPLEMENTATION.json'))
            verify_measured(runtime)
            assert verify_packet() == SOURCE
            if report['status'] == 'SMOKE_REQUEST_COMPLETE_CLEANUP_PENDING':
                report['status'] = 'PASS_GUARDED_EXISTING_INPUT_PROFILE_SMOKE'
        except BaseException as error:
            report.update(status='FAILED_CLEANUP_OR_RESTORATION',cleanup_error=str(error))
            raise
        finally:
            report.update(sampled_child_peaks=peaks,child_exit_codes={label:child.returncode for child,label in children},actual_owner_end_ns=time.time_ns())
            atomic(work/'INVOCATION_ACTUAL.json',report)
            for fd in locks:
                os.close(fd)


if __name__ == '__main__':
    main()

"""Portable source/path/receipt contracts; stdlib only."""
import datetime,hashlib,json,math,os,shutil,stat,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def explicit(path,*,exists=False):
    p=Path(path).expanduser()
    require(p.is_absolute() and '..' not in p.parts,'Explicit absolute path without parent traversal required')
    require(not p.is_symlink(),'Root symlink refused')
    return p.resolve(strict=exists)
def selected_python(path,*,exists=True):
    # A venv executable is normally a symlink. Resolving it for invocation
    # would discard venv discovery and silently select the base interpreter.
    p=Path(path).expanduser()
    require(p.is_absolute() and '..' not in p.parts,'Explicit Python executable required')
    if exists:require(p.is_file() and os.access(p,os.X_OK),'Selected Python executable unavailable')
    return p.absolute()
def separate(output,*inputs):
    output=explicit(output)
    require(not output.exists(),'Fresh output required')
    require(all(not output.is_relative_to(p) for p in inputs),'Output is inside a protected input')
    return output
def disk(path):
    require(shutil.disk_usage(path).free>=32*2**30,'32GiB task filesystem disk floor')
def deadline(utc_value=None,seconds=None):
    require((utc_value is None)!=(seconds is None),'One finite deadline/timeout required')
    now=datetime.datetime.now(datetime.timezone.utc)
    if seconds is not None:
        require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0,'Finite positive timeout required')
        target=now+datetime.timedelta(seconds=seconds)
    else:
        target=datetime.datetime.fromisoformat(utc_value.replace('Z','+00:00'))
        require(target.tzinfo is not None,'Deadline timezone required')
        target=target.astimezone(datetime.timezone.utc)
    require(target>now,'Deadline is expired')
    return target.isoformat()
def before_deadline(value):
    require(datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(value),'Caller deadline reached')
def pressure_fresh(path,ready):
    stamp=ready['preflight']['observations']['essential_finished_monotonic_ns']
    path=Path(path)
    if path.exists() and path.stat().st_size:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-65536));tail=f.read(65536)
        lines=tail.split(b'\n')[:-1]
        if lines:
            row=json.loads(lines[-1]);stamp=row['sample_finished_monotonic_ns']
            require(not row.get('pressure_guard',{}).get('triggered',False),'Pressure guard triggered')
    require(0<=time.monotonic_ns()-stamp<=5*10**9,'Pressure telemetry blind for5s')
def verify_measured(root=None):
    root=HERE/'measured' if root is None else Path(root)
    require(sha(root/'RUNTIME_MANIFEST.json')=='f56ff35852da5c4fbd1396049a0e7324bce652c39555662b3845bb1bc7a423a7','Measured manifest changed')
    rows=load(root/'RUNTIME_MANIFEST.json')['files'];require(len(rows)==36,'Complete runtime subset required')
    for name,spec in rows.items():
        p=root/name;require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root.resolve()),'Measured member path')
        require(p.stat().st_size==spec['bytes'] and sha(p)==spec['sha256'] and (stat.S_IMODE(p.stat().st_mode)&0o111)==(spec['mode']&0o111),'Measured byte/executable-bit changed: '+name)
    return rows
def verify_packet():
    receipt=load(HERE/'SOURCE_SEAL.json')
    for name,spec in receipt['files'].items():
        p=HERE/name;require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(HERE),'Source member path')
        require(sha(p)==spec['sha256'] and (stat.S_IMODE(p.stat().st_mode)&0o111)==(spec['mode']&0o111),'Portable byte/executable-bit changed: '+name)
    verify_measured();return sha(HERE/'SOURCE_SEAL.json')
def guard_tools(root):
    root=explicit(root,exists=True);receipt=load(root/'GUARD_TOOLS_ACTUAL.json')
    require(receipt['status']=='ACTUAL_NATIVE_DARWIN_GUARD_TOOLS_BUILT','Native guard build receipt required')
    for name,digest in receipt['files'].items():require(not (root/name).is_symlink() and sha(root/name)==digest,'Native guard member changed')
    for name in ('memory_guard.py','memory_watch.py','memory_probe.c','process_identity.c'):
        require(sha(root/name)==sha(HERE/'guard-source'/name),'Guard source differs')
    require(set(receipt['files'])=={'memory_guard.py','memory_watch.py','memory_probe.c','process_identity.c','process_identity.dylib','memory_probe'},'Complete guard tool inventory required')
    return root,receipt

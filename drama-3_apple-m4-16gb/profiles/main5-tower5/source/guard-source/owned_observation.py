"""Handle owned Popen normal-exit races without admitting live identity/probe failures."""
import subprocess

def observe_owned(child,birth,identity_read,probe,record):
 code=child.poll()
 if code is not None:return {'terminal':True,'returncode':code}
 identity=identity_read(child.pid)
 if identity==birth and probe.returncode==0:return {'terminal':False,'probe':probe.stdout}
 # task_info/proc_pidinfo may stop exposing a process between poll and waitpid.
 # Only our actual Popen.wait terminal result can resolve this as normal exit.
 try:code=child.wait(timeout=.25)
 except subprocess.TimeoutExpired:
  if identity_read(child.pid)!=birth:raise RuntimeError('Live owned child identity changed or unavailable')
  raise RuntimeError('Live owned child memory probe failed')
 record('normal_exit_observation_race_reaped',pid=child.pid,returncode=code,observed_identity=identity,probe_returncode=probe.returncode)
 return {'terminal':True,'returncode':code}

def cleanup_with_exit_race(children,cleanup,identity_read,group_members,record):
 try:return cleanup(children,record)
 except PermissionError as original:
  # Do not suppress a permission failure against an owned live task/group.
  for child,_ in children:
   try:child.wait(timeout=.25)
   except subprocess.TimeoutExpired:raise original
   if identity_read(child.pid) is not None or group_members(child.pid):raise original
  record('cleanup_permission_normal_exit_race_all_groups_verified_empty',children=[c.pid for c,_ in children])

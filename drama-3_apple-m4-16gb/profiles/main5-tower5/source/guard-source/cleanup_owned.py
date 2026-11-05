import os,signal,subprocess,time
def cleanup_owned_children(children,record,grace_s=3.0):
 """Broadcast before waiting; finish inside the outer guard's five-second grace."""
 started=time.monotonic();live=[(child,label) for child,label in children if child.poll() is None];events=[]
 for child,label in live:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  events.append(('cleanup_TERM_broadcast',{'label':label,'pid':child.pid,'ownership':'Unreaped task-created Popen private process group'}))
 deadline=started+grace_s
 while time.monotonic()<deadline and any(child.poll() is None for child,_ in live):time.sleep(.025)
 for child,label in live:
  if child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   events.append(('cleanup_KILL_broadcast',{'label':label,'pid':child.pid}))
 # At most three live direct groups (server, observer, client); all have already
 # received KILL before any bounded reap. No client can delay the model's TERM.
 for child,label in live:
  remaining=started+4.0-time.monotonic()
  if remaining<=0:break
  try:child.wait(timeout=min(.25,remaining))
  except subprocess.TimeoutExpired:events.append(('cleanup_reap_unresolved_after_KILL',{'label':label,'pid':child.pid}))
 # Delay disk logging until every required signal has been sent.
 for event,values in events:record(event,**values)
 record('shared_cleanup_finished',elapsed_s=time.monotonic()-started,shared_TERM_grace_s=grace_s)

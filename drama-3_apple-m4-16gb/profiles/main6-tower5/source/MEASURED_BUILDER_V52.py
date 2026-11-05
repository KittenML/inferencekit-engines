"""Original BF16 Tower78→affine5/g64; untouchedMAIN6; one projection at a time, streamed into stored NPZ.

The recipe choice is ours. The quantization call is the unchanged public MLX
primitive used by pinned upstream ShardWriter.add. No expert/head/codec is changed.
"""
import argparse,ast,contextlib,datetime,fcntl,hashlib,json,math,os,resource,signal,socket,struct,sys,time,zipfile
from pathlib import Path

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def fsyncdir(path):
 fd=os.open(path,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def npy_header(dtype,shape):
 header=repr({'descr':dtype,'fortran_order':False,'shape':tuple(shape)}).encode('latin1');header+=b' '*(-(10+len(header)+1)%64)+b'\n';assert len(header)<65536
 return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header

@contextlib.contextmanager
def exclusive_locks():
 descriptors=[]
 try:
  for suffix in ('-invocation',''):
   fd=os.open('/tmp/inferencekit-drama3-m4-'+str(os.getuid())+suffix+'.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);descriptors.append(fd)
   assert os.fstat(fd).st_uid==os.getuid();fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield
 finally:
  for fd in descriptors:os.close(fd)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seal',required=True);p.add_argument('--owner-guard-manifest',required=True);a=p.parse_args();seal=json.loads(Path(a.seal).read_text());assert seal['status']=='SEALED_AFTER_CLOSED_V51_AND_V50_ARCHIVE_RESOURCE_BINDING' and sha(__file__)==seal['builder_sha256'];sys.path.insert(0,str(Path('package-draft').resolve()));from memory_guard import KernelIdentity,VerifiedTask,POLICY
 until=time.monotonic()+6
 while time.monotonic()<until:
  try:
   guard=json.loads(Path(a.owner_guard_manifest).read_text())
   if guard.get('memory_guard_readiness',{}).get('ready'):break
  except (OSError,ValueError):pass
  time.sleep(.025)
 else:raise RuntimeError('Outer pressure guard missing')
 assert guard['pid']==os.getpid() and guard['memory_guard_policy']==POLICY;owner=VerifiedTask(os.getpid(),a.owner_guard_manifest);k=KernelIdentity();watch=k.read(guard['memory_watcher_pid']);assert watch
 assert all(guard['memory_guard_readiness']['task_proof'][n]==owner.proof[n] for n in ('pid','uid','birth_sec','birth_usec'))
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda number,frame:(_ for _ in ()).throw(SystemExit(128+number)))
 def check():
  assert k.read(guard['memory_watcher_pid'])==watch,'Outer watcher exited'
  assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat('2026-10-26T08:00:00+00:00')
  assert os.statvfs('.').f_bavail*os.statvfs('.').f_frsize>=34359738368,'Original32GiB floor'
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=1073741824,'Builder1GiB RSS guard'
 def fullhash(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
   i=0
   while block:=f.read(1048576):
    h.update(block);i+=1
    if i%64==0:check()
  return h.hexdigest()
 for item in seal['inputs']:assert fullhash(item['path'])==item['sha256'],item['path']
 closed_spec=seal['closed_primitive'];assert fullhash(closed_spec['path'])==closed_spec['sha256'];closed=json.loads(Path(closed_spec['path']).read_text());assert closed['returncode']==0 and closed['child_closed']
 plan=json.loads(Path(seal['tensor_plan']).read_text());assert len(plan['targets'])==78 and all(t['module'].startswith('tower.') and t['group_size']==64 and t['bits']==5 for t in plan['targets']) and plan['counts']['overlay_raw_payload_bytes']==656015360
 for item in seal['prior_owner_manifests']:
  assert sha(item['path'])==item['sha256'];old=json.loads(Path(item['path']).read_text());proof=old['memory_guard_readiness']['task_proof'];assert k.read(old['pid'])!=tuple(proof[n] for n in ('pid','uid','birth_sec','birth_usec')) and k.read(old['memory_watcher_pid']) is None
 with socket.socket() as port:port.bind(('127.0.0.1',8765))
 check();assert seal['minimum_free_before_build_bytes']>=39650000000 and os.statvfs('.').f_bavail*os.statvfs('.').f_frsize>=seal['minimum_free_before_build_bytes']
 base=Path(seal['base_model']);source=Path('package-default-validation-v14-v1/models/original-fp8');dst=Path(seal['output_model']);work=Path(seal['output_evidence']);assert not dst.exists() and not work.exists();work.mkdir(parents=True);dst.mkdir(parents=True)
 assert base.resolve()!=dst.resolve() and dst.resolve().parent==Path('models').resolve() and not base.is_symlink() and not source.is_symlink();origin=json.loads((base/'origin.json').read_text());assert origin==seal['base_origin'] and json.loads((source/'origin.json').read_text())==plan['original_file_manifest'];began=time.monotonic();audit={'status':'INCOMPLETE','source_plan_sha256':sha(seal['tensor_plan']),'original_used_files':{},'base_verified_files':{},'modules':[]};previous={};obs={};mx=None
 def progress():dump(work/'BUILD_PROGRESS_V1.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modules_completed':len(audit['modules']),'last_module':audit['modules'][-1]['module'] if audit['modules'] else None,'elapsed_s':time.monotonic()-began,'disk_free_bytes':os.statvfs('.').f_bavail*os.statvfs('.').f_frsize})
 try:
  for name,identity in origin['files'].items():
   assert Path(name).name==name;path=base/name;assert path.is_file() and not path.is_symlink() and path.stat().st_size==identity['bytes'] and fullhash(path)==identity['sha256'];audit['base_verified_files'][name]=identity
   if name!='config.json':os.link(path,dst/name)
  selected={t['source']['file']:t['source'] for t in plan['targets']}
  for name,item in selected.items():
   path=Path(name);assert path.resolve().is_relative_to(source.resolve()) and path.is_file() and not path.is_symlink() and path.stat().st_size==item['file_bytes'] and fullhash(path)==item['file_sha256'];audit['original_used_files'][name]={'sha256':item['file_sha256'],'bytes':item['file_bytes'],'mtime_ns':path.stat().st_mtime_ns}
  original_config=json.loads((base/'config.json').read_text());config=json.loads(json.dumps(original_config))
  for target in plan['targets']:
   assert config['quantization'][target['module']]=={'group_size':64,'bits':6};config['quantization'][target['module']]={'group_size':64,'bits':5}
  assert all(config['quantization'][n]==v for n,v in original_config['quantization'].items() if n not in {t['module'] for t in plan['targets']});dump(dst/'config.json',config)
  # Base manifest is exact including the separate codec, without evaluating any
  # model. It also prevents the native dictionary merge hiding duplicate names.
  base_tensors={}
  for path in sorted(base.glob('*.safetensors')):
   with path.open('rb') as f:length=struct.unpack('<Q',f.read(8))[0];assert length<=20*2**20;header=json.loads(f.read(length))
   for key,item in header.items():
    if key=='__metadata__':continue
    assert key not in base_tensors;base_tensors[key]={'shape':item['shape'],'dtype':item['dtype']}
  for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='2'
  import mlx.core as mx
  import numpy as np
  import importlib.metadata
  assert importlib.metadata.version('mlx')=='0.32.4' and mx.metal.is_available() and mx.default_device()==mx.gpu
  previous['memory']=mx.set_memory_limit(536870912);previous['cache']=mx.set_cache_limit(0);previous['wired']=mx.set_wired_limit(0);members={};temp=dst/'tower5-overlay.partial';assert not temp.exists()
  with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
   for target in plan['targets']:
    check();audit['current_module']=target['module'];part=target['source'];path=Path(part['file']);before=audit['original_used_files'][str(path)];assert path.stat().st_size==before['bytes'] and path.stat().st_mtime_ns==before['mtime_ns']
    with path.open('rb') as f:
     length=struct.unpack('<Q',f.read(8))[0];header=f.read(length);assert length==part['header_bytes'] and hashlib.sha256(header).hexdigest()==part['header_sha256'];entry=json.loads(header)[part['key']];assert all(entry[n]==part[n] for n in ('shape','dtype','data_offsets')) and entry['dtype']=='BF16';start,stop=entry['data_offsets'];assert stop-start<=67108864;f.seek(8+length+start);raw=f.read(stop-start);assert len(raw)==stop-start
    raw_sha=hashlib.sha256(raw).hexdigest();host=np.frombuffer(raw,dtype='<u2').reshape(part['shape']);value=mx.view(mx.array(host),mx.bfloat16);qparams=mx.quantize(value,group_size=64,bits=5,mode='affine');mx.eval(qparams);assert qparams[0].dtype==mx.uint32 and all(v.dtype==mx.bfloat16 for v in qparams[1:]);assert list(qparams[0].shape)==target['new_weight_shape'] and all(list(v.shape)==target['new_scale_bias_shape'] for v in qparams[1:]);record={'module':target['module'],'original_BF16_payload_sha256':raw_sha,'outputs':[],'status':'STARTED'};audit['modules'].append(record)
    q6=mx.quantize(value,group_size=64,bits=6,mode='affine');mx.eval(q6)
    for suffix,array in zip(('weight','scales','biases'),q6):
     native=target['native6_tensors'][suffix];native_path=Path(native['file']);assert native_path.resolve().is_relative_to(base.resolve()) and audit['base_verified_files'][native_path.name]['sha256']==native['file_sha256']
     with native_path.open('rb') as stream:
      length=struct.unpack('<Q',stream.read(8))[0];header=stream.read(length);assert length==native['header_bytes'] and hashlib.sha256(header).hexdigest()==native['header_sha256'];spec=json.loads(header)[native['key']];assert all(spec[n]==native[n] for n in ('dtype','shape','data_offsets'));begin,end=spec['data_offsets'];stream.seek(8+length+begin);stored=stream.read(end-begin);assert len(stored)==end-begin
     observed=np.ascontiguousarray(np.asarray(array if suffix=='weight' else mx.view(array,mx.uint16)),dtype='<u4' if suffix=='weight' else '<u2');assert hashlib.sha256(memoryview(observed).cast('B')).hexdigest()==hashlib.sha256(stored).hexdigest();del stored,observed
    record['native6_all_three_raw_buffers_match_protected742']=True;record['coefficient_MSE']={}
    original_f32=value.astype(mx.float32)
    for bits,parameters in ((5,qparams),(6,q6)):
     reconstruction=mx.dequantize(*parameters,group_size=64,bits=bits,mode='affine').astype(mx.float32)
     error=mx.mean(mx.square(original_f32-reconstruction));mx.eval(error);mse=float(error.item());assert math.isfinite(mse)
     record['coefficient_MSE'][str(bits)]=mse;del reconstruction,error;mx.clear_cache()
    del q6,original_f32,parameters
    for suffix,array in zip(('weight','scales','biases'),qparams):
     key=target['module']+'.'+suffix;assert key in base_tensors and key not in members;target_dtype='U32' if suffix=='weight' else 'BF16';stored_dtype='<u4' if suffix=='weight' else '<u2';array=array if suffix=='weight' else mx.view(array,mx.uint16);mx.eval(array);data=np.ascontiguousarray(np.asarray(array),dtype=stored_dtype);header=npy_header(stored_dtype,data.shape);npyhash=hashlib.sha256(header);rawhash=hashlib.sha256();count=len(header)
     with archive.open(key+'.npy','w') as out:
      out.write(header);buffer=memoryview(data).cast('B')
      for offset in range(0,len(buffer),1048576):block=buffer[offset:offset+1048576];out.write(block);npyhash.update(block);rawhash.update(block);count+=len(block)
     members[key]={'shape':list(data.shape),'target_dtype':target_dtype,'stored_dtype':stored_dtype,'raw_le_sha256':rawhash.hexdigest(),'npy_sha256':npyhash.hexdigest(),'npy_bytes':count};record['outputs'].append(key)
    record['status']='COMPLETE';progress();del raw,host,value,qparams,array,data,buffer,block;mx.clear_cache()
  with temp.open('rb') as f:os.fsync(f.fileno())
  assert len(members)==234
  # Independently stream every member from the final ZIP representation before
  # atomic promotion. No second overlay copy or full float model is constructed.
  with zipfile.ZipFile(temp) as archive:
   names=archive.namelist();assert len(names)==len(set(names))==234 and set(names)=={key+'.npy' for key in members}
   for name in names:
    expected=members[name[:-4]];h=hashlib.sha256();size=0
    with archive.open(name) as stream:
     while block:=stream.read(1048576):h.update(block);size+=len(block)
    assert size==expected['npy_bytes'] and h.hexdigest()==expected['npy_sha256'];check()
  overlay=dst/'tower5-overlay.npz';os.replace(temp,overlay);overlay_id={'path':overlay.name,'bytes':overlay.stat().st_size,'sha256':fullhash(overlay)};assert overlay_id['bytes']<=1073741824
  manifest={'schema':'original-BF16-affine5-g64-tower-only-overlay-v52-v1','parent_revision':plan['parent_revision'],'original_revision':plan['original_revision'],'source_plan_sha256':sha(seal['tensor_plan']),'quantization':'Original BF16 Tower78 source directly via native MLX affine5/g64; all250MAIN6 modules unchanged. No expert/head/embedding/norm/codec/top8changes. Changedcoefficients; freshquality required. Recipe choice ours, primitive upstream.','overlay':overlay_id,'base_tensors':base_tensors,'tensors':members};dump(dst/'TOWER5_OVERLAY_MANIFEST_V1.json',manifest)
  selected_keys=sorted(members);selected_modules=sorted({key.rsplit('.',1)[0] for key in members});main_modules=sorted(key for key,value in original_config['quantization'].items() if key.startswith('backbone.') and value=={'bits':6,'group_size':64});assert len(selected_modules)==78 and len(main_modules)==250
  policy={'schema':'original-BF16-nativeMain6-Tower5-overlay-v52-v1','parent_revision':plan['parent_revision'],'source_overlay_manifest':{'path':'TOWER5_OVERLAY_MANIFEST_V1.json','sha256':sha(dst/'TOWER5_OVERLAY_MANIFEST_V1.json')},'selected_override_keys':selected_keys,'selected_override_keys_sha256':hashlib.sha256(canonical(selected_keys).encode()).hexdigest(),'selected_modules':selected_modules,'preserve_native_main_modules':main_modules,'base_config_quantization':original_config['quantization'],'new_config_canonical_sha256':hashlib.sha256(canonical(config).encode()).hexdigest(),'source_plan_sha256':sha(seal['tensor_plan']),'scope':'Fixed78 Tower affine5/g64; allMAIN6 objects retained. Newcoefficients; freshquality mandatory.'};dump(dst/'MAIN6_TOWER5_POLICY_V52.json',policy)
  notice={'recipe':'Main6+Tower5/g64 originalBF16 affine +original742 expert2MSE/E3last8/embedding4/BF16heads and unchangedcodec','builder_sha256':sha(__file__),'source_plan_sha256':sha(seal['tensor_plan']),'weight_loader':'Requiredexplicit native-safetensors then registered NPZoverride; native unmodified loader alone is not this artifact','parent_revision':plan['parent_revision'],'quality':'NOT RUN; originalFAIL not overwritten; no grade inheritance'};dump(dst/'TOWER5_OVERLAY_NOTICE_V1.json',notice)
  identities={}
  for path in sorted(dst.iterdir()):
   assert path.is_file() and not path.is_symlink();identities[path.name]={'sha256':fullhash(path),'bytes':path.stat().st_size}
  revision=hashlib.sha256(canonical({'schema':'inferencekit-native-main6-tower5-artifact-v52','files':identities,'parent':plan['parent_revision']}).encode()).hexdigest();new_origin={'model':'local/drama-3-e2-mse-e3last8-main6-tower5-bf16heads-emb4-g128-v52','revision':revision,'files':identities};dump(dst/'origin.json',new_origin);fsyncdir(dst)
  audit.update(status='PASS',new_revision=revision,overlay=overlay_id,changed_modules=78,replaced_tensors=234,unchanged_base_tensor_objects_at_loader='Runtime check pending',all_source_and_base_files_preserved=True,all234_finalNPYmember_bytes_verified=True,base_tensors=len(base_tensors),physical_duplicate_model_payload_created=False,elapsed_s=time.monotonic()-began,actual_disk_free_bytes=os.statvfs('.').f_bavail*os.statvfs('.').f_frsize);dump(work/'TENSOR_AUDIT_V52.json',audit)
 except BaseException as error:audit.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
 finally:
  if mx is not None:
   for name,fn in [('synchronize',mx.synchronize),('clear_cache',mx.clear_cache),('wired',lambda:mx.set_wired_limit(previous['wired'])),('cache',lambda:mx.set_cache_limit(previous['cache'])),('memory',lambda:mx.set_memory_limit(previous['memory']))]:
    try:
     if name in ('wired','cache','memory') and name not in previous:continue
     fn();obs[name]=True
    except BaseException as error:obs[name]={'error':str(error)}
  audit.update(restoration=obs,all_applied_limits_restored=all(v is True for v in obs.values()),maxrss_kernel_bytes_darwin=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(work/'BUILD_RESULT_V52.json',audit);fsyncdir(work)
 if audit['status']=='PASS' and audit['all_applied_limits_restored']:
  dump(work/'BUILD_COMPLETE_V52.json',{'status':'PASS','model_forward_calls':0,'native_quantization_calls':156,'coefficient_dequantization_calls':156,'builder_sha256':sha(__file__),'source_plan_sha256':sha(seal['tensor_plan']),'model_revision':revision,'origin_sha256':sha(dst/'origin.json'),'overlay_sha256':overlay_id['sha256'],'tensor_audit_sha256':sha(work/'TENSOR_AUDIT_V52.json'),'builder_result_sha256':sha(work/'BUILD_RESULT_V52.json'),'new_model_path':str(dst),'cold_model_admission':'NOT RUN; require actual32GiB+4GiBcoldreserve before launch'});fsyncdir(work)
 else:raise RuntimeError('Build incomplete or restorationfailed; no promotion')
if __name__=='__main__':
 with exclusive_locks():main()

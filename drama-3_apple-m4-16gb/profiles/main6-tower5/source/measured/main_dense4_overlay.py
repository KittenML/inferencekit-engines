"""Explicit registered NPZ overrides after the immutable native safetensors load.

NPZ contains UInt32 packed weights and raw UInt16 BF16 scales/biases. It is
ignored by the native *.safetensors glob, then applied in this explicit order.
Only78 Tower projections change to affine5/group64; all native MAIN6/expert objects remain. No old output/quality grade transfers.
"""
import hashlib,json,re,zipfile,resource,sys
from pathlib import Path
SCHEMA='original-BF16-nativeMain6-Tower5-overlay-v52-v1'
MANIFEST='MAIN6_TOWER5_POLICY_V52.json'
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(root,name):
 if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9_.-]+',name):raise ValueError('Unsafe overlay file name')
 path=root/name
 if path.is_symlink() or not path.is_file() or path.resolve().parent!=root:raise ValueError('Overlay must be a regular local file')
 return path

def memory_observation(mx):
 return {'mlx_active_bytes':mx.get_active_memory(),'mlx_cache_bytes':mx.get_cache_memory(),'mlx_peak_bytes':mx.get_peak_memory(),'kernel_RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}

def load_overlay(model_path,base_loader,mx):
 root=Path(model_path).resolve(strict=True);origin=json.loads(checked(root,'origin.json').read_text());path=checked(root,MANIFEST)
 if digest(path)!=origin['files'][MANIFEST]['sha256']:raise ValueError('Unregistered overlay manifest')
 policy=json.loads(path.read_text())
 if policy['schema']!=SCHEMA or policy['parent_revision']!='742d2f153e5775357faee2692480c7afdfe0b6673837e2183f2632808e7cebc3':raise ValueError('Unregistered MAIN6/Tower5 policy')
 source=checked(root,policy['source_overlay_manifest']['path'])
 if digest(source)!=policy['source_overlay_manifest']['sha256'] or origin['files'][source.name]!=dict(sha256=digest(source),bytes=source.stat().st_size):raise ValueError('Registered native Tower5/g64 tensor source manifest differs')
 spec=json.loads(source.read_text())
 if spec['schema']!='original-BF16-affine5-g64-tower-only-overlay-v52-v1' or len(spec['tensors'])!=234 or spec['parent_revision']!=policy['parent_revision']:raise ValueError('Unregistered source tensor policy')
 targets=set(policy['selected_override_keys']);expected={k for k in spec['tensors'] if k.startswith('tower.')}
 if targets!=expected or len(targets)!=234 or len(policy['selected_override_keys'])!=234:raise ValueError('Exact234 Tower tensor selection required')
 canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
 if hashlib.sha256(canonical(policy['selected_override_keys']).encode()).hexdigest()!=policy['selected_override_keys_sha256']:raise ValueError('Selected-key order/hash differs')
 modules={k.rsplit('.',1)[0] for k in targets};main_modules={k for k,v in policy['base_config_quantization'].items() if k.startswith('backbone.') and v=={'bits':6,'group_size':64}}
 if len(modules)!=78 or sorted(modules)!=policy['selected_modules'] or sorted(main_modules)!=policy['preserve_native_main_modules'] or len(main_modules)!=250:raise ValueError('Projection selection differs')
 if targets!={module+'.'+suffix for module in modules for suffix in ('weight','scales','biases')}:raise ValueError('Exact three buffers per Tower projection required')
 for module in modules:
  weight=spec['base_tensors'][module+'.weight'];scale=spec['base_tensors'][module+'.scales'];bias=spec['base_tensors'][module+'.biases']
  if weight['dtype']!='U32' or scale['dtype']!='BF16' or bias['dtype']!='BF16' or len(weight['shape'])!=2 or len(scale['shape'])!=2 or scale['shape']!=bias['shape']:raise ValueError('Only native dense6/group64 base projections admitted')
  rows,groups=scale['shape'];columns=groups*64
  if weight['shape'][0]!=rows or weight['shape'][1]*32!=columns*6:raise ValueError('Native dense6/group64 geometry differs')
  for suffix,dtype,shape in [('weight','U32',[rows,columns*5//32]),('scales','BF16',[rows,columns//64]),('biases','BF16',[rows,columns//64])]:
   value=spec['tensors'][module+'.'+suffix]
   if value['target_dtype']!=dtype or value['shape']!=shape or value['stored_dtype']!=('<u4' if dtype=='U32' else '<u2'):raise ValueError('Tower5/g64 output tensor geometry/dtype differs')
 config=json.loads(checked(root,'config.json').read_text());canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
 if hashlib.sha256(canonical(config).encode()).hexdigest()!=policy['new_config_canonical_sha256']:raise ValueError('Registered MAIN6/Tower5 config differs')
 if any(config['quantization'][k]!=v for k,v in policy['base_config_quantization'].items() if k not in modules):raise ValueError('Untargeted quantization config differs')
 if any(config['quantization'][k]!={'bits':5,'group_size':64} for k in modules) or any(config['quantization'][k]!={'bits':6,'group_size':64} for k in main_modules):raise ValueError('Main6/Tower5 module precision differs')
 overlay=checked(root,spec['overlay']['path']);identity=spec['overlay']
 if overlay.suffix!='.npz' or overlay.stat().st_size!=identity['bytes'] or digest(overlay)!=identity['sha256'] or origin['files'][overlay.name]!=dict(sha256=identity['sha256'],bytes=identity['bytes']):raise ValueError('Overlay payload differs')
 with zipfile.ZipFile(overlay) as archive:
  names=archive.namelist()
  if len(names)!=len(set(names)) or set(names)!={key+'.npy' for key in spec['tensors']}:raise ValueError('NPZ keys differ from exact overlay inventory')
  for info in archive.infolist():
   entry=spec['tensors'][info.filename[:-4]]
   if info.compress_type!=zipfile.ZIP_STORED or info.file_size!=entry['npy_bytes']:raise ValueError('Unexpected NPZ member representation')
 base=base_loader(root)
 if set(base)!=set(spec['base_tensors']):raise ValueError('Native base tensor inventory differs')
 dtypes={'U32':mx.uint32,'BF16':mx.bfloat16,'F32':mx.float32,'F16':mx.float16,'I32':mx.int32,'I64':mx.int64,'U8':mx.uint8}
 for key,value in base.items():
  expected=spec['base_tensors'][key]
  if list(value.shape)!=expected['shape'] or value.dtype!=dtypes[expected['dtype']]:raise ValueError('Base tensor shape/dtype differs: '+key)
 before={key:id(value) for key,value in base.items() if key not in targets};main_before={key:id(value) for key,value in base.items() if key.startswith('backbone.')};memory={'before_mx_load':memory_observation(mx)};arrays=mx.load(str(overlay));memory['after_mx_load_return']=memory_observation(mx)
 if set(arrays)!=set(spec['tensors']):raise ValueError('Loaded NPZ dictionary differs')
 # Validate all234 Tower buffers; this artifact contains no MAIN4 buffer.
 for key,value in arrays.items():
  entry=spec['tensors'][key]
  if not re.fullmatch('[A-Za-z0-9_.]+',key) or key not in base:raise ValueError('Overlay introduced an unknown parameter')
  dtype=mx.uint32 if entry['target_dtype']=='U32' else mx.uint16
  if value.dtype!=dtype or list(value.shape)!=entry['shape']:raise ValueError('Overlay stored dtype/shape differs: '+key)
 del value
 # Exact inventory is Tower-only; no native MAIN parameter is assigned.
 for key in list(arrays):
  if key not in targets:del arrays[key]
 if set(arrays)!=targets:raise ValueError('Selected NPZ subset differs')
 memory['after_drop_unused']=memory_observation(mx)
 for key,value in arrays.items():
  entry=spec['tensors'][key]
  if entry['target_dtype']=='BF16':value=mx.view(value,mx.bfloat16)
  elif entry['target_dtype']!='U32':raise ValueError('Unexpected overlay target dtype')
  base[key]=value
 if any(id(base[key])!=identifier for key,identifier in before.items()):raise ValueError('Untargeted base tensor object changed')
 if any(id(base[key])!=identifier for key,identifier in main_before.items()):raise ValueError('Native MAIN6 tensor object changed')
 memory['after_override']=memory_observation(mx)
 return base,{'memory_observations':memory,'materialization_scope':'mx.load return andnormal model load; no extra eval toforce unusedsource buffers. Dictionary deletion doesnotprove backingstorage reclaimed; actualpeak/outerOS andnormal model eval remainobserved.','mode':'native-main6-tower5-v1','manifest_sha256':digest(path),'overlay_sha256':identity['sha256'],'overridden_tensor_count':234,'selected_projection_modules':78,'unused_source_main4_buffers':0,'native_main_objects_unchanged':len(main_before),'source_manifest_sha256':digest(source),'untargeted_base_objects_unchanged':len(before),'native_safetensors_first_then_registered_npz':True,'source_dtype_note':'StoredUInt16 is reinterpreted asBF16 by mx.view; no numerical scale/bias cast.'}

def install(mx,utils,base_loader):
 audit={'mode':'native-main6-tower5-v1','files':[],'helper_sha256':digest(__file__)}
 def load(model_path):
  weights,entry=load_overlay(model_path,base_loader,mx);audit['files'].append(entry);return weights
 utils.load_weights=load
 return audit

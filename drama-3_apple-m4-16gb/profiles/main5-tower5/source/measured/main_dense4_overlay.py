"""Native safetensors, then disjoint exact Tower5 and original-BF16 MAIN5 overlays.

Only the declared328 dense projections change from native742 storage. The234
Tower5 buffers are the protected9343 bytes; only750 MAIN5 buffers are new.
"""
import hashlib,json,re,resource,sys,zipfile
from pathlib import Path
SCHEMA='original-BF16-main5-nativeTower5-two-overlays-v62-v1'
POLICY='MAIN5_TOWER5_POLICY_V62.json'
NATIVE='742d2f153e5775357faee2692480c7afdfe0b6673837e2183f2632808e7cebc3'
PARENT='93432d8a6c97faba88309880c3fd2d1b8356bde679dc20f9e0daa57282eb904c'
TOWER_SHA='7b67984c3fa7e4a028186e16b00b3f27a132239daf2778707d7642f16dafe1b2'
NON_QUANT_CONFIG_SHA256='617740f56ef3cac0464d740637fd25bdebce306b58702836d6370a6ddfc0c3f5'
NATIVE_QUANT_CONFIG_SHA256='3b7e1ada6883bd766fb9af812ad0bb806cd612cc35a1b2acbf53d521540bdd5f'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(root,name):
 if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9_.-]+',name):raise ValueError('Unsafe overlay file name')
 p=root/name
 if p.is_symlink() or not p.is_file() or p.resolve().parent!=root:raise ValueError('Overlay must be a regular local file')
 return p
def memory_observation(mx):return {'mlx_active_bytes':mx.get_active_memory(),'mlx_cache_bytes':mx.get_cache_memory(),'mlx_peak_bytes':mx.get_peak_memory(),'kernel_RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
def load_overlay(model_path,base_loader,mx):
 root=Path(model_path).resolve(strict=True);origin=json.loads(checked(root,'origin.json').read_text())
 def registered(name):
  p=checked(root,name)
  if origin['files'][name]!=dict(sha256=digest(p),bytes=p.stat().st_size):raise ValueError('Unregistered artifact file: '+name)
  return p
 path=registered(POLICY);policy=json.loads(path.read_text())
 if policy['schema']!=SCHEMA or policy['native_base_revision']!=NATIVE or policy['parent_revision']!=PARENT:raise ValueError('Unregistered MAIN5/Tower5 policy')
 references=policy['overlay_manifests']
 if [x['path'] for x in references]!=['TOWER5_OVERLAY_MANIFEST_V1.json','MAIN5_OVERLAY_MANIFEST_V1.json'] or [x['scope'] for x in references]!=['tower','backbone']:raise ValueError('Exact two ordered disjoint overlays required')
 native_config=policy['native_config_quantization']
 if hashlib.sha256(canonical(native_config).encode()).hexdigest()!=NATIVE_QUANT_CONFIG_SHA256:raise ValueError('Native742 quantization config differs')
 main=sorted(k for k,v in native_config.items() if k.startswith('backbone.') and v=={'bits':6,'group_size':64});tower=sorted(k for k,v in native_config.items() if k.startswith('tower.') and v=={'bits':6,'group_size':64})
 if len(main)!=250 or len(tower)!=78 or main!=policy['main_modules'] or tower!=policy['tower_modules']:raise ValueError('Exact250 MAIN and78 Tower projection selection required')
 targets={m+'.'+suffix for m in main+tower for suffix in ('weight','scales','biases')}
 keys=policy['selected_override_keys']
 if len(keys)!=984 or len(set(keys))!=984 or set(keys)!=targets or hashlib.sha256(canonical(keys).encode()).hexdigest()!=policy['selected_override_keys_sha256']:raise ValueError('Exact984 selected buffer keys required')
 config=json.loads(registered('config.json').read_text())
 if hashlib.sha256(canonical({k:v for k,v in config.items() if k!='quantization'}).encode()).hexdigest()!=NON_QUANT_CONFIG_SHA256:raise ValueError('Model architecture/reference-independent config changed')
 if hashlib.sha256(canonical(config).encode()).hexdigest()!=policy['new_config_canonical_sha256']:raise ValueError('Registered config differs')
 modules=set(main+tower)
 if set(config['quantization'])!=set(native_config) or any(config['quantization'][k]!=(dict(bits=5,group_size=64) if k in modules else v) for k,v in native_config.items()):raise ValueError('Only selected MAIN/Tower precision may change')
 specs=[];seen=set();base_tensors=None
 for ref in references:
  source=registered(ref['path'])
  if digest(source)!=ref['sha256']:raise ValueError('Overlay source manifest differs')
  spec=json.loads(source.read_text());scope=ref['scope'];selected={m+'.'+suffix for m in (tower if scope=='tower' else main) for suffix in ('weight','scales','biases')}
  want_schema='original-BF16-affine5-g64-tower-only-overlay-v52-v1' if scope=='tower' else 'original-BF16-affine5-g64-main-only-overlay-v62-v1'
  if spec['schema']!=want_schema or spec['parent_revision']!=(NATIVE if scope=='tower' else PARENT) or set(spec['tensors'])!=selected or ref['keys']!=sorted(selected) or seen&selected:raise ValueError('Overlay tensor scope differs')
  if base_tensors is None:base_tensors=spec['base_tensors']
  elif base_tensors!=spec['base_tensors']:raise ValueError('Two overlays disagree on native tensor inventory')
  for module in (tower if scope=='tower' else main):
   weight=base_tensors[module+'.weight'];scales=base_tensors[module+'.scales'];biases=base_tensors[module+'.biases']
   if weight['dtype']!='U32' or scales['dtype']!='BF16' or biases['dtype']!='BF16' or len(weight['shape'])!=2 or len(scales['shape'])!=2 or scales['shape']!=biases['shape']:raise ValueError('Native dense6/g64 base geometry required')
   rows,groups=scales['shape'];cols=groups*64
   if weight['shape']!=[rows,cols*6//32]:raise ValueError('Native dense6 weight geometry differs')
   for suffix,dtype,shape in [('weight','U32',[rows,cols*5//32]),('scales','BF16',[rows,groups]),('biases','BF16',[rows,groups])]:
    v=spec['tensors'][module+'.'+suffix]
    if v['shape']!=shape or v['target_dtype']!=dtype or v['stored_dtype']!=('<u4' if dtype=='U32' else '<u2'):raise ValueError('Registered affine5 output geometry/dtype differs')
  identity=spec['overlay'];overlay=registered(identity['path'])
  if overlay.suffix!='.npz' or overlay.stat().st_size!=identity['bytes'] or digest(overlay)!=identity['sha256']:raise ValueError('Overlay payload differs')
  if scope=='tower' and (identity['sha256']!=TOWER_SHA or identity['bytes']!=656092314):raise ValueError('Protected Tower5 bytes differ')
  with zipfile.ZipFile(overlay) as archive:
   names=archive.namelist()
   if len(names)!=len(set(names)) or set(names)!={k+'.npy' for k in selected}:raise ValueError('NPZ inventory differs')
   for item in archive.infolist():
    if item.compress_type!=zipfile.ZIP_STORED or item.file_size!=spec['tensors'][item.filename[:-4]]['npy_bytes']:raise ValueError('NPZ member representation differs')
  seen.update(selected);specs.append((scope,spec,overlay))
 if seen!=targets:raise ValueError('Incomplete overlay union')
 base=base_loader(root);dtypes={'U32':mx.uint32,'BF16':mx.bfloat16,'F32':mx.float32,'F16':mx.float16,'I32':mx.int32,'I64':mx.int64,'U8':mx.uint8}
 if set(base)!=set(base_tensors):raise ValueError('Native base tensor inventory differs')
 for key,value in base.items():
  wanted=base_tensors[key]
  if list(value.shape)!=wanted['shape'] or value.dtype!=dtypes[wanted['dtype']]:raise ValueError('Native base shape/dtype differs: '+key)
 before={key:id(v) for key,v in base.items() if key not in targets};memory={'before_any_mx_load':memory_observation(mx)};receipts=[]
 for scope,spec,overlay in specs:
  arrays=mx.load(str(overlay));memory['after_'+scope+'_mx_load_return']=memory_observation(mx)
  if set(arrays)!=set(spec['tensors']):raise ValueError('Loaded NPZ dictionary differs')
  for key,value in arrays.items():
   item=spec['tensors'][key]
   if key not in targets or not re.fullmatch('[A-Za-z0-9_.]+',key) or value.dtype!=(mx.uint32 if item['target_dtype']=='U32' else mx.uint16) or list(value.shape)!=item['shape']:raise ValueError('Loaded stored tensor shape/dtype differs: '+key)
  for key,value in arrays.items():base[key]=mx.view(value,mx.bfloat16) if spec['tensors'][key]['target_dtype']=='BF16' else value
  del arrays,value
  memory['after_'+scope+'_override']=memory_observation(mx);receipts.append({'scope':scope,'overlay_sha256':spec['overlay']['sha256'],'tensors':len(spec['tensors'])})
 if any(id(base[key])!=value for key,value in before.items()):raise ValueError('Untargeted tensor object changed')
 return base,{'mode':'native-main5-tower5-v1','manifest_sha256':digest(path),'overlays':receipts,'overridden_tensor_count':984,'MAIN5_tensor_count':750,'Tower5_tensor_count':234,'selected_projection_modules':328,'untargeted_base_objects_unchanged':len(before),'native_safetensors_first_then_registered_npz':True,'memory_observations':memory,'materialization_scope':'Two actual mx.load returns and normal model load; no extra eval or claim that dictionary deletion proves all backing storage reclaimed. Actual outer peak remains observed.','source_dtype_note':'Stored UInt16 coefficients are reinterpreted as BF16 with mx.view; no numerical coefficient cast.'}
def install(mx,utils,base_loader):
 audit={'mode':'native-main5-tower5-v1','files':[],'helper_sha256':digest(__file__)}
 def load(model_path):
  weights,entry=load_overlay(model_path,base_loader,mx);audit['files'].append(entry);return weights
 utils.load_weights=load
 return audit

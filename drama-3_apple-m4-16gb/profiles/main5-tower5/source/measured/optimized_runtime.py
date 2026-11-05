"""Selected experimental runtime: V19 sampler and V21 fixed-GDN compilation."""
import argparse,hashlib,json,os,sys,wave
from pathlib import Path
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--receipt',required=True)
 p.add_argument('--gdn-mode',choices=('native','fixed-block'),required=True);p.add_argument('server_arguments',nargs=argparse.REMAINDER)
 a=p.parse_args();package=Path(a.package).resolve();receipt=Path(a.receipt).resolve()
 source={'driver':digest(Path(__file__).read_bytes()),'helper':digest(Path(__file__).with_name('fixed_gdn.py').read_bytes()),'sampler_helper':digest(Path(__file__).with_name('sampler_sort.py').read_bytes()),'drama3':'902aa196da7d6e11ebbe2076cd43bb3aa8f7db91386fc0605e359cb9de552401','upstream':{'qwen3_5': '66f67b0813657bbb667fa918a00be2c1a6857f953696f75b9249f0fe08aa07af', 'qwen3_next': '57502b0bc872f83a4c6b813485a79531f8e87c84be0f12a8489cd0002d9b670a', 'gated_delta': 'f69cab3c4a94431832e418c46bebd59c95a3ab35c182e482ad38f974c3999aa8', 'cache': 'ab41054c4564784e7e131d7bd9c9e42a91b951bb4478d3a7ce8017ed3028e632'}}
 sys.path.insert(0,str(package));import serve
 from sampler_sort import install
 from fixed_gdn import FixedGDN
 old_init=serve.MlxRuntime.__init__;old_generate=serve.MlxRuntime.generate;health=[];requests=[];runtimes=[]
 def check_codes(r,scope):
  mx,np=r.mx,r.np;spec=json.loads((package/'conditioning-cache-v2/manifest.json').read_text())
  with wave.open(str(package/'conditioning-cache-v2'/spec['wav_file']),'rb') as w:audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
  codes=r.model._encode_audio(audio);mx.eval(codes);mx.synchronize();array=np.ascontiguousarray(np.asarray(codes),dtype='<i8')
  assert array.shape==(10,103) and digest(array.tobytes())=='1332c062e02ae3cc967663559a0234d6db4e04604cd6ff142f1b6ed5f6ac83b4' and int(array.min())==0 and int(array.max())==4027
  mx.synchronize();health.append({'scope':scope,'GPU_sync':'PASS','logical_i64_sha256':digest(array.tobytes())})
 def initialize(r,*args,**kw):
  old_init(r,*args,**kw)
  assert json.loads((package/'selected_recipe.json').read_text())['track']=='e2-mse-e3last8-main5-tower5-bf16heads-emb4-g128-v62' and r.resolved['model_revision']==json.loads((package/'selected_recipe.json').read_text())['revision'] and r.startup['compile_mode']=='none' and r.encoder_residency['enabled']
  check_codes(r,'after_load_before_warmup');observation=install(r,True,source['drama3']);r.fixed_gdn=FixedGDN(r,a.gdn_mode!='native',source['upstream'],source['drama3']);runtimes.append(r)
  policy={'mode':a.gdn_mode,'sources':source,'sort_removal':True,'compile_scope':'Thirty original GDN S1 blocks only, fixed shapes with explicit conv+SSM arrays; original _step scope; all prefill including S1 native; ten full-attention KV blocks native','state_or_attention_padding':False,'native_categorical_and_RNG':True,'production_observation':'Common host invocation counters and existing source telemetry; no per-step GPU-to-host state hashes or completion barriers'}
  parent=r.resolved['runtime_recipe_sha256'];r.runtime_recipe['fixed_gdn_experiment']=policy;revised=digest(serve.canonical(r.runtime_recipe).encode())
  r.resolved['engine_version']=r.resolved['engine_version'].replace('+runtime@'+parent,'')+'+fixed-gdn@'+a.gdn_mode+'+runtime@'+revised
  r.resolved['runtime_recipe_sha256']=revised;r.resolved['fixed_gdn_experiment']=policy;r.startup['runtime_recipe']=r.runtime_recipe
  r.write_telemetry({'event':'fixed_gdn_experiment_installed','runtime_recipe':r.runtime_recipe,'parent_runtime_recipe_sha256':parent,'policy':policy,'observation':observation,'gdn_initial_observation':r.fixed_gdn.snapshot()})
 def generate(r,body,settings,reference):
  before=r.fixed_gdn.snapshot();generated=old_generate(r,body,settings,reference);completed=False
  try:
   for pcm in generated:yield pcm
   completed=True
  finally:
   generated.close();after=r.fixed_gdn.snapshot();steps=after['steps']-before['steps']
   counts={k:[a-b for a,b in zip(after[k],before[k])] for k in ('decode_calls','compiled_calls','prefill_calls','native_prefill_S1_calls')}
   row={'text_utf8_sha256':digest(body['input'].encode()),'seed':body['seed'],'request_kind':r.request_kind,'completed':completed,'steps':steps,'counts':counts,'compiler_traces_before':before['compiler_traces'],'compiler_traces_after':after['compiler_traces']}
   requests.append(row);r.write_telemetry({'event':'fixed_gdn_request_observed',**row});check_codes(r,'after_'+r.request_kind+'_'+str(len(requests)))

 serve.MlxRuntime.__init__=initialize;serve.MlxRuntime.generate=generate
 selected=json.loads((package/'selected_recipe.json').read_text());argv=['serve.py','--model-dir',str(package/'models'/selected['track']),'--track',selected['track']]
 for key,value in selected['runtime'].items():
  if isinstance(value,bool):
   if value:argv.append('--'+key.replace('_','-'))
  else:argv.extend(['--'+key.replace('_','-'),str(value)])
 argv.extend(a.server_arguments[1:] if a.server_arguments[:1]==['--'] else a.server_arguments);sys.argv=argv;os.chdir(package)
 try:serve.main()
 finally:
  final=[r.fixed_gdn.finish() for r in runtimes]
  receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps({'schema':'selected-fixed-gdn-runtime-v30-v1','sources':source,'mode':a.gdn_mode,'fixed_code_health':health,'requests':requests,'final_helpers':final,'scope':'Selected V19 sampler and30original fixed-GDN S1 blocks; allprefill andtenattentionKV native. This processreceipt records actualrequests/health/parameters/hooks. KnownV14qualityFAIL/listeningUP remain separate; no overallPASS inferred.'},indent=2)+'\n')
if __name__=='__main__':main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute a sealed finite speech request plan with the unchanged frozen client."""
import argparse,hashlib,json,time,urllib.request,wave
from pathlib import Path
from inferencekit.serve.audio_client import AudioHTTP,AudioRequest
from inferencekit.serve.client import Endpoint
from inferencekit.ops.bench_audio import load_fixtures,audio_row

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('fixtures','fixtures-sha256','plan','plan-sha256','output','server-telemetry','expected-runtime-sha256'):p.add_argument('--'+name,required=True)
 p.add_argument('--timeout-s',type=float,default=900);p.add_argument('--post-request-quiescence-s',type=float,default=0)
 args=p.parse_args()
 if sha(args.fixtures)!=args.fixtures_sha256 or sha(args.plan)!=args.plan_sha256:p.error('Sealed input bytes differ')
 if args.timeout_s<=0 or args.post_request_quiescence_s<0:p.error('Invalid finite timeout/quiescence')
 fixtures,fixture_hash=load_fixtures(args.fixtures);by_id={f.id:f for f in fixtures};plan=list(map(json.loads,Path(args.plan).read_text().splitlines()))
 for i,item in enumerate(plan):
  if item['request_index']!=i or item['id'] not in by_id or item['text_sha256']!=by_id[item['id']].text_sha256 or type(item['seed']) is not int or not 0<=item['seed']<2**32:p.error('Plan fixture/text/order/seed mismatch')
 root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
 with urllib.request.urlopen('http://127.0.0.1:8765/info',timeout=10) as response:info=json.load(response)['resolved']
 if info['runtime_recipe_sha256']!=args.expected_runtime_sha256:raise RuntimeError('Loaded runtime differs from final seal')
 defaults=info['generation_defaults'];settings={k:defaults[k] for k in ('temperature','top_p','top_k','max_tokens')}
 settings.update({k:info[k] for k in ('model_revision','dtype','quant','engine_version')})
 manifest={'started_time_ns':time.time_ns(),'fixture_file_sha256':sha(args.fixtures),'request_fixture_hash':fixture_hash,'plan_sha256':sha(args.plan),'planned_request_count':len(plan),'resolved':info,'settings':settings,'client_source_sha256':sha(__file__),'timeout_s':args.timeout_s,'post_request_quiescence_s':args.post_request_quiescence_s,'performance':'UNPROVEN','quality':'UNPROVEN'}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 client=AudioHTTP(Endpoint.parse('http://127.0.0.1:8765'),timeout_s=args.timeout_s)
 for item in plan:
  fixture=by_id[item['id']];seed=item['seed'];result=client.generate(AudioRequest(fixture.id,fixture.text,info['model'],seed,44100,1,settings,fixture.reference_audio,fixture.reference_text));row=audio_row(result,fixture,0,seed);row['request_index']=item['request_index']
  prefix=root/('request-%04d'%item['request_index']);prefix.with_suffix('.pcm').write_bytes(result.pcm)
  if result.ok:
   with wave.open(str(prefix.with_suffix('.wav')),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(44100);f.writeframes(result.pcm)
  events=list(map(json.loads,Path(args.server_telemetry).read_text().splitlines()))
  matches=[e for e in events if e.get('event')=='request_finished' and e.get('request_kind')=='api' and e.get('input_sha256')==fixture.text_sha256 and e.get('seed')==seed and (not row['ok'] or e.get('pcm_sha256')==row.get('pcm_sha256'))]
  event=matches[0] if len(matches)==1 else None
  if event is not None and event.get('runtime_recipe_sha256')!=args.expected_runtime_sha256:raise RuntimeError('Actual response runtime differs from final seal')
  status='STRUCTURAL_FAIL' if not result.ok else 'EOS_OBSERVED' if event and event.get('all_segments_eos') and event.get('termination_frame_counts_match') else 'FRAME_CAP' if event and any(x.get('frame_cap_hit') for x in event.get('termination_observations',[])) else 'UNPROVEN'
  meta={'id':fixture.id,'text':fixture.text,'seed':seed,'request_index':item['request_index'],'pcm_sha256':row.get('pcm_sha256'),'wav':prefix.with_suffix('.wav').name if result.ok else None,'generation_status':status,'generation_observation':event,'source_receipt_unique':len(matches)==1,'transport':row,'quality':'UNPROVEN'}
  prefix.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
  with (root/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps({'request_index':item['request_index'],'id':fixture.id,'seed':seed,'ok':result.ok,'generation_status':status,'rtf':row.get('rtf'),'underruns':row.get('underruns')}),flush=True)
  began=time.time_ns()
  if args.post_request_quiescence_s:time.sleep(args.post_request_quiescence_s)
  with (root/'quiescence.jsonl').open('a') as f:f.write(json.dumps({'request_index':item['request_index'],'after_response_time_ns':began,'quiescent_time_ns':time.time_ns(),'requested_quiescence_s':args.post_request_quiescence_s})+'\n')
 manifest.update(finished_time_ns=time.time_ns(),observed_request_count=len(plan));(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()

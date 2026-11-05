"""Project verified original V57 receipts into a public, metadata-only export.
No model, audio, network, hardware or benchmark execution. Original bytes are
hash-bound by role; public summaries have a separate manifest identity.
"""
import argparse,copy,hashlib,json,re
from pathlib import Path

Q='private/quality-resume/'
T=Q+'packaged-paired192-v57-closed-corpus-v1/'
F=Q+'packaged-quality288-v57-closed-corpus-v1/'
I=T+'consumer-output-v1/'
FILES={
 'combined':Q+'mac-v57-fresh288-preservation-v1/ROOT_COMPARATIVE_SPEEDUP_V57_V1.json',
 'direct144':Q+'mac-v57-fresh288-preservation-v1/FRESH288_DIRECT_PRESERVATION_V57_V1.json',
 'root192':T+'ROOT_INDEPENDENT_CURRENT192_VERIFIED_V57_V1.json',
 'exact192':I+'EXACT_BEHAVIOR_COMPARISON_VERIFIED_V1.json',
 'cases192':I+'pairs/CASES192.jsonl','pairs96':I+'pairs/PAIRS96.jsonl',
 'pair_integrity':I+'pairs/PAIR_INTEGRITY_VERIFIED_V1.json',
 'frozen':I+'compare/FROZEN_COMPARE_REPLAY_VERIFIED_V1.json',
 'timing_native_stock':I+'stock-NATIVE_HISTORY_REPLAY_V1.json',
 'timing_native_release':I+'release-NATIVE_HISTORY_REPLAY_V1.json',
 'fresh_native_stock':F+'native-consumption-v2/stock-NATIVE_HISTORY_REPLAY_V57_V1.json',
 'fresh_native_release':F+'native-consumption-v2/release-NATIVE_HISTORY_REPLAY_V57_V1.json',
 'native_failed_stock':F+'native-consumption-v1/stock-NATIVE_HISTORY_REPLAY_V57_V1.json',
 'native_failed_release':F+'native-consumption-v1/release-NATIVE_HISTORY_REPLAY_V57_V1.json',
 'native_repair':F+'NATIVE_INVOCATION_PATH_REPAIR_V2.json',
 'native_both':F+'native-consumption-v2/BOTH_FRESH144_NATIVE_HISTORY_V57_V1.json',
 'native_owner_closure':F+'native-owner-closure/ROOT_WORKER_NATIVE_CLOSURE_V57_V1.json',
 'slo':Q+'mac-v57-slo-actual-v1/WARMED96_PER_ARM_SIGNAL_SLO_V57.json',
 'closed192':T+'CLOSED_CORPUS_V57_V1.json','closed288':F+'CLOSED_CORPUS_V57_QUALITY_V1.json',
 'archive288':F+'LOCAL_FULL_QUALITY288_ARCHIVE_VERIFIED_V57_V1.json',
 'plan':Q+'packaged-paired192-v57-consumer-v1/ACTUAL_RUN_PLAN_V57.json',
 'method_source':Q+'packaged-paired192-v57-consumer-v1/METHOD_SOURCE_SEAL_V57_V1.json',
 'baseline_selection':'mac/tower5-stock-baseline-v55-preparation/retained-balanced-v2/raw/tower5-stock-balanced-v55-v2/BALANCED_STOCK_TOWER5_SELECTION_V55_V1.json',
 'baseline_admission':'mac/tower5-stock-baseline-v55-preparation/retained-admission-v1/V55_STOCK_ADMISSION_ACTUAL_CLOSED_V1.json',
 'baseline_replay':'mac/tower5-stock-baseline-v55-preparation/INDEPENDENT_SELECTION_REPLAY_V55_V1.json',
}
MODEL='93432d8a6c97faba88309880c3fd2d1b8356bde679dc20f9e0daa57282eb904c'
EXPECTED_COMBINED_STATUS='PASS_NARROW_SAME_ARTIFACT_ENGINE_SPEEDUP_AND_FRESH_OUTPUT_PRESERVATION'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def load(path):return json.loads(Path(path).read_text())
def select(x,keys):return {k:copy.deepcopy(x[k]) for k in keys if k in x}
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_rows(path,values):Path(path).write_text(''.join(canonical(v)+'\n' for v in values))

def forbidden_private(text):
    prefixes=[chr(47)+n+chr(47) for n in ('Users','home','workspace','root','tmp')]
    return any(p in text for p in prefixes) or bool(re.search(r'(?:https?://|ssh://|\b(?:\d{1,3}\.){3}\d{1,3}\b|BEGIN [A-Z ]*PRIVATE KEY)',text))

def assert_public(value):
    text=canonical(value)
    assert not forbidden_private(text),'Private route or endpoint in public projection'

def wrapper(role,data,receipts):
    assert_public(data)
    return {'schema':'transformed-public-v57-'+role+'-v1','original_receipt_sha256':receipts[role]['sha256'],
            'transformation':'Public projection; original receipt bytes remain separate and unchanged.','data':data}

def native_summary(value):
    return {**select(value,('status','counts','all_planned_cases_retained','runtime_sha256','consumer_sha256',
        'source_pin','source_config_sha256','source_inference_sha256','tokenizer_sha256','integrity_core_sha256',
        'request_plan_sha256','native_proof_sha256','fixtures_sha256','integer_location_sha256','model_forward_calls','codec_forward_calls','model_weights_loaded','GPU_visible')),
        'requests':len(value['rows']),'actual_segments':sum(row['observed_segments'] for row in value['rows']),
        'scope':'Actual complete prompt/history replay only; no absolute, listening or perceptual grade.'}

def project(run_root,output):
    run_root=Path(run_root).resolve();output=Path(output);output.mkdir(exist_ok=True)
    raw={};receipts={}
    for role,name in FILES.items():
        path=run_root/name;assert path.is_file() and not path.is_symlink() and path.stat().st_size<32*2**20
        data=path.read_bytes();receipts[role]={'original_basename':path.name,'sha256':sha_bytes(data),'bytes':len(data)}
        raw[role]=[json.loads(s) for s in data.splitlines() if s] if name.endswith('.jsonl') else json.loads(data)
    combined=raw['combined'];assert combined['status']==EXPECTED_COMBINED_STATUS and combined['model_revision']==MODEL
    # Verify every source joined by the root combined receipt, without exporting private route names.
    joined=[]
    for name,want in combined['source_results_sha256'].items():
        path=run_root/name;assert path.is_file() and sha(path)==want
        joined.append({'original_basename':path.name,'sha256':want})
    assert combined['timing']['responses']==192 and combined['fresh_quality_preservation']['responses']==288
    assert combined['fresh_quality_preservation']['matching_completion_failures']==0
    assert raw['baseline_replay']['selection_sha256']==receipts['baseline_selection']['sha256']
    assert raw['native_both']['status']=='PASS'
    for role in ('cases192','pairs96','frozen','timing_native_stock','timing_native_release','exact192'):
        name=FILES[role][len(I):]
        assert raw['root192']['retained_output_files_sha256'][name]==receipts[role]['sha256']
    assert raw['baseline_admission']['status']=='CLOSED_TWO_ELIGIBLE_ONE_RECOMMENDED_STARTUP_FAILURE'
    assert raw['baseline_admission']['eligible']==['anchor11','wire12b'] and raw['baseline_admission']['failed']=='recommended'
    for arm in ('stock','release'):
        for scope,count,segments in (('timing',96,128),('fresh',144,144)):
            native=raw[scope+'_native_'+arm]
            assert native['status']=='PASS' and len(native['rows'])==count
            assert native['counts']=={'PASS':count,'FAIL':0,'UNPROVEN':0,'NOT RUN':0}
            assert sum(x['observed_segments'] for x in native['rows'])==segments
        assert combined['fresh_quality_preservation']['native_actual144_each'][arm]['sha256']==receipts['fresh_native_'+arm]['sha256']
    timing={}
    for arm,a in raw['slo']['arms'].items():
        assert len(a['rows'])==96 and a['retained_planned_rows']==96
        for row in a['rows']:
            key=(arm,row['planned']['request_index']);assert key not in timing;timing[key]=row
    cases=[]
    for source in raw['cases192']:
        a=copy.deepcopy(source);a.pop('pcm_source',None)
        observed=timing[(a['arm'],a['request_index'])]
        for k in ('id','seed','text_sha256','round','round_request_ordinal'):assert a[k]==observed['planned'][k]
        client=copy.deepcopy(observed['raw_client']);client.pop('declared_unseen',None)
        assert a['pcm_sha256']==client['pcm_sha256'] and a['original_client_line_sha256']==observed['original_client_line_sha256']
        a['source_cold_warm_label']=a.pop('cold_warm')
        a['startup_scope']='Recorded after startup warmup; source first-recorded label is not cold-voice admission.'
        a['timing']=client;a['source_completion']=observed['source_completion'];a['signal_SLO_checks']=observed['checks']
        a['source_case_canonical_json_sha256']=sha_bytes(canonical(source).encode())
        assert_public(a);cases.append(a)
    assert len(cases)==192 and len({(r['arm'],r['request_index']) for r in cases})==192
    case_map={(r['arm'],r['request_index']):r for r in cases}
    pairs=[]
    for source in raw['pairs96']:
        p=copy.deepcopy(source);p['source_cold_warm_label']=p.pop('cold_warm')
        stock,release=(case_map[(arm,p['request_index'])] for arm in ('stock','release'))
        assert stock['pcm_sha256']==release['pcm_sha256']==p['pcm_sha256']
        assert stock['behavior_sha256']==release['behavior_sha256']==p['observed_behavior_sha256']
        p['timing']={arm:select(case_map[(arm,p['request_index'])]['timing'],('rtf','complete_audio_ms','audio_duration_s','first_pcm_ms','first_signal_ms','first_audio_ms','leading_silence_ms','underruns','underrun_ms')) for arm in ('stock','release')}
        p['source_pair_canonical_json_sha256']=sha_bytes(canonical(source).encode());assert_public(p);pairs.append(p)
    assert len(pairs)==96 and len({p['request_index'] for p in pairs})==96
    write_rows(output/'CASES192.jsonl',cases);write_rows(output/'PAIRS96.jsonl',pairs)
    fresh=raw['direct144'];fresh_rows=fresh['rows']['fresh_stock_vs_release'];assert fresh['status']=='PASS' and len(fresh_rows)==144
    for index,row in enumerate(fresh_rows):assert row['request_index']==index and row['status']=='PASS' and row['completion_status']=='PASS'
    write_rows(output/'FRESH_PAIRS144.jsonl',fresh_rows)
    for arm in ('stock','release'):write_rows(output/('NATIVE144_'+arm.upper()+'.jsonl'),raw['fresh_native_'+arm]['rows'])
    # Public summaries deliberately do not copy interim inherited Tower6 feasibility labels.
    final=select(combined,('status','at','headline','speedup','speedup_ci95','round_pairs','powered','target_1_5x_met','model_revision',
        'actual_runtime_recipes','timing','fresh_quality_preservation','baseline_scope','global_best_engine_claim','listening','human_review',
        'Tower5_vs_original_perceptual_quality','full_absolute_quality','streaming','historical_records','new_verification_model_ASR_speaker_forwards'))
    final['numerical_identity_scope']='Observed on this panel; not a new universal admission/rejection requirement.'
    final['new_untouched_heldout_claim']=False
    final['portable_builder_and_smoke']='Separate public packaging/reproduction; no public checker or fresh install is asserted by this export.'
    final['publication']='Evidence export prepared; repository integration and publication are separate.'
    write(output/'COMPARATIVE_SPEEDUP.json',wrapper('combined',final,receipts))
    exact=select(raw['exact192'],('status','created_utc','scope','case_denominator','pair_denominator','round_pairs','matching_completion_failure_pairs_retained',
       'preserved_observed_behavior','completion_component','frozen_tool_verdict','ratio_candidate_release_over_stock','ratio_ci95',
       'conservative_speedup_from_inverse_upper_ratio_bound','improvement_claim_supported','native_replay_counts','no_new_heldout_claim','new_ASR_speaker_or_model_evaluations','baseline_selection_scope'))
    write(output/'LATEST_COMPARISON.json',wrapper('exact192',exact,receipts))
    frozen=select(raw['frozen'],('status','scope','registered_full_four_round_rule_match','statistical_fields_exact_equal','tool_data','tool_metrics','tool_verdict','wheel_sha256','pair_integrity_sha256','new_model_or_metric_forwards'))
    frozen['historical_intermediate_label']='Original receipt contains legacy known_quality_feasibility wording; it is not a Tower5 absolute-quality grade and is not transferred.'
    write(output/'FROZEN_STATISTICS.json',wrapper('frozen',frozen,receipts))
    direct=select(fresh,('status','created_utc','model_revision','actual_runtime_recipes','planned_fresh_responses','fresh_pair_cases','domain_verdicts','domain_counts',
       'matching_completion_failures_retained','identity_failure_is_perceptual_quality_failure','original_grade_reuse','original_native_output_reuse',
       'overall_quality_pass','perceptual_quality','listening','human_review','generation_seal_sha256','closure_sha256','source_manifest_sha256'))
    direct['scope']='Direct144 fresh pair preservation plus separately completed native144-per-arm replay; no inherited original742 grade/native results.'
    direct['native144_each']={arm:native_summary(raw['fresh_native_'+arm]) for arm in ('stock','release')}
    write(output/'FRESH288_PRESERVATION.json',wrapper('direct144',direct,receipts))
    native={'timing96_each':{arm:native_summary(raw['timing_native_'+arm]) for arm in ('stock','release')},
            'fresh144_each':{arm:native_summary(raw['fresh_native_'+arm]) for arm in ('stock','release')},
            'quality_grade':None,'actual_model_or_codec_forwards':0}
    write(output/'NATIVE_REPLAY.json',{'schema':'transformed-public-native-v57-v1','original_receipt_roles':['timing_native_stock','timing_native_release','fresh_native_stock','fresh_native_release'],'data':native})
    slo=select(raw['slo'],('schema','scope','actual_run_plan_sha256','consumer_source_seal_sha256','original_protocol_sha256','perceived_speech_onset','listening','cold_voice','reliability_100_sequential_requests','reliability_15min_generated_speech','overall_quality_PASS','new_model_ASR_speaker_or_audio_evaluations'))
    slo['arms']={arm:{k:v for k,v in data.items() if k!='rows'} for arm,data in raw['slo']['arms'].items()}
    slo['case_data']='CASES192.jsonl: all first_pcm/signal/audio, leading silence, chunks, RTF, underrun and completion facts retained.'
    slo['old_source_labels']='Source declared_unseen flag is omitted; no new untouched-heldout claim. Source first-recorded label remains descriptive only.'
    write(output/'SIGNAL_SLO.json',wrapper('slo',slo,receipts))
    warm=[]
    for receipt in raw['slo']['all_eight_fresh_process_load_warmup_receipts']:
        row=select(receipt,('arm','round','client_sha256','source_runtime_sha256','scope'))
        row['loaded']=select(receipt['loaded_event'],('event','engine_version','load_seconds_self_reported','load_maxrss_bytes_darwin','load_peak_mlx_bytes','physical_memory_bytes','pre_generation_api_wired_limit_bytes','previous_api_cache_limit_bytes','previous_api_memory_limit_bytes','previous_api_wired_limit_bytes','codec_encoder_residency_release','activation_cast_audit','compilation','compile_mode','dtype_bytes'))
        for event in ('startup_warmup_event','API_ready_event'):
            row[event]=select(receipt[event],('event','ok','status','seconds','elapsed_seconds','sample_count','all_segments_eos','generation_completed','seed','fixture_sha256','native_prompt_proof_sha256','model_revision','engine_version'))
        assert_public(row);warm.append(row)
    assert len(warm)==8
    write(output/'LOAD_WARMUP_RECEIPTS.json',{'schema':'transformed-public-load-warmup-v57-v1','original_receipt_role':'slo','rows':warm,'cold_voice':'NOT_RUN'})
    repair=raw['native_repair']
    provenance={'status':'V1_FAILED_PATH_BEFORE_COMPARISON_V2_NATIVE_PASS',
       'failed_V1_reason':repair['failed_native_V1']['all288_reason'],
       'failed_V1_original_receipt_sha256':{arm:receipts['native_failed_'+arm]['sha256'] for arm in ('stock','release')},
       'repair':select(repair,('status','change','before_runner_sha256','after_runner_sha256','native_algorithm_sha256','actual_current_prompt_sidecar_paths_and_file_hashes_verified')),
       'V2_actual_status':'PASS144_PER_ARM_FROM_SEPARATE_ACTUAL_NATIVE_RECEIPTS',
       'meaning':'V1 bundle route was its parent directory; --bundle was corrected to exact archived package. No validator, algorithm, tolerance, payload, input or case change; failures retained.'}
    write(output/'NATIVE_INVOCATION_PROVENANCE.json',wrapper('native_repair',provenance,receipts))
    selection=copy.deepcopy(raw['baseline_selection']);selection['independent_selection_replay']=raw['baseline_replay']
    selection['supported_setting_admission']=select(raw['baseline_admission'],('status','eligible','failed','owner_sha256','admission_seal_sha256','lifecycle_sha256','actual_owner_watch_and_started_servers_gone','port_free'))
    selection['admission_original_receipt_sha256']=receipts['baseline_admission']['sha256']
    selection['finite_scope']=combined['baseline_scope'];selection['global_best_claim']=False
    write(output/'TESTED_BASELINE_SCOPE.json',wrapper('baseline_selection',selection,receipts))
    plan=raw['plan'];lineage={'model_revision':MODEL,'actual_runtime_recipes':combined['actual_runtime_recipes'],
       'hardware_name':plan['hardware_name'],'physical_memory_bytes':plan['physical_memory_bytes'],
       'actual_plan_controller_sha256':plan['controller_sha256'],'actual_plan_protocol_sha256':plan['protocol']['sha256'],
       'timing_source_and_run_seal_sha256':raw['closed192']['source_and_run_seal_sha256'],
       'timing_source_archive_sha256':raw['closed192']['source_archive_sha256'],
       'timing_source_manifest_sha256':raw['closed192']['source_manifest_sha256'],
       'fresh_generation_seal_sha256':fresh['generation_seal_sha256'],'fresh_source_manifest_sha256':fresh['source_manifest_sha256'],
       'fresh_consumer_sources_sha256':fresh['consumer_sources_sha256'],'frozen_original_methods_unchanged':fresh['original_methods_unchanged'],
       'current_admission_sources':fresh['current_admission_sources'],
       'arm_recipes':{arm:select(data['expected_runtime_recipe'],('schema','model','controls','generation_defaults','activation_dtype_mode','compile_mode','fixed_reference','sources','pcm','fixed_gdn_experiment')) for arm,data in plan['arms'].items()},
       'public_packaging_identity_is_separate':True,'public_checker_execution':'NOT_RUN_BY_THIS_EXPORT'}
    assert_public(lineage);write(output/'MEASURED_SOURCE_LINEAGE.json',{'schema':'transformed-public-source-lineage-v57-v1','data':lineage})
    archives={'timing192':raw['closed192']['archive'],
       'fresh288':select(raw['archive288'],('status','archive_sha256','archive_bytes','inventory_sha256','original_files','regular_members_including_listing','member_bytes','all_member_payloads_hash_verified','extracted_regular_files','all_members_extracted_and_fsynced','original_remote_raw_and_archive_retained')),
       'trust_boundary':'Original root verified retained hashes and lifecycle joins; exclusive hardware owner observed remote closure. This export reads receipts only and does not repeat hardware observations.',
       'original_closure_sha256':{k:receipts[k]['sha256'] for k in ('closed192','closed288','native_owner_closure')}}
    assert_public(archives);write(output/'ARCHIVE_IDENTITY.json',{'schema':'transformed-public-archive-identity-v57-v1','data':archives})
    write(output/'PUBLIC_REPRODUCTION_STATUS.json',{'status':'NOT_RUN_BY_THIS_EXPORT','public_checker':None,'fresh_optional_profile_install':None,'portable_builder_smoke':None,'scope':'Evidence projection only; later public wrapper/build/smoke receipts have their own source identity.'})
    write(output/'SOURCE_RECEIPTS.json',{'schema':'original-v57-receipt-identities-for-transformed-export-v1','receipts':receipts,'combined_root_joined_original_receipts':joined,
      'rule':'Original byte SHA identities, not hashes of these transformed public summaries. Source originals retained separately; private routes omitted.'})
    return {'cases':len(cases),'pairs':len(pairs),'fresh_pairs':len(fresh_rows),'native_fresh_each':144,'original_receipts':len(receipts)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(project(a.run_root,a.output),sort_keys=True))

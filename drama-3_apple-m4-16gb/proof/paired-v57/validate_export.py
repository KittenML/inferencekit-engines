"""Local export structure/privacy/hash checks; never a public model checker."""
import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_evidence import MODEL,assert_public,load,rows,sha

def inspect_data(root):
    root=Path(root);cases=rows(root/'CASES192.jsonl');pairs=rows(root/'PAIRS96.jsonl')
    fresh=rows(root/'FRESH_PAIRS144.jsonl');combined=load(root/'COMPARATIVE_SPEEDUP.json')['data']
    frozen=load(root/'FROZEN_STATISTICS.json')['data'];slo=load(root/'SIGNAL_SLO.json')['data']
    assert len(cases)==192 and len(pairs)==96 and len(fresh)==144
    indexed={(r['arm'],r['request_index']):r for r in cases};assert len(indexed)==192
    assert set(indexed)=={(arm,i) for arm in ('stock','release') for i in range(96)}
    assert combined['model_revision']==MODEL and combined['powered'] is False
    assert combined['speedup']==frozen['tool_data']['decision']['oriented_ratio']
    assert combined['speedup_ci95']==frozen['tool_data']['decision']['oriented_ci95']
    assert frozen['tool_data']['pairs']==4 and frozen['tool_data']['resamples']==10000 and frozen['tool_data']['seed']==0
    assert frozen['tool_data']['threshold']==.02 and frozen['registered_full_four_round_rule_match'] is True
    assert all(frozen['statistical_fields_exact_equal'].values())
    for case in cases:
        t=case['timing'];assert case['pcm_sha256']==t['pcm_sha256']
        assert case['source_completion']=={'all_segments_eos':True,'possible_frame_cap':False,'generation_completed':True}
        assert t['ok'] is True and t['status']==200 and t['error'] is None
        assert t['effective']['model_revision']==MODEL
        for k in ('first_pcm_ms','first_signal_ms','first_audio_ms','complete_audio_ms','audio_duration_s','rtf','leading_silence_ms'):
            assert type(t[k]) in (int,float) and math.isfinite(t[k]) and t[k]>=0
        assert sum(c['sample_count'] for c in t['chunks'])==t['sample_count']
    for i,pair in enumerate(pairs):
        assert pair['request_index']==i and pair['status']==pair['completion_status']=='PASS'
        for arm in ('stock','release'):
            case=indexed[(arm,i)];assert case['pcm_sha256']==pair['pcm_sha256']
            assert case['behavior_sha256']==pair['observed_behavior_sha256']
            assert pair['timing'][arm]['rtf']==case['timing']['rtf']
    for i,r in enumerate(fresh):assert r['request_index']==i and r['status']==r['completion_status']=='PASS'
    native=load(root/'NATIVE_REPLAY.json')['data']
    for arm in ('stock','release'):
        r=rows(root/('NATIVE144_'+arm.upper()+'.jsonl'));assert len(r)==144
        assert [v['request_index'] for v in r]==list(range(144))
        assert all(v['status']=='PASS' and v['source_generation_completed'] and v['source_all_segments_eos'] for v in r)
        assert sum(v['observed_segments'] for v in r)==144
        assert native['fresh144_each'][arm]['actual_segments']==144
        assert native['timing96_each'][arm]['actual_segments']==128
        data=slo['arms'][arm];assert data['status']=='FAIL'
        assert data['components']['every_request_RTF_at_most_1']=='FAIL'
        assert data['components']['every_request_zero_underruns']=='FAIL'
        expected=sorted(indexed[(arm,i)]['timing']['first_audio_ms'] for i in range(96))[math.ceil(.95*96)-1]
        assert expected==data['onset']['p95_ms'] and data['onset']['metric']=='first_audio_ms'
    for p in root.iterdir():
        if p.suffix in ('.json','.jsonl'):
            if p.suffix=='.json':assert_public(load(p))
            else:
                for row in rows(p):assert_public(row)
    return {'status':'PASS_LOCAL_PUBLIC_EXPORT_STRUCTURE_AND_PRIVACY','cases':192,'pairs':96,'fresh_pairs':144,
            'native144_each':True,'public_model_checker_executed':False,'new_hardware_or_ML_calls':0}

def verify_manifest(root):
    root=Path(root);manifest=load(root/'MANIFEST.json')
    for name,item in manifest['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=root/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    return True

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    result=inspect_data(root)
    if (root/'MANIFEST.json').exists():assert verify_manifest(root)
    print(json.dumps(result,sort_keys=True))

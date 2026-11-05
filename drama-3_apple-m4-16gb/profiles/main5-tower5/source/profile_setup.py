"""Stage explicit existing-input MAIN5 profile commands; performs no model work."""
import argparse,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_common import *
def stage(output,python,tower5_model,original_model,timeout_seconds):
    source=verify_packet();end=deadline(seconds=timeout_seconds)
    parent=explicit(tower5_model,exists=True);original=explicit(original_model,exists=True)
    out=explicit(output);require(not out.exists() and not out.is_relative_to(HERE) and not out.is_relative_to(parent) and not out.is_relative_to(original),'Fresh external profile root required')
    require(sha(parent/'origin.json')=='a47ca84ca752d73d07b311ba575e0b8a2817a8751cc50238198f0c7f1cda46ce','Protected Tower5 parent identity required')
    selected=selected_python(python);disk(out.parent);out.mkdir()
    shutil.copytree(HERE/'measured',out/'runtime',copy_function=shutil.copy2);verify_measured(out/'runtime')
    model=out/'model-dc860';evidence=out/'build-evidence';tools=out/'guard-runtime'
    plan={'status':'OPTIONAL_MAIN5_PROFILE_STAGED_NO_NUMERICAL_OR_SERVING_ADMISSION','packaging_source_seal_sha256':source,'measured_runtime_source_manifest_sha256':sha(out/'runtime/RUNTIME_MANIFEST.json'),'profile':'native-main5-tower5-v1','default_repository_profile_changed':False,'input_route':'EXPLICIT_PROTECTED_TOWER5_AND_ORIGINAL_ROOTS_PENDING_GUARDED_VERIFICATION','fresh_install_proof':None,'actual_build':None,'actual_loader_smoke':None,'actual_quality':None,'actual_performance':None,'native_tools_command':[str(selected),'-B',str(HERE/'prepare_guard_tools.py'),'--output',str(tools),'--timeout-seconds',str(timeout_seconds)],'guarded_overlay_command':[str(selected),'-B',str(HERE/'build_guarded.py'),'--python',str(selected),'--guard-runtime',str(tools),'--base-model',str(parent),'--original-model',str(original),'--output-model',str(model),'--receipt-dir',str(evidence),'--deadline-utc',end],'post_build':'Require actual exact750arrays/966241610B MAIN5 NPZ/984combined/dc860 origin, allrestored+closed guard receipts before separately admitted guarded coldloader+oneHTTP/WAV smoke. No current192/288 or crossprecision grades inherited.','fresh_install':'Not supplied or claimed. First reproduce protected9343 via its separately pinned parent route; existing roots never count as fresh install.'}
    atomic(out/'PROFILE_PLAN.json',plan);return plan
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--python',required=True);p.add_argument('--tower5-model',required=True);p.add_argument('--original-model',required=True);p.add_argument('--timeout-seconds',type=float,required=True);a=p.parse_args();stage(a.output,a.python,a.tower5_model,a.original_model,a.timeout_seconds)
if __name__=='__main__':main()

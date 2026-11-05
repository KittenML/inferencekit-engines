"""Stage an OPTIONAL profile and explicit commands; never launches model jobs.

No repository/default modification, model discovery, download or numerical work.
"""
import argparse,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_common import *

TRACK='e2-mse-e3last8-main6-tower5-bf16heads-emb4-g128-v52'
PARENT_TRACK='e2-mse-e3last8-d6-t6-bf16heads-emb4-g128'

def stage(output,python,base_model,original_model,timeout_seconds,*,fresh=False):
    source=verify_packet();end=deadline(seconds=timeout_seconds)
    out=explicit(output);require(not out.exists() and not out.is_relative_to(HERE),'Fresh external profile root required')
    disk(out.parent);out.mkdir()
    # Stage the retained measured runtime bytes and executable modes.
    shutil.copytree(HERE/'measured',out/'runtime',copy_function=shutil.copy2)
    verify_measured(out/'runtime')
    shutil.copytree(HERE/'parent-installer-source',out/'parent-work',copy_function=shutil.copy2)
    parent=load(out/'parent-work/selected_recipe.json')
    require(parent['track']==PARENT_TRACK and parent['revision'].startswith('742d2f153e577535'),'Parent installer must build742')
    if fresh:
        base=out/'parent-work/models'/PARENT_TRACK
        original=out/'parent-work/models/original-fp8'
        selected=out/'parent-work/.venv/bin/python'
        require(base_model is None and original_model is None,'Fresh plan cannot silently use existing roots')
        install=['/bin/sh',str(out/'parent-work/install.sh')]
    else:
        require(base_model is not None and original_model is not None,'Explicit existing input roots required')
        base=explicit(base_model,exists=True);original=explicit(original_model,exists=True)
        selected=selected_python(python);install=None
    model=out/'model9343';evidence=out/'build-evidence';tools=out/'guard-runtime'
    plan={'status':'OPTIONAL_PROFILE_STAGED_NO_NUMERICAL_OR_SERVING_ADMISSION',
          'packaging_source_seal_sha256':source,'measured_runtime_source_manifest_sha256':sha(out/'runtime/RUNTIME_MANIFEST.json'),
          'profile':'native-main6-tower5-v1','default_repository_profile_changed':False,
          'input_route':'FRESH_PARENT_INSTALL_PLAN_NOT_EXECUTED' if fresh else 'EXPLICIT_INPUTS_PENDING_FULL_GUARDED_VERIFICATION',
          'parent_install_command':install,'parent_install_cwd':str(out/'parent-work'),
          'fresh_install_proof':None,'actual_build':None,'actual_loader_smoke':None,'actual_quality':None,'actual_performance':None,
          'native_tools_command':[sys.executable,str(HERE/'prepare_guard_tools.py'),'--output',str(tools),'--timeout-seconds',str(timeout_seconds)],
          'guarded_overlay_command':[str(selected),'-B',str(HERE/'build_guarded.py'),'--python',str(selected),
              '--guard-runtime',str(tools),'--base-model',str(base),'--original-model',str(original),
              '--output-model',str(model),'--receipt-dir',str(evidence),'--deadline-utc',end],
          'proposed_serving_command_requires_separate_existing_runtime_guard':[str(selected),'-B',str(out/'runtime/serve_selected.py'),'--model-dir',str(model)],
          'parent_installer_supervision':'Fresh installer is a separate full cold-install job. Requires its own finite caller deadline,pressure/ownership guard,complete process-tree cleanup and actual install receipt; Tower5 builder512MiB/1GiB cap is not a parent-installer fit claim.',
          'fresh_sequence':['Stage exact parent installer in an empty parent-work; no overlay flags',
                            'Run pinned parent install.sh under independently admitted cold-install supervision; retain actual commands/runtime/logs/cleanup/model-verification',
                            'Verify complete742 and original roots; do not replace fresh proof with V14 or reuse',
                            'Prepare native guard probes; execute exact guarded overlay command',
                            'Join new actual builder/closure/source receipt to exact9343 bytes; keep original derivation metadata unchanged',
                            'Run separately guarded actual loader/checker smoke before any serving/release claim'],
          'semantics':'Staging plan only. No model input hash verification, numerical conversion, install, smoke, serving or current proof is inferred.'}
    atomic(out/'PROFILE_PLAN.json',plan);return plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--python')
    p.add_argument('--base-model');p.add_argument('--original-model');p.add_argument('--fresh-install-plan',action='store_true');p.add_argument('--timeout-seconds',type=float,required=True);a=p.parse_args()
    require(a.fresh_install_plan or a.python is not None,'Explicit existing Python required')
    stage(a.output,a.python,a.base_model,a.original_model,a.timeout_seconds,fresh=a.fresh_install_plan)
if __name__=='__main__':main()

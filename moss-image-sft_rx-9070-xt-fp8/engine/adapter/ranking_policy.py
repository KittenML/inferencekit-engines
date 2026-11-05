"""Explicit existing-native frontend options; no native imports or execution."""
from moss_omni_contract import require

FIELDS=('step_execution','enforce_eager','diffusion_compile_granularity',
        'diffusion_compile_dynamic','enable_diffusion_pipeline_profiler')

def selection(compile_policy,resident_layers,pin_memory,pipeline_profiler):
 require(compile_policy in ('eager','regional'),'Explicit eager/regional policy required')
 require(type(resident_layers) is int and resident_layers in (36,38,40,42),'Resident layers outside finite registered options')
 require(type(pin_memory) is bool and type(pipeline_profiler) is bool,'Explicit boolean frontend options required')
 return {'step_execution':False,'enforce_eager':compile_policy=='eager',
         'diffusion_compile_granularity':'regional','diffusion_compile_dynamic':True,
         'enable_diffusion_pipeline_profiler':pipeline_profiler,
         'resident_layers':resident_layers,'pin_memory':pin_memory}

def require_expected_mode(actual,expected):
 require(expected['step_execution'] is False and actual.step_execution is False,'Native full-forward route required; step remains excluded')
 require(expected['diffusion_compile_granularity']=='regional' and expected['diffusion_compile_dynamic'] is True,'Only pinned native regional granularity/dynamic policy admitted')
 for name in FIELDS:
  value=expected[name];observed=getattr(actual,name)
  require(type(observed) is type(value) and observed==value,'Actual selected native mode differs: '+name)

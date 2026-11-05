"""Observe native-configured main-leaf callables outside their compiled bodies.

Lightweight mode only reads attributes. Diagnostic mode adds one Python count
before entry into each actual Dynamo callable and restores it after the request.
This does not establish that every operator compiled or that no fallback ran.
"""
import functools

def require(value,message):
 if not value:raise RuntimeError(message)

def begin(transformer,*,regional,diagnostic):
 blocks=[(name,module) for name,module in transformer.named_modules() if type(module).__name__=='MOSSImageBlock']
 require(len(blocks)==42,'Expected actual42 native main block instances')
 inventory=[];bindings=[];counts={}
 for index,(name,module) in enumerate(blocks):
  old=getattr(module,'_omni_original_forward',None);forward=getattr(module,'forward',None)
  field='_omni_original_forward' if hasattr(old,'_torchdynamo_orig_callable') else 'forward' if hasattr(forward,'_torchdynamo_orig_callable') else None
  require(getattr(module,'_compiled_call_impl',None) is None,'Unexpected whole-module compiled implementation')
  inventory.append({'index':index,'module':name,'class':type(module).__name__,'actual_dynamo_callable_field':field})
  if field:bindings.append((index,name,module,field,getattr(module,field)))
 require(len(bindings)==(42 if regional else 0),'Actual native compiled leaf inventory differs from selected policy')
 token={'inventory':inventory,'regional':regional,'diagnostic':diagnostic,'bindings':[],'counts':counts,'restored':False}
 if diagnostic:
  try:
   for index,name,module,field,fn in bindings:
    label=name+'.'+field;counts[label]=0
    def wrap(original,label):
     @functools.wraps(original)
     def counted(*args,**kwargs):
      counts[label]+=1
      return original(*args,**kwargs)
     return counted
    counted=wrap(fn,label);setattr(module,field,counted);token['bindings'].append((index,label,module,field,fn,counted))
  except BaseException:
   restore(token);raise
 return token

def restore(token):
 if token['restored']:return
 changed=[]
 for _,label,module,field,original,counted in token['bindings']:
  if getattr(module,field) is not counted:changed.append(label)
  setattr(module,field,original)
 token['restored']=True
 require(not changed,'Callable changed during diagnostic observation: '+str(changed))

def finish(token,*,completed):
 try:
  if completed and token['diagnostic'] and token['regional']:
   for index,label,*_ in token['bindings']:
    require(token['counts'][label]==(50 if index==22 else 100),'Compiled callable entry count differs: '+label)
  return {'actual_main_instances':len(token['inventory']),
          'actual_compiled_leaf_count':sum(row['actual_dynamo_callable_field'] is not None for row in token['inventory']),
          'inventory':token['inventory'],'diagnostic_counter_enabled':token['diagnostic'],
          'compiled_callable_entries':dict(token['counts']) if token['diagnostic'] else None,
          'total_compiled_callable_entries':sum(token['counts'].values()) if token['diagnostic'] else None,
          'completed':completed,
          'definition':'One Python increment before entry into an actual native-configured Dynamo callable. Outside compiled body; does not prove all operators compiled or absence of fallback. Disabled in every lightweight ranking/timing cell.'}
 finally:restore(token)

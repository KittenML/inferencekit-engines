"""Fixed-shape original GDN blocks only; native prefill and attention retained."""
import hashlib,inspect,time
from types import MethodType

class FixedGDN:
 def __init__(self,runtime,enabled,expected_sources,expected_drama_sha):
  from mlx_audio.lm.models import qwen3_5,gated_delta,cache,qwen3_next
  from mlx.utils import tree_flatten
  for name,module in [('qwen3_5',qwen3_5),('gated_delta',gated_delta),('cache',cache),('qwen3_next',qwen3_next)]:
   assert hashlib.sha256(open(inspect.getfile(module),'rb').read()).hexdigest()==expected_sources[name]
  self.runtime=runtime;self.mx=runtime.mx;self.enabled=enabled;self.active=False
  self.layers=runtime.model.backbone.model.layers
  assert len(self.layers)==40 and sum(layer.is_linear for layer in self.layers)==30
  self.blocks=[layer.linear_attn for layer in self.layers if layer.is_linear]
  assert all(type(layer) is qwen3_5.GatedDeltaNet and not layer.training for layer in self.blocks)
  self.indices={id(layer):i for i,layer in enumerate(self.blocks)}
  self.original_call=qwen3_5.GatedDeltaNet.__call__;self.original_step=runtime.model._step
  assert hashlib.sha256(open(inspect.getfile(self.original_call),'rb').read()).hexdigest()==expected_sources['qwen3_5']
  assert self.original_step.__func__ is type(runtime.model)._step and hashlib.sha256(open(inspect.getfile(self.original_step),'rb').read()).hexdigest()==expected_drama_sha
  self.array_cache_type=cache.ArraysCache;self.steps=0;self.decode_calls=[0]*30;self.compiled_calls=[0]*30
  self.prefill_calls=[0]*30;self.prefill_S1_calls=[0]*30;self.traces=[0]*30;self.first_call_host_ns=[None]*30
  self.mx.eval(runtime.model.parameters())
  self.parameters_before={name:(id(value),tuple(value.shape),str(value.dtype)) for name,value in tree_flatten(runtime.model.parameters())}
  self.functions=[]
  original_call=self.original_call;arrays_cache_type=cache.ArraysCache;traces=self.traces
  def create(layer,index):
   def graph(inputs,conv,recurrent):
    traces[index]+=1
    local=arrays_cache_type(2);local.state=[conv,recurrent]
    out=original_call(layer,inputs,mask=None,cache=local)
    return out,local[0],local[1]
   return self.mx.compile(graph,shapeless=False)
  for i,layer in enumerate(self.blocks):self.functions.append(create(layer,i) if enabled else None)
  def observed(layer,inputs,mask=None,cache=None):
   index=self.indices.get(id(layer))
   if index is None:return self.original_call(layer,inputs,mask=mask,cache=cache)
   if not self.active:
    self.prefill_calls[index]+=1;self.prefill_S1_calls[index]+=int(inputs.shape[1]==1)
    return self.original_call(layer,inputs,mask=mask,cache=cache)
   self.decode_calls[index]+=1
   assert inputs.shape==(1,1,2048) and mask is None and type(cache) is self.array_cache_type
   assert cache.lengths is None and cache.left_padding is None and len(cache.state)==2
   assert cache[0].shape==(1,3,8192) and cache[1].shape==(1,32,128,128)
   if not self.enabled:return self.original_call(layer,inputs,mask=mask,cache=cache)
   self.compiled_calls[index]+=1;state_list=cache.state
   first=self.traces[index]==0;began=time.perf_counter_ns() if first else None
   out,conv,recurrent=self.functions[index](inputs,cache[0],cache[1])
   if first:self.first_call_host_ns[index]=time.perf_counter_ns()-began
   cache[0]=conv;cache[1]=recurrent;cache.advance(1)
   assert cache.state is state_list and cache.lengths is None and cache.left_padding is None
   return out
  def step(model,*args,**kwargs):
   assert not self.active,'Nested decode scope is unsupported'
   self.active=True;self.steps+=1
   try:return self.original_step(*args,**kwargs)
   finally:self.active=False
  self.observed=observed;self.step_wrapper=MethodType(step,runtime.model)
  qwen3_5.GatedDeltaNet.__call__=observed
  object.__setattr__(runtime.model,'_step',self.step_wrapper)

 def snapshot(self):
  assert not self.active
  return {'steps':self.steps,'decode_calls':list(self.decode_calls),'compiled_calls':list(self.compiled_calls),
          'prefill_calls':list(self.prefill_calls),'native_prefill_S1_calls':list(self.prefill_S1_calls),
          'compiler_traces':list(self.traces),'ten_full_attention_KV_blocks_native':True,
          'padding_or_attention_change':False,'state_arrays_per_GDN':2,
          'first_graph_call_host_ns':list(self.first_call_host_ns),
          'compile_cost_scope':'First call host span only; later GPU codegen/execution remains included in full disclosed startup warmup.'}

 def finish(self):
  from mlx_audio.lm.models import qwen3_5
  from mlx.utils import tree_flatten
  assert not self.active and qwen3_5.GatedDeltaNet.__call__ is self.observed
  result=self.snapshot()
  result['parameter_objects_shapes_dtypes_unchanged']=self.parameters_before=={name:(id(value),tuple(value.shape),str(value.dtype)) for name,value in tree_flatten(self.runtime.model.parameters())}
  result['all_decode_invocations_accounted']=all(n==self.steps for n in self.decode_calls) and all(n==(self.steps if self.enabled else 0) for n in self.compiled_calls)
  result['trace_counts_valid']=all(n==(1 if self.enabled and self.steps else 0) for n in self.traces)
  qwen3_5.GatedDeltaNet.__call__=self.original_call
  object.__setattr__(self.runtime.model,'_step',self.original_step)
  result['method_hooks_restored']=True
  return result

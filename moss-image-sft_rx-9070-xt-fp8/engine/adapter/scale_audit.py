"""Candidate-only tagging and startup scale audit; no runtime hooks installed here."""
import math
import torch

FACTOR = 1.03125  # exactly representable binary 33/32
NAMES = ('qkv', 'o', 'ffn_up_gate', 'ffn_down')
SHARDS = {'qkv': {'q','k','v'}, 'o': {'whole'}, 'ffn_up_gate': {0,1}, 'ffn_down': {'whole'}}
POLICY = {'kind':'moss_main_dit_fp8_prequant_weight_scale', 'factor':FACTOR,
          'projection_count':168, 'weights':'online_fp8_per_tensor', 'activation':'dynamic',
          'numerics_changed':True}

class ScaleAudit:
    """Wrap already-installed ShardTracker process methods before real loading.

    `online_module` must be the imported, hash-verified patched host module.
    Calling this helper is an integration action; this module does not call it.
    """
    def __init__(self, transformer, tracker, online_module):
        layers=transformer.layers
        if len(layers)!=42 or set(tracker.records)!={(i,n) for i in range(42) for n in NAMES}:
            raise ValueError('Expected exactly the 168 tracked projections')
        plans=[]
        for i,block in enumerate(layers):
            for name in NAMES:
                layer=block.proj[name]; method=layer.quant_method
                if (getattr(layer,'tp_size',None)!=1 or hasattr(layer,'_moss_fp8_scale_factor') or
                    getattr(method,'requantization_source',object()) is not None or
                    type(method) is not online_module.Fp8PerTensorOnlineLinearMethod):
                    raise ValueError(f'Invalid target at {i}.{name}')
                plans.append((i,name,layer,method,method.process_weights_after_loading))
        if len({id(x[2]) for x in plans})!=168 or len({id(x[3]) for x in plans})!=168:
            raise ValueError('Aliased layer or quant method')
        self.rows={}; self.restores=[]; self.methods=[x[3] for x in plans]; self.active=True
        for i,name,layer,method,original in plans:
            state=tracker.records[(i,name)]
            def wrap(i,name,layer,original,state):
                def process(actual_layer,*args,**kwargs):
                    if not self.active or actual_layer is not layer or args or kwargs or (i,name) in self.rows:
                        raise ValueError('Invalid or duplicate processing')
                    seen=set(state['seen'])
                    if state['inflight'] is not None: seen.add(state['inflight'])
                    if (seen!=SHARDS[name] or set(state['replayed'])!=SHARDS[name] or
                        state['finalizations'] or state['processing_active']):
                        raise ValueError('No complete shard replay before scale read')
                    if getattr(layer,'_moss_fp8_scale_factor',None)!=FACTOR or layer.weight.is_meta or layer.weight.dtype!=torch.bfloat16:
                        raise ValueError('Candidate factor or materialized BF16 weight differs')
                    # TP=1 makes upstream amax_for_tp_weight_quant a no-op.
                    base=online_module._fp8_scale(online_module.weight_amax(layer.weight).reshape(1))
                    if base.dtype!=torch.float32 or base.numel()!=1:
                        raise ValueError('Invalid reference scale')
                    before=float(base.item()); expected=float((base*FACTOR).item())
                    if not (math.isfinite(before) and before>0 and math.isfinite(expected) and expected>0):
                        raise ValueError('Non-finite or zero scale')
                    result=original(actual_layer)
                    actual=getattr(layer,'weight_scale',None)
                    if actual is None or actual.dtype!=torch.float32 or actual.numel()!=1 or actual.is_meta:
                        raise ValueError('Processed scale missing')
                    value=float(actual.item())
                    if value!=expected or not getattr(layer,'_already_called_process_weights_after_loading',False):
                        raise ValueError('Processed scale differs from the declared pre-quant recipe')
                    self.rows[(i,name)]={'layer':i,'projection':name,'base_scale_fp32':before,
                                         'factor':FACTOR,'expected_scale_fp32':expected,
                                         'actual_scale_fp32':value,'method_processed':True}
                    return result
                return process
            layer._moss_fp8_scale_factor=FACTOR
            method.process_weights_after_loading=wrap(i,name,layer,original,state)
            self.restores.append((layer,method,original))
    def receipt(self):
        if set(self.rows)!={(i,n) for i in range(42) for n in NAMES}:
            raise ValueError('Missing one or more projection scale observations')
        return {'policy':dict(POLICY),'projection_count':168,
                'projections':[self.rows[k] for k in sorted(self.rows)],
                'verdict':'startup scales observed; execution and quality UNPROVEN'}
    def install_reload_guard(self):
        if self.active or len(self.methods)!=168:
            raise ValueError('Restore auditing before installing reload guard')
        def reject_reload(*args, **kwargs):
            raise RuntimeError('Re-quantization is not admitted in this candidate')
        for method in self.methods:
            method.process_weights_after_loading=reject_reload

    def restore(self):
        for layer,method,original in self.restores:
            method.process_weights_after_loading=original
            # Keep the tag for runtime configuration verification. It must not be
            # applied again unless a separately audited reload is implemented.
        self.active=False; self.restores.clear()


def validate_receipt(receipt):
    """Validate recorded startup factors; still not evidence of inference dispatch."""
    if not isinstance(receipt,dict) or receipt.get('policy')!=POLICY or receipt.get('projection_count')!=168:
        raise ValueError('Wrong scale policy or projection count')
    if receipt.get('verdict')!='startup scales observed; execution and quality UNPROVEN':
        raise ValueError('Invalid scale verdict')
    rows=receipt.get('projections')
    if not isinstance(rows,list) or len(rows)!=168:
        raise ValueError('Missing scale rows')
    expected={(i,n) for i in range(42) for n in NAMES}; seen=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid row')
        key=(row.get('layer'),row.get('projection'))
        if key not in expected or key in seen or row.get('factor')!=FACTOR or row.get('method_processed') is not True:
            raise ValueError('Invalid projection identity or method receipt')
        seen.add(key)
        before=row.get('base_scale_fp32'); target=row.get('expected_scale_fp32'); actual=row.get('actual_scale_fp32')
        if any(not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in (before,target,actual)):
            raise ValueError('Invalid scale values')
        recomputed=float((torch.tensor(before,dtype=torch.float32)*FACTOR).item())
        if target!=recomputed or actual!=target:
            raise ValueError('Scale factor not established')
    if seen!=expected:raise ValueError('Missing projection identities')
    return receipt


def install_model_reload_guard(transformer_class):
    """Reject a second model-wide load before the checkpoint loader runs."""
    def reject_model_reload(*args, **kwargs):
        raise RuntimeError("Model weight reload is not admitted in this candidate")
    transformer_class.load_weights = reject_model_reload


def validate_restored_scales(scale_receipt, projection_audit):
    validate_receipt(scale_receipt)
    produced={(r['layer'],r['projection']):r['actual_scale_fp32'] for r in scale_receipt['projections']}
    restored={(r['layer'],r['projection']):r['scale_min'] for r in projection_audit['projections']}
    if len(produced)!=168 or len(restored)!=168 or produced!=restored:
        raise ValueError('FP8 scales changed during initial offload/resident reload')
    return True

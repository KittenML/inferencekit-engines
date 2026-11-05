"""Candidate-only fail-closed hooks around actual MOSS projection loaders.

Receipts prove calls entered/returned and processing completed in this Python
path, not copied tensor values, checkpoint identity, actual GPU math or quality.
"""
from functools import wraps
import inspect
import torch

# On-disk/native loader before online processing is [out, in]; the selected
# per-tensor online FP8 implementation transposes weight after quantization.
SPECS = {
    'qkv': {'weight': (9216, 3072), 'shards': {'q': (3072,3072), 'k': (3072,3072), 'v': (3072,3072)}},
    'o': {'weight': (3072,3072), 'shards': {'whole': (3072,3072)}},
    'ffn_up_gate': {'weight': (16384,3072), 'shards': {0: (8192,3072), 1: (8192,3072)}},
    'ffn_down': {'weight': (3072,8192), 'shards': {'whole': (3072,8192)}},
}

class ShardTracker:
    def __init__(self, transformer):
        if len(transformer.layers) != 42:
            raise ValueError('Expected 42 main transformer blocks')
        self.records = {}
        self.restores = []
        for idx, block in enumerate(transformer.layers):
            for name, spec in SPECS.items():
                layer = block.proj[name]
                param = layer.weight
                if tuple(param.shape) != spec['weight'] or param.dtype != torch.bfloat16:
                    raise ValueError(f'Unexpected pre-load shape or dtype: {idx}.{name}')
                if getattr(layer, 'tp_size', None) != 1:
                    raise ValueError('Only TP=1 supported')
                method = layer.quant_method
                if not hasattr(method, "requantization_source") or method.requantization_source is not None:
                    raise ValueError("Serialized checkpoint requantization is not admitted")
                orig_loader, orig_process = param.weight_loader, method.process_weights_after_loading
                if getattr(orig_loader, '__name__', None) != 'online_process_loader':
                    raise ValueError('Expected installed deferred online loader')
                underlying = getattr(orig_loader, '__wrapped__', None)
                if underlying is None or getattr(underlying, '__name__', '') == 'online_process_loader':
                    raise ValueError('Unexpected online loader unwrapping chain')
                key = (idx, name)
                state = {'seen': {}, 'replayed': {}, 'inflight': None, 'finalizations': 0, 'processing_active': False}
                self.records[key] = state
                def make_loader(orig, layer, param, spec, state, key):
                    @wraps(orig)
                    def load(*args, **kwargs):
                        if kwargs or len(args) not in (2,3) or args[0] is not param:
                            raise ValueError(f'Unexpected loader call: {key}')
                        shard = args[2] if len(args) == 3 else 'whole'
                        if type(shard) not in (str, int) or shard not in spec['shards']:
                            raise ValueError(f'Unexpected shard: {key}, {shard!r}')
                        tensor = args[1]
                        if tuple(tensor.shape) != spec['shards'][shard] or tensor.dtype != torch.bfloat16:
                            raise ValueError(f'Partial or incorrect BF16 shard: {key}, {shard!r}')
                        if shard in state['seen'] or state['inflight'] is not None or state['finalizations']:
                            raise ValueError(f'Duplicate or late shard: {key}, {shard!r}')
                        state['inflight'] = shard
                        try:
                            result = orig(*args, **kwargs)
                        except BaseException:
                            state['inflight'] = None
                            raise
                        state['seen'][shard] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'returned': True}
                        state['inflight'] = None
                        return result
                    return load
                def make_replay(underlying, spec, state, key):
                    signature = inspect.signature(underlying)
                    @wraps(underlying)
                    def replay(*args, **kwargs):
                        bound = signature.bind(*args, **kwargs)
                        bound.apply_defaults()
                        param, tensor = bound.arguments['param'], bound.arguments['loaded_weight']
                        shard = bound.arguments.get('loaded_shard_id')
                        shard = 'whole' if shard is None else shard
                        if shard not in spec['shards'] or shard in state['replayed']:
                            raise ValueError(f'Unexpected or duplicate actual shard replay: {key}, {shard!r}')
                        if (tuple(tensor.shape) != spec['shards'][shard] or tensor.dtype != torch.bfloat16
                                or getattr(param, 'is_meta', True)):
                            raise ValueError(f'Partial shard or unmaterialized replay: {key}, {shard!r}')
                        result = underlying(*args, **kwargs)
                        state['replayed'][shard] = True
                        return result
                    return replay
                def make_process(orig, layer, spec, state, key):
                    @wraps(orig)
                    def process(actual_layer, *args, **kwargs):
                        if actual_layer is not layer or args or kwargs or state['finalizations'] or state['processing_active']:
                            raise ValueError(f'Unexpected or duplicate processing: {key}')
                        seen = set(state['seen'])
                        if state['inflight'] is not None:
                            seen.add(state['inflight'])
                        if seen != set(spec['shards']) or set(state['replayed']) != set(spec['shards']):
                            raise ValueError(f'Processing before every shard arrived: {key}')
                        state['processing_active'] = True
                        try:
                            result = orig(actual_layer)
                        finally:
                            state['processing_active'] = False
                        if not getattr(layer, '_already_called_process_weights_after_loading', False):
                            raise ValueError(f'Processing did not finalize: {key}')
                        state['finalizations'] += 1
                        return result
                    return process
                orig_loader.__wrapped__ = make_replay(underlying, spec, state, key)
                param.weight_loader = make_loader(orig_loader, layer, param, spec, state, key)
                method.process_weights_after_loading = make_process(orig_process, layer, spec, state, key)
                self.restores.append((param, orig_loader, underlying, method, orig_process))
    def require_shards(self):
        for key, state in self.records.items():
            if state['inflight'] is not None or set(state['seen']) != set(SPECS[key[1]]['shards']):
                raise ValueError(f'Missing or unfinished shard calls: {key}')
    def receipt(self):
        self.require_shards()
        rows=[]
        for (idx,name), state in sorted(self.records.items()):
            if state['finalizations'] != 1 or state['processing_active'] or set(state['replayed']) != set(SPECS[name]['shards']):
                raise ValueError(f'Missing or duplicate processing: {(idx,name)}')
            rows.append({'layer':idx,'projection':name,'shards':[
                {'id':str(shard), **state['seen'][shard], 'replay_returned':state['replayed'][shard]} for shard in SPECS[name]['shards']],
                'processing_completed_events':state['finalizations']})
        return {'verdict':'observed loader and processing calls; value correctness UNPROVEN', 'projections':rows}
    def restore(self):
        for param, loader, underlying, method, process in self.restores:
            loader.__wrapped__ = underlying
            param.weight_loader = loader
            method.process_weights_after_loading = process
        self.restores.clear()

def validate_shard_receipt(receipt):
    if not isinstance(receipt, dict) or receipt.get('verdict') != 'observed loader and processing calls; value correctness UNPROVEN':
        raise ValueError('Missing loader/processing receipt')
    rows = receipt.get('projections')
    if not isinstance(rows,list) or len(rows)!=42*len(SPECS):
        raise ValueError('Incomplete shard coverage')
    expected={(i,n) for i in range(42) for n in SPECS}
    seen=set()
    for row in rows:
        key=(row.get('layer'),row.get('projection'))
        if key not in expected or key in seen or row.get('processing_completed_events') != 1:
            raise ValueError('Invalid projection or processing coverage')
        seen.add(key)
        shards=row.get('shards')
        spec=SPECS[key[1]]['shards']
        wanted={str(k):list(v) for k,v in spec.items()}
        if not isinstance(shards,list) or len(shards)!=len(wanted):
            raise ValueError('Missing or duplicate shard records')
        found={}
        for item in shards:
            if not isinstance(item,dict) or item.get('id') in found or item.get('id') not in wanted:
                raise ValueError('Duplicate or unexpected shard')
            if item.get('shape') != wanted[item['id']] or item.get('dtype') != 'torch.bfloat16' or item.get('returned') is not True or item.get('replay_returned') is not True:
                raise ValueError('Partial or invalid shard record')
            found[item['id']]=True
        if set(found)!=set(wanted):
            raise ValueError('Missing shard')
    if seen != expected:
        raise ValueError('Missing projection')
    return receipt

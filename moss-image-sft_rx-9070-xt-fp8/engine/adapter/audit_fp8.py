"""Candidate-only startup audit of loaded DiT block projection metadata.

This inspects layers and scales; it does not prove runtime activation dtype,
executed kernels, quality, cache discipline, or memory during a request.
"""
import math
import torch
from shard_observer import SPECS

_PROJECTIONS=('qkv','o','ffn_up_gate','ffn_down')
_FP8_DTYPES={'torch.float8_e4m3fn','torch.float8_e4m3fnuz'}


def audit_transformer(transformer, *, expected_layers=42):
    layers=transformer.layers
    if len(layers)!=expected_layers:
        raise ValueError(f'expected {expected_layers} DiT layers, got {len(layers)}')
    rows=[]
    for index,block in enumerate(layers):
        for name in _PROJECTIONS:
            layer=block.proj[name]
            method=getattr(layer,'quant_method',None)
            method_name=type(method).__name__ if method is not None else 'missing'
            if method_name!='Fp8PerTensorOnlineLinearMethod':
                raise ValueError(f'layers.{index}.proj.{name}: expected online FP8 method, got {method_name}')
            weight=layer.weight
            if str(weight.dtype) not in _FP8_DTYPES or weight.device.type=='meta':
                raise ValueError(f'layers.{index}.proj.{name}: FP8 weight absent or still on meta')
            if tuple(weight.shape) != tuple(reversed(SPECS[name]['weight'])):
                raise ValueError(f'layers.{index}.proj.{name}: unexpected processed weight shape={tuple(weight.shape)}, stride={tuple(weight.stride())}, kernel={type(method.fp8_linear).__module__}.{type(method.fp8_linear).__name__}')
            if getattr(layer,'orig_dtype',None)!=torch.bfloat16:
                raise ValueError(f'layers.{index}.proj.{name}: expected BF16 original activation dtype')
            if not getattr(layer,'_already_called_process_weights_after_loading',False):
                raise ValueError(f'layers.{index}.proj.{name}: online weight processing is incomplete')
            scale=getattr(layer,'weight_scale',None)
            if scale is None or scale.device.type=='meta' or scale.numel() != 1 or scale.dtype != torch.float32:
                raise ValueError(f'layers.{index}.proj.{name}: missing weight scale')
            # This is startup-only validation, not part of a timed request.
            values=scale.detach().float().cpu()
            if not bool(torch.isfinite(values).all()) or not bool((values>0).all()):
                raise ValueError(f'layers.{index}.proj.{name}: invalid weight scale')
            kernel=getattr(method,'fp8_linear',None)
            if kernel is None:
                raise ValueError(f'layers.{index}.proj.{name}: FP8 linear dispatch object missing')
            rows.append({'layer':index,'projection':name,'method':method_name,
                         'weight_dtype':str(weight.dtype),'weight_shape':list(weight.shape),
                         'weight_device':str(weight.device),'weight_numel':weight.numel(),
                         'activation_input_configured':str(layer.orig_dtype),
                         'scale_shape':list(scale.shape),'scale_dtype':str(scale.dtype),
                         'scale_min':float(values.min()),'scale_max':float(values.max()),
                         'load_processing_complete':True,
                         'selected_linear_kernel':type(kernel).__module__ + '.' + type(kernel).__name__})
    return {'verdict':'metadata validated; executed path UNPROVEN', 'layers':expected_layers,
            'projection_count':len(rows),'projections':rows,
            'limitations':['Does not observe per-request activation dtypes or GEMM dispatch.',
                           'Does not establish image quality, residency, cache policy or speed.',
                           'Whole-device memory and the actual executing kernel require separate measurement.']}


def validate_audit_receipt(receipt):
    """Reject absent/partial metadata, without turning selection into execution proof."""
    if not isinstance(receipt, dict) or receipt.get('verdict') != 'metadata validated; executed path UNPROVEN':
        raise ValueError('FP8 metadata audit missing')
    rows = receipt.get('projections')
    if receipt.get('layers') != 42 or receipt.get('projection_count') != 168 or not isinstance(rows, list) or len(rows) != 168:
        raise ValueError('FP8 projection audit incomplete')
    identities = {(i, name) for i in range(42) for name in _PROJECTIONS}
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Malformed FP8 projection row')
        key = (row.get('layer'), row.get('projection'))
        if key not in identities or key in observed:
            raise ValueError('Duplicate or unexpected FP8 projection')
        observed.add(key)
        if (row.get('method') != 'Fp8PerTensorOnlineLinearMethod' or
            row.get('weight_dtype') not in _FP8_DTYPES or
            row.get('weight_shape') != list(reversed(SPECS[key[1]]['weight'])) or
            row.get('weight_numel') != math.prod(SPECS[key[1]]['weight']) or
            row.get('activation_input_configured') != 'torch.bfloat16' or
            row.get('scale_dtype') != 'torch.float32' or
            row.get('load_processing_complete') is not True or
            not isinstance(row.get('scale_min'), (int, float)) or not isinstance(row.get('scale_max'), (int, float)) or
            not math.isfinite(row['scale_min']) or not math.isfinite(row['scale_max']) or not 0 < row['scale_min'] <= row['scale_max'] or
            not isinstance(row.get('selected_linear_kernel'), str) or not row['selected_linear_kernel'] or
            not isinstance(row.get('weight_numel'), int) or row['weight_numel'] <= 0 or
            not isinstance(row.get('scale_shape'), list) or math.prod(row['scale_shape']) != 1 or
            not isinstance(row.get('weight_shape'), list) or not row['weight_shape'] or
            not isinstance(row.get('weight_device'), str) or row['weight_device'].startswith('meta')):
            raise ValueError('Invalid FP8 projection metadata')
    if observed != identities:
        raise ValueError('FP8 projection mapping incomplete')
    return receipt

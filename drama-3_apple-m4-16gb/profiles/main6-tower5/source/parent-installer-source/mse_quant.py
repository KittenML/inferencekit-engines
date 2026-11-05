# SPDX-License-Identifier: Apache-2.0
"""Data-free affine-2 conversion experiment using standard MLX packed tensors.

The immutable v1 grid clips about each group's midpoint. Selection compares
reconstruction MSE against the ORIGINAL input coefficients, including the
stored scale/bias precision. It makes no speech-quality guarantee.
"""
import math
import mlx.core as mx

CLIP_GRID_V1 = (0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)


def quantize_mse_clip_v1(value, group_size=128, row_batch=4096):
    if group_size not in (32, 64, 128) or row_batch < 1:
        raise ValueError('supported affine group size and positive row batch required')
    if len(value.shape) < 2 or value.shape[-1] % group_size:
        raise ValueError('last dimension must contain complete groups')
    if value.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError('floating source coefficients required')
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError('nonfinite original coefficients')
    shape=value.shape; width=shape[-1]; rows=value.reshape(-1,width)
    packed=[]; scales=[]; biases=[]
    stats={'algorithm':'affine2-mse-clip-v1','grid':list(CLIP_GRID_V1),'group_size':group_size,
           'coefficient_count':value.size,'group_count':value.size//group_size,
           'baseline_squared_error':0.0,'selected_squared_error':0.0,'improved_groups':0,
           'speech_quality':'UNPROVEN'}
    for start in range(0, rows.shape[0], row_batch):
        original=rows[start:start+row_batch]
        groups=original.astype(mx.float32).reshape(original.shape[0],-1,group_size)
        low=mx.min(groups,axis=-1,keepdims=True); high=mx.max(groups,axis=-1,keepdims=True)
        center=(low+high)*0.5; radius=(high-low)*0.5
        q,s,b=mx.quantize(original,group_size=group_size,bits=2)
        def error(p,sc,bi):
            restored=mx.dequantize(p,sc.astype(mx.float32),bi.astype(mx.float32),group_size=group_size,bits=2).astype(mx.float32).reshape(groups.shape)
            return mx.mean(mx.square(groups-restored),axis=-1)
        baseline=error(q,s,b); best=baseline
        mx.eval(q,s,b,baseline)
        for fraction in CLIP_GRID_V1:
            clipped=mx.minimum(mx.maximum(groups,center-radius*fraction),center+radius*fraction).reshape(original.shape).astype(original.dtype)
            cq,cs,cb=mx.quantize(clipped,group_size=group_size,bits=2)
            candidate_error=error(cq,cs,cb); improve=candidate_error<best
            # A group contains group_size/16 packed uint32 words at 2 bits.
            q=mx.where(improve[...,None],cq.reshape(*best.shape,group_size//16),q.reshape(*best.shape,group_size//16)).reshape(q.shape)
            s=mx.where(improve,cs,s); b=mx.where(improve,cb,b); best=mx.minimum(best,candidate_error)
            mx.eval(q,s,b,best)
        # Recompute from the selected stored tensors; do not trust the selector alone.
        selected=error(q,s,b)
        max_regression=float(mx.max(selected-baseline).item())
        if not math.isfinite(max_regression) or max_regression>1e-12:
            raise RuntimeError('selected stored coefficients increased group MSE')
        stats['baseline_squared_error']+=float(mx.sum(baseline).item())*group_size
        stats['selected_squared_error']+=float(mx.sum(selected).item())*group_size
        stats['improved_groups']+=int(mx.sum(selected<baseline).item())
        packed.append(q);scales.append(s);biases.append(b)
    q=mx.concatenate(packed).reshape(*shape[:-1],width//16)
    s=mx.concatenate(scales).reshape(*shape[:-1],width//group_size)
    b=mx.concatenate(biases).reshape(s.shape)
    mx.eval(q,s,b)
    stats['baseline_mse']=stats['baseline_squared_error']/stats['coefficient_count']
    stats['selected_mse']=stats['selected_squared_error']/stats['coefficient_count']
    return q,s,b,stats

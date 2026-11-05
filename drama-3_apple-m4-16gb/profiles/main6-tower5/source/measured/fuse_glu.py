# SPDX-License-Identifier: Apache-2.0
# Switch forward follows MLX-LM/MLX-Audio, Copyright © 2023-2024 Apple Inc., MIT.
"""Optional packed-row fusion. Quantized values and selected experts are unchanged."""

def install(model):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_audio.lm.models.switch_layers import SwitchGLU,QuantizedSwitchLinear,_gather_sort,_scatter_unsort
    class PairedProjection(nn.Module):
        def __init__(self,first,second):
            super().__init__()
            if not isinstance(first,QuantizedSwitchLinear) or not isinstance(second,QuantizedSwitchLinear):raise TypeError('fusion requires quantized expert projections')
            if (first.bits,first.group_size,first.mode,first.input_dims,first.output_dims)!=(second.bits,second.group_size,second.mode,second.input_dims,second.output_dims):raise ValueError('incompatible projection formats')
            self.bits=first.bits;self.group_size=first.group_size;self.mode=first.mode
            self.weight=mx.concatenate([first.weight,second.weight],axis=-2)
            self.scales=mx.concatenate([first.scales,second.scales],axis=-2)
            self.biases=mx.concatenate([first.biases,second.biases],axis=-2)
            if 'bias' in first or 'bias' in second:raise ValueError('additive projection bias unsupported in this pinned variant')
            mx.eval(self.weight,self.scales,self.biases)
        def __call__(self,x,indices,sorted_indices=False):
            return mx.gather_qmm(x,self.weight,self.scales,self.biases,rhs_indices=indices,transpose=True,group_size=self.group_size,bits=self.bits,mode=self.mode,sorted_indices=sorted_indices)
    class FusedSwitchGLU(nn.Module):
        def __init__(self,original):
            super().__init__();self.pair=PairedProjection(original.up_proj,original.gate_proj);self.activation=original.activation;self.down_proj=original.down_proj;self.eval()
        def __call__(self,x,indices):
            x=mx.expand_dims(x,(-2,-3));do_sort=indices.size>=64;idx=indices;inv_order=None
            if do_sort:x,idx,inv_order=_gather_sort(x,indices)
            if self.training:idx=mx.stop_gradient(idx)
            up,gate=mx.split(self.pair(x,idx,sorted_indices=do_sort),2,axis=-1)
            x=self.down_proj(self.activation(up,gate),idx,sorted_indices=do_sort)
            if do_sort:x=_scatter_unsort(x,inv_order,indices.shape)
            return x.squeeze(-2)
    from mlx_audio.lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    count=0
    # All replacements are children of the sparse MLP. Preserve the parent path.
    parents=[parent for _,parent in model.named_modules() if isinstance(parent,Qwen3NextSparseMoeBlock)]
    for parent in parents:
        value=parent.get('switch_mlp')
        if isinstance(value,SwitchGLU):
            parent['switch_mlp']=FusedSwitchGLU(value);count+=1
            del value
            mx.clear_cache()
    mx.clear_cache()
    return {'fused_switch_glu_count':count,'operation':'concatenate packed up then gate rows; no requantization'}

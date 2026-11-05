# SPDX-License-Identifier: Apache-2.0
"""Optional compilation of stateless MLPs; cache mutation remains in upstream code.

Weights are immutable after installation. Compiled sampling captures MLX RNG
state explicitly so seed resets and successive draws remain operational.
Compilation can alter floating point rounding and requires quality validation.
"""
MODES=('none','stateless-mlp','stateless-mlp-sampler')


def install(model,mode):
    if mode not in MODES:raise ValueError('unknown compile mode')
    if mode=='none':return {'compiled_modules':0,'compiled_sampler':False}
    import mlx.core as mx
    from mlx_audio.lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from mlx_audio.tts.models.fish_drama3.tower import TowerMLP
    count=0
    for name,module in list(model.named_modules()):
        if not isinstance(module,(Qwen3NextSparseMoeBlock,TowerMLP)):continue
        original=module.__call__
        compiled=mx.compile(original)
        # Instance-specific subclass preserves all parameter paths and cache classes.
        base=type(module)
        def forward(self,x):return self._compiled_impl(x)
        subclass=type('Compiled'+base.__name__,(base,),{'__call__':forward})
        object.__setattr__(module,'__class__',subclass)
        module['_compiled_impl']=compiled
        count+=1
    if mode=='stateless-mlp-sampler':
        from mlx_audio.tts.models.fish_drama3 import drama3
        drama3.sample_top_k_top_p=mx.compile(drama3.sample_top_k_top_p,inputs=mx.random.state,outputs=mx.random.state)
    return {'compiled_modules':count,'compiled_sampler':mode=='stateless-mlp-sampler'}

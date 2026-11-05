# SPDX-License-Identifier: Apache-2.0
"""Explicit codec residency/precision variants; built from pinned MLX-Audio MIT code.

The decode-only modes do not support reference conditioning. They remove encoder
residency, not decoder weights or speech generation steps. FP16 is a separate
numerical format and requires waveform/task-quality validation.
"""
import json
import re
from pathlib import Path

MODES=('full-fp32','full-stock-fp16-public','full-stock-fp16-refold','full-stock-bf16-refold','decode-only-fp32','decode-only-fp16')


def removed_key(name):
    return (name.startswith(('encoder.','quantizer.pre_module.','quantizer.downsample.'))
            or re.match(r'quantizer\.(semantic_quantizer|quantizer)\.quantizers\.\d+\.in_proj\.',name) is not None)


def install_loader(mode):
    if mode not in MODES:raise ValueError('unsupported codec mode')
    if mode.startswith('full-'):return
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.codec.models.fish_s1_dac import fish_s1_dac as source

    def load_decode_only(cls,repo_id):
        path=Path(repo_id)
        if not path.is_dir():raise ValueError('codec variants require verified local weights')
        config=json.loads((path/'config.json').read_text())
        codec=source.build_ae(**config)
        codec.pop('encoder')
        codec.quantizer.pop('pre_module')
        codec.quantizer.pop('downsample')
        for collection in (codec.quantizer.semantic_quantizer,codec.quantizer.quantizer):
            for quantizer in collection.quantizers:quantizer.pop('in_proj')
        loaded=codec.sanitize(mx.load(str(path/'codec.safetensors')))
        weights={key:value for key,value in loaded.items() if not removed_key(key)}
        del loaded
        derived=re.compile(r'(.*)\.conv\.conv\.(weight|bias)$')
        missing=[name for name,_ in tree_flatten(codec.parameters()) if name not in weights
                 and not name.endswith('causal_mask')
                 and not ((m:=derived.match(name)) and f'{m[1]}.weight_g' in weights)]
        if missing:raise ValueError('decode-only codec has missing learned parameters')
        codec.load_weights(list(weights.items()),strict=False)
        mx.eval(codec.parameters())
        # Fold in the original float32 arithmetic before any optional cast.
        codec.fold_weight_norm()
        del weights
        if mode=='decode-only-fp16':
            codec.set_dtype(mx.float16)
            folded=[]
            for module in codec.modules():
                value=module.get('_folded')
                if value is not None:
                    module['_folded']=value.astype(mx.float16)
                    folded.append(module['_folded'])
                    if hasattr(module,'_materialize_weight'):module._materialize_weight()
            mx.eval(codec.parameters(),folded)
        codec.reset_streaming_state()
        mx.clear_cache()
        return codec

    source.DAC.from_pretrained=classmethod(load_decode_only)


def apply_stock_dtype(codec,mode):
    """Public MLX Module/DAC APIs only; upstream loader and all encoder paths remain intact."""
    if not mode.startswith('full-stock-'):return
    import mlx.core as mx
    dtype=mx.bfloat16 if mode=='full-stock-bf16-refold' else mx.float16
    codec.set_dtype(dtype)
    if mode.endswith('-refold'):codec.fold_weight_norm()
    mx.eval(codec.parameters())
    mx.clear_cache()

"""Pinned MLX 0.32.4 Metal redundant-sort removal; native sampling retained.

The first argpartition and gather are unchanged. On this pinned Metal backend
argpartition fully sorts, so sorting the gathered prefix again is redundant.
The proof applies to ordinary finite logits and -inf masks with a finite entry.
NaN, +inf and all--inf inputs have no asserted equivalence guarantee.
"""
import hashlib, inspect, importlib.metadata

def build(mx, minimum_temperature):
    def sample_top_k_top_p(logits, temperature, top_k, top_p):
        x = logits.astype(mx.float32) / max(temperature, minimum_temperature)
        x = x - x.max(axis=-1, keepdims=True)
        k = min(top_k, x.shape[-1])
        ids = mx.argpartition(-x, kth=k - 1, axis=-1)[..., :k]
        values = mx.take_along_axis(x, ids, axis=-1)
        probs = mx.exp(values - mx.logsumexp(x, axis=-1, keepdims=True))
        dropped = (mx.cumsum(probs, axis=-1) > top_p) & (mx.arange(k) > 0)
        choice = mx.random.categorical(mx.where(dropped, -mx.inf, values))
        return mx.take_along_axis(ids, choice[..., None], axis=-1)[..., 0]
    return sample_top_k_top_p

def install(runtime, enabled, expected_drama_sha):
    from mlx_audio.tts.models.fish_drama3 import drama3
    assert hashlib.sha256(open(inspect.getfile(drama3),'rb').read()).hexdigest() == expected_drama_sha
    assert importlib.metadata.version('mlx') == '0.32.4' and runtime.mx.metal.is_available() and runtime.mx.default_device()==runtime.mx.gpu
    if enabled:
        drama3.sample_top_k_top_p = build(runtime.mx, drama3.MIN_TEMPERATURE)
    return {'enabled': enabled, 'backend': 'MLX0.32.4 Metal',
            'removed_operations': ['second argsort', 'sorted-values gather', 'sorted-ids gather'] if enabled else [],
            'native_categorical_and_RNG': True,
            'scope': 'No depth compilation; original cache, sampler arithmetic/order, categorical and production observations retained'}

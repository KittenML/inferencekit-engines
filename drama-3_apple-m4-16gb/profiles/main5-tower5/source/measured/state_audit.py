# SPDX-License-Identifier: Apache-2.0
"""Observe existing request reset/prefix-copy behavior without altering tensors."""
from types import MethodType


def install(model, observations):
    original = model._start_caches
    reset = model.codec.reset_streaming_state

    def start(this, frames, system_rows, voice):
        cached_before = voice.prefix is not None
        result = original(frames, system_rows, voice)
        prefix = voice.prefix
        separate = len(result) == len(prefix) and all(len(a) == len(b) and all(x is not y for x, y in zip(a, b)) for a, b in zip(result, prefix))
        observations.append({"event": "fresh_prefix_cache_copy", "system_rows": int(system_rows), "prefix_cached_before": cached_before, "cache_objects_separate_from_voice_snapshot": separate, "cache_types": [[type(c).__name__ for c in group] for group in result], "initial_offsets": [[getattr(c, "offset", None) if type(getattr(c, "offset", None)) is int else None for c in group] for group in result]})
        return result

    def codec_reset(this):
        result = reset()
        observations.append({"event": "codec_streaming_state_reset"})
        return result

    object.__setattr__(model, "_start_caches", MethodType(start, model))
    object.__setattr__(model.codec, "reset_streaming_state", MethodType(codec_reset, model.codec))

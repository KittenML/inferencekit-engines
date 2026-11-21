#!/usr/bin/env python3
"""Compare the two pinned MOSS exports on the same known codec frames."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[1]

def inspect():
    codec = root / "build/payload/vieneu/codec"
    source = root / "assets/voices/adam/ref_codes.txt"
    codes = np.loadtxt(source, dtype=np.int32).reshape(1, -1, 16)
    spec = json.loads((codec / "codec_browser_onnx_meta.json").read_text())["streaming_decode"]
    shape_map = {}
    for offset in spec["transformer_offsets"]:
        shape_map[offset["input_name"]] = (offset["output_name"], offset["shape"], np.int32, 0)
    for layer in spec["attention_caches"]:
        for name, shape, dtype, fill in [
            ("offset", "offset_shape", np.int32, 0),
            ("cached_keys", "cache_shape", np.float32, 0),
            ("cached_values", "cache_shape", np.float32, 0),
            ("cached_positions", "positions_shape", np.int32, -1),
        ]:
            shape_map[layer[name + "_input_name"]] = (layer[name + "_output_name"], layer[shape], dtype, fill)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    full = ort.InferenceSession(str(codec / "moss_audio_tokenizer_decode_full.onnx"), options, providers=["CPUExecutionProvider"])
    step = ort.InferenceSession(str(codec / "moss_audio_tokenizer_decode_step.onnx"), options, providers=["CPUExecutionProvider"])
    def run(session, batch, state):
        named = ["audio", "audio_lengths"] if not state else ["audio", "audio_lengths"] + [s[0] for s in shape_map.values()]
        output = session.run(named, {"audio_codes": batch, "audio_code_lengths": np.asarray([batch.shape[1]], dtype=np.int32), **state})
        if state:
            state.update(dict(zip(shape_map, output[2:], strict=True)))
        return output[0][0].mean(axis=0)[:int(output[1][0])]
    reference = run(full, codes, {})
    cases = []
    for first, follow in [(2, 4), (2, 6), (3, 3), (4, 6), (4, 8)]:
        state = {name: np.full(shape, fill, dtype=dtype) for name, (_, shape, dtype, fill) in shape_map.items()}
        cursor = 0
        chunks = []
        while cursor < codes.shape[1]:
            width = first if cursor == 0 else follow
            chunks.append(run(step, codes[:, cursor:cursor + width], state))
            cursor += width
        wave = np.concatenate(chunks)
        difference = wave.astype(np.float64) - reference
        l2 = np.linalg.norm(difference)
        cases.append({"first": first, "follow": follow, "samples": len(wave), "exact_float": bool(np.array_equal(wave, reference)),
                      "exact_s16": bool(np.array_equal(np.rint(wave * 32767), np.rint(reference * 32767))),
                      "max_abs": float(np.max(np.abs(difference))), "snr_db": None if l2 == 0 else float(20 * np.log10(np.linalg.norm(reference) / l2))})
    return {"host_only": True, "ort": ort.__version__, "codec_frames": codes.shape[1], "cases": cases}

if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2))

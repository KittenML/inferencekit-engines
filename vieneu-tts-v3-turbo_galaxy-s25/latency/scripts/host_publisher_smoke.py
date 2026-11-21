#!/usr/bin/env python3
"""Run the pinned unmodified Python INT8 path as a host support check."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine

root = Path(__file__).resolve().parents[1]
assets = root / "assets"
voice = assets / "voices/adam"
codes = np.loadtxt(voice / "ref_codes.txt", dtype=np.int64)
embedding = np.loadtxt(voice / "speaker.emb.txt", dtype=np.float32, delimiter=",")
engine = OnnxV3LiteEngine(onnx_dir=str(assets / "vieneu-int8"), codec_dir=str(root / "build/payload/vieneu/codec"), threads=4)
assert engine.sess_codec_step is not None, "The publisher runtime fell back to whole-utterance decode"
phonemes = "sˈin tʃˈaː2w kaːk5 ɓˈaːn6."
sample_rate = engine.SAMPLE_RATE
timings = {}
for name, stream in [("full", False), ("step", True)]:
    np.random.seed(42)
    start = perf_counter()
    arguments = {"phonemes": phonemes, "ref_codes": codes, "speaker_emb": embedding, "max_new_frames": 100}
    if stream:
        raw = iter(engine.infer_stream(**arguments))
        first = next(raw)
        first_ms = (perf_counter() - start) * 1000
        pieces = [first, *raw]
    else:
        pieces = [engine.infer(**arguments)]
        first_ms = (perf_counter() - start) * 1000
    pcm = np.concatenate(pieces)
    assert pcm.size and np.isfinite(pcm).all()
    timings[name] = {"first_pcm_ms": first_ms, "complete_ms": (perf_counter() - start) * 1000,
                     "sample_count": len(pcm), "chunks": len(pieces), "sample_rate": sample_rate}
print(json.dumps({"host_only": True, "resolved_threads": engine.ort_intra_op_threads, "timings": timings}, indent=2))

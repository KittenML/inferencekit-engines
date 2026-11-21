"""Compare greedy ONNX adapter output against the publisher on device-study prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine


HERE = Path(__file__).resolve().parent
LANE = HERE.parent
MODEL = LANE / "assets/vieneu-int8"
VOICE = LANE / "assets/voices/adam"
CODEC = LANE / "build/payload/vieneu/codec"
BUILD = HERE / "build"


def score(observed, expected):
    result = {"samples": len(observed), "publisher_samples": len(expected)}
    if observed.shape == expected.shape:
        error = expected - observed
        norm = float(np.linalg.norm(error))
        result.update(
            max_abs_error=float(np.abs(error).max()),
            residual_snr_db=float(20 * np.log10(np.linalg.norm(expected) / norm)) if norm else None,
            adapter_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
            publisher_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-build", type=Path, required=True)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    support = args.host_build.resolve()
    engine = OnnxV3LiteEngine(onnx_dir=str(MODEL), codec_dir=str(CODEC), threads=2)
    if engine.sess_codec_step is None:
        raise RuntimeError("publisher codec silently fell back to full decode")
    speaker = np.loadtxt(VOICE / "speaker.emb.txt", delimiter=",", dtype=np.float32)
    codes = np.loadtxt(VOICE / "ref_codes.txt", dtype=np.int64)
    prepared = json.loads((BUILD / "voices.json").read_text())["presets"]["adam"]
    assert np.array_equal(np.array(prepared["speaker_anchor"], dtype=np.float32), engine._speaker_anchor(speaker))
    assert np.array_equal(prepared["codes"], codes)

    env = os.environ.copy()
    env.update(
        LD_LIBRARY_PATH=f"{BUILD}/lib:{support}/bin:{support}",
        VIENEU_SEA_G2P_DICT=str(CODEC.parent / "sea_g2p.bin"),
        VIENEU_SEA_G2P_DEBUG="1",
        VIENEU_ACOUSTIC_BACKEND="onnx",
    )
    fixtures = [line.split("\t", 3) for line in (LANE / "workload.tsv").read_text().splitlines()]
    fixtures.insert(0, ["controlled", "short", "phonemes", "həlˈoʊ wˈɜɹld."])
    failed = False
    for fixture, bucket, mode, text in fixtures:
        if not args.all and fixture not in {"controlled", "en-s1", "en-m1", "en-l1", "vi-s1", "vi-m1"}:
            continue
        out_pcm = BUILD / f"verify-{fixture}.f32"
        result = subprocess.run(
            [BUILD / "smoke", MODEL, CODEC, BUILD / "voices.json", text, out_pcm, mode, "4"],
            cwd=support, env=env, text=True, capture_output=True,
        )
        (BUILD / f"verify-{fixture}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            print(json.dumps({"id": fixture, "mode": mode, "exit": result.returncode, "error": result.stderr.strip()}), flush=True)
            failed = True
            continue
        expected_chunks = []
        token_parity = []
        for line in result.stdout.splitlines():
            if not line.startswith("phonemes="):
                continue
            phonemes, ids = line.removeprefix("phonemes=").split("\tids=")
            cpp_ids = [int(token) for token in ids.split(",")] if ids else []
            py_ids = engine.tokenizer.encode(phonemes, add_special_tokens=False).ids
            token_parity.append(cpp_ids == py_ids)
            if expected_chunks:
                expected_chunks.append(np.zeros(7200, dtype=np.float32))
            expected_chunks.extend(engine.infer_stream(
                phonemes=phonemes, speaker_emb=speaker, ref_codes=codes,
                temperature=0.0, frame_cap=False,
            ))
        expected = np.concatenate(expected_chunks)
        expected.tofile(BUILD / f"publisher-{fixture}.f32")
        observed = np.fromfile(out_pcm, dtype=np.float32)
        print(json.dumps({"id": fixture, "bucket": bucket, "mode": mode,
                          "token_parity_by_chunk": token_parity, **score(observed, expected)}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

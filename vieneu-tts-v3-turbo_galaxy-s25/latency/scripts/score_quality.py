#!/usr/bin/env python3
"""Offline transcript, predicted naturalness and paired PCM checks."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import unicodedata
import wave

from faster_whisper import WhisperModel
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EN_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
VI_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
CACHE = Path.home() / ".cache/huggingface/hub"
HUB = Path.home() / ".cache/torch/hub"
UTMOS = HUB / "tarepan_SpeechMOS_v1.2.0"
UTMOS_CKPT = HUB / "checkpoints/utmos22_strong_step7459_v1.pt"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_pcm(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected mono signed 16-bit PCM")
        data = wav.readframes(wav.getnframes())
        return wav.getframerate(), data, np.frombuffer(data, dtype="<i2")

def words(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in normalized).split()

def edit_distance(ref, hyp):
    costs = list(range(len(hyp) + 1))
    for row, r in enumerate(ref, 1):
        previous, costs[0] = costs[0], row
        for col, h in enumerate(hyp, 1):
            diagonal, previous = previous, costs[col]
            costs[col] = min(diagonal + (r != h), costs[col - 1] + 1, costs[col] + 1)
    return costs[-1]

def checkpoint(language):
    model = "faster-whisper-small.en" if language == "en" else "faster-whisper-large-v3"
    revision = EN_REVISION if language == "en" else VI_REVISION
    return CACHE / f"models--Systran--{model}" / "snapshots" / revision

def evaluate(capture, baseline, candidate, no_mos):
    source = {}
    for line in (ROOT / "workload.tsv").read_text().splitlines():
        name, _, lang, text = line.split("\t")
        source[name] = (lang, text)
    scorer = {}
    scorer_info = {}
    naturalness = None
    if not no_mos:
        import torch
        torch.set_num_threads(4)
        naturalness = torch.hub.load(str(UTMOS), "utmos22_strong", source="local", pretrained=True)
        scorer_info["predicted_mos"] = {"version": "SpeechMOS v1.2.0 UTMOS22 strong", "checkpoint_sha256": sha(UTMOS_CKPT)}
    results, saved, totals = [], {}, defaultdict(lambda: {"words": 0, "word_errors": 0, "characters": 0, "character_errors": 0, "predicted_mos": []})
    for path in sorted(capture.rglob("requests.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "wav" not in row:
                continue
            identity = (row["label"], row["round"], row["id"])
            if identity in saved:
                raise ValueError(f"Duplicate audio {identity}")
            saved[identity] = path.with_name(row["wav"])
            lang, reference = source[row["id"]]
            if lang not in scorer:
                model = checkpoint(lang)
                scorer[lang] = WhisperModel(str(model), device="cpu", compute_type="int8", cpu_threads=6,
                                            num_workers=1, local_files_only=True)
                scorer_info[lang] = {"revision": model.name, "model_sha256": sha(model / "model.bin")}
            wav = saved[identity]
            segments, _ = scorer[lang].transcribe(str(wav), language=lang, beam_size=5, temperature=0,
                                                  vad_filter=False, condition_on_previous_text=False)
            text = " ".join(segment.text.strip() for segment in segments)
            ref, hyp = words(reference), words(text)
            character_ref = "".join(ref)
            character_hyp = "".join(hyp)
            rate, pcm, values = read_pcm(wav)
            if len(values) != row["source_frames"]:
                raise ValueError(f"Truncated WAV {wav}")
            row_metrics = {"file": str(wav.relative_to(capture)), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                           "language": lang, "label": row["label"], "round": row["round"], "id": row["id"],
                           "transcript": text, "reference": reference, "words": len(ref),
                           "word_errors": edit_distance(ref, hyp), "characters": len(character_ref),
                           "character_errors": edit_distance(character_ref, character_hyp)}
            aggregate = totals[(row["label"], lang)]
            for key in ("words", "word_errors", "characters", "character_errors"):
                aggregate[key] += row_metrics[key]
            if naturalness and lang == "en":
                # Pass the entire raw request, preserving the trailing and inter-chunk pauses.
                with torch.inference_mode():
                    prediction = naturalness(torch.from_numpy(values.astype(np.float32) / 32767).unsqueeze(0), rate)
                row_metrics["predicted_mos"] = float(prediction[0])
                aggregate["predicted_mos"].append(row_metrics["predicted_mos"])
            results.append(row_metrics)
    wave_pairs = []
    if baseline and candidate:
        for (label, round_number, fixture_id), path in saved.items():
            if label != baseline or (candidate, round_number, fixture_id) not in saved:
                continue
            rate_a, pcm_a, a = read_pcm(path)
            rate_b, pcm_b, b = read_pcm(saved[(candidate, round_number, fixture_id)])
            same_shape = rate_a == rate_b and a.shape == b.shape
            record = {"id": fixture_id, "round": round_number, "same_shape": same_shape, "exact_s16": same_shape and pcm_a == pcm_b,
                      "baseline_frames": len(a), "candidate_frames": len(b)}
            if same_shape:
                d = a.astype(np.float64) - b
                difference = np.linalg.norm(d)
                record.update(max_abs_lsb=int(np.max(np.abs(d))), differing_samples=int(np.count_nonzero(d)),
                              signal_to_error_db=float(20 * np.log10(np.linalg.norm(a) / difference)) if difference else None)
            wave_pairs.append(record)
    return {"scorers": scorer_info, "faster_whisper_version": version("faster-whisper"),
            "asr_config": {"device": "cpu", "compute": "int8", "temperature": 0, "beam": 5, "vad": False},
            "totals": [{**agg, "label": label, "language": lang,
                        "wer": agg["word_errors"] / agg["words"],
                        "cer": agg["character_errors"] / agg["characters"],
                        "mean_predicted_mos": float(np.mean(agg["predicted_mos"])) if agg["predicted_mos"] else None}
                       for (label, lang), agg in sorted(totals.items())],
            "per_request": results, "pairs": wave_pairs,
            "exact_verdict": ("PASS" if len(wave_pairs) >= 12 and all(p["exact_s16"] for p in wave_pairs)
                              else "FAIL" if any(not p["exact_s16"] for p in wave_pairs) else "UNPROVEN"),
            "listening": "NOT RUN"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--no-mos", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.capture, args.baseline, args.candidate, args.no_mos), indent=2))

if __name__ == "__main__":
    main()

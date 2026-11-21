#!/usr/bin/env python3
"""Audit confirmation coverage, route, artifact identity and serial cell order."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]

def verify(capture):
    manifest = json.loads((ROOT / "runs/vieneu-v3-turbo-galaxy-s25-20261120/manifest.json").read_text())
    index = json.loads((capture / "index.json").read_text())
    session = json.loads((capture / "session.json").read_text())
    assert index["model"].startswith("SM-S931") and index["soc_model"] == "SM8750"
    assert index["android_release"] == "15" and index.get("complete")
    assert session["apk_sha256"] == manifest["artifacts"]["apk_sha256"]
    assert hashlib.sha256((ROOT / "workload.tsv").read_bytes()).hexdigest() == manifest["workload"]["fixture_sha256"]
    assert hashlib.sha256((ROOT / "app/assets/confirmation.tsv").read_bytes()).hexdigest() == manifest["artifacts"]["plan_sha256"]
    actual_order = [(cell["engine"], int(cell["round"])) for cell in index["cells"]]
    expected_order = [(label, int(round_number)) for label, round_number, *_ in
                      (line.split("\t") for line in (ROOT / "app/assets/confirmation.tsv").read_text().splitlines())]
    assert actual_order == expected_order
    source = {}
    for line in (ROOT / "workload.tsv").read_text().splitlines():
        name, _, lang, _ = line.split("\t")
        source[name] = lang
    output = Counter()
    condition_rows = []
    expected_configs = {e["label"]: e["resolved_config"] for e in manifest["engines"]}
    engines = {}
    issues = []
    process_ids = set()
    retained_wavs = 0
    for cell in index["cells"]:
        label, round_number = cell["engine"], int(cell["round"])
        folder = capture / cell["directory"]
        condition_rows.append({"label": label, "round": round_number,
                               "thermal_status_before": cell["before"]["thermal_status"],
                               "thermal_status_after": cell["after"]["thermal_status"],
                               "music_volume": cell["before"]["music_volume"]})
        if cell["exit_code"] != 0:
            issues.append({"label": label, "round": round_number, "exit_code": cell["exit_code"]})
        runtime = json.loads((folder / "cell.json").read_text())
        assert runtime["label"] == label and runtime["round"] == round_number
        assert runtime["pid"] not in process_ids
        process_ids.add(runtime["pid"])
        assert runtime["threads"] == expected_configs[label]["threads"]
        assert runtime["start_output_on_admission"]
        rows = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
        language = "vi" if round_number == -3 else "en"
        expected_ids = [name for name in source if source[name] == language]
        if round_number >= 0:
            rotation = (round_number % 3) * 3
            expected_ids = expected_ids[rotation:] + expected_ids[:rotation]
        assert [row["id"] for row in rows] == expected_ids
        engine = engines.setdefault(label, {"condition_count": 0, "framed_requests": 0, "pss_requests": 0,
                                            "rendered_requests": 0, "audible_timestamps": 0,
                                            "gapped_requests": 0, "requests_with_xruns": 0, "clipped_samples": 0})
        engine["condition_count"] += 1
        routes = {r["id"]: r["type"] for r in cell["before"].get("audio_output_routes", [])}
        for row in rows:
            assert row["label"] == label and row["round"] == round_number and row["language"] == language
            if "error" in row:
                issues.append({"label": label, "round": round_number, "id": row["id"], "error": row["error"]})
                continue
            audio = row["playback"]
            assert audio["source_frames"] == row["source_frames"] and audio["start_on_admission"]
            if audio["first_audible_ms"] is not None:
                anchor = audio["onset_timestamp_monotonic_ns"]
                from_onset = (audio["onset_timestamp_frame"] - audio["onset_frame"]) * 1e9 / audio["output_rate"]
                reconstructed = (anchor - from_onset - audio["request_monotonic_ns"]) / 1e6
                assert abs(reconstructed - audio["first_audible_ms"]) < 0.001
            if audio["status"] == "ok":
                expected_frames = audio["source_frames"] * audio["output_rate"] // audio["source_rate"]
                assert audio["consumed_pcm_frames"] == expected_frames
                assert audio["stream_start_requested_ms"] < audio["first_pcm_ms"] < audio["first_audible_ms"]
            else:
                issues.append({"label": label, "round": round_number, "id": row["id"], "playback": audio["status"]})
            engine["framed_requests"] += row["source_frames"] > 0
            engine["pss_requests"] += row["memory"]["peak"]["pss_kib"] is not None
            engine["rendered_requests"] += audio["status"] == "ok"
            engine["audible_timestamps"] += audio["first_audible_ms"] is not None
            engine["gapped_requests"] += audio["app_gap_count"] > 0
            engine["requests_with_xruns"] += (audio["device_xruns"] or 0) > 0
            engine["clipped_samples"] += row["clipped_samples"]
            output[(audio["device_id"], routes.get(audio["device_id"]), audio["output_rate"])] += 1
            if "wav" in row:
                retained_wavs += 1
                with wave.open(str(folder / row["wav"]), "rb") as wav:
                    assert wav.getnframes() == row["source_frames"]
                    assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
    return {"engine": engines, "conditions": condition_rows, "issues": issues,
            "unique_processes": len(process_ids), "retained_wavs": retained_wavs,
            "device": index["model"], "apk_sha256": session["apk_sha256"],
            "audio_routes": [{"device_id": ident, "type": kind, "rate": rate, "requests": count}
                             for (ident, kind, rate), count in output.items()]}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
args = parser.parse_args()
print(json.dumps(verify(args.capture), indent=2))

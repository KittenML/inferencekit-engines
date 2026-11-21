#!/usr/bin/env python3
"""Aggregate fresh process cells and bootstrap over paired device rounds."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median

def quantile(values, probability):
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)] if values else None

def summary(values):
    return {"n": len(values), "median": median(values) if values else None, "p95": quantile(values, .95)}

METRICS = {
    "first_pcm_ms": lambda row: row["playback"]["first_pcm_ms"],
    "first_audible_ms": lambda row: row["playback"]["first_audible_ms"],
    "synthesis_ms": lambda row: row["playback"]["synthesis_complete_ms"],
    "playback_ms": lambda row: row["playback"]["playback_complete_ms"],
    "rtf": lambda row: row["rtf"],
    "audio_s": lambda row: row["audio_s"],
    "peak_pss_kib": lambda row: row["memory"]["peak"]["pss_kib"],
    "peak_rss_kib": lambda row: row["memory"]["peak"]["rss_kib"],
}

def bootstrap_ratio(baseline, candidate, metric, expected_ids, expected_rounds):
    aggregate = lambda rows: quantile([metric(row) for row in rows if metric(row) is not None], .95)
    def complete(rows):
        return (sorted(row["id"] for row in rows) == expected_ids
                and all(row["playback"]["status"] == "ok" and metric(row) is not None for row in rows))

    pairs = [(aggregate(baseline[k]), aggregate(candidate[k])) for k in expected_rounds
             if complete(baseline[k]) and complete(candidate[k])]
    ratios = [a / b for a, b in pairs if a is not None and b]
    if len(ratios) != len(expected_rounds):
        return {"rounds": len(ratios), "expected_rounds": len(expected_rounds), "complete": False}
    generator = random.Random(42)
    estimates = [median(generator.choices(ratios, k=len(ratios))) for _ in range(10000)]
    return {"rounds": len(ratios), "complete": True, "ratio": median(ratios),
            "ci95": [quantile(estimates, .025), quantile(estimates, .975)], "raw_ratios": ratios}

def calculate(root, baseline, candidate):
    observations = defaultdict(list)
    failures, files, cells = [], {}, defaultdict(list)
    if not (root / "index.json").is_file():
        raise ValueError("Specify a single device capture directory with its index")
    for path in sorted(root.rglob("requests.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            if "error" in row:
                failures.append({"file": str(path.relative_to(root)), "line": number, **row})
            else:
                observations[(row["label"], row["language"], row["bucket"])].append(row)
        with path.open("rb") as stream:
            files[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    for path in sorted(root.rglob("cell.json")):
        cell = json.loads(path.read_text())
        life = path.with_name("lifetime-memory.json")
        if life.exists():
            cell["lifetime"] = json.loads(life.read_text())
        cells[cell["label"]].append(cell)

    metrics = []
    for (label, language, bucket), rows in sorted(observations.items()):
        aggregates = {name: summary([fn(r) for r in rows if fn(r) is not None]) for name, fn in METRICS.items()}
        metrics.append({"label": label, "language": language, "bucket": bucket, "requests": len(rows), **aggregates,
                        "gapped_requests": sum(r["playback"]["app_gap_count"] > 0 for r in rows),
                        "app_gap_frames": sum(r["playback"]["app_gap_frames"] for r in rows),
                        "xrun_requests": sum((r["playback"]["device_xruns"] or 0) > 0 for r in rows),
                        "xrun_unknown_requests": sum(r["playback"]["device_xruns"] is None for r in rows),
                        "playback_statuses": sorted({r["playback"]["status"] for r in rows}),
                        "clipped_samples": sum(r["clipped_samples"] for r in rows)})
    comparisons = []
    if baseline and candidate:
        workload = Path(__file__).resolve().parents[1] / "workload.tsv"
        ids_by_bucket = defaultdict(list)
        for line in workload.read_text().splitlines():
            name, bucket, lang, _ = line.split("\t")
            if lang == "en":
                ids_by_bucket[bucket].append(name)
        for bucket in ("short", "medium", "long"):
            def round_groups(label):
                grouped = defaultdict(list)
                for row in observations.get((label, "en", bucket), []):
                    if row["round"] >= 0:
                        grouped[row["round"]].append(row)
                return grouped
            rounds_a = round_groups(baseline)
            rounds_b = round_groups(candidate)
            def compare(round_indices):
                return {metric: bootstrap_ratio(rounds_a, rounds_b, METRICS[metric],
                                                sorted(ids_by_bucket[bucket]), round_indices)
                        for metric in ("first_audible_ms", "synthesis_ms", "rtf")}
            comparisons.append({"bucket": bucket, "baseline": baseline, "candidate": candidate,
                                **compare(range(6)), "exclude_pcm_retention_round": compare(range(1, 6))})
    return {"metrics": metrics, "device_cell_records": cells, "comparisons": comparisons,
            "failures": failures, "raw_sha256": files}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    print(json.dumps(calculate(args.capture, args.baseline, args.candidate), indent=2))

if __name__ == "__main__":
    main()

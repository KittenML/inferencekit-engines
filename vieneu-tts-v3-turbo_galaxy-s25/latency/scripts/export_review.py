#!/usr/bin/env python3
"""Copy retained source audio and transcript diagnostics into InferenceKit's review layout."""
import argparse
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("quality", type=Path)
parser.add_argument("run", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
buckets = {}
for line in (root / "workload.tsv").read_text().splitlines():
    name, length, language, text = line.split("\t")
    bucket = length if language == "en" else f"vi-{length}"
    buckets[name] = bucket
    entry = args.run / "inputs" / bucket / f"{name}.json"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(json.dumps({"text": text, "reference_text": text}, ensure_ascii=False) + "\n")
quality = json.loads(args.quality.read_text())
for row in quality["per_request"]:
    name = row["id"]
    stem = args.run / "outputs" / row["label"] / buckets[name] / name
    stem.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.capture / row["file"], stem.with_suffix(".wav"))
    stem.with_suffix(".asr.txt").write_text(row["transcript"] + "\n")

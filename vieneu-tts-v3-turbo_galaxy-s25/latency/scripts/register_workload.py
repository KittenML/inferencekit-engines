#!/usr/bin/env python3
"""Register the source text and the exact same-format screen."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fixtures = []
limits = {"short": (12, 60), "medium": (61, 160), "long": (161, 400)}
for line in (root / "workload.tsv").read_text().splitlines():
    name, bucket, locale, prompt = line.split("\t")
    low, high = limits[bucket]
    assert low <= len(prompt) <= high, (name, bucket, len(prompt))
    fixtures.append({"id": name, "text": prompt})
with (root / "quality-fixtures.jsonl").open("w") as output:
    for fixture in fixtures:
        output.write(json.dumps(fixture, ensure_ascii=False) + "\n")

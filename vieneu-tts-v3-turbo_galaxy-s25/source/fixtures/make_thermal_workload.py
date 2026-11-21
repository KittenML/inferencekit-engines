#!/usr/bin/env python3
"""Extend the current paired speech fixtures for longer, open-loop runs."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_plans", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--waves", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--period-ms", type=int, default=3000)
    args = parser.parse_args()
    if min(args.levels, args.waves, args.rounds, args.period_ms) < 1:
        parser.error("levels, waves, rounds and period must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sentences = [
        row.split("\t")
        for row in Path(__file__).with_name("extra-sentences.tsv").read_text().splitlines()
    ]
    for family, column, sample_rate in (("kokoro", 1, 24000), ("vieneu", 2, 48000)):
        path = args.study_plans / f"{family}.tsv"
        inputs = [tuple(row.split("\t")) for row in path.read_text().splitlines()]
        *existing, warmup = inputs
        if warmup[0] != "warmup":
            raise ValueError(f"{path}: expected warm-up in the final row")
        unique = {(a, b) for _, a, b in existing}
        first = list(dict.fromkeys([a for _, a, _ in existing] +
                                   [row[column] for row in sentences if row[0] == "first"]))
        second = list(dict.fromkeys([b for _, _, b in existing] +
                                    [row[column] for row in sentences if row[0] == "second"]))
        added = [(a, b) for a in first for b in second if (a, b) not in unique]
        random.Random(61118).shuffle(added)
        rows = existing + [(f"{family}-{index:03}", a, b)
                           for index, (a, b) in enumerate(added, start=len(existing))]
        count = args.waves * args.levels
        if count > len(rows):
            parser.error(f"{family} needs {count} paired fixtures and has {len(rows)}")
        output_file = args.output / path.name
        output_file.write_text("".join("\t".join(row) + "\n" for row in (*rows[:count], warmup)),
                               encoding="utf-8")
        digest = hashlib.sha256(output_file.read_bytes()).hexdigest()
        for mode, chunks in (("stream", 2), ("full", 1)):
            plan = {
                "levels": list(range(1, args.levels + 1)),
                "rounds": args.rounds,
                "waves": args.waves,
                "period_ms": args.period_ms,
                "sample_rate": sample_rate,
                "mode": mode,
                "chunks_per_request": chunks,
                "fixture_ids": [row[0] for row in rows[:count]],
                "fixture_sha256": digest,
            }
            output_path = args.output / f"thermal-{family}-{mode}.json"
            output_path.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()

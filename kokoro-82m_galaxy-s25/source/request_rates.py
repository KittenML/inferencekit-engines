"""Measure completed and on-time request rates for registered native timing cells."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

PLANS = Path(__file__).resolve().parent / "plans"


def summarize(rows, concurrency, plan):
    waves = plan["waves"]
    period_ms = plan["period_ms"]
    keys = {(row["wave"], row["stream"]) for row in rows}
    expected = {(wave, stream) for wave in range(waves) for stream in range(concurrency)}
    if len(rows) != len(expected) or keys != expected or any(
            row["id"] != plan["fixture_ids"][row["wave"] * concurrency + row["stream"]]
            for row in rows):
        return {"coverage": "UNPROVEN", "observations": len(rows), "expected": len(expected)}
    completed = [row for row in rows if row["complete_ms"] is not None
                 and len(row["chunks"]) == plan["chunks_per_request"]]
    on_time = [row for row in completed if row["complete_ms"] <= period_ms]
    release_window_ms = waves * period_ms
    drain_to_ms = max((row["wave"] * period_ms + row["complete_ms"] for row in completed), default=0)
    elapsed_ms = max(release_window_ms, drain_to_ms)
    completed_frames = sum(chunk[1] for row in completed for chunk in row["chunks"])
    on_time_frames = sum(chunk[1] for row in on_time for chunk in row["chunks"])
    return {
        "coverage": "PASS", "observations": len(rows), "completed": len(completed), "on_time": len(on_time),
        "release_window_ms": release_window_ms, "observation_window_ms": elapsed_ms,
        "completed_requests_s": len(completed) * 1000 / elapsed_ms,
        "on_time_requests_s": len(on_time) * 1000 / release_window_ms,
        "completed_audio_s_per_s": completed_frames * 1000 / (plan["sample_rate"] * elapsed_ms),
        "on_time_audio_s_per_s": on_time_frames * 1000 / (plan["sample_rate"] * release_window_ms),
    }


def observations(capture):
    manifest = json.loads((capture / "run-plan.json").read_bytes())
    if manifest.get("quality_only"):
        raise ValueError("quality-only captures are outside the registered timing protocol")
    for label, model, mode, _, _, *options in manifest["arms"]:
        config = options[0] if options else {}
        family = model.split("-", 1)[0]
        fixture = config.get("INFERENCEKIT_PLAN", f'{manifest["phase"]}-{family}-{mode}.json')
        registration = (PLANS / fixture).read_bytes()
        plan = json.loads(registration)
        path = capture / f"{label}.jsonl"
        if not path.exists():
            yield {"label": label, "capture": "NOT RUN"}
            continue
        raw = path.read_bytes()
        grouped = defaultdict(list)
        for row in map(json.loads, raw.splitlines()):
            grouped[row["round"], row["concurrency"]].append(row)
        yield {
            "label": label, "timings_sha256": hashlib.sha256(raw).hexdigest(),
            "plan_sha256": hashlib.sha256(registration).hexdigest(),
            "cells": [{"round": round_number, "concurrency": concurrency,
                       **summarize(rows, concurrency, plan)}
                      for (round_number, concurrency), rows in sorted(grouped.items())],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    arguments = parser.parse_args()
    for record in observations(arguments.capture):
        print(json.dumps(record))


if __name__ == "__main__":
    main()

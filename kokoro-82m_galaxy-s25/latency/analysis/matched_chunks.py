"""Verify and summarize the matched three-sentence Galaxy S25 benchmark."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

ARMS = ("kokoro_stream", "pico_stream")
ROUNDS = 6
RATE = 24_000


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def distribution(values):
    ordered = sorted(values)
    return {
        "median": median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def check_row(row, fixture):
    events = row["chunk_events"]
    ready = [event["ready_ms"] for event in events]
    waits = row["observed_queue_empty_wait_ms"]
    if (row["status"] != "OK" or row["audio_playback_error"] is not None
            or row["first_presented_signal_ms"] is None
            or row["sample_rate"] != RATE or row["callbacks"] != 3
            or [event["text"] for event in events] != fixture["segments"]
            or any(event["frames"] <= 0 for event in events)
            or sum(event["frames"] for event in events) != row["frames"]
            or row["audio_written_frames"] != row["frames"]
            or not 0 < ready[0] < ready[1] < ready[2] <= row["synthesis_complete_ms"]
            or len(waits) != 2 or any(wait < 0 for wait in waits)):
        raise ValueError(f"Incomplete or invalid playback for {fixture['id']}")


def chunk_stats(rows):
    events = [row["chunk_events"] for row in rows]
    durations = [[event["frames"] * 1000 / RATE for event in request] for request in events]
    ready = [[event["ready_ms"] for event in request] for request in events]
    gaps = [row["observed_queue_empty_wait_ms"] for row in rows]
    return {
        "chunk_audio_ms": [distribution([values[i] for values in durations]) for i in range(3)],
        "next_chunk_ready_interval_ms": [
            distribution([values[i + 1] - values[i] for values in ready]) for i in range(2)
        ],
        "observed_queue_empty_wait_ms": [
            distribution([values[i] for values in gaps]) for i in range(2)
        ],
        "requests_with_observed_empty_wait": sum(any(gap > 0 for gap in row) for row in gaps),
        "requests_with_audiotrack_underrun": sum(row["audio_write_underruns"] > 0 for row in rows),
        "audiotrack_underruns": sum(row["audio_write_underruns"] for row in rows),
    }


def summarize(rows):
    return {
        **chunk_stats(rows),
        "requests": len(rows),
        "first_presented_signal_ms": distribution([row["first_presented_signal_ms"] for row in rows]),
        "complete_synthesis_ms": distribution([row["synthesis_complete_ms"] for row in rows]),
        "playback_drained_ms": distribution([row["playback_drained_ms"] for row in rows]),
        "thermal_statuses": sorted({row["thermal_status"] for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixtures = {row["id"]: row for row in read_rows(args.fixtures)}
    if any(len(fixture["segments"]) != 3 or " ".join(fixture["segments"]) != fixture["text"]
           for fixture in fixtures.values()):
        raise ValueError("Every fixture must cover its full text in three segments")
    device = json.loads((args.results / "device-start.json").read_text())
    if not device["model"].startswith("SM-S931"):
        raise ValueError("Expected Galaxy S25 evidence")
    expected_order = [
        f"{arm}/round-{round_id}" for round_id in range(ROUNDS)
        for arm in (ARMS if round_id % 2 == 0 else reversed(ARMS))
    ]
    schedule = json.loads((args.results / "schedule.json").read_text())
    if [cell["label"] for cell in schedule] != expected_order:
        raise ValueError("Unexpected or incomplete arm order")
    if len({cell["pid"] for cell in schedule}) != len(schedule):
        raise ValueError("Expected a fresh worker per cell")

    rows_by_arm = {arm: [] for arm in ARMS}
    fixture_order = {}
    settings = set()
    for cell in schedule:
        arm, round_label = cell["label"].split("/")
        directory = args.results / cell["label"]
        done = json.loads((directory / "done.json").read_text())
        if (done["status"] != "OK" or not done["stream"] or not done["play"]
                or done["threads"] != 2 or done["screen"]):
            raise ValueError(f"Unexpected run settings for {cell['label']}")
        rows = read_rows(directory / "requests.jsonl")
        order = [row["id"] for row in rows]
        if len(order) != len(fixtures) or set(order) != set(fixtures):
            raise ValueError(f"Unexpected measured fixtures for {cell['label']}")
        if fixture_order.setdefault(round_label, order) != order:
            raise ValueError(f"Unpaired fixture order for {round_label}")
        for row in rows:
            check_row(row, fixtures[row["id"]])
            settings.add((row["audio_buffer_frames"], row["audio_start_threshold_frames"],
                          row["audio_route_type"], done["media_volume"], done["media_volume_max"]))
        rows_by_arm[arm].extend(rows)
    if len(settings) != 1:
        raise ValueError(f"Playback settings differed across cells: {settings}")

    buckets = sorted({fixture["bucket"] for fixture in fixtures.values()})
    summary = {"device": device, "rounds": ROUNDS, "audio_settings": sorted(settings)}
    summary["arms"] = {
        arm: {"pooled": summarize(rows), "buckets": {
            bucket: summarize([row for row in rows if row["bucket"] == bucket])
            for bucket in buckets
        }} for arm, rows in rows_by_arm.items()
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

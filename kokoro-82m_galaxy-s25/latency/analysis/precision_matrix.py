"""Reduce a paired Galaxy S25 TTS precision capture to per-arm summaries."""
import argparse
import json
from pathlib import Path
from random import Random
from statistics import median

from matched_chunks import chunk_stats, distribution, read_rows

ARMS = ("kokoro_fp32", "kokoro_int8", "mini_fp32", "mini_int8", "mini_int4",
        "pico_fp32", "pico_int8", "pico_int4")
ORDER = (0, 1, 7, 2, 6, 3, 5, 4)


def audit_chunks(row, fixture):
    events = row["chunk_events"]
    ready = [event["ready_ms"] for event in events]
    if (row["callbacks"] != 3 or row["sample_rate"] != 24000
            or row["bucket"] != fixture["bucket"]
            or [event["text"] for event in events] != fixture["segments"]
            or any(event["frames"] <= 0 for event in events)
            or sum(event["frames"] for event in events) != row["frames"]
            or not 0 < ready[0] < ready[1] < ready[2] <= row["synthesis_complete_ms"]):
        raise ValueError(f"Invalid three-chunk capture for {row['id']}")


def speed_stats(rows):
    # Broken or silent audio can have valid synthesis timings and no speech onset.
    stats = {}
    for name in ("first_callback_ms", "first_presented_signal_ms", "synthesis_complete_ms",
                 "playback_drained_ms", "rtf", "vm_hwm_kib", "post_request_pss_kib"):
        valid = [row[name] for row in rows if row[name] is not None]
        stats[name] = {**distribution(valid), "observed": len(valid)} if valid else None
    played = [row for row in rows if row["audio_playback_error"] is None
              and row["audio_written_frames"] == row["frames"]
              and len(row["observed_queue_empty_wait_ms"]) == 2]
    if played:
        stats.update(chunk_stats(played))
    stats["requests"] = len(rows)
    stats["valid_playback"] = len(played)
    stats["requests_without_speech_onset"] = sum(row["onset_frame"] < 0 for row in rows)
    stats["requests_with_nonfinite_pcm"] = sum(row["nonfinite_samples"] > 0 for row in rows)
    stats["requests_with_clipping"] = sum(row["clipped_samples"] > 0 for row in rows)
    stats["request_thermal_status"] = sorted({row["thermal_status"] for row in rows})
    return stats


def paired_ratio(rounds, baseline, candidate, metric):
    ratios = []
    for index in range(len(ARMS)):
        a, b = rounds.get((baseline, index)), rounds.get((candidate, index))
        if a is None or b is None:
            return None
        a_values, b_values = ([row[metric] for row in arm] for arm in (a, b))
        if any(value is None or value <= 0 for value in a_values + b_values):
            return None
        ratios.append(median(a_values) / median(b_values))
    rng = Random(20261121)
    draws = sorted(median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return {"median": median(ratios), "ci95": [draws[249], draws[9749]], "paired_rounds": len(ratios)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixtures = {row["id"]: row for row in read_rows(args.fixtures)}
    schedule = json.loads((args.results / "schedule.json").read_text())
    expected = [f"{ARMS[(index + round_id) % len(ARMS)]}/round-{round_id}"
                for round_id in range(len(ARMS)) for index in ORDER]
    if [cell["label"] for cell in schedule] != expected[:len(schedule)]:
        raise ValueError("Unexpected on-device round order")
    mode = json.loads((args.results / "capture-mode.json").read_text())
    if mode["mode"] != "precision-matrix":
        raise ValueError("Invalid capture mode")
    device = json.loads((args.results / "device-start.json").read_text())
    if not device["model"].startswith("SM-S931"):
        raise ValueError("Expected Galaxy S25 evidence")

    rows_by_arm = {arm: [] for arm in ARMS}
    errors = {arm: [] for arm in ARMS}
    rounds, fixture_order, settings = {}, {}, set()
    for label in expected:
        arm, round_label = label.split("/")
        number = int(round_label.removeprefix("round-"))
        directory = args.results / label
        if not (directory / "done.json").is_file():
            errors[arm].append({"label": label, "verdict": "NOT RUN",
                                "error": "Device capture contains no terminal worker record"})
            continue
        done = json.loads((directory / "done.json").read_text())
        if not (directory / "requests.jsonl").is_file():
            errors[arm].append({"label": label, "verdict": "NOT RUN", "error": done.get("error")})
            continue
        if (done["threads"] != 2 or not done["play"] or not done["stream"]
                or done["matrix_arm"] != arm):
            raise ValueError(f"Unexpected worker settings for {label}")
        measured = read_rows(directory / "requests.jsonl")
        order = [row["id"] for row in measured]
        if len(order) != len(fixtures) or set(order) != set(fixtures):
            raise ValueError(f"Incomplete on-device fixture order for {label}")
        if fixture_order.setdefault(round_label, order) != order:
            raise ValueError(f"Unpaired fixture order for {round_label}")
        rows = []
        for row in measured:
            if row["status"] != "OK":
                errors[arm].append({"label": label, "id": row["id"], "verdict": "NOT RUN",
                                    "error": row["error"]})
                continue
            audit_chunks(row, fixtures[row["id"]])
            if row["audio_playback_error"] is None:
                settings.add((row["audio_buffer_frames"], row["audio_start_threshold_frames"],
                              row["audio_route_type"], done["media_volume"], done["media_volume_max"]))
            rows.append(row)
        if len(rows) == len(fixtures):
            rounds[(arm, number)] = rows
        rows_by_arm[arm].extend(rows)
    if len(settings) > 1:
        raise ValueError(f"Playback settings differed across cells: {settings}")

    summary = {"device": device, "expected_rounds": len(ARMS), "completed_cells": len(schedule),
               "audio_settings": sorted(settings), "arms": {}, "paired": {}}
    buckets = sorted({fixture["bucket"] for fixture in fixtures.values()})
    for arm, rows in rows_by_arm.items():
        summary["arms"][arm] = {"pooled": speed_stats(rows) if rows else None, "errors": errors[arm],
                               "buckets": {bucket: speed_stats(subset)
                                           for bucket in buckets
                                           if (subset := [row for row in rows if row["bucket"] == bucket])}}
    for arm in ARMS:
        model, precision = arm.split("_")
        if precision == "fp32":
            continue
        baselines = [f"{model}_fp32"]
        if precision == "int4":
            baselines.append(f"{model}_int8")
        for baseline in baselines:
            summary["paired"][f"{baseline}/{arm}"] = {
                metric: paired_ratio(rounds, baseline, arm, metric)
                for metric in ("first_presented_signal_ms", "synthesis_complete_ms", "rtf")
            }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

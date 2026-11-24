"""Audit and summarize the published Kokoro configurations on a Galaxy S25."""
import argparse
import json
from pathlib import Path

from matched_chunks import read_rows
from precision_matrix import audit_chunks, speed_stats

ARMS = ("sherpa_official_int8", "mediapipe_opencl", "adreno_fp16", "pico_int8")
ORDER = (0, 1, 3, 2)


def summarize(capture, fixtures):
    results = capture / "results"
    device = json.loads((results / "device-start.json").read_text())
    if not device["model"].startswith("SM-S931"):
        raise ValueError("Expected a physical base Galaxy S25")
    mode = json.loads((results / "capture-mode.json").read_text())
    if mode != {"mode": "release-panel", "kokoro_seeded": False}:
        raise ValueError(f"Unexpected capture mode: {mode}")
    schedule = json.loads((results / "schedule.json").read_text())
    expected = [f"{ARMS[(index + round_id) % len(ARMS)]}/round-{round_id}"
                for round_id in range(len(ARMS)) for index in ORDER]
    if [cell["label"] for cell in schedule] != expected[:len(schedule)]:
        raise ValueError("Unexpected arm order")
    if len(schedule) != len({cell["pid"] for cell in schedule}):
        raise ValueError("Expected a fresh process per cell")
    routing = capture / "backend-placement.txt"
    placement = routing.read_text().splitlines() if routing.is_file() else []
    arm_rows = {arm: [] for arm in ARMS}
    by_round = {}
    failures = {arm: [] for arm in ARMS}
    backends = {arm: [] for arm in ARMS}
    fixture_orders = {}
    playback_settings = set()
    metadata = json.loads((capture / "device-metadata.json").read_text())
    for label in expected:
        arm, round_name = label.split("/")
        cell = results / label
        if not (cell / "done.json").is_file():
            failures[arm].append({"round": round_name, "error": metadata.get("status"),
                                  "detail": "Capture lacks the terminal worker record"})
            continue
        done = json.loads((cell / "done.json").read_text())
        if not (cell / "requests.jsonl").is_file():
            failures[arm].append({"round": round_name, "error": done.get("error")})
            continue
        if done["release_arm"] != arm or not done["stream"] or not done["play"]:
            raise ValueError(f"Unexpected worker settings: {label}")
        if arm in {"sherpa_official_int8", "pico_int8"} and done["threads"] != 2:
            raise ValueError(f"Unexpected CPU thread count: {label}")
        rows = read_rows(cell / "requests.jsonl")
        order = [row["id"] for row in rows]
        if len(order) != len(fixtures) or set(order) != set(fixtures):
            raise ValueError(f"Missing fixtures: {label}")
        if fixture_orders.setdefault(round_name, order) != order:
            raise ValueError(f"Input order changed in {label}")
        pid = str(done["pid"])
        worker_logs = [line for line in placement if pid in line.split()[2:3]]
        backends[arm].append({"round": round_name, "pid": done["pid"],
                              "runtime": done["runtime"], "log": worker_logs})
        valid = []
        for row in rows:
            if row["status"] != "OK":
                failures[arm].append({"round": round_name, "id": row["id"], "error": row["error"]})
                continue
            audit_chunks(row, fixtures[row["id"]])
            if row["audio_playback_error"] is None:
                playback_settings.add((row["audio_buffer_frames"], row["audio_start_threshold_frames"],
                                       row["audio_route_type"], done["media_volume"], done["media_volume_max"]))
            valid.append(row)
        arm_rows[arm].extend(valid)
        if len(valid) == len(fixtures):
            by_round[arm, round_name] = valid
    if len(playback_settings) > 1:
        raise ValueError(f"Playback settings changed: {playback_settings}")
    rounds = [f"round-{i}" for i in range(len(ARMS))
              if all((arm, f"round-{i}") in by_round for arm in ARMS)]
    return {"device": device, "cells": len(schedule), "complete_matched_rounds": rounds,
            "audio_settings": sorted(playback_settings),
            "arms": {arm: {"pooled": speed_stats(rows) if rows else None,
                           "matched": speed_stats([row for r in rounds for row in by_round[arm, r]])
                           if rounds else None,
                           "timing_failures": failures[arm], "backend": backends[arm]}
                     for arm, rows in arm_rows.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixtures = {row["id"]: row for row in read_rows(args.fixtures)}
    args.output.write_text(json.dumps(summarize(args.capture, fixtures), indent=2) + "\n")


if __name__ == "__main__":
    main()

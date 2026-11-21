#!/usr/bin/env python3
"""Measure lead-in levels in matched mono speech clips before listening."""
import argparse
from array import array
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles
import sys

from diagnose_pcm_words import observe


def analyze_first_chunk(path, rate, thresholds):
    pcm = path.read_bytes()
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    first_sample = next((index for index, sample in enumerate(samples) if sample), None)
    window = rate // 50
    powers = [
        sum(sample * sample for sample in samples[start:start + window]) / window
        for start in range(0, len(samples) - window + 1, window)
    ]
    threshold_ms = {}
    for decibels in thresholds:
        minimum_power = 32768 ** 2 * 10 ** (decibels / 10)
        first_block = next(
            (index for index, (left, right) in enumerate(zip(powers, powers[1:]))
             if min(left, right) >= minimum_power),
            None,
        )
        threshold_ms[str(decibels)] = None if first_block is None else first_block * 20
    return {
        "sha256": sha256(pcm).hexdigest(),
        "first_nonzero_ms": None if first_sample is None else first_sample * 1000 / rate,
        "energy_onset_offset_ms": threshold_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--rate", type=int, default=24000)
    parser.add_argument("--all-waves", action="store_true")
    args = parser.parse_args()
    if args.rate < 50 or args.rate % 50:
        parser.error("sample rate must be a positive multiple of 50")
    thresholds = (-60, -50, -45, -40, -35)
    baseline = observe(args.capture, args.baseline, args.all_waves)
    candidate = observe(args.capture, args.candidate, args.all_waves)
    pairs = []
    for key in sorted(baseline.keys() & candidate.keys()):
        row = {"request": key}
        for label, (paths, channels) in (
                ("baseline", baseline[key]), ("candidate", candidate[key])):
            if channels != 1:
                raise ValueError(f"{key}: signal offsets require mono PCM")
            row[label] = analyze_first_chunk(paths[0], args.rate, thresholds)
        pairs.append(row)
    if not pairs:
        raise ValueError("No complete matched waveform pairs")
    summary = {}
    for db in map(str, thresholds):
        summary[db] = {}
        for label in ("baseline", "candidate"):
            values = [arm["energy_onset_offset_ms"][db] - arm["first_nonzero_ms"]
                      for pair in pairs if (arm := pair[label])["energy_onset_offset_ms"][db] is not None
                      and arm["first_nonzero_ms"] is not None]
            summary[db][label] = {
                "clips": len(values),
                "lead_in_ms_median": median(values) if values else None,
                "lead_in_ms_p95": quantiles(values, n=20, method="inclusive")[-1]
                if len(values) > 1 else (values[0] if values else None),
            }
    print(json.dumps({
        "scope": "PCM first-chunk offset diagnostic",
        "windows": "Two consecutive nonoverlapping 20 ms RMS windows, full scale 32768",
        "rate": args.rate,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "unpaired_baseline": sorted(baseline.keys() - candidate.keys()),
        "unpaired_candidate": sorted(candidate.keys() - baseline.keys()),
        "summary": summary,
        "rows": pairs,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Derive latency ratios from paired, complete native capacity rounds."""
import argparse
from itertools import product
import json
from math import ceil, floor
from pathlib import Path
from statistics import median


def ratio(baseline, candidate):
    return median(baseline) / median(candidate)


def matched_ratio(baseline, candidate):
    """Enumerate paired bootstrap draws over three to six fresh rounds."""
    rounds = len(baseline)
    draws = [ratio([baseline[i] for i in draw], [candidate[i] for i in draw])
             for draw in product(range(rounds), repeat=rounds)]
    draws.sort()
    last = len(draws) - 1
    return {"baseline": baseline, "candidate": candidate,
            "ratio": ratio(baseline, candidate),
            "ci95": [draws[floor(last * 0.025)], draws[ceil(last * 0.975)]]}


def complete(cell):
    return cell["requests"] == cell["expected_requests"] and cell["failed_requests"] == 0


def paired_level(left, right):
    baseline, candidate = left["rounds"], right["rounds"]
    enough_rounds = 3 <= len(baseline) == len(candidate) <= 6
    if not enough_rounds or not all(complete(cell) for cell in baseline + candidate):
        return {"coverage": "UNPROVEN"}
    result = {"coverage": "PASS", "rounds": len(baseline),
              "baseline_level_verdict": left["verdict"], "candidate_level_verdict": right["verdict"]}
    for name in ("first_audio_p95_ms", "rtf_p95"):
        before = [max(stream[name] for stream in cell["streams"]) for cell in baseline]
        after = [max(stream[name] for stream in cell["streams"]) for cell in candidate]
        result[name] = matched_ratio(before, after)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="Both arms from one interleaved capture import")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--concurrency", action="append", type=int,
                        help="Compare a shared tested level when registered grids differ")
    arguments = parser.parse_args()
    rows = {row["label"]: row for row in map(json.loads, arguments.summary.read_text().splitlines())}
    baseline, candidate = (rows[name] for name in (arguments.baseline, arguments.candidate))
    common = ({level["concurrency"]: level for level in arm["levels"]}
              for arm in (baseline, candidate))
    left, right = common
    selected = set(arguments.concurrency or left.keys())
    if arguments.concurrency and not selected <= left.keys() & right.keys():
        raise ValueError("Every selected level must be present in both captures")
    if not arguments.concurrency and set(left) != set(right):
        raise ValueError("Compare arms from the same registered load plan")
    print(json.dumps({"baseline_evidence": baseline["evidence_id"],
                      "candidate_evidence": candidate["evidence_id"],
                      "method": "ratio of median worst-stream p95; exact paired round bootstrap; 95% percentile bounds",
                      "levels": {n: paired_level(left[n], right[n]) for n in sorted(selected)}}, indent=2))


if __name__ == "__main__":
    main()

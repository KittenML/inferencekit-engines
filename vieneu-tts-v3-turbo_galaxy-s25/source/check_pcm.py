"""Check matched Android captures for valid PCM and byte equality within one precision."""
import argparse
import array
import hashlib
import json
from pathlib import Path


def collect(capture, arm, selected_round, selected_level=None, waves=1):
    files = {}
    for row in map(json.loads, (capture / f"{arm}.jsonl").read_text().splitlines()):
        if row["wave"] >= waves or (selected_round is not None and row["round"] != selected_round):
            continue
        if selected_level is not None and row["concurrency"] != selected_level:
            continue
        for chunk, (_, frames, leading) in enumerate(row["chunks"]):
            fixture = row["id"]
            stem = f'{fixture}-r{row["round"]}-n{row["concurrency"]}-s{row["stream"]}-c{chunk}'
            input_file = capture / "pcm" / arm / f"{stem}.s16le"
            pcm = input_file.read_bytes()
            channels, remainder = divmod(len(pcm), frames * 2)
            if channels not in (1, 2) or remainder:
                raise ValueError(f"{stem}: PCM byte count does not agree with timing record")
            samples = array.array("h", pcm)
            first = next((index // channels for index, value in enumerate(samples) if value), None)
            if first != leading:
                raise ValueError(f"{stem}: first PCM signal does not agree with timing record")
            key = stem if selected_round is None else f'{fixture}-n{row["concurrency"]}-s{row["stream"]}-c{chunk}'
            files[key] = {
                "sha256": hashlib.sha256(pcm).hexdigest(),
                "frames": frames, "channels": channels,
            }
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--same-precision", action="store_true")
    parser.add_argument("--reference-round", type=int)
    parser.add_argument("--candidate-round", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--waves", type=int, default=1)
    arguments = parser.parse_args()
    if (arguments.reference_round is None) != (arguments.candidate_round is None):
        parser.error("provide both round indices for a matched repeat check")
    if arguments.waves < 1:
        parser.error("--waves must be positive")
    reference = collect(arguments.capture, arguments.reference, arguments.reference_round,
                        arguments.concurrency, arguments.waves)
    candidate = collect(arguments.capture, arguments.candidate, arguments.candidate_round,
                        arguments.concurrency, arguments.waves)
    paired = reference.keys() & candidate.keys()
    identical = sorted(key for key in paired if reference[key] == candidate[key])
    structural = "PASS" if len(paired) and paired == reference.keys() == candidate.keys() else "UNPROVEN"
    exact = ("PASS" if len(identical) == len(paired) else "FAIL") if structural == "PASS" else "UNPROVEN"
    report = {"objective": "exact PCM within the retained request waves", "retained_waves": arguments.waves,
              "reference": arguments.reference, "candidate": arguments.candidate,
              "reference_round": arguments.reference_round, "candidate_round": arguments.candidate_round,
              "paired_chunks": len(paired), "byte_identical_chunks": len(identical),
              "structure": structural, "byte_equivalence": exact,
              "quality": exact if arguments.same_precision else "UNPROVEN",
              "mismatched": sorted(paired - set(identical)),
              "reference_pcm": reference, "candidate_pcm": candidate}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

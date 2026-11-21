"""Export saved native requests to the speech quality importer."""
import argparse
import json
from pathlib import Path


def export(capture, arm, level, rounds, waves, output):
    rows = [row for row in map(json.loads, (capture / f"{arm}.jsonl").read_text().splitlines())
            if row["wave"] < waves and row["concurrency"] == level]
    if not rows:
        raise ValueError("no requests for this configuration and level")
    keys = {(row["round"], row["wave"], row["stream"]) for row in rows}
    expected = {(index, wave, stream) for index in range(rounds)
                for wave in range(waves) for stream in range(level)}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError("incomplete retained request coverage")

    for row in rows:
        if row["complete_ms"] is None or len(row["chunks"]) != 2:
            raise ValueError("incomplete speech request")
        name = row["id"]
        stem = f'{name}-r{row["round"]}-n{level}-s{row["stream"]}'
        destination = output / f'r{row["round"]}' if rounds > 1 else output
        destination.mkdir(parents=True, exist_ok=True)
        chunks = [(capture / "pcm" / arm / f"{stem}-c{index}.s16le").read_bytes()
                  for index in range(2)]
        channel_counts = {len(chunk) / (frames * 2)
                          for chunk, (_, frames, _) in zip(chunks, row["chunks"])}
        if len(channel_counts) != 1 or not channel_counts <= {1, 2}:
            raise ValueError(f"{stem}: waveform size disagrees with timing")
        (destination / f"{name}.pcm").write_bytes(b"".join(chunks))
        yield int(channel_counts.pop())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("arm")
    parser.add_argument("concurrency", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--waves", type=int, default=1)
    arguments = parser.parse_args()
    channels = set(export(arguments.capture, arguments.arm, arguments.concurrency,
                          arguments.rounds, arguments.waves, arguments.output))
    if len(channels) != 1:
        raise ValueError("channel layout changed between requests")
    print(channels.pop())


if __name__ == "__main__":
    main()

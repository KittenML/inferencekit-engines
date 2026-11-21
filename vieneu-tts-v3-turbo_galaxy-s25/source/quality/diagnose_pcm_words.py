#!/usr/bin/env python3
"""Score paired saved PCM without changing timing inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unicodedata
import wave


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(" " if unicodedata.category(char).startswith("P") else char
                   for char in text).split()


def edit_distance(reference, candidate):
    previous = list(range(len(candidate) + 1))
    for index, expected in enumerate(reference, start=1):
        current = [index]
        for offset, actual in enumerate(candidate, start=1):
            current.append(min(previous[offset] + 1, current[-1] + 1,
                               previous[offset - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def observe(capture, arm, all_waves):
    observations = {}
    with (capture / f"{arm}.jsonl").open() as stream:
        for row in map(json.loads, stream):
            if (row["wave"] and not all_waves) or len(row["chunks"]) != 2 or row["complete_ms"] is None:
                continue
            key = f'{row["id"]}-r{row["round"]}-n{row["concurrency"]}-s{row["stream"]}'
            paths = [capture / "pcm" / arm / f"{key}-c{index}.s16le" for index in range(2)]
            layouts = set()
            for path, (_, frames, _) in zip(paths, row["chunks"], strict=True):
                channels, remainder = divmod(path.stat().st_size, frames * 2)
                if channels not in (1, 2) or remainder:
                    raise ValueError(f"{path}: PCM differs from the recorded 16-bit waveform")
                layouts.add(channels)
            if len(layouts) != 1:
                raise ValueError(f"{key}: PCM channel layout changed across chunks")
            observations[key] = paths, layouts.pop()
    return observations


def transcribe(pcm, channels, rate, scratch, whisper, weights, language):
    with wave.open(str(scratch / "native.wav"), "wb") as wav:
        wav.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(scratch / "native.wav"),
                    "-ac", "1", "-ar", "16000", str(scratch / "speech.wav")], check=True)
    output = scratch / "asr"
    subprocess.run([str(whisper), "-m", str(weights), "-l", language, "-t", "3", "-ng", "-nf",
                    "-nt", "-tp", "0", "-bs", "5", "-otxt", "-of", str(output),
                    str(scratch / "speech.wav")], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    return output.with_suffix(".txt").read_text().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--language", required=True, choices=("en", "vi"))
    parser.add_argument("--rate", required=True, type=int)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--all-waves", action="store_true")
    args = parser.parse_args()
    fixtures = {fixture: " ".join(sentences) for fixture, *sentences in
                (line.split("\t") for line in args.fixtures.read_text().splitlines())}
    baseline = observe(args.capture, args.baseline, args.all_waves)
    candidate = observe(args.capture, args.candidate, args.all_waves)
    common = baseline.keys() & candidate.keys()
    if not common:
        raise ValueError("No paired, complete PCM requests")
    rows = []
    transcripts = {}
    with TemporaryDirectory() as directory:
        for key in sorted(common):
            expected = fixtures[key.rsplit("-r", 1)[0]]
            expected_tokens = tokens(expected)
            expected_chars = "".join(expected_tokens)
            pair = {"request": key, "expected": expected,
                    "expected_space_delimited_tokens": len(expected_tokens),
                    "expected_characters": len(expected_chars)}
            for label, (paths, channels) in (("baseline", baseline[key]), ("candidate", candidate[key])):
                pcm = b"".join(path.read_bytes() for path in paths)
                digest = hashlib.sha256(pcm).hexdigest()
                cache_key = (digest, channels)
                if cache_key not in transcripts:
                    transcripts[cache_key] = transcribe(
                        pcm, channels, args.rate, Path(directory), args.whisper.resolve(),
                        args.weights.resolve(), args.language)
                text = transcripts[cache_key]
                actual = tokens(text)
                pair[label] = {"pcm_sha256": digest, "channels": channels, "transcript": text,
                               "token_edits": edit_distance(expected_tokens, actual),
                               "character_edits": edit_distance(expected_chars, "".join(actual))}
            rows.append(pair)
    with args.weights.open("rb") as checkpoint:
        weights_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    print(json.dumps({"objective": "exploratory ASR content diagnostic",
                      "language": args.language, "input_rate": args.rate, "weights_sha256": weights_sha256,
                      "capture_scope": "all waves" if args.all_waves else "wave zero",
                      "scorer": "whisper.cpp 4979e04f5dca, CPU beam 5, temperature 0, no fallback",
                      "baseline": args.baseline, "candidate": args.candidate,
                      "unpaired_baseline": sorted(baseline.keys() - candidate.keys()),
                      "unpaired_candidate": sorted(candidate.keys() - baseline.keys()),
                      "distinct_fixtures": len({row["request"].rsplit("-r", 1)[0] for row in rows}),
                      "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

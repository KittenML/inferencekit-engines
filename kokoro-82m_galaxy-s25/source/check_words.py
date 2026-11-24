"""Compare offline ASR transcripts for the saved Kokoro wave-zero requests."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from check_pcm import collect


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def edit_distance(reference, candidate):
    previous = list(range(len(candidate) + 1))
    for index, expected in enumerate(reference, start=1):
        current = [index]
        for offset, actual in enumerate(candidate, start=1):
            current.append(min(previous[offset] + 1, current[-1] + 1,
                               previous[offset - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def transcript(audio, sample_rate, scratch, recognizer, weights):
    wav = scratch / "speech.wav"
    pcm = b"".join(path.read_bytes() for path in audio)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sample_rate),
         "-ac", "1", "-i", "pipe:0", "-ar", "16000", str(wav)],
        input=pcm, check=True,
    )
    output = scratch / "transcript"
    subprocess.run(
        [str(recognizer), "-m", str(weights), "-l", "en", "-t", "3",
         "-ng", "-nt", "-otxt", "-of", str(output), str(wav)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return output.with_suffix(".txt").read_text().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--concurrency", type=int)
    arguments = parser.parse_args()
    fixture_file = Path(__file__).resolve().parent / "plans/kokoro.tsv"
    fixtures = {fixture: " ".join(sentences) for fixture, *sentences in
                (line.split("\t") for line in fixture_file.read_text().splitlines())}
    reference = collect(arguments.capture, arguments.reference, None, arguments.concurrency)
    candidate = collect(arguments.capture, arguments.candidate, None, arguments.concurrency)
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Recognize matched, complete wave-zero PCM captures")
    rows = []
    with TemporaryDirectory() as scratch:
        for first_chunk in sorted(stem for stem in reference if stem.endswith("-c0")):
            request = first_chunk.removesuffix("-c0")
            fixture = request.split("-r")[0]
            chunks = [f"{request}-c{index}" for index in range(2)]
            expected = fixtures[fixture]
            results = []
            for arm, files in [(arguments.reference, reference), (arguments.candidate, candidate)]:
                if any(files[chunk]["channels"] != 1 for chunk in chunks):
                    raise ValueError(f"{request}: expected monaural PCM")
                paths = [arguments.capture / "pcm" / arm / f"{chunk}.s16le" for chunk in chunks]
                results.append(transcript(paths, 24000, Path(scratch),
                                          arguments.whisper.resolve(), arguments.weights.resolve()))
            rows.append({"request": request, "expected": expected, "reference": results[0],
                         "candidate": results[1], "expected_words": len(words(expected)),
                         "reference_edits": edit_distance(words(expected), words(results[0])),
                         "candidate_edits": edit_distance(words(expected), words(results[1])),
                         "matched_words": words(results[0]) == words(results[1])})
    with arguments.weights.open("rb") as weight_file:
        weights_sha256 = hashlib.file_digest(weight_file, "sha256").hexdigest()
    print(json.dumps({"objective": "offline ASR content diagnostic on saved wave-zero PCM",
                      "reference": arguments.reference, "candidate": arguments.candidate,
                      "weights_sha256": weights_sha256, "measurements": rows}, indent=2))


if __name__ == "__main__":
    main()

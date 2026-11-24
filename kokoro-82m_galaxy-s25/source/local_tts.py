"""Install the selected Android runtime and render a WAV on an attached phone."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import wave

from prepare_payload import ARTIFACTS, digest_of, fetch
from run_device import (
    DEVICE, MEDIAPIPE_APP, adb, capture_mediapipe, media_runtime_log, selected_binary, stage_mediapipe,
)

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "build/local-optimized"
V_CPP = ROOT / "build/vieneu-cpp/android-kit"
PROFILES = {"kokoro-opencl": ("kokoro", 24000), "vieneu-cpp": ("vieneu", 48000)}


def stage(profile):
    adb("get-state")
    adb("shell", f"mkdir -p {DEVICE}/plans")
    if profile == "vieneu-cpp":
        if not (V_CPP / "capture-vieneu-cpp").is_file():
            raise ValueError("run ./install.sh vieneu-cpp first")
        adb("shell", f"mkdir -p {DEVICE}/vieneu-cpp")
        adb("push", f"{V_CPP}/.", f"{DEVICE}/vieneu-cpp", timeout=360)
        adb("shell", f"chmod 755 {DEVICE}/vieneu-cpp/capture-vieneu-cpp")
        return

    apk = ROOT / "build/mediapipe-capture.apk"
    if not apk.is_file():
        raise ValueError("run ./install.sh kokoro-opencl first")
    (PAYLOAD / "application").mkdir(parents=True, exist_ok=True)
    (PAYLOAD / "models/mediapipe").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(apk, PAYLOAD / "application/mediapipe-capture.apk")
    asset = next(item for item in ARTIFACTS if item[0] == "mediapipe/kokoro-GPU.litertlm")
    model = PAYLOAD / "models" / asset[0]
    if not model.is_file() or digest_of(model) != asset[2]:
        with tempfile.TemporaryDirectory() as scratch:
            fetch(PAYLOAD, Path(scratch), asset)
    adb("shell", f"mkdir -p {DEVICE}/models/mediapipe")
    adb("push", str(model), f"{DEVICE}/models/{asset[0]}", timeout=180)
    adb("push", str(ROOT / "source/plans/kokoro.tsv"), f"{DEVICE}/plans/kokoro.tsv")
    stage_mediapipe(PAYLOAD, ["kokoro-mediapipe-gpu"])


def fixture_file(family, sentences, location):
    if len(sentences) > 2 or any(not text.strip() or any(char in text for char in "\n\r\t\0")
                                 for text in sentences):
        raise ValueError("provide up to two nonempty input sentences without control characters")
    fixtures = (ROOT / f"source/plans/{family}.tsv").read_text().splitlines()
    mode = "stream"
    if sentences:
        mode = "full" if len(sentences) == 1 else "stream"
        request = ["sample", *sentences, " "][:3]
    else:
        request = ["sample", *fixtures[0].split("\t")[1:]]
    location.write_text("\t".join(request) + "\n" + fixtures[-1] + "\n")
    return mode


def media_pcm(fixture, mode, logs):
    adb("push", str(fixture), f"{DEVICE}/plans/kokoro.tsv")
    adb("shell", f"cat {DEVICE}/plans/kokoro.tsv | run-as {MEDIAPIPE_APP} sh -c "
                 + shlex.quote("cat > files/kokoro.tsv"))
    result = capture_mediapipe("kokoro-mediapipe-gpu", 0, 1, 1, 3000, mode)
    if result.returncode or b"status=complete" not in result.stdout:
        raise RuntimeError(result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"))
    delegate_log = media_runtime_log("kokoro-opencl", result.stdout, logs)
    if (b"Created OpenCL device" not in delegate_log or
            b"Initializing OpenCL-based API" not in delegate_log):
        raise RuntimeError("OpenCL initialization could not be verified in the Android delegate log")
    observation = adb("exec-out", f"run-as {MEDIAPIPE_APP} cat files/capture.jsonl").stdout
    return observation, f"run-as {MEDIAPIPE_APP} cat files/pcm/"


def vieneu_pcm(fixture, mode):
    adb("push", str(fixture), f"{DEVICE}/plans/vieneu.tsv")
    binary, environment = selected_binary("vieneu-cpp", "stock")
    command = shlex.join((binary, "vieneu", "vieneu-cpp/model", "plans/vieneu.tsv",
                          mode, "1", "0", "0", "1", "1", "3000", "pcm/sample"))
    observation = adb("shell", f"cd {DEVICE} && rm -rf pcm/sample && mkdir -p pcm/sample && "
                                f"{environment} && {command}").stdout
    return observation, f"cat {DEVICE}/pcm/sample/"


def sample(profile, sentences, output):
    family, rate = PROFILES[profile]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        fixture = Path(scratch) / f"{family}.tsv"
        mode = fixture_file(family, sentences, fixture)
        observation, pcm_prefix = (media_pcm(fixture, mode, output.parent) if family == "kokoro"
                                   else vieneu_pcm(fixture, mode))
    row = json.loads(observation)
    expected = 2 if mode == "stream" else 1
    if row["id"] != "sample" or row["complete_ms"] is None or len(row["chunks"]) != expected:
        raise RuntimeError("the Android runtime did not return a complete sample")
    waveform = bytearray()
    for chunk, (_, frames, leading) in enumerate(row["chunks"]):
        if frames <= 0 or leading >= frames:
            raise RuntimeError("the Android runtime returned a silent sample")
        pcm = adb("exec-out", f"{pcm_prefix}sample-r0-n1-s0-c{chunk}.s16le").stdout
        if len(pcm) != frames * 2:
            raise RuntimeError("the returned waveform is not 16-bit mono PCM")
        waveform.extend(pcm)
    with wave.open(str(output), "wb") as recording:
        recording.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        recording.writeframes(waveform)
    print(f"{output} · {rate} Hz · {len(waveform) / 2 / rate:.2f} s")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "sample"):
        command = commands.add_parser(name)
        command.add_argument("profile", choices=PROFILES)
        if name == "sample":
            command.add_argument("sentences", nargs="*")
    arguments = parser.parse_args()
    if arguments.command == "stage":
        stage(arguments.profile)
    else:
        sample(arguments.profile, arguments.sentences, ROOT / f"build/output/{arguments.profile}.wav")


if __name__ == "__main__":
    main()

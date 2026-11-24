"""Import the raw native timing records into InferenceKit."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL_PINS = {
    "kokoro-fp32": (
        "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987",
        "sherpa-onnx-v1.13.6", "fp32",
    ),
    "kokoro-int8": (
        "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987",
        "sherpa-onnx-v1.13.6", "int8",
    ),
    "kokoro-audiocpp-fp32": (
        "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987",
        "audio.cpp-749fffec6ad4a", "FP32 checkpoint and voice weights",
    ),
    "kokoro-adreno-fp16": (
        "a8nova/adreno-llms-weights/Kokoro-82M", "7a5f7a1d4fe10589d4b7984b6161591065902707",
        "adreno-llms-2b1aee9fdd7d-CLBlast", "FP16 weights, OpenCL mixed compute",
    ),
    "kokoro-soniqo-cpu": (
        "soniqo/Kokoro-82M-ONNX", "a49a07fb772d5281a42570e8645bf238f7f73bf2",
        "soniqo-speech-core-59b9d7902ea6-ORT-1.27.0-cpu", "FP16 external weights, realtime graph",
    ),
    "kokoro-soniqo-nnapi": (
        "soniqo/Kokoro-82M-ONNX", "a49a07fb772d5281a42570e8645bf238f7f73bf2",
        "soniqo-speech-core-59b9d7902ea6-ORT-1.27.0-nnapi-requested", "FP16 external weights, realtime graph",
    ),
    "kokoro-mediapipe-cpu": (
        "litert-community/Kokoro-82M", "bf1c2405642aa310bd210205c22d3ae1abc253fb",
        "MediaPipe-Tasks-Audio-1.2.2-LiteRT-cpu", "INT8 model, CPU vocoder",
    ),
    "kokoro-mediapipe-gpu": (
        "litert-community/Kokoro-82M", "bf1c2405642aa310bd210205c22d3ae1abc253fb",
        "MediaPipe-Tasks-Audio-1.2.2-LiteRT-gpu", "INT8 model, requested GPU FP16 vocoder",
    ),
    "kokoro-mediapipe-gpu-bundle-on-cpu": (
        "litert-community/Kokoro-82M", "bf1c2405642aa310bd210205c22d3ae1abc253fb",
        "MediaPipe-Tasks-Audio-1.2.2-LiteRT-cpu", "INT8 GPU bundle with CPU vocoder",
    ),
    "kokoro-litert-omni-gpu": (
        "litert-community/Kokoro-82M", "bf1c2405642aa310bd210205c22d3ae1abc253fb",
        "LiteRT-LM-Omni-v0.19.1-7a03724cece4", "INT8 model, requested GPU FP16 vocoder",
    ),
    "vieneu-bf16": (
        "pnnbao-ump/VieNeu-TTS-v3-Turbo", "61b85e3d937fbbacb387714180e8182823512523",
        "audio.cpp-749fffec6ad4a", "bf16 backbone, f16 codec",
    ),
    "vieneu-q8": (
        "pnnbao-ump/VieNeu-TTS-v3-Turbo", "61b85e3d937fbbacb387714180e8182823512523",
        "audio.cpp-749fffec6ad4a", "q8_0 matmuls, bf16 norms and embeddings, f16 codec",
    ),
    "vieneu-cpp": (
        "pnnbao-ump/VieNeu-TTS-v3-Turbo", "61b85e3d937fbbacb387714180e8182823512523",
        "VieNeu-TTS.cpp-838979faf0-llama-ebd048fc5e4b-ORT-1.27.0",
        "Q8_0 talker backbone, native acoustic weights, ONNX codec",
    ),
}
for backend in ("cpu", "htp", "opencl", "mixed"):
    model, revision, _, dtype = MODEL_PINS["vieneu-cpp"]
    MODEL_PINS[f"vieneu-cpp-snap-{backend}"] = (
        model, revision, f"VieNeu-TTS.cpp-838979faf0-llama-b12503-snapdragon-{backend}", dtype,
    )
for backend in ("gpu", "npu"):
    model, revision, _, dtype = MODEL_PINS["kokoro-mediapipe-gpu"]
    MODEL_PINS[f"kokoro-mediapipe-{backend}-123"] = (
        model, revision, f"MediaPipe-Tasks-Audio-1.2.3-LiteRT-{backend}-requested", dtype,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inferencekit", default="inferencekit")
    parser.add_argument("--hardware", default="Samsung Galaxy S25 / Android 16")
    parser.add_argument("--first-audio-limit-ms", type=float)
    parser.add_argument("--buffered-plan", action="store_true", help="Use the registered buffered playback plan")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((arguments.capture / "run-plan.json").read_bytes())
    if manifest.get("quality_only"):
        parser.error("quality-only capture; use the speech PCM importer")
    plan_directory = ROOT / "source/plans"
    fixture_directory = plan_directory / manifest["fixture_set"] if manifest.get("fixture_set") else plan_directory
    with (arguments.output / "summary.jsonl").open("w", encoding="utf-8") as summary:
        for arm in manifest["arms"]:
            label, model, mode, _, _, *options = arm
            config = options[0] if options else {}
            model_id, revision, engine, precision = MODEL_PINS[model]
            family = model.split("-")[0]
            timings = arguments.capture / f"{label}.jsonl"
            if not timings.is_file():
                summary.write(json.dumps({"label": label, "capture": "NOT RUN"}) + "\n")
                continue
            env = {**os.environ, "INFERENCEKIT_RUNS_DIR": str(arguments.output / "evidence")}
            suffix = config.get("INFERENCEKIT_PLAN", manifest["phase"] + f"-{family}-{mode}.json")
            if arguments.buffered_plan:
                suffix = suffix.removesuffix(".json") + "-buffered.json"
            runtime = config.get("INFERENCEKIT_AUDIORUNTIME")
            if runtime in ("backbone-hidden", "kleidiai", "kokoro-im2col", "kokoro-vulkan", "openmp",
                           "ort-pool", "vieneu-cpp-batched", "vieneu-cpp-shared", "vieneu-shared", "vulkan"):
                engine += "-" + runtime
            if config.get("KOKORO_AUDIOCPP_NEON") == "1":
                engine += "-neon-vocoder"
            if config.get("INFERENCEKIT_MEDIAPIPE_POOL_SIZE"):
                engine += "-worker-pool"
            if config.get("INFERENCEKIT_OMNI_PROCESS_PER_STREAM") == "1":
                engine += "-process-per-stream"
            onset_rule = ("--first-audio-as-metric",)
            if arguments.first_audio_limit_ms is not None:
                onset_rule = ("--slo-first-audio-p95-ms", str(arguments.first_audio_limit_ms))
            command = (
                arguments.inferencekit, "import", "audio-capacity", "--timings", str(timings),
                "--plan", str(plan_directory / suffix),
                "--fixtures", str(fixture_directory / f"{family}.tsv"),
                "--label", label, "--model", model_id, "--model-revision", revision,
                "--engine-version", engine, "--hardware-name", arguments.hardware,
                "--model-dtype", precision, *onset_rule, "--json",
            )
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode:
                summary.write(json.dumps({"label": label, "capture": "INVALID", "error": result.stderr.strip()}) + "\n")
                continue
            record = json.loads(result.stdout)
            data = record["data"]
            summary.write(json.dumps({"label": label, "evidence_id": record.get("id", record.get("evidence_id")),
                                      **data}) + "\n")
            print(label, data["verdict"], data["highest_passing_tested"], data["n_star"], flush=True)


if __name__ == "__main__":
    main()

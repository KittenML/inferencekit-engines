"""Zip the scripts and native binaries for Device Farm."""
import argparse
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("scenario", type=Path)
parser.add_argument("archive", type=Path)
settings = parser.parse_args()
scenario = json.loads(settings.scenario.read_bytes())
phase = scenario["phase"]
models = {arm[1] for arm in scenario["arms"]}
fixture_set = scenario.get("fixture_set")
if fixture_set is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", fixture_set):
    parser.error("invalid fixture set name")
plans = ROOT / "source/plans"
fixture_directory = plans / fixture_set if fixture_set else plans
registered_plans = {arm[5]["INFERENCEKIT_PLAN"] for arm in scenario["arms"]
                    if len(arm) > 5 and "INFERENCEKIT_PLAN" in arm[5]}
if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}\.json", name) or not (plans / name).is_file()
       for name in registered_plans):
    parser.error("invalid registered plan filename")

with zipfile.ZipFile(settings.archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    archive.write(settings.scenario, "run-plan.json")
    archive.write(ROOT / "source" / "run_device.py", "tests/test_capture.py")
    archive.write(ROOT / "source" / "prepare_payload.py", "tests/prepare_payload.py")
    if scenario.get("local_reproduction"):
        for filename in (
                "serve.sh", "test.sh", "source/local_tts.py", "source/prepare_payload.py",
                "source/run_device.py", "source/plans/kokoro.tsv", "source/plans/vieneu.tsv"):
            archive.write(ROOT / filename, "local-reproduction/" + filename)
    archive.writestr("requirements.txt", "pytest==7.4.4\n")
    for wheel in sorted((ROOT / "build/wheelhouse").glob("*.whl")):
        archive.write(wheel, f"wheelhouse/{wheel.name}")
    archive.write(ROOT / "build/stripped/capture", "payload/bin/iktts7s3a5q7o")
    archive.write(ROOT / "build/stripped/libaudiocpp.so", "payload/lib/libaudiocpp.so")
    runtimes = {arm[5].get("INFERENCEKIT_AUDIORUNTIME") for arm in scenario["arms"] if len(arm) > 5}
    for runtime in sorted(runtimes & {"backbone-hidden", "kleidiai", "kokoro-im2col", "kokoro-vulkan",
                                     "openmp", "vieneu-shared", "vulkan"}):
        archive.write(ROOT / f"build/stripped/{runtime}/libaudiocpp.so",
                      f"payload/lib/{runtime}/libaudiocpp.so")
    if "vieneu-shared" in runtimes:
        archive.write(ROOT / "build/stripped/vieneu-shared/capture", "payload/lib/vieneu-shared/capture")
    if "ort-pool" in runtimes:
        archive.write(ROOT / "build/stripped/ort-pool/libsherpa-onnx-c-api.so",
                      "payload/lib/ort-pool/libsherpa-onnx-c-api.so")
    archive.write(ROOT / "build/stripped/libsea_g2p_rs.so", "payload/models/vieneu/libsea_g2p_rs.so")
    if any(model.startswith("kokoro-mediapipe-") and not model.endswith("-123") for model in models):
        app = os.environ.get("INFERENCEKIT_MEDIAPIPE_APK", ROOT / "build/mediapipe-capture.apk")
        archive.write(app, "payload/application/mediapipe-capture.apk")
    if any(model.startswith("kokoro-mediapipe-") and model.endswith("-123") for model in models):
        archive.write(ROOT / "build/mediapipe-capture-123.apk", "payload/application/mediapipe-capture-123.apk")
    if "kokoro-audiocpp-fp32" in models:
        path = os.environ.get("INFERENCEKIT_KOKORO_CPP_MODEL")
        if not path or not Path(path).is_file():
            parser.error("set INFERENCEKIT_KOKORO_CPP_MODEL to the converted official FP32 GGUF")
        archive.write(Path(path), "payload/models/kokoro-audiocpp/kokoro-82m-f32.gguf")
        for filename in ("libaudiocpp.so", "libespeak-ng.so"):
            archive.write(ROOT / f"build/stripped/kokoro/{filename}", f"payload/lib/kokoro/{filename}")
    if "kokoro-adreno-fp16" in models:
        kit = ROOT / "build/kokoro-adreno-kit"
        if not (kit / "Kokoro_82M_inference_fp16").is_file():
            parser.error("build the Adreno Kokoro kit first")
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                archive.write(path, "payload/kokoro-adreno/" + path.relative_to(kit).as_posix())
    if "kokoro-litert-omni-gpu" in models:
        kit = Path(os.environ.get("INFERENCEKIT_OMNI_KOKORO_KIT", ROOT / "build/litert-lm-direct/android-kit"))
        if not (kit / "capture-omni-kokoro").is_file():
            parser.error("build the pinned LiteRT-LM Omni Kokoro kit first")
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                archive.write(path, "payload/omni-kokoro/" + path.relative_to(kit).as_posix())
    if any(model.startswith("kokoro-soniqo-") for model in models):
        kit = Path(os.environ.get("INFERENCEKIT_SONIQO_KOKORO_KIT", ROOT / "build/soniqo-kokoro/android-kit"))
        if not (kit / "capture-soniqo-kokoro").is_file():
            parser.error("build the pinned Soniqo Kokoro kit first")
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                archive.write(path, "payload/soniqo-kokoro/" + path.relative_to(kit).as_posix())
    snapdragon = any(model.startswith("vieneu-cpp-snap-") for model in models)
    if "vieneu-cpp" in models or snapdragon:
        kit_path = os.environ.get("INFERENCEKIT_VIENEU_CPP_KIT")
        if not kit_path:
            parser.error("set INFERENCEKIT_VIENEU_CPP_KIT to the alternate Android kit")
        kit = Path(kit_path)
        if not (kit / "capture-vieneu-cpp").is_file() or not (kit / "model/config.json").is_file():
            parser.error("alternate Android kit is incomplete")
        capture = os.environ.get("INFERENCEKIT_VIENEU_CPP_CAPTURE")
        if scenario.get("quality_only") and not capture:
            parser.error("set INFERENCEKIT_VIENEU_CPP_CAPTURE to the complete-PCM capture build")
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                relative = path.relative_to(kit).as_posix()
                source = Path(capture) if capture and relative == "capture-vieneu-cpp" else path
                archive.write(source, "payload/vieneu-cpp/" + relative)
        for runtime in sorted(runtimes & {"vieneu-cpp-batched", "vieneu-cpp-shared"}):
            override = runtime.removeprefix("vieneu-cpp-")
            archive.write(ROOT / f"build/stripped/{runtime}/libvieneu-tts.so",
                          f"payload/vieneu-cpp/lib/{override}/libvieneu-tts.so")
    if snapdragon:
        kit_variable = ("INFERENCEKIT_VIENEU_SNAPDRAGON_MIXED_KIT" if "vieneu-cpp-snap-mixed" in models
                        else "INFERENCEKIT_VIENEU_SNAPDRAGON_KIT")
        kit_path = os.environ.get(kit_variable)
        if not kit_path:
            parser.error(f"set {kit_variable} to the Snapdragon Android kit")
        kit = Path(kit_path)
        binary = kit / "capture-vieneu-cpp"
        if not binary.is_file() or not (kit / "lib").is_dir():
            parser.error("Snapdragon Android kit is incomplete")
        archive.write(binary, "payload/vieneu-snapdragon/capture-vieneu-cpp")
        for path in sorted((kit / "lib").glob("*.so")):
            archive.write(path, f"payload/vieneu-snapdragon/lib/{path.name}")
    for family in ("kokoro", "vieneu"):
        name = f"{family}.tsv"
        archive.write(fixture_directory / name, f"payload/plans/{name}")
    for path in sorted(plans.iterdir()):
        if path.is_file() and (path.suffix == ".conf" or path.name.startswith(phase + "-") or
                               path.name in registered_plans):
            archive.write(path, f"payload/plans/{path.name}")

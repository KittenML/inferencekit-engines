"""Prepare the native runtime and selected model assets for a directly attached phone."""
import argparse
from pathlib import Path
import shutil
import tempfile

from prepare_payload import ARTIFACTS, digest_of, fetch

ROOT = Path(__file__).resolve().parent.parent
LIBS = ("libsherpa-onnx-c-api.so", "libonnxruntime.so")
SHERPA = ROOT / "dependencies/jniLibs/arm64-v8a"
PAYLOAD = ROOT / "build/local-payload"


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def runtime():
    if all((SHERPA / library).is_file() for library in LIBS):
        return
    (PAYLOAD / "lib").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as store:
        asset = next(item for item in ARTIFACTS if item[0] == "sherpa-runtime")
        fetch(PAYLOAD, Path(store), asset)
    for library in LIBS:
        copy(PAYLOAD / "lib" / library, SHERPA / library)


def model_assets(model):
    vieneu_voice = "vieneu/sea_g2p.bin", "vieneu/ref_codes.txt", "vieneu/speaker.emb.txt"
    wanted = {
        "kokoro": ("kokoro-fp32",),
        "vieneu": ("vieneu/vieneu-v3-turbo-bf16.gguf", *vieneu_voice),
        "vieneu-q8": ("vieneu/vieneu-v3-turbo-q8_0.gguf", *vieneu_voice),
        "vieneu-voice": ("vieneu/ref_codes.txt", "vieneu/speaker.emb.txt"),
    }[model]
    (PAYLOAD / "models/vieneu").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as store:
        for asset in ARTIFACTS:
            name = asset[0]
            if name in wanted:
                existing = PAYLOAD / "models" / name
                if name == "kokoro-fp32":
                    existing = PAYLOAD / "models/kokoro-multi-lang-v1_0/model.onnx"
                expected = (
                    "b40f62b166ac8164b0627ef48a0b358eda0985e272fb03ef5252e7206305da11"
                    if name == "kokoro-fp32" else asset[2]
                )
                if not existing.is_file() or digest_of(existing) != expected:
                    fetch(PAYLOAD, Path(store), asset)
    if model in {"vieneu", "vieneu-q8"}:
        copy(ROOT / "build/stripped/libsea_g2p_rs.so", PAYLOAD / "models/vieneu/libsea_g2p_rs.so")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("runtime", "kokoro", "vieneu", "vieneu-q8", "vieneu-voice", "both"))
    args = parser.parse_args()
    if args.model == "vieneu-voice":
        model_assets(args.model)
        return
    runtime()
    if args.model == "runtime":
        return
    families = ("kokoro", "vieneu") if args.model == "both" else (args.model,)
    for name in families:
        model_assets(name)
    for name in LIBS:
        copy(SHERPA / name, PAYLOAD / "lib" / name)
    copy(ROOT / "build/stripped/libaudiocpp.so", PAYLOAD / "lib/libaudiocpp.so")
    copy(ROOT / "build/stripped/capture", PAYLOAD / "bin/ik-speech")
    for fixture in (ROOT / "source/plans").glob("*.tsv"):
        copy(fixture, PAYLOAD / "plans" / fixture.name)


if __name__ == "__main__":
    main()

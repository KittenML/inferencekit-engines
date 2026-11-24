"""Package pinned Kokoro deployments and the existing Pico native INT8 control."""
import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED


def verify(path, want):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != want:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")


def append_tree(archive, source, target):
    for child in sorted(source.rglob("*")):
        if child.is_file():
            archive.write(child, f"{target}/{child.relative_to(source).as_posix()}")


def main():
    kit = Path(sys.argv[1])
    assets = Path("build/release-assets")
    sherpa = assets / "kokoro-int8-multi-lang-v1_0"
    media = assets / "kokoro-GPU.litertlm"
    verify(media, "1ef178b127ec48c737daf17eabb4cea5f6b3ffbc510aced270fe5c267bbeac52")
    verify(assets / "kokoro-int8-v1_0.tar.bz2",
           "72c347230107ef8a31098ec4d468a605244beef5f1ed00db97efa13518800653")
    verify(kit / "weights/model.fp16.bin",
           "94725a9a8f70300e5fc600523bd664960f0fd0126b8ad9349ac28bdb04b77f7c")
    with ZipFile(sys.argv[2], "w", ZIP_DEFLATED, compresslevel=1) as archive:
        for name in ("model.int8.onnx", "tokens.txt", "voices.bin", "lexicon-us-en.txt"):
            archive.write(sherpa / name, f"sherpa/{name}")
        append_tree(archive, sherpa / "espeak-ng-data", "sherpa/espeak-ng-data")
        archive.write(media, "mediapipe/kokoro-GPU.litertlm")
        for directory in ("assets", "kernels", "weights"):
            append_tree(archive, kit / directory, f"adreno/{directory}")
        append_tree(archive, sherpa / "espeak-ng-data", "adreno/assets/espeak-ng-data")
        pico = Path("build/matrix-models/pico")
        for name in ("kitten_tts_2_pico_en_int8.bin", "kitten_tts_2_pico_en_int8_arch.json",
                     "vocab.json", "eleanor.bin"):
            archive.write(pico / name, f"pico/{name}")
        append_tree(archive, pico / "g2p", "pico/g2p")


if __name__ == "__main__":
    main()

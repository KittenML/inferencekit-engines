"""Pack the official English Kokoro checkpoint without weight quantization."""
import argparse
import hashlib
from pathlib import Path
from types import ModuleType


ASSETS = {
    "config.json": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "kokoro-v1_0.pth": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "voices/af_heart.pt": "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("official_checkpoint", type=Path)
    parser.add_argument("audio_cpp_checkout", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    source = arguments.official_checkpoint
    for name, expected in ASSETS.items():
        with (source / name).open("rb") as checkpoint:
            observed = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        if observed != expected:
            raise ValueError(f"checkpoint asset hash mismatch for {name}")
    if {path.name for path in (source / "voices").glob("*.pt")} != {"af_heart.pt"}:
        raise ValueError("copy only the requested af_heart voice into the input directory")

    checkout = arguments.audio_cpp_checkout
    converter = checkout / "tools/models/prepare_kokoro_gguf.py"
    data = converter.read_bytes()
    if hashlib.sha256(data).hexdigest() != "024984e46ef7bf9acd9a193795925752a0d098280b05551df9ad79dd7a148c65":
        raise ValueError("use the pinned upstream audio.cpp converter")
    text = data.decode("utf-8")
    storage_rule = "if array.ndim >= 2 and all(d > 1 for d in array.shape):"
    languages = "['en-us', 'en-gb', 'es', 'fr-fr', 'hi', 'it', 'ja', 'pt-br', 'zh']"
    if text.count(storage_rule) != 1 or text.count(languages) != 1:
        raise ValueError("unexpected upstream audio.cpp converter")
    text = text.replace(storage_rule, "if precision != 'f32' and " + storage_rule[3:])
    text = text.replace(languages, "['en-us']")
    upstream = ModuleType("kokoro_gguf_converter")
    upstream.__file__ = str(converter)
    exec(compile(text, str(converter), "exec"), upstream.__dict__)
    # English uses the separate eSpeak runtime and data during synthesis.
    upstream.load_g2p_tables = dict
    upstream.convert(source, arguments.output, "f32", model_spec=checkout / "model_specs/kokoro_tts.json")


if __name__ == "__main__":
    main()

"""Prepare the publisher's fixed preset conditioning with unmodified checkpoint data."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("preset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    speaker = np.loadtxt(args.preset / "speaker.emb.txt", dtype=np.float32, delimiter=",").reshape(-1)
    ref_codes = np.loadtxt(args.preset / "ref_codes.txt", dtype=np.int64, ndmin=2)
    if speaker.size != config["speaker_embedding_dim"] or not np.isfinite(speaker).all() or not speaker.any():
        raise ValueError("invalid preset speaker embedding")
    if ref_codes.shape[1] != config["n_vq"] or np.any((ref_codes < 0) | (ref_codes >= config["audio_vocab_size"])):
        raise ValueError("invalid preset reference codes")
    with np.load(args.model / "vieneu_v3_heads.npz") as heads:
        anchor = speaker @ heads["xvec_w"].astype(np.float32).T + heads["xvec_b"].astype(np.float32)
        anchor = (anchor - anchor.mean()) / np.sqrt(anchor.var() + float(heads["xvec_ln_eps"]))
        anchor = (anchor * heads["xvec_ln_w"] + heads["xvec_ln_b"]).astype(np.float32)
    if anchor.shape != (config["hidden_size"],) or not np.isfinite(anchor).all():
        raise ValueError("invalid projected speaker anchor")
    preset = {"speaker_anchor": anchor.tolist(), "codes": ref_codes.tolist()}
    args.output.write_text(json.dumps({"default_voice": args.preset.name, "presets": {args.preset.name: preset}}) + "\n")


if __name__ == "__main__":
    main()

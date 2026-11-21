#!/usr/bin/env python3
"""Package the publisher's Adam preset in the pinned native runtime's format."""
import json
from pathlib import Path
import sys

voice, output = map(Path, sys.argv[1:])
codes = [[int(x) for x in row.split()] for row in (voice / "ref_codes.txt").read_text().splitlines()]
embedding = [float(x) for x in (voice / "speaker.emb.txt").read_text().replace(",", " ").split()]
if len(embedding) != 192 or not codes or any(len(row) != 16 for row in codes):
    raise ValueError("Checkpoint preset does not match the public Turbo configuration")
output.write_text(json.dumps({"default_voice": "adam", "presets": {"adam": {"speaker_emb": embedding, "codes": codes}}}, separators=(",", ":")) + "\n")

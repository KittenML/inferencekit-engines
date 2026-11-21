#!/usr/bin/env python3
"""Add the publisher's matched benchmark voice to the alternate model pack."""
import json
from pathlib import Path
import sys

presets_file, codes_file, embedding_file, destination = map(Path, sys.argv[1:])
voices = json.loads(presets_file.read_text())
codes = [list(map(int, line.split())) for line in codes_file.read_text().splitlines()]
embedding = list(map(float, embedding_file.read_text().strip().split(',')))
if not codes or any(len(row) != 16 for row in codes) or len(embedding) != 192:
    raise ValueError("Expected nonempty 16-code rows and a 192-dimensional embedding")
voices["presets"]["minh_quan_pro"] = {"codes": codes, "speaker_emb": embedding}
destination.write_text(json.dumps(voices, ensure_ascii=False) + "\n")

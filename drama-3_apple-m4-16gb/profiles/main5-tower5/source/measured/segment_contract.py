# SPDX-License-Identifier: Apache-2.0
"""Prove ordered text/role coverage while retaining exact native chunk hashes."""
import hashlib
import json
import re
from pathlib import Path
import frozen_chunking

SPEAKER = re.compile(r"<\|speaker:(\d+)\|>")
NATIVE_CHUNKING_SHA256 = "8b31655839a3c7112040657911b16cfe6fb7978fd98e35975c3d1cfa6196fbd7"
WHITESPACE_TRANSLATION = {ord(char): None for char in frozen_chunking.WHITE_SPACE}


def role_units(text):
    parts = SPEAKER.split(text)
    pairs = [(0, parts[0])] + [(int(role), body) for role, body in zip(parts[1::2], parts[2::2])]
    return [(role, body.translate(WHITESPACE_TRANSLATION)) for role, body in pairs if body.translate(WHITESPACE_TRANSLATION)]


def merge(units):
    rows = []
    for role, body in units:
        if rows and rows[-1][0] == role:
            rows[-1][1] += body
        else:
            rows.append([role, body])
    return rows


def partition_proof(text, chunks):
    if hashlib.sha256(Path(frozen_chunking.__file__).read_bytes()).hexdigest() != NATIVE_CHUNKING_SHA256:
        raise ValueError("Pinned pure native chunking source changed")
    if chunks != frozen_chunking.split_text(text):
        raise ValueError("Segment bytes differ from exact pinned native split_text output")
    if not chunks or any(not isinstance(part, str) or not part.strip() for part in chunks):
        raise ValueError("Empty native text partition")
    expected = merge(role_units(text))
    observed = merge([unit for chunk in chunks for unit in role_units(chunk)])
    if expected != observed:
        raise ValueError("Native text partition omits, duplicates, reorders or changes role/content")
    return {
        "schema": "inferencekit-ordered-native-text-partition-v1",
        "native_chunking_source_sha256": NATIVE_CHUNKING_SHA256,
        "exact_native_segment_bytes_match": True,
        "full_input_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "all_ordered_role_content_preserved": True,
        "projection": "Exact native split_text(full_input) byte partition first; upstream flatten_lines and edge-trim behavior retained. Additional role/content check removes only upstream WHITE_SPACE and merges adjacent same-role units. Word boundaries cannot be changed independently of the pinned partition.",
        "role_content_projection_sha256": hashlib.sha256(json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "segments": [{"segment_index": i, "text_utf8_sha256": hashlib.sha256(part.encode()).hexdigest(), "text_utf8_bytes": len(part.encode())} for i, part in enumerate(chunks)],
    }

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run only on an approved host after plugin readiness; downloads pinned weights."""
import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=("upstream-3bit", "upstream-4bit", "original-fp8"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--acknowledge-approved-host", action="store_true", required=True)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    lock = json.loads(Path(__file__).with_name("sources.lock.json").read_text())
    spec = lock["original_model"] if args.track == "original-fp8" else lock["models"][args.track]
    root = Path(snapshot_download(spec["id"], revision=spec["revision"], local_dir=args.output))
    files = {}
    for item in spec["files"]:
        file = root / item["path"]
        if not file.is_file() or file.stat().st_size != item["bytes"]:
            raise RuntimeError("downloaded file size differs from pinned metadata")
        digest = file_hash(file)
        if item["lfs_sha256"]:
            valid = digest == item["lfs_sha256"]
        else:
            valid = file_hash(file, "sha1", git_blob=True) == item["git_blob_sha1"]
        if not valid:
            raise RuntimeError("downloaded file hash differs from pinned metadata")
        files[item["path"]] = {"sha256": digest, "bytes": item["bytes"]}
    (root / "origin.json").write_text(json.dumps({"model": spec["id"], "revision": spec["revision"], "files": files}, indent=2) + "\n")


if __name__ == "__main__":
    main()

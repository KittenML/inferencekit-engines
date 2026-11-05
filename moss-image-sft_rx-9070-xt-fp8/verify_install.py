#!/usr/bin/env python3
"""Verify the installed model and the generated configuration before startup."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

MODEL_REVISION = 'f972bd4b4b9e3852133cc5da954d87bc32cd8fb9'
SOURCE_COMMIT = '7cbc776c16a39f73978ae5ce400729d94e0fda35'
MANIFEST_SHA256 = '70803d3a3acd8580a98094ddebf49d6f5c8f4aee2e05d08e2f0b88a9bea883c0'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(runtime, manifest_file, config_file, expected_manifest_sha=MANIFEST_SHA256):
    runtime = runtime.resolve(strict=True)
    model = runtime / 'model'
    if model.is_symlink() or not model.is_dir():
        raise ValueError('model directory missing or is a symlink')
    manifest_digest = sha(manifest_file)
    if manifest_digest != expected_manifest_sha:
        raise ValueError('model manifest differs from package pin')
    receipt_file = runtime / 'install-complete.json'
    if receipt_file.is_symlink():
        raise ValueError('install receipt is a symlink')
    receipt = json.loads(receipt_file.read_text())
    if (receipt.get('source_commit') != SOURCE_COMMIT or
        receipt.get('model_revision') != MODEL_REVISION or
        receipt.get('model_manifest_sha256') != manifest_digest):
        raise ValueError('install receipt does not bind this model manifest')
    config = json.loads(config_file.read_text())
    if (config.get('model_revision') != MODEL_REVISION or
        config.get('source_commit') != SOURCE_COMMIT or
        config.get('native_source_root') != str((runtime / 'vllm-omni').resolve()) or
        config.get('engine_args', {}).get('model') != str(model.resolve()) or
        config.get('expected_worker_config', {}).get('model') != str(model.resolve())):
        raise ValueError('configuration is not bound to the installed model and source')
    entries = json.loads(manifest_file.read_text())
    expected = set()
    for item in entries:
        rel = PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or str(rel) != item['path']:
            raise ValueError('invalid model manifest path')
        if str(rel) in expected:
            raise ValueError('duplicate model artifact')
        expected.add(str(rel))
        target = model.joinpath(*rel.parts)
        current = model
        for component in rel.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError('model artifact path contains a symlink')
        if not target.is_file() or target.stat().st_size != item['size'] or sha(target) != item['sha256']:
            raise ValueError('model artifact missing or changed: ' + str(rel))
    actual = set()
    for base, dirs, files in os.walk(model, followlinks=False):
        for name in dirs + files:
            if (Path(base) / name).is_symlink():
                raise ValueError('model directory contains a symlink')
        for name in files:
            actual.add((Path(base) / name).relative_to(model).as_posix())
    if actual != expected:
        raise ValueError('installed model contains missing or unlisted files')
    return {'status': 'PASS', 'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model_revision': MODEL_REVISION, 'model_manifest_sha256': manifest_digest,
            'install_receipt_sha256': sha(receipt_file), 'configuration_sha256': sha(config_file),
            'model_path': str(model), 'verified_files': len(expected),
            'qualification': 'snapshot verification; files must remain unchanged during the run'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True, type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    result = verify(package / 'runtime', package / 'model-artifacts.json', args.configuration)
    output = args.configuration.parent / 'model-verification.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('verified pinned model and generated configuration')

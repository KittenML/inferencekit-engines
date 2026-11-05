# SPDX-License-Identifier: Apache-2.0
"""Read-only, exact-byte safetensors views through public MLX CPU adoption.

Requires MLX 0.32.4 on unified-memory Metal. Every tensor must be adopted without
a copy; there is no silent dtype conversion or native-loader fallback. The
checkpoint files must remain unchanged for the lifetime of the model.
"""
import json
import hashlib
import math
import mmap
import os
import struct
import tempfile
from pathlib import Path

_native_loader = None


def aligned_mirror(path, cache_dir, expected_sha256):
    """Pad only the header; prove that every tensor payload byte is unchanged."""
    path, cache_dir = Path(path), Path(cache_dir)
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(100 * 2**20, path.stat().st_size - 8):
            raise ValueError('Invalid safetensors header length')
        header = source.read(length)
        if length % 8 == 0:
            return path, {'header_repacked': False, 'source_sha256': expected_sha256}
        padded = header + b' ' * (-length % 8)
        new_prefix = struct.pack('<Q', len(padded))
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / (path.stem + '-' + expected_sha256 + '.safetensors')
        creating = not destination.exists()
        descriptor, temporary = tempfile.mkstemp(prefix='.align-', dir=cache_dir) if creating else (None, None)
        source_hash = hashlib.sha256(prefix + header)
        expected_target_hash = hashlib.sha256(new_prefix + padded)
        payload_hash = hashlib.sha256()
        try:
            output = os.fdopen(descriptor, 'wb') if creating else None
            try:
                if output:
                    output.write(new_prefix + padded)
                while chunk := source.read(8 * 2**20):
                    source_hash.update(chunk)
                    expected_target_hash.update(chunk)
                    payload_hash.update(chunk)
                    if output:
                        output.write(chunk)
            finally:
                if output:
                    output.close()
            if source_hash.hexdigest() != expected_sha256:
                raise RuntimeError('Original checkpoint changed before aligned mapping')
            candidate = Path(temporary) if creating else destination
            actual_target_hash, actual_payload_hash = hashlib.sha256(), hashlib.sha256()
            with candidate.open('rb') as check:
                actual_target_hash.update(check.read(8 + len(padded)))
                while chunk := check.read(8 * 2**20):
                    actual_target_hash.update(chunk)
                    actual_payload_hash.update(chunk)
            if (actual_target_hash.digest() != expected_target_hash.digest()
                    or actual_payload_hash.digest() != payload_hash.digest()):
                raise RuntimeError('Aligned cache differs from exact original payload')
            if creating:
                os.replace(temporary, destination)
                temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    return destination, {'header_repacked': True, 'source_sha256': expected_sha256,
                         'aligned_sha256': expected_target_hash.hexdigest(),
                         'payload_sha256_both_files': payload_hash.hexdigest(),
                         'header_padding_added_bytes': len(padded) - length,
                         'cache_created': creating}


def load_file(path):
    import mlx.core as mx
    import numpy as np
    dtypes = {'F32': (np.dtype('<f4'), mx.float32),
              'F16': (np.dtype('<f2'), mx.float16),
              'BF16': (np.dtype('<u2'), mx.bfloat16),
              'U32': (np.dtype('<u4'), mx.uint32),
              'I32': (np.dtype('<i4'), mx.int32),
              'I64': (np.dtype('<i8'), mx.int64),
              'U8': (np.dtype('u1'), mx.uint8),
              'BOOL': (np.dtype('?'), mx.bool_)}
    path = Path(path)
    with path.open('rb') as stream:
        size = path.stat().st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors prefix')
        header_bytes = struct.unpack('<Q', prefix)[0]
        if not 2 <= header_bytes <= min(100 * 2**20, size - 8):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(header_bytes))
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    if not isinstance(header, dict):
        raise ValueError('Invalid safetensors header object')
    base = 8 + header_bytes
    result = {}
    audit = {'file': path.name, 'tensor_count': 0, 'mapped_tensor_bytes': 0,
             'readonly': True, 'zero_copy_pointer_checks': 0, 'dtypes': {}}
    spans = []
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(spec, dict) or spec.get('dtype') not in dtypes:
            raise ValueError('Unsupported safetensors tensor dtype')
        shape = spec.get('shape')
        offsets = spec.get('data_offsets')
        if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)):
            raise ValueError('Invalid safetensors shape or offsets')
        start, end = offsets
        dtype, target = dtypes[spec['dtype']]
        if not 0 <= start <= end <= size - base or end - start != math.prod(shape) * dtype.itemsize:
            raise ValueError('Safetensors extent does not match file or shape')
        if (base + start) % dtype.itemsize:
            raise ValueError('Tensor pointer is not dtype-aligned; no copy fallback')
        spans.append((start, end))
        host = np.ndarray(shape=shape, dtype=dtype, buffer=mapped, offset=base + start)
        array = mx.asarray(host, copy=False)
        if spec['dtype'] == 'BF16':
            array = array.view(mx.bfloat16)
        if array.dtype != target or array.shape != tuple(shape):
            raise RuntimeError('Mapped array shape or dtype changed')
        pointer_view = array.view(mx.uint16) if spec['dtype'] == 'BF16' else array
        if np.asarray(pointer_view).ctypes.data != host.ctypes.data:
            raise RuntimeError('MLX did not retain the mapped CPU pointer')
        result[name] = array
        audit['tensor_count'] += 1
        audit['zero_copy_pointer_checks'] += 1
        audit['mapped_tensor_bytes'] += end - start
        audit['dtypes'][spec['dtype']] = audit['dtypes'].get(spec['dtype'], 0) + end - start
    previous = 0
    for start, end in sorted(spans):
        if start != previous:
            raise ValueError('Safetensors contains overlap or unaccounted tensor bytes')
        previous = end
    if previous != size - base:
        raise ValueError('Safetensors trailing bytes are not accounted for')
    return result, audit


def install(mode, cache_dir=None, verified_files=None):
    global _native_loader
    from mlx_audio import utils
    if _native_loader is None:
        _native_loader = utils.load_weights
    if mode == 'native':
        utils.load_weights = _native_loader
        return {'mode': mode, 'files': []}
    if mode == 'native-main6-tower5-v1':
        import mlx.core as mx
        from main_dense4_overlay import install as install_main4
        if hashlib.sha256(Path(__file__).with_name('main_dense4_overlay.py').read_bytes()).hexdigest() != '334cfddc88fabf94d55fac9143c6405639c0b4e5324a5e81c9d06fcafaf50519':
            raise ValueError('Main4/Tower6 loader helper differs from pinned source')
        return install_main4(mx, utils, _native_loader)
    if mode != 'mmap-readonly-v1':
        raise ValueError('Unsupported weight loading mode')
    if cache_dir is None or verified_files is None:
        raise ValueError('Mapped loader needs a separate cache and verified source hashes')
    audit = {'mode': mode, 'files': [], 'codec_loader': 'unchanged',
             'warning': 'File-backed mapped payload is not guaranteed resident; process and OS memory must be observed.'}

    def load_weights(model_path):
        files = sorted(Path(model_path).glob('*.safetensors'))
        if not files:
            raise ValueError('Mapped loader requires local safetensors')
        weights = {}
        for path in files:
            # Drama3.sanitize removes these before loading the backbone; its
            # separate codec loader and fold arithmetic remain unchanged.
            if path.name == 'codec.safetensors':
                continue
            mapped_path, transform = aligned_mirror(path, cache_dir, verified_files[path.name]['sha256'])
            arrays, item = load_file(mapped_path)
            item['file'] = path.name
            item['alignment_transform'] = transform
            if set(arrays) & set(weights):
                raise ValueError('Duplicate checkpoint tensor name')
            weights.update(arrays)
            audit['files'].append(item)
        return weights

    utils.load_weights = load_weights
    return audit

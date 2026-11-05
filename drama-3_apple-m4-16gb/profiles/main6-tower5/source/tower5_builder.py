"""Portable reproduction of the measured Tower5 overlay; no model forwards.

The caller owns process, memory-pressure and deadline supervision. This module
accepts explicit input roots and a live guard callback; it has no ambient model
search or standalone unguarded entry point. Original derivation metadata remains
byte-identical. A separate reproduction receipt identifies this builder.
"""
from pathlib import Path, PurePosixPath
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
BASE_PREFIX = PurePosixPath("package-default-validation-v14-v1/models/e2-mse-e3last8-d6-t6-bf16heads-emb4-g128")
ORIGINAL_PREFIX = PurePosixPath("package-default-validation-v14-v1/models/original-fp8")
BASE_REVISION = "742d2f153e5775357faee2692480c7afdfe0b6673837e2183f2632808e7cebc3"
TARGET_REVISION = "93432d8a6c97faba88309880c3fd2d1b8356bde679dc20f9e0daa57282eb904c"
PLAN_SHA256 = "b20ca2ccfc93c8bd470109f98d0a3654a69262da2160833ff205f11bcbceff2d"
PARENT_LOCK_SHA256 = "d6cf466b86eae2b17c857ef2a42967e0eca44c565ae4c20cf8b9f8c8af41a7e0"
TARGET_ORIGIN_SHA256 = "a47ca84ca752d73d07b311ba575e0b8a2817a8751cc50238198f0c7f1cda46ce"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, check):
    before = file_identity(path)
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
            check()
    require(file_identity(path) == before, "Input changed during hashing")
    return value.hexdigest()


def file_identity(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def regular_member(root, name):
    require(isinstance(name, str) and name not in ("", ".", "..") and
            PurePosixPath(name).name == name and "\\" not in name,
            "Expected a flat model member name")
    path = root / name
    require(not path.is_symlink() and path.is_file() and path.resolve().parent == root,
            "Model member must be a regular file inside its declared root")
    return path


def relocated(root, historical, prefix):
    require(isinstance(historical, str) and "\\" not in historical,
            "Invalid historical tensor path")
    path = PurePosixPath(historical)
    require(str(path) == historical and ".." not in path.parts and path.is_relative_to(prefix),
            "Tensor path is outside its explicit registered prefix")
    relative = path.relative_to(prefix)
    require(len(relative.parts) == 1, "Nested model member refused")
    return regular_member(root, relative.name)


def npy_header(dtype, shape):
    header = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)}).encode("latin1")
    header += b" " * (-(10 + len(header) + 1) % 64) + b"\n"
    require(len(header) < 65536, "NPY v1 header too large")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def read_tensor(root, prefix, spec, check):
    path = relocated(root, spec["file"], prefix)
    before = file_identity(path)
    require(path.stat().st_size == spec["file_bytes"], "Tensor shard size changed")
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "Truncated safetensors header length")
        length = struct.unpack("<Q", length_bytes)[0]
        require(length == spec["header_bytes"] and length <= 20 * 2**20,
                "Unregistered safetensors header length")
        header = stream.read(length)
        require(hashlib.sha256(header).hexdigest() == spec["header_sha256"], "Tensor header changed")
        entry = json.loads(header)[spec["key"]]
        require(all(entry[key] == spec[key] for key in ("shape", "dtype", "data_offsets")),
                "Tensor declaration changed")
        start, stop = entry["data_offsets"]
        require(0 <= start < stop and stop - start <= 64 * 2**20 and
                8 + length + stop <= path.stat().st_size, "Tensor bounds invalid")
        stream.seek(8 + length + start)
        raw = stream.read(stop - start)
        require(len(raw) == stop - start, "Truncated tensor")
    require(file_identity(path) == before, "Tensor source changed during read")
    check()
    return raw


def audit_npz(path, tensors, check):
    """Independently reread every stored NPY header/payload, bounded to1MiB."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(names == [name + ".npy" for name in tensors] and len(names) == len(set(names)),
                "Final NPZ member order/set/duplicates differ")
        for name, spec in tensors.items():
            raw_hash = hashlib.sha256()
            npy_hash = hashlib.sha256()
            expected_header = npy_header(spec["stored_dtype"], spec["shape"])
            count = 0
            with archive.open(name + ".npy") as stream:
                header = stream.read(len(expected_header))
                require(header == expected_header, "Final NPY header differs")
                npy_hash.update(header); count += len(header)
                while block := stream.read(1024 * 1024):
                    raw_hash.update(block); npy_hash.update(block); count += len(block)
                    check()
            require(count == spec["npy_bytes"] and npy_hash.hexdigest() == spec["npy_sha256"] and
                    raw_hash.hexdigest() == spec["raw_le_sha256"], "Final NPY stored bytes differ")
            check()
    return {"members": len(tensors), "all_raw_and_NPY_hashes_reread": True}


def share_file(source, destination, expected, check):
    """Create a new link/copy, never open an existing shared inode for writing."""
    require(not destination.exists() and not destination.is_symlink(), "Output member already exists")
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        with source.open("rb") as src, destination.open("xb") as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)
                check()
            dst.flush()
            os.fsync(dst.fileno())
        mode = "copy"
    require(destination.stat().st_size == expected["bytes"] and
            digest(destination, check) == expected["sha256"], "Shared base member differs")
    return mode


def build(base_model, original_model, output_model, *, mx, np, check, report=None):
    """Reproduce exact registered bytes under the caller's active resource guard."""
    require(callable(check), "A live resource guard callback is required")
    require(report is None or callable(report), "Invalid progress callback")
    check()
    base = Path(base_model).resolve(strict=True)
    original = Path(original_model).resolve(strict=True)
    output = Path(output_model).absolute()
    require(".." not in Path(output_model).parts, "Output parent traversal refused")
    require(base.is_dir() and original.is_dir() and base != original, "Separate input roots required")
    require(not output.exists() and not output.is_symlink(), "Fresh model output required")
    require(not output.resolve().is_relative_to(base) and not output.resolve().is_relative_to(original),
            "Output cannot be inside either protected input root")
    require(digest(HERE / "TENSOR_PLAN_V52_V1.json", check) == PLAN_SHA256, "Frozen tensor plan changed")
    plan = json.loads((HERE / "TENSOR_PLAN_V52_V1.json").read_text())
    origin = json.loads(regular_member(base, "origin.json").read_text())
    require(origin["revision"] == BASE_REVISION, "Wrong parent model revision")
    require(json.loads(regular_member(original, "origin.json").read_text()) == plan["original_file_manifest"],
            "Wrong original checkpoint manifest")
    require(digest(HERE / "model-metadata/origin.json", check) == TARGET_ORIGIN_SHA256,
            "Measured target origin changed")
    expected = json.loads((HERE / "model-metadata/origin.json").read_text())
    require(expected["revision"] == TARGET_REVISION, "Wrong expected Tower5 revision")
    metadata = HERE / "model-metadata"
    for name in ("config.json", "MAIN6_TOWER5_POLICY_V52.json", "TOWER5_OVERLAY_MANIFEST_V1.json",
                 "TOWER5_OVERLAY_NOTICE_V1.json"):
        require(digest(metadata / name, check) == expected["files"][name]["sha256"],
                "Registered metadata changed: " + name)
    overlay_spec = json.loads((metadata / "TOWER5_OVERLAY_MANIFEST_V1.json").read_text())
    targets = plan["targets"]
    require(len(targets) == 78 and len({row["module"] for row in targets}) == 78 and
            all(row["module"].startswith("tower.") and row["bits"] == 5 and row["group_size"] == 64
                for row in targets), "Exact Tower78 selection required")
    require(list(overlay_spec["tensors"]) == [row["module"] + "." + suffix for row in targets
            for suffix in ("weight", "scales", "biases")], "Registered tensor order changed")
    # Parent values must match the locked measured parent, not a self-declared origin.
    require(digest(HERE / "PARENT_SOURCES_LOCK.json", check) == PARENT_LOCK_SHA256,
            "Measured parent lock changed")
    lock = json.loads((HERE / "PARENT_SOURCES_LOCK.json").read_text())
    parents = [spec for spec in lock["models"].values() if spec.get("revision") == BASE_REVISION]
    require(len(parents) == 1, "Exactly one locked parent is required")
    parent_files = {item["path"]: {"sha256": item["lfs_sha256"], "bytes": item["bytes"]}
                    for item in parents[0]["files"]}
    require(origin["files"] == parent_files and origin["model"] == parents[0]["id"],
            "Parent origin differs from the complete measured lock")
    for name, spec in parent_files.items():
        path = regular_member(base, name)
        require(path.stat().st_size == spec["bytes"] and digest(path, check) == spec["sha256"],
                "Protected parent member differs: " + name)
    sources = {row["source"]["file"]: row["source"] for row in targets}
    source_identities = {}
    for historical, spec in sources.items():
        path = relocated(original, historical, ORIGINAL_PREFIX)
        require(digest(path, check) == spec["file_sha256"],
                "Original source shard differs")
        source_identities[historical] = file_identity(path)
    for row in targets:
        for spec in row["native6_tensors"].values():
            path = relocated(base, spec["file"], BASE_PREFIX)
            require(parent_files[path.name] == {"sha256": spec["file_sha256"], "bytes": spec["file_bytes"]},
                    "Native6 calibration shard is outside the verified parent")
    require(importlib.metadata.version("mlx") == "0.32.4", "Pinned MLX 0.32.4 required")
    require(mx.metal.is_available() and mx.default_device() == mx.gpu, "Native Metal required")
    previous = {}
    restoration = {}
    result = {"status": "INCOMPLETE", "model_forwards": 0, "modules": [], "shared_members": {},
              "input_roots": {"base": str(base), "original": str(original)},
              "source_plan_sha256": PLAN_SHA256, "parent_lock_sha256": PARENT_LOCK_SHA256}
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, spec in parent_files.items():
            if name != "config.json":
                result["shared_members"][name] = share_file(base / name, output / name, spec, check)
        previous["memory"] = mx.set_memory_limit(512 * 2**20)
        previous["cache"] = mx.set_cache_limit(0)
        previous["wired"] = mx.set_wired_limit(0)
        partial = output / "tower5-overlay.partial"
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for row in targets:
                check()
                source_path = relocated(original, row["source"]["file"], ORIGINAL_PREFIX)
                require(file_identity(source_path) == source_identities[row["source"]["file"]],
                        "Verified source shard changed before use")
                raw = read_tensor(original, ORIGINAL_PREFIX, row["source"], check)
                host = np.frombuffer(raw, dtype="<u2").reshape(row["source"]["shape"])
                value = mx.view(mx.array(host), mx.bfloat16)
                q5 = mx.quantize(value, group_size=64, bits=5, mode="affine")
                mx.eval(q5)
                q6 = mx.quantize(value, group_size=64, bits=6, mode="affine")
                mx.eval(q6)
                require(q5[0].dtype == mx.uint32 and all(v.dtype == mx.bfloat16 for v in q5[1:]),
                        "Unexpected native quantization dtype")
                for suffix, array in zip(("weight", "scales", "biases"), q6):
                    native = row["native6_tensors"][suffix]
                    require(list(array.shape) == native["shape"] and
                            array.dtype == (mx.uint32 if suffix == "weight" else mx.bfloat16),
                            "Native6 calibration dtype/shape differs")
                    saved = read_tensor(base, BASE_PREFIX, row["native6_tensors"][suffix], check)
                    actual = np.ascontiguousarray(np.asarray(array if suffix == "weight" else mx.view(array, mx.uint16)),
                                                  dtype="<u4" if suffix == "weight" else "<u2")
                    require(hashlib.sha256(memoryview(actual).cast("B")).digest() == hashlib.sha256(saved).digest(),
                            "Native6 reproduction calibration differs")
                    del saved, actual
                errors = {}
                original_f32 = value.astype(mx.float32)
                for bits, parameters in ((5, q5), (6, q6)):
                    reconstructed = mx.dequantize(*parameters, group_size=64, bits=bits, mode="affine").astype(mx.float32)
                    error = mx.mean(mx.square(original_f32 - reconstructed))
                    mx.eval(error)
                    errors[str(bits)] = float(error.item())
                    require(math.isfinite(errors[str(bits)]), "Nonfinite conversion diagnostic")
                    del reconstructed, error
                    mx.clear_cache()
                del q6, original_f32, parameters
                array_receipts = []
                for suffix, array in zip(("weight", "scales", "biases"), q5):
                    key = row["module"] + "." + suffix
                    spec = overlay_spec["tensors"][key]
                    data = np.ascontiguousarray(np.asarray(array if suffix == "weight" else mx.view(array, mx.uint16)),
                                                dtype=spec["stored_dtype"])
                    require(list(data.shape) == spec["shape"], "Registered output shape differs")
                    header = npy_header(spec["stored_dtype"], data.shape)
                    raw_hash = hashlib.sha256()
                    npy_hash = hashlib.sha256(header)
                    count = len(header)
                    with archive.open(key + ".npy", "w") as stream:
                        stream.write(header)
                        buffer = memoryview(data).cast("B")
                        for offset in range(0, len(buffer), 1024 * 1024):
                            block = buffer[offset:offset + 1024 * 1024]
                            stream.write(block); raw_hash.update(block); npy_hash.update(block)
                            count += len(block)
                            check()
                    require(raw_hash.hexdigest() == spec["raw_le_sha256"] and
                            npy_hash.hexdigest() == spec["npy_sha256"] and count == spec["npy_bytes"],
                            "Reproduced Tower5 array differs from the measured artifact")
                    array_receipts.append({"key": key, "raw_le_sha256": raw_hash.hexdigest(),
                                           "npy_sha256": npy_hash.hexdigest(), "npy_bytes": count})
                result["modules"].append({"module": row["module"], "native6_calibration": "PASS", "MSE": errors,
                                          "original_BF16_payload_sha256": hashlib.sha256(raw).hexdigest(),
                                          "arrays": array_receipts})
                if report is not None:
                    report(result)
                del raw, host, value, q5, array, data, buffer, block
                mx.clear_cache()
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        result["actual_final_NPZ_array_audit"] = audit_npz(partial, overlay_spec["tensors"], check)
        container = {"sha256": digest(partial, check), "bytes": partial.stat().st_size}
        result["container_observation"] = {"actual": container,
            "expected": expected["files"]["tower5-overlay.npz"], "all234_array_hashes_matched": True,
            "status": "EXACT_CONTAINER_MATCH" if container == expected["files"]["tower5-overlay.npz"]
                      else "ARRAYS_MATCH_CONTAINER_MISMATCH_RETAINED"}
        require(container == expected["files"]["tower5-overlay.npz"],
                "Complete NPZ representation differs")
        partial.rename(output / "tower5-overlay.npz")
        for name in ("config.json", "MAIN6_TOWER5_POLICY_V52.json", "TOWER5_OVERLAY_MANIFEST_V1.json",
                     "TOWER5_OVERLAY_NOTICE_V1.json"):
            with (output / name).open("xb") as stream:
                stream.write((metadata / name).read_bytes()); stream.flush(); os.fsync(stream.fileno())
        require({path.name for path in output.iterdir()} == set(expected["files"]), "Unexpected output members")
        for name, spec in expected["files"].items():
            path = regular_member(output.resolve(), name)
            require(path.stat().st_size == spec["bytes"] and digest(path, check) == spec["sha256"],
                    "Final measured artifact mismatch: " + name)
        for historical, spec in sources.items():
            path = relocated(original, historical, ORIGINAL_PREFIX)
            require(file_identity(path) == source_identities[historical] and
                    digest(path, check) == spec["file_sha256"], "Original source changed during build")
        for name, spec in parent_files.items():
            require(digest(regular_member(base, name), check) == spec["sha256"],
                    "Protected parent changed during build")
        result.update(status="ARTIFACT_BYTES_VERIFIED_AWAITING_RESTORATION", model_revision=TARGET_REVISION, quantize_calls=156,
                      coefficient_dequantize_calls=156, unchanged_parent_members=len(parent_files) - 1)
    except BaseException as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary = sys.exc_info()[1]
        for name, callback in (("synchronize", mx.synchronize), ("clear_cache", mx.clear_cache),
                               ("wired", mx.set_wired_limit), ("cache", mx.set_cache_limit), ("memory", mx.set_memory_limit)):
            if name in ("wired", "cache", "memory") and name not in previous:
                continue
            try:
                callback(previous[name]) if name in previous else callback()
                restoration[name] = True
            except BaseException as error:
                restoration[name] = {"error": str(error)}
        result["restoration"] = restoration
        if not all(value is True for value in restoration.values()):
            result["status"] = "FAILED_RESTORATION"
        if report is not None:
            try:
                report(result)
            except BaseException as error:
                if primary is not None:
                    primary.add_note("Progress/restoration receipt failed: " + str(error))
                else:
                    raise
    require(all(value is True for value in restoration.values()), "MLX limit restoration failed")
    check()
    # origin is the final completion marker; partial failures never receive it.
    with (output / "origin.partial").open("xb") as stream:
        stream.write((metadata / "origin.json").read_bytes()); stream.flush(); os.fsync(stream.fileno())
    require(not (output / "origin.json").exists(), "Completion marker already exists")
    (output / "origin.partial").rename(output / "origin.json")
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    result["builder_sha256"] = digest(Path(__file__), check)
    result["original_derivation_metadata_preserved"] = True
    result["physical_duplicate_model_payload_created"] = any(mode == "copy" for mode in result["shared_members"].values())
    result["status"] = "PASS"
    if report is not None:
        report(result)
    return result

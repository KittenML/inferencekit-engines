"""Refuse accidental integrated graphics or a different rescheduled worker."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path


class DeadlineExpired(RuntimeError):
    pass


def check_run_deadline():
    value = os.environ.get("MOSS_RUN_DEADLINE")
    if value:
        deadline = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if (
            deadline.tzinfo is None
            or datetime.datetime.now(datetime.timezone.utc) >= deadline
        ):
            raise DeadlineExpired("the configured run deadline has passed")


def assert_target():
    check_run_deadline()
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "working ROCm with an explicitly selected RX 9070 XT is required"
        )
    # On mixed WSL hosts AMD SMI may count fewer adapters than HIP. Initialize
    # HIP before relying on Torch's management-library discovery fast path.
    torch.cuda.init()
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            "isolate exactly one RX 9070 XT with the documented ROCr visibility selector"
        )
    p = torch.cuda.get_device_properties(0)
    arch = str(getattr(p, "gcnArchName", ""))
    if "RX 9070 XT" not in p.name or "gfx1201" not in arch:
        raise RuntimeError("selected device is not the approved RX 9070 XT gfx1201")
    if not 15 * 2**30 <= p.total_memory <= 17 * 2**30:
        raise RuntimeError(
            "selected device memory differs from the nominal 16 GiB target"
        )
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = None
    uuid = getattr(p, "uuid", None)
    uuid = str(uuid) if isinstance(uuid, (str, bytes)) else None
    session_fingerprint = hashlib.sha256(
        json.dumps({"boot": boot, "uuid": uuid, "name": p.name, "gfx": arch,
                    "memory": p.total_memory}, sort_keys=True).encode()
    ).hexdigest() if boot else None
    return {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_name": p.name,
        "gfx": arch,
        "torch_total_memory_bytes": p.total_memory,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "count": count,
        "HSA_ENABLE_DXG_DETECTION": os.environ.get("HSA_ENABLE_DXG_DETECTION"),
        "host_cpu": host_cpu_facts(torch),
        "hardware_session_fingerprint": session_fingerprint,
        "hardware_session_fingerprint_source": "kernel boot identity plus selected GPU facts; UUID only if exposed",
        "dedicated_residency_verdict": "UNPROVEN without external WSL residency telemetry",
    }


def host_cpu_facts(torch):
    fields = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                fields["model_name"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                fields[key + "_bytes"] = int(value.split()[0]) * 1024
    except OSError:
        fields["proc_identity_status"] = "UNPROVEN"
    fields["process_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    fields["torch_intraop_threads"] = torch.get_num_threads()
    fields["torch_interop_threads"] = torch.get_num_interop_threads()
    return fields


def configure_tunable(torch):
    """Load an immutable offline solver map; no online tuning during requests."""
    selected = os.environ.get("MOSS_TUNING_FILE")
    tunable = torch.cuda.tunable
    if not selected:
        return {"explicit_file": None, "enabled": tunable.is_enabled(),
                "tuning_enabled": tunable.tuning_is_enabled()}
    source = Path(selected).resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = os.environ.get("MOSS_TUNING_SHA256")
    if not expected or digest != expected:
        raise RuntimeError("Offline GEMM solver map must match its explicit SHA256")
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    if not tunable.read_file(str(source)):
        raise RuntimeError("Torch rejected the offline GEMM solver map or validators")
    return {"explicit_file": str(source), "sha256": digest,
            "enabled": tunable.is_enabled(), "tuning_enabled": tunable.tuning_is_enabled(),
            "validators": tunable.get_validators(), "loaded_result_count": len(tunable.get_results())}


def math_settings(torch):
    """Report the actual process controls; these are not GPU kernel traces."""
    cuda = torch.backends.cuda
    matmul = cuda.matmul
    result = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": matmul.allow_tf32,
        "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
        "miopen_enabled": torch.backends.cudnn.enabled,
        "miopen_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        "miopen_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdp_enabled": cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": cuda.math_sdp_enabled(),
        "explicit_miopen_controls": {
            name: os.environ[name]
            for name in (
                "MIOPEN_FIND_MODE", "MIOPEN_FIND_ENFORCE",
                "MIOPEN_DEBUG_CONV_GEMM", "MIOPEN_DEBUG_CONV_FFT",
                "MIOPEN_DEBUG_CONV_WINOGRAD", "MIOPEN_DEBUG_CONV_DIRECT",
                "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM",
                "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX",
            ) if name in os.environ
        },
    }
    try:
        result["preferred_blas_library_raw_torch_enum"] = str(cuda.preferred_blas_library())
    except RuntimeError:
        result["preferred_blas_library_raw_torch_enum"] = None
    try:
        result["preferred_rocm_flash_attention_raw_torch_enum"] = str(cuda.preferred_rocm_fa_library())
    except (AttributeError, RuntimeError):
        result["preferred_rocm_flash_attention_raw_torch_enum"] = None
    from torch._inductor import config

    result["inductor_emulate_precision_casts"] = config.emulate_precision_casts
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(assert_target(), indent=2))

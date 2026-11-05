# SPDX-License-Identifier: Apache-2.0
"""Explicit startup warmup and portable runtime recipe identity; no model imports."""
import base64
import hashlib
import json
import time
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    return digest(Path(path).read_bytes())


def generation_defaults(args, original):
    result = dict(original)
    for key in original:
        result[key] = getattr(args, "default_" + key, original[key])
    return result


def warmup_contract(args):
    fields = ("startup_warmup_fixture", "startup_warmup_fixture_sha256", "startup_warmup_native_proof", "startup_warmup_native_proof_sha256", "startup_warmup_receipt")
    values = [getattr(args, key, None) for key in fields]
    if not any(values):
        return {"enabled": False}
    if not all(values):
        raise ValueError("Startup warmup requires the complete fixture/native-proof/receipt contract")
    raw = Path(values[0]).read_bytes()
    native_raw = Path(values[2]).read_bytes()
    if digest(raw) != values[1] or digest(native_raw) != values[3]:
        raise ValueError("Startup warmup fixture/native proof digest mismatch")
    rows = list(map(json.loads, raw.decode().splitlines()))
    native = list(map(json.loads, native_raw.decode().splitlines()))
    if len(rows) != 1 or len(native) != 1:
        raise ValueError("Exactly one registered warmup input and native prompt proof required")
    fixture, proof = rows[0], native[0]
    if fixture.get("id") != "startup_warmup_public_v13" or fixture.get("text") != "A soft breeze moved the curtain beside the window." or fixture.get("seed") != 20261023:
        raise ValueError("Unregistered startup warmup input")
    if proof.get("id") != fixture["id"] or proof.get("response_text_sha256") != digest(fixture["text"].encode()):
        raise ValueError("Warmup native proof does not match exact text/id")
    return {"enabled": True, "fixture_sha256": values[1], "native_prompt_proof_sha256": values[3], "fixture": fixture, "native_prompt": proof, "attempts_per_fresh_process": 1, "readiness": "Bind HTTP only after successful warmup and point EOS/prompt/reset validation; no retry"}


def recipe(args, resolved, defaults, sources, warmup):
    record = {
        "schema": "inferencekit-drama3-runtime-recipe-v13",
        "sources": sources,
        "model": {key: resolved[key] for key in ("model", "model_revision", "dtype", "quant")},
        "controls": {key: resolved[key] for key in ("cache_limit_bytes", "memory_limit_bytes", "wired_limit_bytes", "codec_mode", "load_wired_limit_bytes", "weight_loading_mode")},
        "generation_defaults": defaults,
        "compile_mode": getattr(args, "compile_mode", "none"),
        "activation_dtype_mode": getattr(args, "activation_dtype_mode", "checkpoint"),
        "fuse_switch_glu": getattr(args, "fuse_switch_glu", False),
        "diagnostic_code_capture": getattr(args, "capture_code_frames", False),
        "fixed_reference": resolved.get("fixed_reference"),
        "startup_warmup": warmup,
        "sealed_runtime_policy": getattr(args, "sealed_runtime_policy", False),
        "integer_observation_policy": "conditioned-all-segment-prompt-and-origin-joined-history-int64le-v1",
        "pcm": {"sample_rate": 44100, "channels": 1, "encoding": "pcm_s16le", "rounding": "rint(clip(float32,-1,1)*32767)", "trimming_or_normalization": False},
    }
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return record, digest(raw)


def run_warmup(runtime, args, contract, check_body):
    if not contract["enabled"]:
        return None
    if runtime.fixed_reference is None:
        raise ValueError("Registered fixed reference required for this warmup")
    fixture, native = contract["fixture"], contract["native_prompt"]
    actual = runtime.fixed_reference.initial_prompt_proof(runtime.model, args.startup_warmup_fixture, runtime.mx)
    row = actual["rows"][0]
    for a, b in (("text_utf8_sha256", "response_text_sha256"), ("first_segment_text_utf8_sha256", "first_chunk_text_sha256"), ("shape", "shape"), ("system_rows", "system_rows"), ("native_text_segment_count", "native_segment_count"), ("logical_int64le_sha256", "logical_i64_sha256")):
        if row[a] != native[b]:
            raise ValueError("Actual warmup initial prompt differs from native proof: " + a)
    path = Path(args.fixed_reference_manifest).resolve()
    manifest = json.loads(path.read_text())
    wav = (path.parent / manifest["wav_file"]).read_bytes()
    runtime.fixed_reference.validate_request(wav, runtime.fixed_reference.text)
    body = {"input": fixture["text"], "model": runtime.resolved["model"], "seed": fixture["seed"], "sample_rate": 44100, "channels": 1, "encoding": "pcm_s16le", "stream": True, "settings": {}, "reference_audio_b64": base64.b64encode(wav).decode(), "reference_text": runtime.fixed_reference.text}
    settings, reference, effective = check_body(body, runtime.resolved, runtime.fixed_reference)
    began = time.perf_counter_ns()
    runtime.request_kind = "startup_warmup"
    h, count = hashlib.sha256(), 0
    runtime.write_telemetry({"event": "startup_warmup_started", "fixture_sha256": contract["fixture_sha256"], "native_prompt_proof_sha256": contract["native_prompt_proof_sha256"], "attempt": 1, "API_bound": False, "actual_initial_prompt": row})
    try:
        for pcm in runtime.generate(body, settings, reference):
            if not pcm or len(pcm) % 2:
                raise RuntimeError("Invalid startup warmup PCM")
            h.update(pcm)
            count += len(pcm) // 2
        if not count or not runtime.generation_terminations or not all(item["eos_observed"] for item in runtime.generation_terminations):
            raise RuntimeError("Startup warmup missing nonempty uncapped source EOS")
        observed = runtime.conditioned_initial_prompt
        if observed is None or observed["logical_int64le_sha256"] != row["logical_int64le_sha256"] or observed["shape"] != row["shape"]:
            raise RuntimeError("Actual warmup first build differs from preflight/native prompt")
        elapsed_s = (time.perf_counter_ns() - began) / 1e9
        result = {"event": "startup_warmup_finished", "fixture_sha256": contract["fixture_sha256"], "native_prompt_proof_sha256": contract["native_prompt_proof_sha256"], "elapsed_s": elapsed_s, "sample_count": count, "pcm_sha256": h.hexdigest(), "audio_duration_s": count / 44100, "internal_generation_RTF": elapsed_s / (count / 44100), "effective": effective, "source_EOS": True, "attempts": 1, "API_bound": False, "actual_initial_prompt": observed, "state_reset_observations": runtime.state_observations}
        Path(args.startup_warmup_receipt).write_text(json.dumps(result, indent=2) + "\n")
        runtime.write_telemetry(result)
        return result
    finally:
        runtime.request_kind = "api"

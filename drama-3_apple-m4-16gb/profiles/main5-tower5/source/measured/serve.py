#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private unvalidated Drama 3 adapter; no MLX imports until native runtime construction."""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import signal
import sys
import subprocess
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MAX_BODY = 16 * 1024 * 1024
GEN_DEFAULTS = {"temperature": 0.7, "top_p": 0.7, "top_k": 30, "max_tokens": 1024,
                "streaming_interval": 0.32, "first_chunk_frames": 4}
PIN_KEYS = ("model_revision", "dtype", "quant", "engine_version", "cache_limit_bytes", "memory_limit_bytes", "wired_limit_bytes", "codec_mode", "load_wired_limit_bytes", "weight_loading_mode")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_body(body, resolved, fixed_reference=None):
    if not isinstance(body, dict) or set(body) - {"input", "model", "seed", "sample_rate", "channels", "encoding", "stream", "settings", "reference_audio_b64", "reference_text"}:
        raise ValueError("invalid request fields")
    if not isinstance(body.get("input"), str) or not body["input"].strip() or len(body["input"].encode()) > 100000:
        raise ValueError("nonempty bounded input text required")
    for key, value in (("model", resolved["model"]), ("sample_rate", 44100), ("channels", 1), ("encoding", "pcm_s16le"), ("stream", True)):
        if body.get(key) != value or (key in ("sample_rate", "channels") and type(body.get(key)) is not int):
            raise ValueError("request differs from loaded model/PCM format")
    if type(body.get("seed")) is not int or not 0 <= body["seed"] < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    requested = body.get("settings", {})
    if not isinstance(requested, dict) or set(requested) - (set(GEN_DEFAULTS) | set(PIN_KEYS)):
        raise ValueError("unsupported settings; ignored knobs are refused")
    defaults = {**GEN_DEFAULTS, **resolved.get("generation_defaults", {})}
    settings = {**defaults, **requested}
    if resolved.get("runtime_policy_sealed") and any(key in requested and requested[key] != defaults[key] for key in ("first_chunk_frames", "streaming_interval")):
        raise ValueError("requested runtime setting differs from sealed recipe")
    if any(type(settings[k]) is not int for k in ("top_k", "max_tokens", "first_chunk_frames")):
        raise ValueError("integer generation settings required")
    if not 1 <= settings["top_k"] <= 4097 or not 1 <= settings["max_tokens"] <= 4096 or not 1 <= settings["first_chunk_frames"] <= 64:
        raise ValueError("generation integer setting out of bounds")
    for k, lo, hi in (("temperature", 0, 2), ("top_p", 0, 1), ("streaming_interval", 0, 2)):
        v = settings[k]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not lo < v <= hi:
            raise ValueError("generation numeric setting out of bounds")
    for key in PIN_KEYS:
        if key in requested and canonical(requested[key]) != canonical(resolved[key]):
            raise ValueError("requested pin differs from loaded runtime")
    if resolved.get("codec_mode","full-fp32").startswith("decode-only") and "reference_audio_b64" in body and fixed_reference is None:
        raise ValueError("decode-only codec does not support reference conditioning")
    reference = None
    if "reference_audio_b64" in body:
        if not isinstance(body["reference_audio_b64"], str) or not isinstance(body.get("reference_text"), str) or not body["reference_text"].strip():
            raise ValueError("reference bytes need their transcript")
        reference = base64.b64decode(body["reference_audio_b64"], validate=True)
        with wave.open(io.BytesIO(reference), "rb") as f:
            if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype()) != (44100, 1, 2, "NONE"):
                raise ValueError("reference must be mono 44.1kHz PCM16 WAV")
            frames = f.getnframes()
            if not 44100 <= frames <= 30 * 44100 or len(f.readframes(frames)) != 2 * frames:
                raise ValueError("reference must be complete and 1..30 seconds")
        if fixed_reference is not None:
            fixed_reference.validate_request(reference, body["reference_text"])
    elif body.get("reference_text") is not None:
        raise ValueError("reference transcript without audio")
    effective = {"model": resolved["model"], "seed": body["seed"], **defaults,
                 **{k: settings[k] for k in GEN_DEFAULTS}, **{k: resolved[k] for k in PIN_KEYS}}
    if fixed_reference is not None and reference is not None:
        effective["fixed_reference"] = fixed_reference.metadata
    return settings, reference, effective


class MlxRuntime:
    def __init__(self, args):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("native arm64 macOS is required; CPU preparation cannot load the model")
        import mlx.core as mx
        import numpy as np
        from mlx_audio.tts.utils import load_model
        from mlx.utils import tree_flatten
        self.mx, self.np = mx, np
        self.fixed_reference = None
        if getattr(args, "fixed_reference_manifest", None):
            from fixed_reference import FixedReference
            self.fixed_reference = FixedReference(args.fixed_reference_manifest, args.fixed_reference_sha256)
        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable")
        self.device_info = mx.metal.device_info()
        lock = json.loads(Path(args.lock).read_text())
        spec = lock["models"][args.track]
        direct = importlib.metadata.distribution("mlx-audio").read_text("direct_url.json")
        installed = json.loads(direct or "{}")
        if installed.get("vcs_info", {}).get("commit_id") != lock["mlx_audio_commit"]:
            raise RuntimeError("installed mlx-audio source commit is not the pinned revision")
        if importlib.metadata.version("mlx") != lock["mlx_version"]:
            raise RuntimeError("installed MLX version differs from lock")
        model_path = Path(args.model_dir).resolve()
        origin = json.loads((model_path / "origin.json").read_text())
        if origin.get("model") != spec["id"] or origin.get("revision") != spec["revision"]:
            raise RuntimeError("local checkpoint origin does not match pinned model")
        from download_model import file_hash
        expected_files = {item["path"]: item for item in spec["files"]}
        if {f.name for f in model_path.glob("*.safetensors")} != {name for name in expected_files if name.endswith(".safetensors")}:
            raise RuntimeError("unregistered checkpoint shard present or pinned shard missing")
        if set(origin.get("files", {})) != set(expected_files):
            raise RuntimeError("checkpoint content manifest differs from pinned files")
        for name, identity in origin.get("files", {}).items():
            file = model_path / name
            expected = expected_files[name]
            if not file.is_file() or file.stat().st_size != expected["bytes"] or file_hash(file) != identity["sha256"]:
                raise RuntimeError("local checkpoint content changed after download")
            if expected["lfs_sha256"] and identity["sha256"] != expected["lfs_sha256"]:
                raise RuntimeError("checkpoint weight digest differs from source lock")
            if not expected["lfs_sha256"] and file_hash(file, "sha1", git_blob=True) != expected["git_blob_sha1"]:
                raise RuntimeError("checkpoint metadata differs from source lock")
        if not origin.get("files"):
            raise RuntimeError("checkpoint content manifest missing")
        config = json.loads((model_path / "config.json").read_text())
        if sha(canonical(config).encode()) != spec["config_canonical_sha256"]:
            raise RuntimeError("checkpoint configuration differs from pinned conversion")
        def wired_setting():
            try:
                return subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"],stderr=subprocess.DEVNULL,timeout=10,text=True).strip()
            except (OSError,subprocess.SubprocessError):
                return "UNPROVEN (setting not exposed)"
        wired_before = wired_setting()
        self.cache_limit = args.cache_limit_bytes
        mx.set_default_device(mx.gpu)
        self.previous_cache_limit = mx.set_cache_limit(self.cache_limit) if self.cache_limit is not None else None
        self.previous_memory_limit = mx.set_memory_limit(args.memory_limit_bytes) if args.memory_limit_bytes is not None else None
        self.load_wired_limit = args.wired_limit_bytes if args.load_wired_limit_bytes is None else args.load_wired_limit_bytes
        self.previous_wired_limit = mx.set_wired_limit(self.load_wired_limit)
        mx.reset_peak_memory()
        began = time.perf_counter()
        from mmap_weights import install as install_weight_loader
        weight_loading_mode = getattr(args, "weight_loading_mode", "native")
        if weight_loading_mode != 'native' and (getattr(args, 'fuse_switch_glu', False)
                or getattr(args, 'activation_dtype_mode', 'checkpoint') != 'checkpoint'):
            raise ValueError('Mapped exact-weight prototype requires unfused checkpoint dtypes')
        self.weight_loading = install_weight_loader(weight_loading_mode,
            cache_dir=model_path.parent / '.inferencekit-mmap' / spec['revision'],
            verified_files=origin['files'])
        from codec_modes import install_loader
        install_loader(args.codec_mode)
        self.model = load_model(str(model_path), lazy=False, strict=True)
        if self.fixed_reference is not None:
            self.fixed_reference.install(self.model, mx)
        from codec_modes import apply_stock_dtype
        apply_stock_dtype(self.model.codec,args.codec_mode)
        mx.eval(self.model.parameters())
        mx.synchronize()
        if self.model.sample_rate != 44100:
            raise RuntimeError("unexpected codec sample rate")
        modules = dict(self.model.named_modules())
        quant = config["quantization"]
        checked = {}
        for name, recipe in quant.items():
            if not isinstance(recipe, dict):
                continue
            module = modules.get(name)
            if module is None or getattr(module, "bits", None) != recipe["bits"] or getattr(module, "group_size", None) != recipe["group_size"]:
                raise RuntimeError("loaded quantized module does not match checkpoint recipe")
            checked[name] = {"bits": module.bits, "group_size": module.group_size}
        del modules
        module=None
        from codec_encoder_release import release as release_codec_encoder
        if args.release_codec_encoder and args.codec_mode != 'full-stock-fp16-refold':
            raise ValueError('This public residency experiment requires the exact selected stock FP16/refold decoder')
        self.encoder_residency = release_codec_encoder(self.model,self.fixed_reference,mx,
            enabled=args.release_codec_encoder,audit_payloads=args.audit_codec_retained_payloads)
        self.fusion = {"fused_switch_glu_count":0}
        if getattr(args,"fuse_switch_glu",False):
            from fuse_glu import install as install_fusion
            self.fusion=install_fusion(self.model)
        dtype_mode = getattr(args, "activation_dtype_mode", "checkpoint")
        cast_audit = {"mode":dtype_mode,"converted_tensors":0,"changed_elements":0,"max_abs_change":0.0}
        if dtype_mode == "fp16-from-bf16":
            for name, value in tree_flatten(self.model.parameters()):
                if value.dtype != mx.bfloat16:continue
                half=value.astype(mx.float16)
                finite=mx.all(mx.isfinite(half))
                difference=mx.abs(half.astype(mx.float32)-value.astype(mx.float32))
                changed=mx.sum(difference != 0); maximum=mx.max(difference)
                mx.eval(finite,changed,maximum)
                if not bool(finite.item()):raise RuntimeError("BF16 to FP16 cast would overflow")
                cast_audit["converted_tensors"]+=1
                cast_audit["changed_elements"]+=int(changed.item())
                cast_audit["max_abs_change"]=max(cast_audit["max_abs_change"],float(maximum.item()))
            self.model.set_dtype(mx.float16,predicate=lambda dtype:dtype == mx.bfloat16)
            mx.eval(self.model.parameters());mx.clear_cache()
        mx.synchronize()
        self.pre_generation_wired_limit = mx.set_wired_limit(args.wired_limit_bytes)
        params = tree_flatten(self.model.parameters())
        dtype_bytes = {}
        for _, value in params:
            key = str(value.dtype).replace("mlx.core.", "")
            dtype_bytes[key] = dtype_bytes.get(key, 0) + value.nbytes
        buckets = {}
        for name, value in params:
            bucket = name.split('.')[0]
            buckets[bucket] = buckets.get(bucket, 0) + value.nbytes
        hidden_codec = [(name, module.get('_folded')) for name, module in self.model.codec.named_modules()
                        if module.get('_folded') is not None]
        hidden_dtype_bytes = {}
        for _, value in hidden_codec:
            dtype = str(value.dtype)
            hidden_dtype_bytes[dtype] = hidden_dtype_bytes.get(dtype, 0) + value.nbytes
        dtype_label = "mixed:" + ",".join(sorted(dtype_bytes))
        from compile_modes import install
        self.compilation = install(self.model, args.compile_mode)
        self.generation_terminations=[]
        self.generation_inputs=[]
        self.capture_code_frames=getattr(args,"capture_code_frames",False)
        self.captured_code_frames={};self.captured_prompt_frames={};self.captured_pieces=[]
        self.conditioned_initial_prompt=None
        self.segment_prompt_observations=[]
        self.state_observations=[]
        self.request_kind="api"
        self.using_fixed_reference=False
        self.integer_sidecar_dir=None
        self.integer_code_observations=[]
        self.integer_history_observations=[]
        self.integer_segment_buffers={}
        def save_integer(kind,index,value):
            mx.eval(value)
            logical=np.ascontiguousarray(np.asarray(value),dtype='<i8')
            path=self.integer_sidecar_dir/(kind+'-'+str(index)+'.npy')
            np.save(path,logical,allow_pickle=False)
            return {'file':str(path),'file_sha256':sha(path.read_bytes()),'shape':list(logical.shape),'logical_int64le_sha256':sha(logical.tobytes(order='C'))}
        def observe_history(index,history,target):
            if not self.using_fixed_reference:return
            turns=[]
            for text,codes in history:
                mx.eval(codes)
                array=np.ascontiguousarray(np.asarray(codes),dtype='<i8')
                digest=sha(array.tobytes(order='C'))
                candidates=[row for row in self.integer_code_observations if row['logical_int64le_sha256']==digest and row['text_utf8_sha256']==sha(text.encode())]
                if not candidates:raise RuntimeError('Actual history does not join an originating emitted segment')
                origin=candidates[0] if len(turns)==0 else candidates[-1]
                turns.append({'text':text,'text_utf8_sha256':sha(text.encode()),'origin_segment_index':origin['segment_index'],'codes':origin})
            self.integer_history_observations.append({'segment_index':index,'target_text':target,'target_text_utf8_sha256':sha(target.encode()),'history':turns})
        def emitted_frame(index,frame):
            if self.capture_code_frames:self.captured_code_frames.setdefault(index,[]).append(frame)
            if self.using_fixed_reference:self.integer_segment_buffers.setdefault(index,[]).append(frame)
        def completed_segment(index):
            frames=self.integer_segment_buffers.pop(index,[])
            if not self.using_fixed_reference or not frames:return
            codes=save_integer('emitted-codes',index,mx.stack(frames,axis=1))
            self.integer_code_observations.append({'segment_index':index,'text_utf8_sha256':self.generation_inputs[index]['text_utf8_sha256'],**codes})
        from generation_audit import install_input_audit
        def observe_prompt(index,frames):
            if self.capture_code_frames:self.captured_prompt_frames[index]=frames
            if self.using_fixed_reference:
                mx.eval(frames)
                logical=np.ascontiguousarray(np.asarray(frames),dtype="<i8")
                observed={"shape":list(logical.shape),"logical_int64le_sha256":sha(logical.tobytes(order="C"))}
                system_rows=self.generation_inputs[index]['system_rows']
                self.segment_prompt_observations.append({"segment_index":index,**observed,"system_rows":system_rows,"system_prefix_logical_int64le_sha256":sha(logical[:system_rows].tobytes(order="C"))})
                self.segment_prompt_observations[-1]['integer_artifact']=save_integer('prompt',index,frames)
                if index==0:self.conditioned_initial_prompt=observed
        prompt_sink=observe_prompt if self.capture_code_frames or self.fixed_reference is not None else None
        install_input_audit(self.model,self.generation_inputs,prompt_sink,observe_history)
        from state_audit import install as install_state_audit
        install_state_audit(self.model,self.state_observations)
        from eos_audit import install as install_eos_audit
        install_eos_audit(self.model,self.generation_terminations,mx,emitted_frame,completed_segment)
        # This is resolved provenance; requested pins are checked against it, never echoed as proof.
        wrapper = sha(Path(__file__).read_bytes() + Path(__file__).with_name("codec_modes.py").read_bytes() + Path(__file__).with_name("compile_modes.py").read_bytes() + Path(__file__).with_name("fuse_glu.py").read_bytes() + Path(__file__).with_name("eos_audit.py").read_bytes() + Path(__file__).with_name("generation_audit.py").read_bytes() + Path(__file__).with_name("mmap_weights.py").read_bytes() + Path(__file__).with_name("fixed_reference.py").read_bytes() + Path(__file__).with_name("startup_policy.py").read_bytes() + Path(__file__).with_name("segment_contract.py").read_bytes() + Path(__file__).with_name("state_audit.py").read_bytes() + Path(__file__).with_name("frozen_chunking.py").read_bytes() + Path(__file__).with_name("codec_encoder_release.py").read_bytes() + Path(__file__).with_name("main_dense4_overlay.py").read_bytes())[:16]
        self.resolved = {"model": spec["id"], "model_revision": spec["revision"], "dtype": dtype_label,
                         "quant": spec["quant"], "engine_version": f"mlx-audio@{lock['mlx_audio_commit']}+adapter@{wrapper}+compile@{args.compile_mode}+dtype@{dtype_mode}+fuse@{int(getattr(args,'fuse_switch_glu',False))}+codec@{args.codec_mode}+codes@{int(self.capture_code_frames)}+weights@{weight_loading_mode}",
                         "cache_limit_bytes": self.cache_limit, "memory_limit_bytes": args.memory_limit_bytes, "wired_limit_bytes":args.wired_limit_bytes, "codec_mode":args.codec_mode, "load_wired_limit_bytes":self.load_wired_limit, "weight_loading_mode":weight_loading_mode}
        self.resolved['codec_encoder_residency']={'release_enabled':args.release_codec_encoder,
            'retained_payload_audit':args.audit_codec_retained_payloads,
            'helper_source_sha256':sha(Path(__file__).with_name('codec_encoder_release.py').read_bytes())}
        self.resolved['engine_version'] += '+encoder-release@'+str(int(args.release_codec_encoder))+'+retained-payload-audit@'+str(int(args.audit_codec_retained_payloads))
        if self.fixed_reference is not None:
            self.resolved["engine_version"] += "+fixed-reference@" + self.fixed_reference.metadata["manifest_sha256"]
            self.resolved["fixed_reference"] = self.fixed_reference.metadata
        from startup_policy import generation_defaults,recipe,warmup_contract
        self.resolved['generation_defaults']=generation_defaults(args,GEN_DEFAULTS)
        self.resolved['runtime_policy_sealed']=getattr(args,'sealed_runtime_policy',False)
        self.warmup_contract=warmup_contract(args)
        self.runtime_recipe,recipe_sha=recipe(args,self.resolved,self.resolved['generation_defaults'],
            {'adapter_source_sha256_prefix':wrapper,'sources_lock_sha256':sha(Path(args.lock).read_bytes()),'mlx_version':lock['mlx_version'],'mlx_audio_commit':lock['mlx_audio_commit'],'python':platform.python_version(),'checkpoint_file_manifest_sha256':sha(canonical({name:{'sha256':value['sha256'],'bytes':expected_files[name]['bytes']} for name,value in origin['files'].items()}).encode()),'checkpoint_config_sha256':sha(canonical(config).encode()),'codec_encoder_residency':self.resolved['codec_encoder_residency']},self.warmup_contract)
        self.resolved['runtime_recipe_sha256']=recipe_sha
        self.resolved['engine_version']+='+runtime@'+recipe_sha
        def projection_format(module):
            return {"bits":getattr(module,"bits",None),"group_size":getattr(module,"group_size",None),"weight_dtype":str(module.weight.dtype),"weight_shape":list(module.weight.shape)}
        self.head_formats={"semantic_eos":projection_format(self.model.semantic_head),"residual_codebook_heads":[projection_format(module) for module in self.model.tower.codebook_heads]}
        self.startup = {"codec_encoder_residency_observation":self.encoder_residency,"resolved": self.resolved,"runtime_recipe":self.runtime_recipe,"head_formats":self.head_formats, "weight_loading":self.weight_loading, "compile_mode": args.compile_mode, "compilation":self.compilation,"fusion":self.fusion,"diagnostic_code_capture":self.capture_code_frames, "activation_cast_audit":cast_audit, "pre_generation_api_wired_limit_bytes":self.pre_generation_wired_limit, "previous_api_cache_limit_bytes":self.previous_cache_limit,"dtype_bytes": dtype_bytes, "quantized_modules": checked,
                        "load_seconds_self_reported": time.perf_counter() - began,
                        "public_parameter_tensor_bytes_by_top_module":buckets,
                        "hidden_codec_folded_tensor_bytes_by_dtype":hidden_dtype_bytes,
                        "hidden_codec_folded_count":len(hidden_codec),
                        "tensor_accounting_warning":"Public parameters and hidden kernels can alias; sums are not physical residency or true allocator peaks",
                        "load_peak_mlx_bytes": mx.get_peak_memory(), "load_maxrss_bytes_darwin": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "metal": self.device_info,
                        "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=10).strip()),
                        "wired_limit_changed_by_adapter":self.previous_wired_limit != args.wired_limit_bytes, "previous_api_wired_limit_bytes":self.previous_wired_limit,"previous_api_memory_limit_bytes":self.previous_memory_limit, "wired_setting_before":wired_before,"wired_setting_after":wired_setting(), "checkpoint_config_sha256": sha(canonical(config).encode())}
        self.telemetry = Path(args.telemetry)
        self.telemetry.parent.mkdir(parents=True, exist_ok=True)
        self.write_telemetry({"event": "loaded", **self.startup})

    def write_telemetry(self, row):
        with self.telemetry.open("a") as f:
            f.write(canonical({"time_ns": time.time_ns(), "pid": os.getpid(), "engine_version":self.resolved['engine_version'],"runtime_recipe_sha256":self.resolved['runtime_recipe_sha256'],"request_kind":self.request_kind,"codec_encoder_residency_release":self.encoder_residency["enabled"],**row}) + "\n")

    def generate(self, body, settings, reference):
        mx, np = self.mx, self.np
        self.generation_terminations.clear();self.generation_inputs.clear()
        self.captured_code_frames.clear();self.captured_prompt_frames.clear();self.captured_pieces.clear()
        self.segment_prompt_observations.clear();self.state_observations.clear()
        self.conditioned_initial_prompt=None
        self.using_fixed_reference=reference is not None and self.fixed_reference is not None
        self.integer_code_observations.clear();self.integer_history_observations.clear();self.integer_segment_buffers.clear()
        self.integer_sidecar_dir=None
        if self.using_fixed_reference:
            self.integer_sidecar_dir=self.telemetry.with_name(self.telemetry.stem+'-integer-sidecars')/(str(time.time_ns())+'-seed-'+str(body['seed'])+'-input-'+sha(body['input'].encode()))
            self.integer_sidecar_dir.mkdir(parents=True,exist_ok=False)
        mx.random.seed(body["seed"])
        mx.reset_peak_memory()
        ref_audio = None
        if reference is not None:
            with wave.open(io.BytesIO(reference), "rb") as f:
                ref_audio = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
        generated = self.model.generate(text=body["input"], ref_audio=ref_audio, ref_text=body.get("reference_text"),
                                        stream=True, **{k: settings[k] for k in GEN_DEFAULTS})
        from segment_contract import partition_proof
        from importlib import import_module
        partition=partition_proof(body['input'],import_module(self.model.__class__.__module__).split_text(body['input']))
        clipped, samples, segment_frames = 0, 0, {}
        pcm_digest=hashlib.sha256()
        completed=False
        audio_dtypes=set()
        try:
            for result in generated:
                segment_frames[result.segment_idx] = segment_frames.get(result.segment_idx, 0) + result.token_count
                mx.eval(result.audio)
                audio = np.asarray(result.audio.astype(mx.float32) if result.audio.dtype == mx.bfloat16 else result.audio, dtype=np.float32).reshape(-1)
                if not np.isfinite(audio).all():
                    raise RuntimeError("nonfinite generated audio")
                if not len(audio):
                    continue  # upstream emits an empty terminal flush; transport final closes the stream
                audio_dtypes.add(str(result.audio.dtype))
                clipped += int(np.count_nonzero(np.abs(audio) > 1))
                samples += len(audio)
                pcm=np.rint(np.clip(audio,-1,1)*32767).astype("<i2").tobytes()
                if self.capture_code_frames:
                    self.captured_pieces.append({"segment_index":int(result.segment_idx),"frame_count":int(result.token_count),"sample_count":len(audio)})
                pcm_digest.update(pcm)
                yield pcm
            if [row['text_utf8_sha256'] for row in self.generation_inputs] != [row['text_utf8_sha256'] for row in partition['segments']]:
                raise RuntimeError('Observed native segments do not match complete ordered text partition')
            if self.generation_inputs and self.generation_inputs[0]['history_count']!=0:
                raise RuntimeError('First native segment carries previous request history')
            copies=[row for row in self.state_observations if row['event']=='fresh_prefix_cache_copy']
            if len(copies)!=len(partition['segments']) or not all(row['cache_objects_separate_from_voice_snapshot'] for row in copies) or not any(row['event']=='codec_streaming_state_reset' for row in self.state_observations):
                raise RuntimeError('Upstream request reset/prefix-cache copy observation failed')
            completed=True
        finally:
            generated.close()
            mx.synchronize()
            code_artifact=None
            if self.capture_code_frames and self.captured_code_frames:
                code_dir=self.telemetry.with_name(self.telemetry.stem+"-codes");code_dir.mkdir(parents=True,exist_ok=True)
                prefix=code_dir/(pcm_digest.hexdigest()+"-seed-"+str(body["seed"]))
                arrays={}
                for index,frames in self.captured_code_frames.items():
                    array=mx.stack(frames,axis=1);mx.eval(array);arrays["segment_"+str(index)]=np.asarray(array,dtype=np.int32)
                for index,frames in self.captured_prompt_frames.items():
                    mx.eval(frames);arrays["prompt_"+str(index)]=np.asarray(frames,dtype=np.int32)
                np.savez_compressed(prefix.with_suffix(".npz"),**arrays)
                meta={"kind":"diagnostic_code_replay_only","consumer_performance_use":False,"pcm_sha256":pcm_digest.hexdigest(),"sample_count":samples,"sample_rate":44100,"frame_length":self.model.codec.frame_length,"pieces":self.captured_pieces,"generation_inputs":self.generation_inputs,"terminations":self.generation_terminations,"generation_completed":completed,"resolved":self.resolved,"archive_sha256":sha(prefix.with_suffix(".npz").read_bytes())}
                prefix.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
                code_artifact=str(prefix.with_suffix(".json"))
            self.write_telemetry({"event": "request_finished", "input_sha256": sha(body["input"].encode()),
                                  "fixed_reference":self.fixed_reference.metadata if reference is not None and self.fixed_reference is not None else None,
                                  "conditioned_initial_prompt":self.conditioned_initial_prompt,
                                  "segment_prompt_observations":list(self.segment_prompt_observations),"ordered_text_partition":partition,"state_reset_observations":list(self.state_observations),"runtime_generation_settings":{k:settings[k] for k in GEN_DEFAULTS},
                                  "integer_history_observations":list(self.integer_history_observations),"integer_emitted_code_observations":list(self.integer_code_observations),"integer_observation_policy":"conditioned-all-segment-prompt-and-origin-joined-history-int64le-v1",
                                  "seed": body["seed"], "sample_count": samples, "clipped_samples": clipped,
                                  "generation_inputs":list(self.generation_inputs),"diagnostic_code_artifact":code_artifact,
                                  "generation_completed":completed,"pcm_sha256":pcm_digest.hexdigest(),"audio_dtypes":sorted(audio_dtypes),
                                  "termination_observations":list(self.generation_terminations),
                                  "all_segments_eos":completed and bool(self.generation_terminations) and all(row["eos_observed"] for row in self.generation_terminations),
                                  "termination_frame_counts_match":{row["segment_index"]:row["emitted_audio_frames"] for row in self.generation_terminations} == segment_frames,
                                  "segment_frames": segment_frames, "possible_frame_cap": any(n >= settings["max_tokens"] for n in segment_frames.values()),
                                  "active_mlx_bytes": mx.get_active_memory(), "cache_mlx_bytes": mx.get_cache_memory(),
                                  "peak_mlx_bytes": mx.get_peak_memory(), "maxrss_bytes_darwin": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def server_for(runtime, port=8765):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass  # no request text, endpoints, headers or credentials in logs

        def do_GET(self):
            if self.path not in ("/health", "/info"):
                self.send_error(404)
                return
            data = {"ready": True, "resolved": runtime.resolved} if self.path == "/info" else {"ready": True}
            raw = canonical(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            if self.path != "/v1/audio/speech":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("request body bound exceeded")
                body = json.loads(self.rfile.read(length))
                settings, reference, effective = check_body(body, runtime.resolved, getattr(runtime, "fixed_reference", None))
            except (ValueError, TypeError, wave.Error, EOFError):
                self.send_error(400, "invalid speech request")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(doc):
                self.wfile.write(canonical(doc).encode() + b"\n")
                self.wfile.flush()

            samples = 0
            try:
                send({"type": "metadata", "sample_rate": 44100, "channels": 1, "encoding": "pcm_s16le"})
                for pcm in runtime.generate(body, settings, reference):
                    if not pcm or len(pcm) % 2:
                        raise RuntimeError("malformed runtime PCM")
                    samples += len(pcm) // 2
                    send({"type": "audio", "pcm_b64": base64.b64encode(pcm).decode("ascii")})
                if not samples:
                    raise RuntimeError("runtime produced no audio")
                send({"type": "final", "sample_count": samples, "effective": effective})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                # A generation failure deliberately has no final event, so client correctness fails.
                runtime.write_telemetry({"event": "generation_failed", "input_sha256": sha(body["input"].encode())})
            finally:
                self.close_connection = True

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    main_started=time.perf_counter_ns()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--lock", default=str(Path(__file__).with_name("sources.lock.json")))
    p.add_argument("--track", default="upstream-3bit")
    p.add_argument("--codec-mode", choices=("full-fp32","full-stock-fp16-public","full-stock-fp16-refold","full-stock-bf16-refold","decode-only-fp32","decode-only-fp16"), default="full-fp32")
    p.add_argument("--activation-dtype-mode", choices=("checkpoint","fp16-from-bf16"), default="checkpoint")
    p.add_argument("--weight-loading-mode", choices=("native","mmap-readonly-v1","native-main5-tower5-v1"), default="native")
    p.add_argument("--capture-code-frames",action="store_true")
    p.add_argument("--fixed-reference-manifest")
    p.add_argument("--fixed-reference-sha256")
    p.add_argument("--fixed-prompt-fixtures")
    p.add_argument("--fixed-prompt-output")
    for key,value in GEN_DEFAULTS.items():
        p.add_argument('--default-'+key.replace('_','-'),type=type(value),default=value)
    p.add_argument('--sealed-runtime-policy',action='store_true')
    p.add_argument('--startup-warmup-fixture')
    p.add_argument('--startup-warmup-fixture-sha256')
    p.add_argument('--startup-warmup-native-proof')
    p.add_argument('--startup-warmup-native-proof-sha256')
    p.add_argument('--startup-warmup-receipt')
    p.add_argument('--release-codec-encoder',action='store_true',help='Development-only exact fixed-reference decoder residency refactor')
    p.add_argument('--audit-codec-retained-payloads',action='store_true',help='Hash retained decoder/folded tensors before and after the residency operation; startup cost disclosed')
    p.add_argument("--fuse-switch-glu",action="store_true")
    p.add_argument("--compile-mode", choices=("none","stateless-mlp","stateless-mlp-sampler"), default="none")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cache-limit-bytes", type=int)
    p.add_argument("--memory-limit-bytes", type=int)
    p.add_argument("--load-wired-limit-bytes",type=int)
    p.add_argument("--wired-limit-bytes", type=int, default=0)
    p.add_argument("--telemetry", default="telemetry/runtime.jsonl")
    args = p.parse_args()
    if bool(args.fixed_reference_manifest) != bool(args.fixed_reference_sha256):
        p.error("fixed-reference manifest and digest must be supplied together")
    if bool(args.fixed_prompt_fixtures) != bool(args.fixed_prompt_output) or (args.fixed_prompt_fixtures and not args.fixed_reference_manifest):
        p.error("prompt proof requires paired fixture/output paths and a registered fixed reference")
    if args.cache_limit_bytes is not None and args.cache_limit_bytes < 0:
        p.error("cache limit must be nonnegative")
    from startup_policy import generation_defaults,warmup_contract,run_warmup
    try:
        contract=warmup_contract(args)
        if contract['enabled'] and not args.fixed_reference_manifest:
            raise ValueError('This startup warmup requires the registered fixed reference')
        dummy={key:None for key in PIN_KEYS}
        dummy.update(model='startup-argument-validation',codec_mode='full-fp32',generation_defaults=generation_defaults(args,GEN_DEFAULTS))
        check_body({'input':'Validate only.','model':dummy['model'],'seed':0,'sample_rate':44100,'channels':1,'encoding':'pcm_s16le','stream':True,'settings':{}},dummy)
    except (OSError,ValueError,TypeError) as error:
        p.error(str(error))
    runtime = MlxRuntime(args)
    service=None
    previous_signal=signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.fixed_prompt_fixtures:
            proof = runtime.fixed_reference.initial_prompt_proof(runtime.model,args.fixed_prompt_fixtures,runtime.mx)
            proof["resolved"] = runtime.resolved
            proof["tokenizer_json_sha256"] = sha((Path(args.model_dir)/"tokenizer.json").read_bytes())
            Path(args.fixed_prompt_output).write_text(json.dumps(proof,indent=2)+"\n")
            runtime.write_telemetry({"event":"conditioned_initial_prompts_sealed","proof_sha256":sha(Path(args.fixed_prompt_output).read_bytes())})
        run_warmup(runtime,args,runtime.warmup_contract,check_body)
        service = server_for(runtime, args.port)
        runtime.write_telemetry({'event':'API_ready','main_entry_to_HTTP_bind_s':(time.perf_counter_ns()-main_started)/1e9,'startup_warmup_enabled':runtime.warmup_contract['enabled'],'full_process_startup_cost_source':'Join this event time_ns to external task launch/kernel birth receipt; main-entry timer excludes Python import startup'})
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        if service is None:runtime.write_telemetry({'event':'startup_failed','error_type':type(error).__name__,'error':str(error),'API_bound':False})
        raise
    finally:
        if service is not None:service.server_close()
        restored={}
        actions=[('synchronize',runtime.mx.synchronize),('wired_limit',lambda:runtime.mx.set_wired_limit(runtime.previous_wired_limit))]
        if runtime.previous_memory_limit is not None:actions.append(('memory_limit',lambda:runtime.mx.set_memory_limit(runtime.previous_memory_limit)))
        if runtime.previous_cache_limit is not None:actions.append(('cache_limit',lambda:runtime.mx.set_cache_limit(runtime.previous_cache_limit)))
        for name,action in actions:
            try:action();restored[name]=True
            except Exception as error:restored[name]={'status':'UNPROVEN','error_type':type(error).__name__}
        runtime.write_telemetry({"event": "process_stopped", "per_process_limits_restored":all(value is True for value in restored.values()),'restoration_observations':restored})
        signal.signal(signal.SIGTERM,previous_signal)


if __name__ == "__main__":
    main()

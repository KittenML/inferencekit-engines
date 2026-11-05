"""Observe pinned native request-mode MOSS; no model arithmetic is replaced.

Heavy imports occur only when the remote native worker calls configure().
Diagnostic byte copies and trajectory writes are explicitly excluded from
performance qualification. Lightweight observations remain inside client time.
"""
from ranking_policy import require_expected_mode

import functools
import hashlib
import inspect
import json
import os
import time
import native_quality_dump as quality_dump
from pathlib import Path

from moss_omni_contract import (NONCE_KEY, PIPELINE, SOURCE_COMMIT, TRANSFORMER,
                               canonical, nonce_value, request_fact, require, sha, validate_receipt)

_config = None
_config_sha = None
_current = None
_branch = None
_cache_refs = {}
_verified_sources = {}
_negative_fingerprint = None


def configuration():
    return _config


def config_sha():
    return _config_sha


def source_fact(function):
    path = Path(inspect.getsourcefile(inspect.unwrap(function))).resolve()
    name = path.relative_to(Path(_config["native_source_root"]).resolve()).as_posix()
    if name not in _verified_sources:
        digest = sha(path)
        require(digest == _config["native_sources"].get(name), "Actual imported callable source differs: " + name)
        _verified_sources[name] = digest
    return name


def tensor_fact(value):
    import torch
    require(isinstance(value, torch.Tensor), "Expected actual worker tensor")
    fact = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "bytes": value.numel() * value.element_size()}
    if _config["mode"] == "diagnostic":
        # Flatten handles scalar timestep tensors; BF16 bytes need uint8 view.
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        fact["sha256"] = hashlib.sha256(raw).hexdigest()
    return fact


def persist(status, error=None):
    if _current is None:
        return
    _current.update(status=status, finished_time_ns=time.time_ns(), worker_source_files=dict(_verified_sources))
    if error is not None:
        _current.update(error_type=type(error).__name__, error=str(error))
    target = Path(_config["proof_directory"]) / (_current["nonce"] + ".json")
    temporary = target.with_suffix(".tmp")
    with temporary.open("x") as f:
        json.dump(_current, f, indent=2, allow_nan=False)
        f.write("\n")
    temporary.replace(target)


def actual_request(sampling, prompt):
    require(isinstance(prompt, dict) and isinstance(prompt.get("prompt"), str), "One explicit native prompt required")
    extra = sampling.extra_args
    require(prompt.get("negative_prompt") == "", "Fixed empty negative prompt required")
    require(sampling.generator_device == "cpu" and sampling.latents is None, "Fresh CPU noise policy differs")
    if sampling.generator is not None:
        require(str(sampling.generator.device) == "cpu" and sampling.generator.initial_seed() == sampling.seed,
                "Native request generator seed/device differs")
    require(sampling.num_outputs_per_prompt == 1 and sampling.quality == "lossless", "Native output policy differs")
    require(sampling.guidance_scale == _config["expected_guidance_scale"] and extra.get("flow_shift") == 5 and extra.get("uncond_skip_layers") == [22],
            "Fixed CFG/shift/skip policy differs")
    require(extra.get("slg_start") == 0 and extra.get("slg_stop") == 1 and
            extra.get("slg_scale") == 0 and extra.get("slg_layers") == [], "Unadmitted conditional SLG")
    return {"prompt": prompt["prompt"], "negative_prompt": "", "seed": sampling.seed,
            "height": sampling.height, "width": sampling.width, "steps": sampling.num_inference_steps,
            "guidance_scale": sampling.guidance_scale, "flow_shift": extra["flow_shift"],
            "uncond_skip_layers": extra["uncond_skip_layers"], "max_sequence_length": sampling.max_sequence_length}


def forward(original, self, req):
    global _current, _branch, _cache_refs
    source_fact(original)
    require(len(req.requests) == 1, "Observer admits exactly one native request")
    sampling = req.sampling_params_list[0]
    nonce = (sampling.extra_args or {}).get(NONCE_KEY)
    if nonce is None and req.is_dummy_run():
        # Native dummy loading/profiling does not count as a served request.
        return original(self, req)
    nonce = nonce_value(nonce)
    require(_current is None, "Overlapping or incomplete request")
    require(not (Path(_config["proof_directory"]) / (nonce + ".json")).exists(), "Reused observation nonce")
    request = actual_request(sampling, req.prompts[0])
    import torch
    _current = {"status": "STARTED", "nonce": nonce, "native_request_id": req.request_id,
                "worker_pid": os.getpid(), "source_commit": SOURCE_COMMIT,
                "config_sha256": _config_sha, "observer_source_sha256": sha(__file__),
                "mode": _config["mode"], "route": "request_forward", "request": request_fact(request),
                "started_time_ns": time.time_ns(), "encodes": [], "first_forward": {},
                "branch_counts": {"positive": 0, "negative": 0}, "guidance_calls": 0,
                "scheduler_calls": 0, "vae_calls": 0, "text_refiner_forwards": {"positive": 0, "negative": 0},
                "trajectory": [], "cache_contract_verified": False, "fixed_negative_cache": _config["fixed_negative_cache"],
                "torch": {"version": torch.__version__, "git_version": torch.version.git_version,
                          "hip": torch.version.hip, "matmul_precision": torch.get_float32_matmul_precision(),
                          "cudnn_enabled": torch.backends.cudnn.enabled}}
    _cache_refs = {}
    def encoder_called(_module, _inputs):
        require(_branch in ("positive", "negative"), "Encoder outside registered branch")
        _current["encodes"][-1]["actual_encoder_forwards"] += 1
    def refiner_called(_module, _inputs):
        require(_branch in ("positive", "negative"), "Refiner outside actual transformer branch")
        _current["text_refiner_forwards"][_branch] += 1
    import compiled_dispatch
    dispatch_token = None
    handles = [self.text_encoder.register_forward_pre_hook(encoder_called),
               self.transformer.text_embedder.register_forward_pre_hook(refiner_called)]
    try:
        dispatch_token = compiled_dispatch.begin(self.transformer, regional=not _config["expected_worker_config"]["enforce_eager"], diagnostic=_config["mode"] == "diagnostic")
        result = original(self, req)
        _current["compile_dispatch"] = compiled_dispatch.finish(dispatch_token, completed=True)
        dispatch_token = None
        require(_current["text_refiner_forwards"] == {"positive": 1, "negative": 1}, "Unexpected conditioning recomputation")
        if _config["save_trajectory"]:
            _current["quality_bundle"] = quality_dump.finish_omni(
                _config["proof_directory"], nonce, _current["quality_inputs"], _current["trajectory"])
        _current["fixed_negative_cache_stats"] = negative_policy(self).stats() if _config["fixed_negative_cache"] else None
        _current["cache_contract_verified"] = True
        _current["status"] = "COMPLETE"
        _current["worker_source_files"] = dict(_verified_sources)
        validate_receipt(_current, nonce=nonce, request=request, config_sha=_config_sha,
                         observer_sha=sha(__file__), mode=_config["mode"], trajectories=_config["save_trajectory"])
        persist("COMPLETE")
        return result
    except BaseException as error:
        if dispatch_token is not None:
            _current["compile_dispatch"] = compiled_dispatch.finish(dispatch_token, completed=False)
        persist("FAILED", error)
        raise
    finally:
        if dispatch_token is not None:
            compiled_dispatch.restore(dispatch_token)
        for handle in handles:
            handle.remove()
        _current = None
        _branch = None
        _cache_refs = {}


def encode_prompt(original, self, prompt, max_sequence_length=4096):
    global _branch, _negative_fingerprint
    if _current is None:
        return original(self, prompt, max_sequence_length)
    source_fact(original)
    index = len(_current["encodes"])
    require(index in (0, 1) and _branch is None, "Unexpected text-encoder topology")
    branch = "positive" if index == 0 else "negative"
    values = [prompt] if isinstance(prompt, str) else prompt
    require(len(values) == 1 and isinstance(values[0], str), "Expected one actual encoded text")
    key = "prompt_sha256" if index == 0 else "negative_prompt_sha256"
    require(hashlib.sha256(values[0].encode()).hexdigest() == _current["request"][key], "Encoded text differs")
    require(max_sequence_length == _current["request"]["max_sequence_length"], "Encoder length policy differs")
    row = {"branch": branch, "actual_encoder_forwards": 0}
    _current["encodes"].append(row)
    _branch = branch
    try:
        result = original(self, prompt, max_sequence_length)
        row.update(embeds=tensor_fact(result[0]), mask=tensor_fact(result[1]))
        if _config["fixed_negative_cache"]:
            policy = negative_policy(self)
            require(policy is not None and policy.last_call is not None, "Missing actual fixednegative routing receipt")
            row["fixed_negative_cache_call"] = policy.last_call
            before, after = policy.last_call["native_cache_before"], policy.last_call["native_cache_after"]
            require(policy.last_call["branch"] == branch, "Cache branch order differs")
            if branch == "positive":
                require(row["actual_encoder_forwards"] == 1 and before == after, "Positive encoder reused or touchednegativecache")
            else:
                forwards = row["actual_encoder_forwards"]
                require(forwards in (0, 1) and after["hits"] - before["hits"] == 1 - forwards and after["misses"] - before["misses"] == forwards,
                        "Actual negative encoder/cache accounting differs")
                require(list(result[0].shape) == [1, 10, 4096] and str(result[0].dtype) == "torch.bfloat16" and list(result[1].shape) == [1, 10] and str(result[1].dtype) == "torch.bool",
                        "Native emptynegative shape/precision differs")
        else:
            require(row["actual_encoder_forwards"] == 1, "Text encoder was reused/skipped")
        if branch == "negative" and _config["mode"] == "diagnostic":
            fingerprint = (row["embeds"]["sha256"], row["mask"]["sha256"])
            require(_negative_fingerprint in (None, fingerprint), "Fixed emptynegative bytes changed withinworker")
            _negative_fingerprint = fingerprint
        return result
    finally:
        _branch = None


def guidance(original, self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, **kwargs):
    if _current is None:
        return original(self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, **kwargs)
    source_fact(original)
    require(do_true_cfg is True and true_cfg_scale == _current["request"]["guidance_scale"] == _config["expected_guidance_scale"], "Consumed CFG scalar differs")
    require(negative_kwargs is not None and positive_kwargs["hidden_states"] is negative_kwargs["hidden_states"],
            "CFG branches do not consume the same latent")
    require(kwargs["latents"] is positive_kwargs["hidden_states"], "SLG latent differs")
    slg = kwargs["slg_config"]
    require(list(slg.uncond_skip_layers) == [22] and not slg.cond_enabled, "Actual SLG branch policy differs")
    require(list(kwargs["active"]) == [True], "Negative skip layer not active for this step")
    if _current["guidance_calls"] == 0:
        require(len(_current["encodes"]) == 2, "Missing actual text encodes")
        _current["prepared"] = {
            "prompt_embeds": tensor_fact(positive_kwargs["encoder_hidden_states"]),
            "prompt_attention_mask": tensor_fact(positive_kwargs["encoder_attention_mask"]),
            "negative_prompt_embeds": tensor_fact(negative_kwargs["encoder_hidden_states"]),
            "negative_attention_mask": tensor_fact(negative_kwargs["encoder_attention_mask"]),
            "latents": tensor_fact(positive_kwargs["hidden_states"]),
            "timesteps": tensor_fact(self.scheduler.timesteps), "sigmas": tensor_fact(self.scheduler.sigmas)}
        if _config["save_trajectory"]:
            _current["quality_inputs"] = quality_dump.capture_inputs(
                _config["proof_directory"], _current["nonce"],
                {"initial_latent": positive_kwargs["hidden_states"],
                 "positive_embedding": positive_kwargs["encoder_hidden_states"],
                 "negative_embedding": negative_kwargs["encoder_hidden_states"],
                 "positive_attention_mask": positive_kwargs["encoder_attention_mask"],
                 "negative_attention_mask": negative_kwargs["encoder_attention_mask"]}, _current["prepared"])
        _current["scheduler_values"] = {"timesteps": self.scheduler.timesteps.detach().cpu().tolist(),
                                         "sigmas": self.scheduler.sigmas.detach().cpu().tolist()}
        if _config["mode"] == "diagnostic":
            for index, prefix in ((0, "prompt"), (1, "negative")):
                embed = "prompt_embeds" if index == 0 else "negative_prompt_embeds"
                mask = "prompt_attention_mask" if index == 0 else "negative_attention_mask"
                require(_current["encodes"][index]["embeds"]["sha256"] == _current["prepared"][embed]["sha256"] and
                        _current["encodes"][index]["mask"]["sha256"] == _current["prepared"][mask]["sha256"],
                        "Consumed conditioning differs from actual encoder result")
    result = original(self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, **kwargs)
    require(str(result.dtype) == "torch.float32", "CFG did not produce FP32")
    _current["guidance_calls"] += 1
    return result


def transformer(original, self, *args, **kwargs):
    global _branch, _negative_fingerprint
    if _current is None:
        return original(self, *args, **kwargs)
    source_fact(original)
    names = ("hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask",
             "return_dict", "_conditioning_cache", "skip_layers")
    bound = {**dict(zip(names, args)), **kwargs}
    layers = list(bound.get("skip_layers", ()))
    require(layers in ([], [22]) and _branch is None, "Unexpected actual transformer branch")
    branch = "negative" if layers else "positive"
    cache = bound.get("_conditioning_cache")
    require(isinstance(cache, dict), "Expected native within-request conditioning dictionary")
    index = _current["branch_counts"][branch]
    if index == 0:
        require(not cache, "Cross-request refined conditioning reuse is forbidden")
        require(all(cache is not other for other in _cache_refs.values()), "CFG branches share a conditioning cache")
        _cache_refs[branch] = cache
        _current["first_forward"][branch] = {
            "hidden_states": tensor_fact(bound["hidden_states"]),
            "encoder_hidden_states": tensor_fact(bound["encoder_hidden_states"]),
            "encoder_attention_mask": tensor_fact(bound["encoder_attention_mask"]),
            "skip_layers": layers, "cache_empty_at_first_forward": True}
    else:
        require(cache is _cache_refs[branch] and set(cache) == {"conditioning"}, "Native request-local cache changed")
    _branch = branch
    try:
        result = original(self, *args, **kwargs)
        require(set(cache) == {"conditioning"}, "Native conditioning cache was not populated")
        _current["branch_counts"][branch] += 1
        return result
    finally:
        _branch = None


def scheduler_step(original, self, noise_pred, timestep, latents, per_request_scheduler=None, generator=None):
    if _current is None:
        return original(self, noise_pred, timestep, latents, per_request_scheduler, generator)
    source_fact(original)
    index = _current["scheduler_calls"]
    require(per_request_scheduler is None, "Unexpected stepwise scheduler route")
    require(_current["branch_counts"] == {"positive": index + 1, "negative": index + 1} and
            _current["guidance_calls"] == index + 1, "CFG/Euler dispatch order differs")
    require(str(noise_pred.dtype) == str(latents.dtype) == "torch.float32", "Euler inputs are not FP32")
    result = original(self, noise_pred, timestep, latents, per_request_scheduler, generator)
    require(str(result.dtype) == "torch.float32", "Euler output is not FP32")
    if _config["save_trajectory"]:
        import numpy as np
        directory = Path(_config["proof_directory"]) / _current["nonce"]
        directory.mkdir(exist_ok=True)
        path = directory / ("state-%03d.npy" % index)
        array = result.detach().contiguous().cpu().numpy()
        with path.open("xb") as f:
            np.save(f, array, allow_pickle=False)
        fact = tensor_fact(result)
        fact.update(index=index, path=path.relative_to(Path(_config["proof_directory"])).as_posix(), file_sha256=sha(path))
        _current["trajectory"].append(fact)
    _current["scheduler_calls"] += 1
    return result


def vae_decode(original, self, latents):
    if _current is None:
        return original(self, latents)
    source_fact(original)
    require(_current["scheduler_calls"] == _current["request"]["steps"], "Native denoising ended early")
    require(str(self.vae.dtype) == str(latents.dtype) == "torch.float32", "Native VAE precision differs")
    _current["vae_input"] = tensor_fact(latents)
    result = original(self, latents)
    _current["vae_output"] = tensor_fact(result)
    require(str(result.dtype) == "torch.float32", "Native decoded output is not FP32")
    _current["vae_calls"] += 1
    return result


def reject_stepwise(original, self, *args, **kwargs):
    raise RuntimeError("Stepwise MOSS is outside this request-mode observer admission; no silent fallback")


def negative_policy(pipeline):
    return getattr(pipeline, "_ik_fixed_empty_negative_policy", None)


def install_fixed_negative_policy(cls):
    if not _config["fixed_negative_cache"]:
        return
    from fixed_empty_negative import FixedEmptyNegativePolicy
    from vllm_omni.diffusion.cache.prompt_embed_cache import install_prompt_embed_cache
    source_fact(install_prompt_embed_cache)
    native_forward, native_encode = cls.forward, cls.encode_prompt
    source_fact(native_forward)
    source_fact(native_encode)
    def policy(pipeline):
        current = negative_policy(pipeline)
        if current is None:
            current = FixedEmptyNegativePolicy(native_encode.__get__(pipeline, type(pipeline)), model_tag=canonical({
                "config_sha256": _config_sha, "native_commit": SOURCE_COMMIT, "model_revision": _config["model_revision"],
                "pinned_model_api_sha256": _config["pinned_model_api_sha256"]}))
            pipeline._ik_fixed_empty_negative_policy = current
        return current
    @functools.wraps(native_encode)
    def cached_encode(self, prompt, max_sequence_length=4096):
        return policy(self).encode(prompt, max_sequence_length)
    @functools.wraps(native_forward)
    def cached_forward(self, req):
        require(len(req.requests) == 1, "Fixed negative policy admits one request")
        sampling = req.sampling_params_list[0]
        nonce = (sampling.extra_args or {}).get(NONCE_KEY)
        if nonce is None and req.is_dummy_run():
            return native_forward(self, req)
        nonce_value(nonce)
        request = actual_request(sampling, req.prompts[0])
        with policy(self).request(request["prompt"]):
            return native_forward(self, req)
    cls.encode_prompt, cls.forward = cached_encode, cached_forward


def configure(path, od_config):
    global _config, _config_sha
    require(_config is None, "One observed native worker per process")
    _config_sha = sha(path)
    _config = json.loads(path.read_text())
    require(_config["source_commit"] == SOURCE_COMMIT and _config["mode"] in ("off", "diagnostic", "lightweight"),
            "Unexpected observer configuration")
    require(not _config["save_trajectory"] or _config["mode"] == "diagnostic", "Trajectory collection requires diagnostic mode")
    for name, digest in _config["adapter_sources"].items():
        require(sha(Path(__file__).with_name(name)) == digest, "Worker adapter source differs: " + name)
    root = Path(_config["native_source_root"]).resolve()
    for name, digest in _config["native_sources"].items():
        require(sha(root / name) == digest, "Worker pinned source differs: " + name)
    require_expected_mode(od_config, _config["expected_worker_config"])
    from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline
    from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageTransformer
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    source_fact(DiffusionModelRunner.__init__)
    source_fact(MOSSImagePipeline.forward)
    source_fact(MOSSImageTransformer.forward)
    install_fixed_negative_policy(MOSSImagePipeline)
    if _config["mode"] == "off":
        return
    def install(cls, name, callback):
        original = getattr(cls, name)
        source_fact(original)
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return callback(original, *args, **kwargs)
        setattr(cls, name, wrapper)
    for name, callback in (("forward", forward), ("encode_prompt", encode_prompt),
                           ("predict_noise_with_slg", guidance), ("scheduler_step", scheduler_step),
                           ("_vae_decode", vae_decode), ("prepare_encode", reject_stepwise)):
        install(MOSSImagePipeline, name, callback)
    install(MOSSImageTransformer, "forward", transformer)

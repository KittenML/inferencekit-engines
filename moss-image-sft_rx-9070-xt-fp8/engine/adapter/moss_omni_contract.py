"""Pure-stdlib contracts for a single-request native Omni observer."""
import hashlib
import json
import re
from pathlib import Path

SOURCE_COMMIT = "7cbc776c16a39f73978ae5ce400729d94e0fda35"
CONFIG_ENV = "INFERENCEKIT_MOSS_OMNI_CONFIG"
NONCE_KEY = "inferencekit_observation_nonce"
RUNNER = "moss_omni_runner.ObservedMOSSRunner"
PIPELINE = "vllm_omni/diffusion/models/mossimage/pipeline_mossimage.py"
TRANSFORMER = "vllm_omni/diffusion/models/mossimage/mossimage_transformer.py"


def require(condition, explanation):
    if not condition:
        raise RuntimeError(explanation)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonce_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{32}", value), "Invalid request nonce")
    return value


def request_fact(request):
    return {"prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(request["negative_prompt"].encode()).hexdigest(),
            **{name: request[name] for name in ("seed", "width", "height", "steps", "guidance_scale",
                                               "flow_shift", "uncond_skip_layers", "max_sequence_length")}}


def validate_receipt(receipt, *, nonce, request, config_sha, observer_sha, mode, trajectories):
    require(receipt.get("status") == "COMPLETE", "Worker observation incomplete")
    require(receipt.get("nonce") == nonce_value(nonce), "Worker nonce differs")
    require(receipt.get("source_commit") == SOURCE_COMMIT, "Native source pin differs")
    require(receipt.get("config_sha256") == config_sha, "Worker configuration differs")
    require(receipt.get("observer_source_sha256") == observer_sha, "Worker observer source differs")
    require(receipt.get("mode") == mode and mode in ("diagnostic", "lightweight"), "Invalid observer mode")
    require(type(receipt.get("worker_pid")) is int and receipt["worker_pid"] > 0, "Worker PID missing")
    require(receipt.get("route") == "request_forward", "Unadmitted native execution route")
    require(receipt.get("request") == request_fact(request), "Consumed worker request differs")
    steps = request["steps"]
    require(receipt.get("branch_counts") == {"positive": steps, "negative": steps}, "Actual transformer dispatch differs")
    require(receipt.get("guidance_calls") == steps and receipt.get("scheduler_calls") == steps,
            "Incomplete actual CFG/Euler execution")
    require(receipt.get("vae_calls") == 1, "Expected exactly one native VAE decode")
    require(receipt.get("cache_contract_verified") is True, "Within-request conditioning cache unproven")
    encodes = receipt.get("encodes", [])
    require([row.get("branch") for row in encodes] == ["positive", "negative"], "Encoding order differs")
    if receipt.get("fixed_negative_cache", False):
        for index, row in enumerate(encodes):
            call = row.get("fixed_negative_cache_call", {})
            require(call.get("branch") == ("positive" if index == 0 else "negative"), "Negative policy branch receipt missing")
            before, after = call.get("native_cache_before", {}), call.get("native_cache_after", {})
            forwards = row.get("actual_encoder_forwards")
            if index == 0:
                require(forwards == 1 and before == after, "Positive conditioning reused or touchednegativecache")
            else:
                require(forwards in (0, 1) and before.get("max_size") == after.get("max_size") == after.get("size") == 1 and
                        after.get("hits", 0) - before.get("hits", 0) == 1 - forwards and
                        after.get("misses", 0) - before.get("misses", 0) == forwards, "Negative cache hit/miss differs from actualencoder")
    else:
        require(all(row.get("actual_encoder_forwards") == 1 for row in encodes), "Text conditioning reused/skipped")
    require(receipt.get("text_refiner_forwards") == {"positive": 1, "negative": 1}, "Native refinement cache differs")
    prepared = receipt.get("prepared", {})
    require(set(prepared) == {"prompt_embeds", "prompt_attention_mask", "negative_prompt_embeds",
                              "negative_attention_mask", "latents", "timesteps", "sigmas"}, "Consumed input facts missing")
    require(set(receipt.get("first_forward", {})) == {"positive", "negative"}, "Actual model input boundary missing")
    latent_shape = [1, 128, request["height"] // 16, request["width"] // 16]
    require(prepared["latents"].get("dtype") == "torch.float32" and prepared["latents"].get("shape") == latent_shape,
            "Initial latent shape/precision differs")
    require(prepared["timesteps"].get("dtype") == "torch.float32" and prepared["timesteps"].get("shape") == [steps] and
            prepared["sigmas"].get("dtype") == "torch.float32" and prepared["sigmas"].get("shape") == [steps + 1],
            "Consumed scheduler shape/precision differs")
    for branch, embed_key, mask_key in (("positive", "prompt_embeds", "prompt_attention_mask"),
                                      ("negative", "negative_prompt_embeds", "negative_attention_mask")):
        embed, mask = prepared[embed_key], prepared[mask_key]
        shape = embed.get("shape", [])
        require(len(shape) == 3 and shape[0] == 1 and shape[2] == 4096 and type(shape[1]) is int and 1 <= shape[1] <= 4096
                and embed.get("dtype") == "torch.bfloat16", "Consumed text embedding shape/precision differs")
        require(mask.get("dtype") == "torch.bool" and mask.get("shape") == shape[:2], "Consumed text mask differs")
        first = receipt["first_forward"][branch]
        require(first.get("skip_layers") == ([] if branch == "positive" else [22]) and
                first.get("cache_empty_at_first_forward") is True, "Actual branch/cache admission differs")
        if mode == "diagnostic":
            require(all(first.get(first_key, {}).get("sha256") == prepared[prepared_key].get("sha256")
                        for first_key, prepared_key in (("hidden_states", "latents"), ("encoder_hidden_states", embed_key),
                                                       ("encoder_attention_mask", mask_key))), "Actual transformer inputs differ from prepared inputs")
    if mode == "diagnostic":
        require(all(re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", ""))) for row in prepared.values()),
                "Diagnostic consumed-tensor hashes missing")
    if receipt.get("fixed_negative_cache", False):
        require(prepared["negative_prompt_embeds"].get("shape") == [1,10,4096] and
                prepared["negative_attention_mask"].get("shape") == [1,10], "Fixedemptynegative shape differs")
    observed_sources = receipt.get("worker_source_files", {})
    require(all(name in observed_sources for name in (PIPELINE, TRANSFORMER,
                "vllm_omni/diffusion/guidance/skip_layer_guidance.py")), "Actual callable source proof missing")
    states = receipt.get("trajectory", [])
    require(len(states) == (steps if trajectories else 0), "Trajectory count differs")
    if trajectories:
        require([row.get("index") for row in states] == list(range(steps)), "Trajectory order differs")
        require(all(row.get("dtype") == "torch.float32" and row.get("shape") ==
                    latent_shape for row in states),
                "Trajectory shape/dtype differs")
    return receipt

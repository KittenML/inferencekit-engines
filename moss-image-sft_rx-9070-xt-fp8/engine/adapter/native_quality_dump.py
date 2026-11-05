"""Diagnostic-only exact tensor exports; no model imports at module import time.

These files are evidence, never model inputs or timing results. Each request
retains its own initial state and all 50 native post-Euler states. CPU copies do
not share storage with live inference tensors. No casting of BF16 embeddings.
"""
import hashlib
import json
import re
from pathlib import Path

INPUT_NAMES = {
    "initial_latent": "latents",
    "positive_embedding": "prompt_embeds",
    "negative_embedding": "negative_prompt_embeds",
    "positive_attention_mask": "prompt_attention_mask",
    "negative_attention_mask": "negative_attention_mask",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory(root, nonce, *, create=False):
    require(isinstance(nonce, str) and re.fullmatch(r"[a-f0-9]{32}", nonce), "Invalid dump nonce")
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and root.resolve() == root, "Dump root must be an existing absolute nonsymlink directory")
    target = root / nonce
    if create:
        target.mkdir(exist_ok=False)
    require(target.is_dir() and not target.is_symlink(), "Dump request directory missing or symlinked")
    return target


def cpu_copy(value):
    import torch
    require(isinstance(value, torch.Tensor), "Expected an actual native tensor")
    require(str(value.dtype) in ("torch.float32", "torch.bfloat16", "torch.bool"), "Unexpected diagnostic tensor dtype")
    require(0 < value.numel() * value.element_size() <= 128 * 1024 * 1024, "Unexpected diagnostic tensor size")
    result = value.detach().to("cpu").contiguous().clone()
    require(bool(torch.isfinite(result).all().item()), "Nonfinite diagnostic tensor")
    return result


def fact(value):
    import torch
    require(str(value.device) == "cpu", "Evidence hashing requires a CPU tensor")
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def validate_input_facts(facts, observed):
    require(set(facts) == set(INPUT_NAMES), "Incomplete native input dump")
    for name, source in INPUT_NAMES.items():
        require(source in observed, "Missing observed native input: " + source)
        require(all(facts[name].get(key) == observed[source].get(key)
                    for key in ("shape", "dtype", "bytes", "sha256")),
                "Saved input differs from actual consumed tensor: " + name)
    require(facts["initial_latent"]["shape"] == [1, 128, 64, 64]
            and facts["initial_latent"]["dtype"] == "torch.float32", "Unregistered latent shape/dtype")
    for role in ("positive", "negative"):
        embed, mask = facts[role + "_embedding"], facts[role + "_attention_mask"]
        require(embed["dtype"] == "torch.bfloat16" and mask["dtype"] == "torch.bool"
                and embed["shape"][:1] == [1] and embed["shape"][-1:] == [4096]
                and mask["shape"] == embed["shape"][:-1], "Conditioning shape/dtype mismatch")


def save_new(path, tensors, metadata):
    from safetensors.torch import save_file
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not path.is_symlink(), "Refusing to replace existing tensor evidence")
    # Exclusive creation prevents accidental reuse even after a failed export.
    with temporary.open("xb"):
        pass
    save_file(tensors, str(temporary), metadata={key: str(value) for key, value in metadata.items()})
    temporary.rename(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def capture_inputs(root, nonce, tensors, observed):
    require(set(tensors) == set(INPUT_NAMES), "Missing native input tensor")
    target = directory(root, nonce, create=True)
    copies = {name: cpu_copy(value) for name, value in tensors.items()}
    facts = {name: fact(value) for name, value in copies.items()}
    validate_input_facts(facts, observed)
    record = save_new(target / "inputs.safetensors", copies,
                      {"status": "INPUTS_ONLY_NOT_COMPLETE", "nonce": nonce})
    record.update(tensors=facts, helper_sha256=sha(__file__), diagnostic_only=True)
    return record


def load_inputs(root, nonce, record):
    from safetensors.torch import load_file
    path = directory(root, nonce) / "inputs.safetensors"
    require(record.get("path") == path.name and record.get("sha256") == sha(path)
            and record.get("bytes") == path.stat().st_size, "Initial input evidence changed")
    tensors = load_file(str(path), device="cpu")
    require(set(tensors) == set(INPUT_NAMES), "Input evidence keys changed")
    require({name: fact(value) for name, value in tensors.items()} == record["tensors"], "Input bytes changed")
    return tensors


def validate_states(states, initial_shape):
    require(len(states) == 50, "Expected exactly 50 native post-Euler states")
    for index, row in enumerate(states):
        require(row.get("index") == index and row.get("shape") == initial_shape
                and row.get("dtype") == "torch.float32", "Missing, reordered or incompatible Euler state")
        require(isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64, "Missing Euler state hash")


def complete(root, nonce, inputs, states, *, source, extra=None):
    require(len(states) == 50, "Expected exactly 50 native post-Euler tensors")
    target = directory(root, nonce)
    tensors = load_inputs(root, nonce, inputs)
    rows = []
    for index, value in enumerate(states):
        copied = cpu_copy(value)
        row = dict(fact(copied), index=index)
        rows.append(row)
        tensors[f"latent_{index + 1:03d}"] = copied
    validate_states(rows, list(tensors["initial_latent"].shape))
    record = save_new(target / "trajectory.safetensors", tensors,
                      {"status": "COMPLETE", "nonce": nonce, "source": source,
                       "initial_is_separate_from_post_euler": "true", "diagnostic_only": "true"})
    record.update(inputs_sha256=inputs["sha256"], states=rows, source=source,
                  helper_sha256=sha(__file__), diagnostic_only=True,
                  tensor_keys=sorted(tensors), extra=extra)
    # Equal consecutive states are retained and reported, never replaced.
    sequence = [inputs["tensors"]["initial_latent"]["sha256"]] + [row["sha256"] for row in rows]
    record["consecutive_equal_states"] = [i for i in range(1, len(sequence)) if sequence[i] == sequence[i - 1]]
    with (target / "trajectory-manifest.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    record["path"] = nonce + "/trajectory.safetensors"
    return record


def finish_sglang(root, nonce, inputs, trajectory, timesteps, final_latent):
    require(list(trajectory.shape) == [1, 50, 128, 64, 64]
            and str(trajectory.dtype) == "torch.float32", "Native SGLang trajectory shape/dtype differs")
    require(list(timesteps.shape) == [50], "Native timestep count differs")
    last = fact(cpu_copy(trajectory[:, -1]))
    require(last == fact(cpu_copy(final_latent)), "Final native trajectory does not match denoising output")
    return complete(root, nonce, inputs, [trajectory[:, i] for i in range(50)],
                    source="native SGLang return_trajectory_latents; post-Euler; single worker",
                    extra={"trajectory_timesteps": timesteps.detach().cpu().tolist(),
                           "final_latent_equal": True})


def finish_omni(root, nonce, inputs, trajectory):
    import numpy as np
    import torch
    validate_states(trajectory, inputs["tensors"]["initial_latent"]["shape"])
    target = directory(root, nonce)
    states = []
    for index, row in enumerate(trajectory):
        path = target / ("state-%03d.npy" % index)
        require(row.get("path") == nonce + "/" + path.name and not path.is_symlink()
                and sha(path) == row.get("file_sha256"), "Native Omni state file changed/reordered")
        require(path.stat().st_size < 3 * 1024 * 1024, "Native state file exceeds registered bound")
        value = torch.from_numpy(np.load(path, allow_pickle=False))
        actual = fact(value)
        require(all(actual[key] == row.get(key) for key in ("shape", "dtype", "bytes", "sha256")), "Native state tensor bytes changed")
        states.append(value)
    return complete(root, nonce, inputs, states, source="native Omni scheduler post-Euler NPY observations")

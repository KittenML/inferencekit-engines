"""Create an immutable task configuration; no native imports or hardware use."""
import argparse
import json
from pathlib import Path
from moss_omni_contract import RUNNER, SOURCE_COMMIT, require, sha
from ranking_policy import selection
from scale_audit import POLICY

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source-root", required=True)
p.add_argument("--fp8-overlay-root", required=True)
p.add_argument("--model-dir", required=True)
p.add_argument("--run-directory", required=True)
p.add_argument("--transport-file", required=True)
p.add_argument("--mode", choices=["off", "diagnostic", "lightweight"], required=True)
p.add_argument("--save-trajectory", action="store_true")
p.add_argument("--fixed-negative", action="store_true")
p.add_argument("--deadline-utc", required=True)
p.add_argument("--compile-policy", choices=["eager", "regional"], required=True)
p.add_argument("--resident-layers", type=int, choices=[36, 38, 40, 42], required=True)
p.add_argument("--pin-memory", choices=["false", "true"], required=True)
p.add_argument("--pipeline-profiler", choices=["false", "true"], required=True)
p.add_argument("--guidance-scale", type=float, choices=[3.0, 3.5, 4.0], default=3.0)
a = p.parse_args()
selected = selection(a.compile_policy, a.resident_layers, a.pin_memory == "true", a.pipeline_profiler == "true")
require(not a.save_trajectory or a.mode == "diagnostic", "Trajectory requires diagnostic mode")
root = Path(__file__).resolve().parent
run = Path(a.run_directory).resolve()
run.mkdir(parents=True, exist_ok=False)
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
require(manifest["commit"] == SOURCE_COMMIT, "Source audit pin differs")
offload = {"mode": "layer", "components": ["dit", "text_encoder"],
           "layer_options": {"text_encoder": {"weight_transfer": "rank-local"},
                             "dit": {"weight_transfer": "rank-local", "resident_layers": selected["resident_layers"]}}, "pin_memory": selected["pin_memory"]}
parallel = {name: 1 for name in ("tensor_parallel_size", "ulysses_degree", "ring_degree", "cfg_parallel_size", "vae_patch_parallel_size")}
engine_args = {"quantization_config": {"transformer": {"method": "fp8", "activation_scheme": "dynamic"}, "default": None}, "model": str(Path(a.model_dir).resolve()), "model_class_name": "MOSSImagePipeline", "dtype": "torch.bfloat16",
               "enforce_eager": selected["enforce_eager"], "diffusion_compile_granularity": selected["diffusion_compile_granularity"],
               "diffusion_compile_dynamic": selected["diffusion_compile_dynamic"], "diffusion_attention_backend": "TORCH_SDPA", **parallel,
               "max_num_seqs": 1, "cache_backend": None, "enable_prompt_embed_cache": False,
               "diffusion_offload_config": offload, "vae_use_tiling": False, "vae_use_slicing": False,
               "enable_diffusion_pipeline_profiler": selected["enable_diffusion_pipeline_profiler"], "log_stats": True, "num_weight_load_threads": 1,
               "step_execution": False, "diffusion_model_runner_cls": RUNNER}
expected = {"step_execution": False, "enforce_eager": selected["enforce_eager"],
            "diffusion_compile_granularity": selected["diffusion_compile_granularity"],
            "diffusion_compile_dynamic": selected["diffusion_compile_dynamic"],
            "enable_diffusion_pipeline_profiler": selected["enable_diffusion_pipeline_profiler"], "cache_backend": "none", "enable_prompt_embed_cache": False,
            "vae_use_tiling": False, "vae_use_slicing": False, "diffusion_model_runner_cls": RUNNER,
            "diffusion_offload_config": offload, "dtype": "torch.bfloat16", "model": engine_args["model"], "parallel": parallel}
cfg = {"source_commit": SOURCE_COMMIT, "candidate_patch_sha256": manifest["candidate_patch_sha256"], "precision_policy": "main_dit_online_fp8_dynamic_bf16_input", "fp8_scale_policy": POLICY, "fp8_overlay_root": str(Path(a.fp8_overlay_root).resolve()), "fp8_scale_patch_sha256": "95aee4ac3b9aca81327777619f93119e18c299a70c9e4e69fa074561102a78ac", "fp8_scale_source_sha256": "8bcc9a37d448ed271bc5994b1211876a16d082fa1f3fdc84ea95d99465a6776d", "mode": a.mode, "save_trajectory": a.save_trajectory,
       "fixed_negative_cache": a.fixed_negative, "expected_guidance_scale": a.guidance_scale, "explicit_ranking_frontend_selection": selected, "model_revision": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
       "pinned_model_api_sha256": sha(root / "pinned-model-api.json"),
       "native_source_root": str(Path(a.source_root).resolve()),
       "native_sources": {r["path"]: r["sha256"] for r in manifest["files"]},
       "adapter_sources": {name: sha(root / name) for name in (
           "moss_omni_adapter.py", "moss_omni_observer.py", "moss_omni_contract.py", "moss_omni_runner.py", "fixed_empty_negative.py", "native_quality_dump.py", "ranking_policy.py", "compiled_dispatch.py", "audit_fp8.py", "shard_observer.py", "scale_audit.py")},
       "common_transport_sha256": sha(a.transport_file), "deadline_utc": a.deadline_utc,
       "proof_directory": str(run / "worker-proofs"), "engine_args": engine_args, "expected_worker_config": expected}
target = run / "configuration.json"
with target.open("x") as f:
    json.dump(cfg, f, indent=2, allow_nan=False)
    f.write("\n")
print(json.dumps({"configuration": str(target), "sha256": sha(target), "status": "PREPARED_NOT_EXECUTED"}))

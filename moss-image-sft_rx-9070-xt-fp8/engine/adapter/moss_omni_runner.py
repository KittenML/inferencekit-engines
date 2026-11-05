"""Native extension point; import this module only on the remote GPU worker.

All execution methods remain inherited from the pinned native runner. This
subclass installs observational hooks, verifies resolved settings, and reports
readiness. Native dummy profiling stays unobserved and is not a served request.
"""
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

from moss_omni_contract import CONFIG_ENV, RUNNER, canonical, require, sha
import moss_omni_observer as observer


def distribution_versions():
    result = {}
    for name in ("vllm", "vllm-omni", "torch", "triton", "pytorch-triton-rocm", "diffusers", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class ObservedMOSSRunner(DiffusionModelRunner):
    def __init__(self, vllm_config, od_config, device):
        observer.configure(Path(os.environ[CONFIG_ENV]), od_config)
        super().__init__(vllm_config=vllm_config, od_config=od_config, device=device)

    def load_model(self, *args, **kwargs):
        import functools
        from shard_observer import ShardTracker
        from scale_audit import ScaleAudit, install_model_reload_guard, validate_receipt as validate_scale_receipt, validate_restored_scales
        import vllm.model_executor.model_loader.reload.layerwise as layerwise_module
        import vllm.model_executor.model_loader.reload.meta as meta_module
        import vllm.model_executor.model_loader.utils as loader_utils_module
        import vllm.model_executor.parameter as parameter_module
        import vllm.model_executor.layers.linear as linear_module
        import vllm.model_executor.layers.quantization.online.fp8 as online_fp8_module
        pinned_loader_files = {
            "layerwise": (layerwise_module.__file__, "0d395b3889a3d5e8f34a7f7849edcc80821d0de4bc8146774fdc0763d5170770"),
            "meta": (meta_module.__file__, "749d03cc94639c6e0133c1f0e326eaec2b43ee7d8fe3288963982bf50a1aeed4"),
            "loader_utils": (loader_utils_module.__file__, "3091a9f64cbe42ad0e7fc0907fbbcb9197d2d56d5da4008779c5a16d9b8649a6"),
            "parameter": (parameter_module.__file__, "1a4bbd7400fd1ba79e8e1e8d19666cd22881ec83c35e6e01b3fdc78faee23171"),
            "linear": (linear_module.__file__, "7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0"),
            "online_fp8": (online_fp8_module.__file__, "8bcc9a37d448ed271bc5994b1211876a16d082fa1f3fdc84ea95d99465a6776d"),
        }
        loader_sources = {name: {"file": str(Path(path).resolve()), "sha256": sha(path)}
                          for name, (path, _) in pinned_loader_files.items()}
        require(all(loader_sources[name]["sha256"] == expected for name, (_, expected) in pinned_loader_files.items()),
                "Installed vLLM loader source differs from reviewed implementation")
        cfg_at_load = observer.configuration()
        require(Path(online_fp8_module.__file__).resolve() == Path(cfg_at_load["fp8_overlay_root"]).resolve() /
                "vllm/model_executor/layers/quantization/online/fp8.py",
                "FP8 method did not import from the isolated candidate overlay")
        from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageTransformer
        original_load_weights = MOSSImageTransformer.load_weights
        trackers = []
        scale_auditors = []
        @functools.wraps(original_load_weights)
        def observed_load_weights(transformer, weights):
            require(not trackers, "Expected exactly one transformer weight loading call")
            tracker = ShardTracker(transformer)
            trackers.append(tracker)
            scale_auditors.append(ScaleAudit(transformer, tracker, online_fp8_module))
            loaded = original_load_weights(transformer, weights)
            tracker.require_shards()
            return loaded
        MOSSImageTransformer.load_weights = observed_load_weights
        try:
            result = super().load_model(*args, **kwargs)
            require(len(trackers) == 1, "Actual transformer shard loading was not observed")
            shard_receipt = trackers[0].receipt()
            require(len(scale_auditors) == 1, "Expected exactly one scale auditor")
            scale_receipt = validate_scale_receipt(scale_auditors[0].receipt())
        finally:
            MOSSImageTransformer.load_weights = original_load_weights
            for scale_auditor in scale_auditors:
                scale_auditor.restore()
            for tracker in trackers:
                tracker.restore()
        scale_auditors[0].install_reload_guard()
        # This disposable candidate process supports only the initial checkpoint
        # load. Reject a second model-wide load before any weight can mutate.
        install_model_reload_guard(MOSSImageTransformer)
        from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline
        require(type(self.pipeline) is MOSSImagePipeline, "Actual native pipeline class differs")
        require(self.cache_backend is None and self.prompt_embed_cache is None, "Cross-request caches not admitted")
        cfg = observer.configuration()
        od = self.od_config
        actual = {name: getattr(od, name) for name in (
            "step_execution", "enforce_eager", "diffusion_compile_granularity", "diffusion_compile_dynamic",
            "enable_diffusion_pipeline_profiler", "cache_backend", "enable_prompt_embed_cache",
            "vae_use_tiling", "vae_use_slicing", "diffusion_model_runner_cls", "diffusion_offload_config")}
        actual["dtype"] = str(od.dtype)
        actual["model"] = str(Path(od.model).resolve())
        parallel = od.parallel_config
        actual["parallel"] = {name: getattr(parallel, name) for name in (
            "tensor_parallel_size", "ulysses_degree", "ring_degree", "cfg_parallel_size", "vae_patch_parallel_size")}
        # Check the effective configuration, not merely parent kwargs.
        require(canonical(actual) == canonical(cfg["expected_worker_config"]), "Resolved worker configuration differs")
        precision = {name: str(next(getattr(self.pipeline, name).parameters()).dtype)
                     for name in ("transformer", "text_encoder", "vae")}
        require(precision == {"transformer": "torch.bfloat16", "text_encoder": "torch.bfloat16", "vae": "torch.float32"},
                "Actual loaded component precision differs")
        from audit_fp8 import audit_transformer
        from vllm_omni.quantization.component_config import ComponentQuantizationConfig, resolve_component_quant_config
        from vllm_omni.quantization.fp8_config import DiffusionFp8Config
        qc = od.quantization_config
        require(isinstance(qc, ComponentQuantizationConfig), "Actual quantization is not component scoped")
        selected_qc = resolve_component_quant_config(qc, "transformer")
        require(type(selected_qc) is DiffusionFp8Config and selected_qc.activation_scheme == "dynamic"
                and not selected_qc.is_checkpoint_fp8_serialized
                and resolve_component_quant_config(qc, "text_encoder") is None
                and resolve_component_quant_config(qc, "vae") is None,
                "Resolved quantization policy differs")
        import inspect
        import vllm.envs as vllm_envs
        require(not vllm_envs.VLLM_BATCH_INVARIANT, "Batch-invariant mode may dequantize FP8 weights to BF16")
        from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
        actual_fp8_method_source = Path(inspect.getsourcefile(Fp8PerTensorOnlineLinearMethod)).resolve()
        fp8_method_identity = {"file": str(actual_fp8_method_source), "sha256": sha(actual_fp8_method_source),
                               "batch_invariant": bool(vllm_envs.VLLM_BATCH_INVARIANT)}
        require(all(type(block.proj[name].quant_method) is Fp8PerTensorOnlineLinearMethod
                    and block.proj[name].quant_method.requantization_source is None
                    for block in self.pipeline.transformer.layers
                    for name in ("qkv", "o", "ffn_up_gate", "ffn_down")),
                "Loaded FP8 methods do not match imported vLLM class")
        # The DLO resident group stores zero-sized placeholders between stages.
        # Audit the actual restored device views and always restore the original
        # offloaded state before reporting readiness.
        controller = getattr(self.pipeline, "_dlo_residency_controller", None)
        require(controller is not None and selected_qc is not None, "Expected stage-scoped DiT offload")
        controller.load_resident_layers()
        try:
            fp8_audit = audit_transformer(self.pipeline.transformer)
        finally:
            controller.offload_resident_layers()
        validate_restored_scales(scale_receipt, fp8_audit)
        scale_receipt["initial_offload_reload_scales_equal"] = True
        require(od.diffusion_model_runner_cls == RUNNER, "Native extension point not selected")
        from vllm_omni.diffusion.attention.layer import Attention
        attention = []
        for name, module in self.pipeline.transformer.named_modules():
            if isinstance(module, Attention):
                backend = module.attn_backend.get_name() if module.attn_backend else None
                implementation = type(module.attention)
                require(backend == "TORCH_SDPA" or backend == "SDPA", "Actual attention backend differs")
                observer.source_fact(implementation.forward)
                attention.append({"module": name, "backend": backend,
                                  "implementation": implementation.__module__ + "." + implementation.__name__})
        require(len(attention) == 44, "Expected 42 main and two text-refiner attention modules")
        import torch
        properties = torch.cuda.get_device_properties(self.device)
        actual_hardware = {"gpu_name": properties.name, "gfx": str(getattr(properties, "gcnArchName", "")),
                           "torch_total_memory_bytes": properties.total_memory, "visible_device_count": torch.cuda.device_count()}
        require("RX 9070 XT" in actual_hardware["gpu_name"] and "gfx1201" in actual_hardware["gfx"] and
                actual_hardware["visible_device_count"] == 1 and 15 * 2**30 <= properties.total_memory <= 17 * 2**30,
                "Actual native worker device differs from approved RX9070XT16GB")
        tunable = torch.cuda.tunable
        actual_math = {"allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                       "miopen_enabled": torch.backends.cudnn.enabled, "miopen_version": torch.backends.cudnn.version(),
                       "miopen_benchmark": torch.backends.cudnn.benchmark, "miopen_deterministic": torch.backends.cudnn.deterministic,
                       "preferred_blas_library_raw_torch_enum": str(torch.backends.cuda.preferred_blas_library()),
                       "preferred_rocm_flash_attention_raw_torch_enum": str(torch.backends.cuda.preferred_rocm_fa_library()),
                       "tunable_enabled": tunable.is_enabled(), "tunable_tuning_enabled": tunable.tuning_is_enabled(),
                       "tunable_record_untuned_enabled": tunable.record_untuned_is_enabled(), "tunable_result_count": len(tunable.get_results())}
        # Collection identity observations only; inherited native execution is unchanged.
        process_fields = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        process_identity = {"PID":os.getpid(),"birth_ticks":int(process_fields[19]),
                            "UID":os.geteuid(),"GID":os.getegid(),"CPU_affinity":sorted(os.sched_getaffinity(0))}
        ready = {"actual_worker_process":process_identity,
                 "actual_torch_origin":str(Path(torch.__file__).resolve()),"actual_torch_init_sha256":sha(torch.__file__),
                 "actual_torch_version":str(torch.__version__),"actual_torch_git":torch.version.git_version,"actual_torch_HIP":torch.version.hip,
                 "status": "READY", "worker_pid": os.getpid(), "config_sha256": observer.config_sha(),
                 "mode": cfg["mode"], "fixed_negative_cache_configured": cfg["fixed_negative_cache"], "actual_worker_config": actual, "component_precision": precision,
                 "fp8_projection_audit": fp8_audit, "fp8_shard_receipt": shard_receipt, "fp8_scale_receipt": scale_receipt, "observed_loader_sources": loader_sources, "fp8_method_source_identity": fp8_method_identity, "resolved_quantization": {"class": type(selected_qc).__module__ + "." + type(selected_qc).__name__, "activation_scheme": selected_qc.activation_scheme, "checkpoint_serialized": selected_qc.is_checkpoint_fp8_serialized, "scope": "transformer blocks only"},
                 "installed_attention_modules": attention,
                 "attention_kernel_execution": "Not inferred from configuration; hardware kernel profiling remains separate",
                 "versions": distribution_versions(), "actual_hardware": actual_hardware, "actual_math": actual_math,
                 "native_sources": cfg["native_sources"], "adapter_sources": cfg["adapter_sources"],
                 "imported_native_runner_file": str(Path(__import__(DiffusionModelRunner.__module__, fromlist=["_"]).__file__).resolve()),
                 "observed_requests": "Required separately; readiness is not execution proof"}
        path = Path(cfg["proof_directory"]) / ("ready-" + str(os.getpid()) + ".json")
        with path.open("x") as f:
            json.dump(ready, f, indent=2, allow_nan=False)
            f.write("\n")
        return result

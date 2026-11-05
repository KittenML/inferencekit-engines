"""Native Omni API behind the frozen common MOSS HTTP image transport."""
import argparse
import datetime
import io
import json
import os
import sys
import threading
import uuid
from pathlib import Path

from moss_omni_contract import (CONFIG_ENV, NONCE_KEY, RUNNER, SOURCE_COMMIT, canonical,
                               require, sha, validate_receipt)


class Engine:
    def __init__(self, configuration):
        self.path = Path(configuration).resolve()
        self.config_sha = sha(self.path)
        self.config = json.loads(self.path.read_text())
        self.engine = None
        self.failed = False
        self.lock = threading.Lock()
        self.resolved = None

    def check_deadline(self):
        deadline = datetime.datetime.fromisoformat(self.config["deadline_utc"])
        require(deadline.tzinfo is not None, "Explicit task deadline required")
        require(datetime.datetime.now(datetime.timezone.utc) < deadline, "Task deadline reached")

    def load(self):
        self.check_deadline()
        cfg = self.config
        require(cfg["source_commit"] == SOURCE_COMMIT, "Unexpected native source pin")
        require(cfg["mode"] == "lightweight" and cfg["save_trajectory"] is False, "Timing requires original lightweight/no-trajectory mode")
        require(not cfg["save_trajectory"] or cfg["mode"] == "diagnostic", "Invalid trajectory policy")
        names = {"moss_omni_adapter.py", "moss_omni_observer.py", "moss_omni_contract.py", "moss_omni_runner.py", "fixed_empty_negative.py", "native_quality_dump.py", "ranking_policy.py", "compiled_dispatch.py", "audit_fp8.py", "shard_observer.py", "scale_audit.py"}
        require(set(cfg["adapter_sources"]) == names, "Incomplete adapter source identity")
        for name, digest in cfg["adapter_sources"].items():
            require(sha(Path(__file__).with_name(name)) == digest, "Adapter source differs: " + name)
        source_root = Path(cfg["native_source_root"]).resolve()
        require(len(cfg["native_sources"]) >= 23, "Incomplete native source boundary")
        for name, digest in cfg["native_sources"].items():
            require(sha(source_root / name) == digest, "Native source differs: " + name)
        proof = Path(cfg["proof_directory"])
        require(proof.is_absolute(), "Absolute proof directory required")
        proof.mkdir(parents=True, exist_ok=False)
        os.environ[CONFIG_ENV] = str(self.path)
        os.environ["OMNI_DIFFUSION_PROMPT_EMBED_CACHE"] = "0"
        # Spawned native processes inherit this path and import the observational
        # runner through Omni's explicit diffusion_model_runner_cls extension.
        adapter_path = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] = os.pathsep.join([adapter_path] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
        import torch
        from vllm_omni.entrypoints.omni import Omni
        import inspect
        require(Path(inspect.getsourcefile(Omni)).resolve() == source_root / "vllm_omni/entrypoints/omni.py",
                "Imported Omni differs from pinned checkout")
        torch.backends.cudnn.enabled = True
        args = dict(cfg["engine_args"])
        require(cfg.get("precision_policy") == "main_dit_online_fp8_dynamic_bf16_input" and
                args.get("quantization_config") == {"transformer": {"method": "fp8", "activation_scheme": "dynamic"}, "default": None},
                "Unexpected candidate quantization policy")
        require(args["dtype"] == "torch.bfloat16" and args["diffusion_model_runner_cls"] == RUNNER,
                "Native dtype/observer extension differs")
        args["dtype"] = torch.bfloat16
        self.engine = Omni(**args)
        ready_files = list(proof.glob("ready-*.json"))
        require(len(ready_files) == 1, "Expected exactly one actual native worker readiness receipt")
        ready = json.loads(ready_files[0].read_text())
        require(ready.get("status") == "READY" and ready.get("config_sha256") == self.config_sha,
                "Native worker readiness mismatch")
        require(ready.get("adapter_sources") == cfg["adapter_sources"] and ready.get("native_sources") == cfg["native_sources"],
                "Native worker source identity differs")
        require(canonical(ready.get("actual_worker_config")) == canonical(cfg["expected_worker_config"]),
                "Native worker resolved config differs")
        from audit_fp8 import validate_audit_receipt
        validate_audit_receipt(ready.get("fp8_projection_audit"))
        from shard_observer import validate_shard_receipt
        validate_shard_receipt(ready.get("fp8_shard_receipt"))
        from scale_audit import POLICY, validate_receipt as validate_scale_receipt, validate_restored_scales
        require(cfg.get("fp8_scale_policy") == POLICY and
                cfg.get("fp8_scale_patch_sha256") == "95aee4ac3b9aca81327777619f93119e18c299a70c9e4e69fa074561102a78ac" and
                cfg.get("fp8_scale_source_sha256") == "8bcc9a37d448ed271bc5994b1211876a16d082fa1f3fdc84ea95d99465a6776d",
                "Candidate scale policy or source pin differs")
        validate_scale_receipt(ready.get("fp8_scale_receipt"))
        validate_restored_scales(ready["fp8_scale_receipt"], ready["fp8_projection_audit"])
        require(ready["fp8_scale_receipt"].get("initial_offload_reload_scales_equal") is True,
                "Candidate scales did not survive the initial offload/reload check")
        self.resolved = {"engine": "vllm-omni", "source_commit": SOURCE_COMMIT,
                         "config_sha256": self.config_sha, "mode": cfg["mode"],
                         "actual_worker": ready, "readiness_sha256": sha(ready_files[0]),
                         "qualification": "UNPROVEN until off/on, consumed-state, quality, and client timing admission"}

    def describe(self):
        return self.resolved

    def generate(self, request):
        self.check_deadline()
        require(self.engine is not None and not self.failed, "Engine unavailable or prior request failed")
        require(self.lock.acquire(blocking=False), "Concurrent native requests are not admitted")
        try:
            require(request["negative_prompt"] == "" and request["uncond_skip_layers"] == [22] and
                    request["guidance_scale"] == self.config["expected_guidance_scale"] and request["flow_shift"] == 5, "Unadmitted fixed sampling policy")
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
            from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
            from PIL import Image
            nonce = uuid.uuid4().hex
            proof = Path(self.config["proof_directory"]) / (nonce + ".json")
            require(not proof.exists(), "Request nonce collision")
            params = {"height": request["height"], "width": request["width"], "seed": request["seed"],
                      "generator_device": "cpu", "num_inference_steps": request["steps"],
                      "guidance_scale": request["guidance_scale"], "num_outputs_per_prompt": 1,
                      "max_sequence_length": request["max_sequence_length"], "quality": "lossless",
                      "extra_args": {"flow_shift": request["flow_shift"], "uncond_skip_layers": request["uncond_skip_layers"],
                                     "slg_start": 0.0, "slg_stop": 1.0, "slg_layers": [], "slg_scale": 0.0, NONCE_KEY: nonce}}
            outputs = self.engine.generate({"prompt": request["prompt"], "negative_prompt": "", "modalities": ["image"]},
                                           sampling_params_list=[OmniDiffusionSamplingParams(**params)])
            require(outputs, "Native engine returned no outputs")
            images = []
            for output in outputs:
                value = getattr(output, "images", None)
                if value:
                    images = value
                    break
            if not images:
                images = extract_images_from_outputs(outputs)
            require(len(images) == 1, "Expected exactly one native output image")
            value = images[0]
            if not isinstance(value, Image.Image):
                from diffusers.utils import numpy_to_pil
                converted = numpy_to_pil(value)
                require(len(converted) == 1, "Unexpected native image conversion batch")
                value = converted[0]
            require(value.size == (request["width"], request["height"]), "Native output dimensions differ")
            buffer = io.BytesIO()
            value.save(buffer, format="PNG")
            value.close()
            image = buffer.getvalue()
            require(len(image) <= 32 * 1024 * 1024 and image.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid bounded native PNG")
            if self.config["mode"] == "off":
                receipt = None
                require(not proof.exists(), "Observer-off unexpectedly produced execution proof")
            else:
                require(proof.is_file() and proof.stat().st_size <= 1024 * 1024, "Actual native worker proof missing/oversized")
                receipt = validate_receipt(json.loads(proof.read_text()), nonce=nonce, request=request,
                    config_sha=self.config_sha, observer_sha=self.config["adapter_sources"]["moss_omni_observer.py"],
                    mode=self.config["mode"], trajectories=self.config["save_trajectory"])
                if self.config["save_trajectory"]:
                    bundle = receipt.get("quality_bundle", {})
                    expected_bundle = Path(self.config["proof_directory"]) / nonce / "trajectory.safetensors"
                    require(bundle.get("path") == nonce + "/trajectory.safetensors"
                            and bundle.get("sha256") == sha(expected_bundle)
                            and bundle.get("diagnostic_only") is True, "Native quality bundle missing/changed")
                require(receipt.get("fixed_negative_cache") == self.config["fixed_negative_cache"], "Actual fixednegative policy differs")
                require(receipt["worker_pid"] == self.resolved["actual_worker"]["worker_pid"], "Request worker differs from readiness")
                require(all(self.config["native_sources"].get(name) == digest for name, digest in receipt["worker_source_files"].items()),
                        "Actual execution source hash differs")
            return image, receipt["scheduler_calls"], {"actual_steps": receipt["scheduler_calls"], "cache": (["fixed_negative_embedding"] if receipt and receipt["encodes"][1]["actual_encoder_forwards"] == 0 else []), "observer_mode": self.config["mode"],
                "native_worker_receipt_sha256": sha(proof) if receipt else None,
                "observed_execution": receipt, "resolved_runtime": self.resolved,
                "cache_policy": "Fixedemptynegativeonly when selected; positivesalwaysfresh; nativefreshperrequestrefinement caches retained",
                "timing": "Frozen common HTTP client measures through final decoded PNG; no internal timing substituted"}
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def shutdown(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--transport-root", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    require(1 <= args.port <= 65535, "Invalid port")
    engine = Engine(args.configuration)
    sys.path.insert(0, str(Path(args.transport_root).resolve()))
    from moss_amd import server as transport
    require(sha(transport.__file__) == engine.config["common_transport_sha256"], "Common image transport differs")
    server = None
    try:
        engine.load()
        server = transport.make_server(engine, host="127.0.0.1", port=args.port)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        engine.shutdown()


if __name__ == "__main__":
    main()

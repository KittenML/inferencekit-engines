"""Strict, dependency-light image HTTP transport; model libraries load only in Engine.load()."""

from __future__ import annotations

import argparse
import base64
import datetime
import uuid
import hashlib
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "OpenMOSS-Team/MOSS-Image-SFT"
REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
PRECISION = {
    "transformer": "bf16_except_main_block_projections_fp8_scale_1.03125",
    "text_encoder": "bf16",
    "vae": "fp32",
    "latents": "fp32",
    "rope": "fp32",
}
REQUIRED = {
    "model",
    "model_revision",
    "dtype",
    "quant",
    "prompt",
    "width",
    "height",
    "steps",
    "guidance_scale",
    "seed",
}
OPTIONAL = {
    "size",
    "scheduler",
    "flow_shift",
    "negative_prompt",
    "uncond_skip_layers",
    "max_sequence_length",
    "precision",
}


class BadRequest(ValueError):
    pass


def parse(body):
    if (
        not isinstance(body, dict)
        or set(body) - REQUIRED - OPTIONAL
        or REQUIRED - set(body)
    ):
        raise BadRequest("unsupported or missing request fields")
    if body["model"] != MODEL or body["model_revision"] != REVISION:
        raise BadRequest(
            "model identifier or revision differs from the loaded checkpoint"
        )
    if body["dtype"] != "bf16" or body["quant"] != "fp8-scale-1.03125":
        raise BadRequest("candidate requires FP8 main DiT projection quantization with BF16 inputs")
    if (
        not isinstance(body["prompt"], str)
        or not body["prompt"].strip()
        or len(body["prompt"]) > 100000
    ):
        raise BadRequest("prompt must be non-empty bounded text")
    for k in ("width", "height", "steps", "seed"):
        if type(body[k]) is not int:
            raise BadRequest(k + " must be an integer")
    if any(not 16 <= body[k] <= 4096 or body[k] % 16 for k in ("width", "height")):
        raise BadRequest("dimensions must be positive multiples of 16 within [16,4096]")
    if body["width"] * body["height"] > 4_194_304:
        raise BadRequest("image exceeds 4 megapixels")
    if not 1 <= body["steps"] <= 1000 or not 0 <= body["seed"] < 2**31:
        raise BadRequest("steps or seed outside bounds")
    gs = body["guidance_scale"]
    if (
        isinstance(gs, bool)
        or not isinstance(gs, (int, float))
        or not math.isfinite(gs)
        or not 0 <= gs <= 100
    ):
        raise BadRequest("guidance_scale outside finite bounds")
    if (
        body.get("size", f"{body['width']}x{body['height']}")
        != f"{body['width']}x{body['height']}"
    ):
        raise BadRequest("size disagrees with dimensions")
    if body.get("scheduler", "euler") != "euler":
        raise BadRequest("upstream baseline supports its Euler scheduler only")
    shift = body.get("flow_shift", 5.0)
    if (
        isinstance(shift, bool)
        or not isinstance(shift, (int, float))
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise BadRequest("flow_shift must be positive and finite")
    negative = body.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > 100000:
        raise BadRequest("negative_prompt must be bounded text")
    layers = body.get("uncond_skip_layers", [22])
    if (
        not isinstance(layers, list)
        or len(set(layers)) != len(layers)
        or any(type(i) is not int or not 0 <= i < 42 for i in layers)
    ):
        raise BadRequest("uncond_skip_layers must contain distinct indices in [0,42)")
    seq = body.get("max_sequence_length", 4096)
    if type(seq) is not int or not 1 <= seq <= 4096:
        raise BadRequest("max_sequence_length outside bounds")
    if body.get("precision", PRECISION) != PRECISION:
        raise BadRequest("precision settings differ from candidate policy")
    return {
        **body,
        "scheduler": "euler",
        "flow_shift": float(shift),
        "negative_prompt": negative,
        "uncond_skip_layers": layers,
        "max_sequence_length": seq,
        "precision": dict(PRECISION),
    }


def execution_identity(req, extra=None):
    return {
        **(extra or {}),
        "prompt_sha256": hashlib.sha256(req["prompt"].encode("utf-8")).hexdigest(),
        "negative_prompt_sha256": hashlib.sha256(req["negative_prompt"].encode("utf-8")).hexdigest(),
        "prompt_identity_source": "exact accepted UTF-8 request text passed to engine.generate",
    }


def response(req, image_bytes, steps_run, extra=None):
    effective = {k: v for k, v in req.items() if k not in ("prompt", "size", "steps")}
    effective["steps_run"] = steps_run
    return {
        "image_b64": base64.b64encode(image_bytes).decode(),
        "format": "png",
        "effective": effective,
        "cache": (extra or {}).get("cache", []),
        "execution": execution_identity(req, extra),
    }


def make_server(engine, host="127.0.0.1", port=0):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(1800)

        def send(self, status, doc):
            data = json.dumps(doc, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def do_GET(self):
            if self.path == "/health":
                self.send(200, {"status": "ok", "model": MODEL, "revision": REVISION})
            elif self.path == "/info":
                self.send(200, engine.describe())
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/images/generations":
                self.send(404, {"error": "not found"})
                return
            try:
                n = self.headers.get("Content-Length", "")
                if not n.isdigit() or not 0 < int(n) <= 1_048_576:
                    raise BadRequest("bounded Content-Length required")
                if self.headers.get("Transfer-Encoding"):
                    raise BadRequest("request Transfer-Encoding unsupported")
                raw = self.rfile.read(int(n))
                if len(raw) != int(n):
                    raise BadRequest("truncated request")
                req = parse(json.loads(raw))
            except (ValueError, TypeError, RecursionError):
                self.send(400, {"error": "invalid image request"})
                return
            if not lock.acquire(blocking=False):
                self.send(409, {"error": "one image request at a time"})
                return
            try:
                from .hardware import check_run_deadline

                check_run_deadline()
                request_id = uuid.uuid4().hex
                started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                image, steps, extra = engine.generate(req)
                extra = execution_identity(req, extra)
                proof_dir = os.environ.get("MOSS_EXECUTION_LOG_DIR")
                if proof_dir:
                    output_sha = hashlib.sha256(image).hexdigest()
                    facts = {
                        "request_id": request_id,
                        "started_at": started_at,
                        "completed_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "measurement_source": "instrumented engine; self-reported, not the client timer",
                        "request_sha256": hashlib.sha256(
                            json.dumps(
                                req, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        "output_sha256": output_sha,
                        "execution": extra,
                    }
                    directory = Path(proof_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / (output_sha + "-" + request_id + ".json")).write_text(
                        json.dumps(facts, indent=2) + "\n"
                    )
                self.send(200, response(req, image, steps, extra))
            except Exception as exc:
                from .hardware import DeadlineExpired

                if isinstance(exc, DeadlineExpired):
                    self.send(503, {"error": "configured run deadline has passed"})
                    return
                # Keep request text, host paths and credentials out of HTTP errors.
                print("generation failed: " + type(exc).__name__, flush=True)
                if os.environ.get("MOSS_DEBUG") == "1":
                    import traceback

                    traceback.print_exc()
                self.send(500, {"error": "image generation failed"})
            finally:
                lock.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-dir", required=True)
    p.add_argument(
        "--memory-mode",
        choices=[
            "model",
            "sequential",
            "text_cpu",
            "text_sequential",
            "staged_text_sequential",
        ],
        default="staged_text_sequential",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dump-dir")
    p.add_argument("--fixed-negative", action="store_true")
    a = p.parse_args(argv)
    from .upstream import Engine

    engine = Engine(a.model_dir, a.memory_mode, a.compile, a.dump_dir, a.fixed_negative)
    engine.load()
    server = make_server(engine, a.host, a.port)
    print(json.dumps({"event": "ready", **engine.describe()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

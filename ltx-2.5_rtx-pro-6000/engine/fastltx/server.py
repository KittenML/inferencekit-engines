"""HTTP clip server: GET /health, POST /v1/clips (one text-to-video clip per request)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIELDS = {"task", "prompt", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed", "noise_seed",
          "audio"}
REQUIRED = {"task", "prompt", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed"}
ENGINE = None
READY = threading.Event()
LOCK = threading.Lock()


class BadRequest(Exception):
    pass


def _int(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
        raise BadRequest(f"{name} must be an integer")
    return int(v)


def parse(body: dict) -> dict:
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        raise BadRequest(f"unsupported field(s): {unknown}")
    missing = sorted(k for k in REQUIRED if body.get(k) is None)
    if missing:
        raise BadRequest(f"missing field(s): {missing}")
    task = os.environ.get("CLIP_TASK", "t2v")
    if body["task"] != "t2v" or task != "t2v":
        raise BadRequest(f"this server serves task t2v only, got {body['task']!r}")
    if not isinstance(body["prompt"], str) or not body["prompt"].strip():
        raise BadRequest("prompt must be a non-empty string")
    w, h, n = _int(body["width"], "width"), _int(body["height"], "height"), _int(body["num_frames"], "num_frames")
    steps = _int(body["steps"], "steps")
    seed = _int(body["seed"], "seed")
    ns = body.get("noise_seed")
    ns = seed if ns is None else _int(ns, "noise_seed")
    fps = body["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != int(fps) or fps <= 0:
        raise BadRequest("fps must be a positive whole number (the muxer writes an integer frame rate)")
    gs = body["guidance_scale"]
    if isinstance(gs, bool) or not isinstance(gs, (int, float)):
        raise BadRequest("guidance_scale must be a number")
    if w <= 0 or h <= 0 or w % 32 or h % 32:
        raise BadRequest("width and height must be positive multiples of 32")
    if n < 1 or (n - 1) % 8:
        raise BadRequest("num_frames must be 8k+1")
    if steps < 1 or steps > 1000:
        raise BadRequest("steps must be in 1..1000")
    audio = body.get("audio")
    if audio is not None and audio is not True:
        raise BadRequest("this model always makes an audio track: audio must be true or absent")
    if not (0 <= ns < 2 ** 63):
        raise BadRequest("noise_seed out of range")
    return dict(task="t2v", prompt=body["prompt"], width=w, height=h, num_frames=n, fps=float(fps), steps=steps,
                guidance_scale=float(gs), seed=seed, noise_seed=ns, audio=audio)


def request_record(req: dict) -> dict:
    """The request fields a step record holds (noise_seed defaults to seed)."""
    return {k: req.get(k) for k in ("task", "prompt", "width", "height", "num_frames", "fps", "steps",
                                    "guidance_scale", "seed", "noise_seed", "audio")}


def make_clip(req: dict) -> dict:
    import torch

    from .stepdump import StepRecorder

    t0 = time.perf_counter()
    dump_dir = os.environ.get("CLIP_STEP_DUMP_DIR")
    rec = StepRecorder(request_record(req))
    box = {}

    def start_writer(r):
        # the record is complete once denoising ends: copy it to the CPU and write it under a temporary name
        # while the decoders and the muxer run; it is renamed to <sha256 of the mp4> before the answer
        def work():
            try:
                r.to_cpu()
                box["tmp"] = r.write_tmp(dump_dir)
            except BaseException as e:  # noqa: BLE001
                box["err"] = e
        th = threading.Thread(target=work, daemon=True)
        th.start()
        return th

    out = ENGINE.generate(req["prompt"], req["width"], req["height"], req["num_frames"], req["fps"], req["steps"],
                          req["guidance_scale"], req["noise_seed"], recorder=rec,
                          on_denoised=start_writer if dump_dir else None)
    st = out["stages"]
    from diffusers.utils import encode_video

    with tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as td:
        path = os.path.join(td, "clip.mp4")
        encode_video(out["frames"], fps=int(req["fps"]), output_path=path, audio=out["wave"],
                     audio_sample_rate=out["rate"])
        data = Path(path).read_bytes()
    st.mark("mux", sync=False)
    sha = hashlib.sha256(data).hexdigest()
    if dump_dir:
        out["writer"].join()
        if "err" in box:
            raise box["err"]
        StepRecorder.commit(box["tmp"], Path(dump_dir) / f"{sha}.safetensors")
    st.mark("step_dump", sync=False)
    frames, h, w = out["shape"]
    return {"video_b64": base64.b64encode(data).decode("ascii"), "format": "mp4",
            "stage_ms": {k: round(v, 1) for k, v in st.ms.items()},
            "cache": ["negative_text_encode"],
            "effective": {"steps_run": out["steps_run"], "width": w, "height": h, "frames": frames,
                          "fps": req["fps"], "dtype": "bfloat16", "attention": ENGINE.attention_name,
                          "noise_seed": req["noise_seed"], "video_sha256": sha,
                          "total_ms": round((time.perf_counter() - t0) * 1000.0, 1)}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            if READY.is_set():
                self._send(200, {"status": "ok"})
            else:
                self._send(503, {"status": "loading"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/clips":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"null")
            req = parse(body)
        except BadRequest as e:
            self._send(400, {"error": str(e)})
            return
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"bad JSON: {e}"})
            return
        if not READY.is_set():
            self._send(503, {"error": "loading"})
            return
        try:
            with LOCK:
                res = make_clip(req)
        except BadRequest as e:
            self._send(400, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._send(200, res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    global ENGINE
    from .pipeline import Engine

    t0 = time.time()
    ENGINE = Engine()
    ENGINE.negative()
    ENGINE.warmup()
    print(f"engine ready in {time.time() - t0:.1f} s", flush=True)
    READY.set()
    th.join()


if __name__ == "__main__":
    main()

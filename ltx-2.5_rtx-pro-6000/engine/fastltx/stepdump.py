"""Per-step record (format clip-steps.v1): step.<i>.<stream>.{x,out,x_next}, final.<stream>, sigma, sigma_next."""

from __future__ import annotations

import json
import os
from pathlib import Path

FORMAT = "clip-steps.v1"


class StepRecorder:
    def __init__(self, request: dict):
        self.request = dict(request)
        self.tensors = {}
        self.sigma, self.sigma_next = [], []
        self.count = {}
        self.finals = {}

    def step(self, stream, sigma, sigma_next, x, out, x_next):
        i = self.count.get(stream, 0)
        self.count[stream] = i + 1
        if i == len(self.sigma):
            self.sigma.append(float(sigma))
            self.sigma_next.append(float(sigma_next))
        # kept on the GPU (a copy, no sync) and moved to the CPU in one go by to_cpu()
        for name, t in (("x", x), ("out", out), ("x_next", x_next)):
            self.tensors[f"step.{i}.{stream}.{name}"] = t.detach().clone().contiguous()

    def final(self, stream, latent):
        self.finals[stream] = latent.detach().clone().contiguous()

    def to_cpu(self):
        """Copy every recorded tensor to the CPU (call on the stream that produced them, or after a sync)."""
        import torch

        side = torch.cuda.Stream() if torch.cuda.is_available() else None
        if side is not None:
            with torch.cuda.stream(side):
                self.tensors = {k: v.to("cpu") for k, v in self.tensors.items()}
                self.finals = {k: v.to("cpu") for k, v in self.finals.items()}
            side.synchronize()
        else:
            self.tensors = {k: v.to("cpu") for k, v in self.tensors.items()}
            self.finals = {k: v.to("cpu") for k, v in self.finals.items()}

    def write_tmp(self, directory):
        """Write the record under a temporary name in ``directory``; returns that path (rename with commit())."""
        import tempfile

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".safetensors.partial")
        os.close(fd)
        self._save(tmp)
        return tmp

    @staticmethod
    def commit(tmp, path):
        os.replace(tmp, path)

    def save(self, path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        self._save(tmp)
        os.replace(tmp, path)
        return path.stat().st_size

    def _save(self, tmp):
        import torch
        from safetensors.torch import save_file

        tensors = dict(self.tensors)
        for s, n in self.count.items():
            f = self.finals.get(s)
            tensors[f"final.{s}"] = f if f is not None else tensors[f"step.{n - 1}.{s}.x_next"].clone()
        tensors["sigma"] = torch.tensor(self.sigma, dtype=torch.float64)
        tensors["sigma_next"] = torch.tensor(self.sigma_next, dtype=torch.float64)
        save_file(tensors, str(tmp), metadata={"format": FORMAT, "request": json.dumps(self.request, sort_keys=True)})

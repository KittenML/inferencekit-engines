#!/bin/sh
# Install the LTX-2.5 clip server into ./.venv and download the model weights it needs.
# Run it from anywhere, as a normal user (no root). Safe to run again: finished steps are skipped.
set -eu
cd "$(dirname "$0")"

REPO=Lightricks/LTX-2.5-Diffusers
REV=da0b690960026eab794b8ceaf841233cf7064478

say() { printf '[install] %s\n' "$*"; }
die() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

# Triton (used by torch.compile) builds small C helpers when the server starts, so a C compiler is needed.
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
  || die "no C compiler found. Install one first (Ubuntu: sudo apt-get install -y gcc), then run ./install.sh again."
command -v nvidia-smi >/dev/null 2>&1 || say "warning: nvidia-smi not found; the server needs an NVIDIA GPU and driver."

# 1. uv (Python package manager), installed for this user only if it is missing.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || die "uv is not installed and curl is missing. Install uv (https://docs.astral.sh/uv/) and run again."
    say "installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh
  fi
fi

# 2. Python 3.12 virtual environment in ./.venv with pinned packages (about 10 GB).
if [ ! -x .venv/bin/python ]; then
  say "creating .venv (Python 3.12)"
  uv venv .venv --python 3.12
fi
say "installing PyTorch 2.13.0 (CUDA 13.2 build)"
uv pip install --python .venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
say "installing the other packages (engine/requirements.txt)"
uv pip install --python .venv/bin/python -r engine/requirements.txt

# 3. Model weights: only the files the engine loads (about 66 GB), at a pinned revision,
#    into the Hugging Face cache (~/.cache/huggingface, or $HF_HOME if you set it).
say "checking the model weights ($REPO at $REV)"
REPO="$REPO" REV="$REV" .venv/bin/python - <<'EOF'
import os
import sys

from huggingface_hub import get_token, snapshot_download

repo, rev = os.environ["REPO"], os.environ["REV"]
patterns = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors.index.json",
    "connectors/diffusion_pytorch_model-0000?-of-00002.safetensors",
    "transformer_full/*",
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
]
HELP = f"""
This model is gated. Before the download can start:
  1. Open https://huggingface.co/{repo} while logged in to Hugging Face and accept the licence
     (LTX-2.x Community License).
  2. Log in on this machine with a token that can read that repository:
       .venv/bin/hf auth login
     or set it for this shell only:  export HF_TOKEN=<your token>
  3. Run ./install.sh again.
"""


def done(path):
    with open(os.path.join(".venv", "ltx25_diffusers_snapshot"), "w") as f:
        f.write(path + "\n")
    print(f"[install] weights ready: {path}")


def complete(path):
    """True when every weight file the engine loads is in the cached snapshot."""
    import json

    need = ["model_index.json", "vae/diffusion_pytorch_model.safetensors",
            "audio_vae/diffusion_pytorch_model.safetensors", "vocoder/diffusion_pytorch_model.safetensors",
            "tokenizer/tokenizer.json", "scheduler/scheduler_config.json"]
    for idx in ("transformer_full/diffusion_pytorch_model.safetensors.index.json",
                "text_encoder/model.safetensors.index.json",
                "connectors/diffusion_pytorch_model.safetensors.index.json"):
        if not os.path.isfile(os.path.join(path, idx)):
            return False
        with open(os.path.join(path, idx)) as f:
            shards = set(json.load(f)["weight_map"].values())
        need += [os.path.join(os.path.dirname(idx), s) for s in shards]
    return all(os.path.isfile(os.path.join(path, n)) for n in need)


try:
    cached = snapshot_download(repo, revision=rev, allow_patterns=patterns, local_files_only=True)
except Exception:  # noqa: BLE001 - not cached yet
    cached = None
if cached and complete(cached):
    done(cached)
    sys.exit(0)

if not (os.environ.get("HF_TOKEN") or get_token()):
    print("[install] ERROR: no Hugging Face token found." + HELP, file=sys.stderr)
    sys.exit(1)
print("[install] downloading (about 66 GB; an interrupted download resumes when you run ./install.sh again)")
try:
    path = snapshot_download(repo, revision=rev, allow_patterns=patterns)
except Exception as e:  # noqa: BLE001
    name = type(e).__name__
    if name in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(e) or "403" in str(e):
        print(f"[install] ERROR: Hugging Face refused access ({name})." + HELP, file=sys.stderr)
        sys.exit(1)
    raise
done(path)
EOF

say "done. Start the server with ./serve.sh"

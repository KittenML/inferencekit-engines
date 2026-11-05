#!/usr/bin/env bash
# Sets up everything inside this folder; no root needed, safe to run again.
#   .venv/                    Python 3.12, PyTorch 2.14.1 (CUDA 13.0), transformers, Triton
#   .venv/cuda/               CUDA 13.0 compiler from PyPI, only if no CUDA 13 toolkit is installed
#   models/Qwen3-8B-FP8/      the weights, at a pinned revision (about 10 GB)
#   .venv/torch_extensions/   the compiled CUDA kernels
# Settings (environment): CUDA_HOME=<a CUDA 13 toolkit>, MODEL_DIR=<where the weights go>.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD

MODEL_ID=Qwen/Qwen3-8B-FP8
MODEL_REV=220b46e3b2180893580a4454f21f22d3ebb187d3
MODEL_DIR=${MODEL_DIR:-$HERE/models/Qwen3-8B-FP8}
TORCH_INDEX=https://download.pytorch.org/whl/cu130

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# 1. GPU and driver. CUDA 13 needs NVIDIA driver 580 or newer.
say "Checking the GPU and driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Install the NVIDIA driver (580 or newer) first."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU: $GPU, driver $DRIVER"
[ "${DRIVER%%.*}" -ge 580 ] 2>/dev/null || die "driver $DRIVER is too old for CUDA 13. Install driver 580 or newer."
case "$GPU" in *"RTX 5090"*) ;; *) echo "WARNING: this engine was built and tuned for an RTX 5090. On a '$GPU' it may not build or may be slow." ;; esac

# 2. A C++ compiler for the kernel build.
command -v g++ >/dev/null || command -v c++ >/dev/null \
  || die "no C++ compiler found. Install one, e.g. 'sudo apt-get install -y build-essential'."

# 3. uv (Python package manager), installed for this user if missing.
if ! command -v uv >/dev/null; then
  if [ ! -x "$HOME/.local/bin/uv" ]; then
    command -v curl >/dev/null || die "uv is not installed and curl is missing. Install uv (https://docs.astral.sh/uv/) or curl and run ./install.sh again."
    say "Installing uv into ~/.local/bin"; curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

# 4. Python environment.
say "Creating .venv"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" "torch==2.14.1"
uv pip install --python .venv/bin/python \
  "transformers==5.19.0" "safetensors==0.8.0" "huggingface_hub==2.2.0" "ninja==1.13.2" "numpy==2.5.4"

# 5. CUDA 13 compiler. PyTorch builds the kernels with nvcc, and nvcc must be CUDA 13 like PyTorch.
nvcc_major() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1; }
say "Looking for a CUDA 13 compiler (nvcc)"
FOUND=""
for c in ${CUDA_HOME:+"$CUDA_HOME"} /usr/local/cuda-13* /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && [ "$(nvcc_major "$c/bin/nvcc")" = 13 ] && { FOUND=$c; break; }
done
if [ -z "$FOUND" ] && command -v nvcc >/dev/null && [ "$(nvcc_major "$(command -v nvcc)")" = 13 ]; then
  FOUND=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
fi
if [ -z "$FOUND" ]; then
  echo "No CUDA 13 toolkit found (checked \$CUDA_HOME, /usr/local/cuda-13*, /usr/local/cuda, PATH)."
  echo "Installing NVIDIA's CUDA 13.0 compiler from PyPI into .venv/cuda (no root needed)."
  uv pip install --python .venv/bin/python --target .venv/cuda-pkgs "nvidia-cuda-nvcc==13.0.*" \
    "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*"
  ln -sfn "$HERE/.venv/cuda-pkgs/nvidia/cu13" .venv/cuda
  ln -sfn lib .venv/cuda/lib64                         # PyTorch looks for lib64
  ln -sf libcudart.so.13 .venv/cuda/lib/libcudart.so   # and links -lcudart
  FOUND=$HERE/.venv/cuda
  [ "$(nvcc_major "$FOUND/bin/nvcc")" = 13 ] || die "the CUDA 13 compiler from PyPI did not install correctly.
Install the CUDA 13.0 toolkit instead (https://developer.nvidia.com/cuda-downloads) and run
  CUDA_HOME=/usr/local/cuda-13.0 ./install.sh"
fi
echo "Using CUDA_HOME=$FOUND ($("$FOUND/bin/nvcc" --version | grep release))"
cat > .venv/engine.env <<EOF
export CUDA_HOME="$FOUND"
export PATH="$HERE/.venv/bin:\$PATH"   # ninja, used by the kernel build
export TORCH_EXTENSIONS_DIR="$HERE/.venv/torch_extensions"
export MODEL_DIR="\${MODEL_DIR:-$MODEL_DIR}"
EOF
# shellcheck disable=SC1091
. .venv/engine.env

# 6. Weights (resumes a partial download).
say "Downloading $MODEL_ID at revision ${MODEL_REV:0:12} to $MODEL_DIR"
.venv/bin/hf download "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"

# 7. Build the CUDA kernels once (a few minutes the first time).
say "Building the CUDA kernels"
.venv/bin/python - <<'EOF'
import sys, time, torch
sys.path.insert(0, ".")
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
t = time.time()
from kernels import ops
ops()
print(f"kernels ready ({time.time() - t:.0f} s), torch {torch.__version__}, GPU {torch.cuda.get_device_name()}")
EOF

say "Done. Start the server with ./serve.sh"

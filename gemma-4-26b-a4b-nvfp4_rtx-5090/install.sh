#!/usr/bin/env bash
# Install the engine into ./.venv (inside this folder): vLLM 0.31.0 plus the g4fast plugin (./engine),
# and the Gemma 4 chat template, checked by sha256. No root needed. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

VLLM_VERSION=0.31.0
TEMPLATE_REPO=google/gemma-4-26B-A4B-it
TEMPLATE_REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
TEMPLATE_SHA256=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
TEMPLATE=.venv/gemma-4-26B-A4B-it.chat_template.jinja

fail() { echo "install.sh: $*" >&2; exit 1; }

# uv (Python package manager), installed for this user into ~/.local/bin if it is missing.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || fail "uv is not installed and curl is missing. Install uv (https://docs.astral.sh/uv/) and run ./install.sh again."
    echo "== installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh
  fi
fi
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found: this engine needs an NVIDIA GPU and its driver."
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
  || fail "no C compiler found (Triton needs one). On Ubuntu: sudo apt-get install -y build-essential"

if [ ! -x .venv/bin/python ]; then
  echo "== creating .venv (Python 3.12)"
  uv venv --python 3.12 .venv
fi

if .venv/bin/python -c "import vllm, g4fast, sys; sys.exit(vllm.__version__ != '$VLLM_VERSION')" 2>/dev/null; then
  echo "== vLLM $VLLM_VERSION already installed"
else
  echo "== installing vLLM $VLLM_VERSION (a few GB; this takes a while)"
  uv pip install --python .venv/bin/python "vllm==$VLLM_VERSION" --torch-backend=auto
fi
echo "== installing the g4fast plugin and ninja"
uv pip install --python .venv/bin/python --reinstall-package g4fast ./engine ninja
.venv/bin/python -c "import vllm, g4fast; assert vllm.__version__ == '$VLLM_VERSION', vllm.__version__"

# FlashInfer (part of vLLM) compiles some of its kernels on first use, so it needs a CUDA compiler whose
# version matches the CUDA headers. vLLM's own environment mixes versions, so a matching compiler and the
# headers it needs go into .venv/cuda, pinned to the CUDA version of the installed PyTorch. serve.sh uses it.
CUDA_SPEC=$(.venv/bin/python - <<'EOF'
import importlib.metadata as m, sys, torch
cuda = torch.version.cuda or ""
if not cuda.startswith("13."):
    sys.exit(f"install.sh: expected a CUDA 13 build of PyTorch, got {cuda or 'none'}; is the driver 580 or newer?")
v = m.version
nvrtc = v("nvidia-cuda-nvrtc")  # the compiler release that matches this PyTorch build
print(" ".join([f"nvidia-cuda-nvcc=={nvrtc}", f"nvidia-cuda-crt=={nvrtc}", f"nvidia-nvvm=={nvrtc}",
                f"nvidia-cuda-nvrtc=={nvrtc}", f"nvidia-cuda-runtime=={v('nvidia-cuda-runtime')}"]
               + [f"{n}=={v(n)}" for n in ("nvidia-cublas", "nvidia-curand", "nvidia-cusparse",
                                          "nvidia-cusolver", "nvidia-cufft")]))
EOF
)
if [ "$(cat .venv/cuda/.spec 2>/dev/null)" = "$CUDA_SPEC" ] && [ -x .venv/cuda/nvidia/cu13/bin/nvcc ]; then
  echo "== CUDA compiler already installed"
else
  echo "== installing a matching CUDA compiler into .venv/cuda"
  rm -rf .venv/cuda
  # shellcheck disable=SC2086
  uv pip install --target .venv/cuda $CUDA_SPEC
  echo "$CUDA_SPEC" > .venv/cuda/.spec
fi
# the library names and folders the FlashInfer build links against (lib64, unversioned .so, libcuda)
C=.venv/cuda/nvidia/cu13
ln -sfn lib "$C/lib64"
ln -sf libcudart.so.13 "$C/lib/libcudart.so"
ln -sf libnvrtc.so.13 "$C/lib/libnvrtc.so"
mkdir -p "$C/lib/stubs"
if ! /sbin/ldconfig -p | grep -q 'libcuda\.so '; then
  libcuda=$(/sbin/ldconfig -p | awk '/libcuda\.so\.1 / {print $NF; exit}')
  [ -n "$libcuda" ] || fail "libcuda.so.1 (from the NVIDIA driver) not found."
  ln -sf "$libcuda" "$C/lib/stubs/libcuda.so"
fi
"$C/bin/nvcc" --version | tail -1

if ! echo "$TEMPLATE_SHA256  $TEMPLATE" | sha256sum -c --status - 2>/dev/null; then
  echo "== downloading the chat template ($TEMPLATE_REPO)"
  .venv/bin/python - "$TEMPLATE_REPO" "$TEMPLATE_REV" "$TEMPLATE" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
repo, rev, dest = sys.argv[1:]
shutil.copyfile(hf_hub_download(repo, "chat_template.jinja", revision=rev), dest)
EOF
  echo "$TEMPLATE_SHA256  $TEMPLATE" | sha256sum -c --status - \
    || { rm -f "$TEMPLATE"; fail "the chat template's sha256 does not match; not using it."; }
fi

echo "== done. Start the server with ./serve.sh (the model, about 17 GB, downloads on the first start)."

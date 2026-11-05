#!/usr/bin/env bash
# Start the OpenAI-compatible server on 127.0.0.1:8000 (set PORT or HOST to change). Extra arguments go to
# vllm serve. The first start downloads the model and compiles kernels (30 minutes or more); later starts take minutes.
set -euo pipefail
cd "$(dirname "$0")"

TEMPLATE=.venv/gemma-4-26B-A4B-it.chat_template.jinja
[ -x .venv/bin/vllm ] && [ -f "$TEMPLATE" ] && [ -x .venv/cuda/nvidia/cu13/bin/nvcc ] \
  || { echo "serve.sh: run ./install.sh first." >&2; exit 1; }

# The CUDA compiler from install.sh, for the kernels FlashInfer compiles on first use (and ninja from .venv).
export CUDA_HOME="$PWD/.venv/cuda/nvidia/cu13"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
# FlashInfer's compiled kernels go in .venv too, so another vLLM install cannot make it rebuild them.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$PWD/.venv}"
# Each compile job can use about 8 GB of RAM: run one job per 10 GB available (at most one per CPU core).
if [ -z "${MAX_JOBS:-}" ]; then
  avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  MAX_JOBS=$(( avail_gb / 10 )); [ "$MAX_JOBS" -ge 1 ] || MAX_JOBS=1
  [ "$MAX_JOBS" -le "$(nproc)" ] || MAX_JOBS=$(nproc)
  export MAX_JOBS
fi

REV=a19cfe00be84568a6867111c9a68c9c44fdcffe6
# G4F_*: the engine's fast paths. VLLM_DISABLE_COMPILE_CACHE=1 is required: vLLM's compile cache does not
# notice the plugin, so a cached graph would leave out the packed-weight path.
exec env VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  G4F_SAMPLER=1 G4F_MOE=1 G4F_HEAD=1 G4F_PACK=1 VLLM_DISABLE_COMPILE_CACHE=1 \
  .venv/bin/vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
  --revision "$REV" --tokenizer-revision "$REV" \
  --served-model-name gemma-4-26b-a4b-nvfp4 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --seed 0 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --language-model-only \
  --chat-template "$TEMPLATE" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  -cc '{"compile_sizes": [1]}' \
  "$@"

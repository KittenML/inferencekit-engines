#!/usr/bin/env bash
# Run one exactness check with the same environment as serve.sh. Needs the GPU, so stop ./serve.sh first.
#   tools/run.sh test_sampler.py         sampler: same tokens as vLLM's top-k/top-p + Gumbel sampling
#                                        (100 mismatches expected: rows whose logits are all equal, where
#                                        vLLM's kernel does not follow its own lowest-id rule)
#   tools/run.sh test_keptset.py         sampler: same kept token set as vLLM's top-k/top-p mask
#   tools/run.sh test_softcap_monotone.py  the soft-cap is non-decreasing over every BF16 value
#   tools/run.sh test_packed.py          packed weights: every matrix round-trips bit for bit
#   G4F_MOE=1 G4F_MOE_CHECK=1 tools/run.sh test_moe.py   MoE kernels vs vLLM's FlashInfer MoE, per layer
#   G4F_SAMPLER=1 G4F_MOE=1 G4F_HEAD=1 G4F_HEAD_CHECK=1 tools/run.sh test_head.py   pruned output layer vs the full one
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
[ -n "${1:-}" ] || { sed -n 2,10p "$SELF"; exit 2; }
[ -x .venv/cuda/nvidia/cu13/bin/nvcc ] || { echo "tools/run.sh: run ./install.sh first." >&2; exit 1; }
export CUDA_HOME="$PWD/.venv/cuda/nvidia/cu13"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$PWD/.venv}"
if [ -z "${MAX_JOBS:-}" ]; then  # one compile job per 10 GB of available RAM, as serve.sh
  MAX_JOBS=$(( $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) / 10 ))
  [ "$MAX_JOBS" -ge 1 ] || MAX_JOBS=1
  export MAX_JOBS
fi
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
script=$1; shift
exec .venv/bin/python "tools/${script#tools/}" "$@"

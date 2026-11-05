#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
runtime="$root/runtime"
: "${MOSS_RUN_DEADLINE:?set an explicit UTC deadline}"
if [[ -v CONFIG || -v MODEL_DIR || -v OMNI_SOURCE || -v FP8_OVERLAY_ROOT ]]; then
  echo 'CONFIG, MODEL_DIR and OMNI_SOURCE overrides are not supported by this pinned package' >&2
  exit 2
fi
env -i HOME="${HOME:?HOME must be set}" PATH=/usr/bin:/bin LANG=C.UTF-8 python3 -E -s "$root/verify_snapshot.py"
export OMNI_SOURCE="$runtime/vllm-omni" MODEL_DIR="$runtime/model" FP8_OVERLAY_ROOT="$runtime/fp8-overlay"
export ROCM_PATH="$runtime/rocm/opt/rocm-7.2.4" HIP_PATH="$runtime/rocm/opt/rocm-7.2.4"
export PATH="$runtime/venv/bin:$ROCM_PATH/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/opt/rocm-host/lib:/usr/lib/wsl/lib:/opt/rocm-wsl/lib:$ROCM_PATH/lib:$runtime/profiler-sdk/opt/rocm-7.2.4/lib:$runtime/aqlprofile/opt/rocm-7.2.4/lib:$runtime/runtime-libs/usr/lib/x86_64-linux-gnu"
export ROCBLAS_TENSILE_LIBPATH="$ROCM_PATH/lib/rocblas/library" HIPBLASLT_TENSILE_LIBPATH="$ROCM_PATH/lib/hipblaslt/library"
export HSA_ENABLE_DXG_DETECTION=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export VLLM_TARGET_DEVICE=rocm VLLM_OMNI_TARGET_DEVICE=rocm VLLM_ROCM_USE_AITER=0 DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export ROCPROFILER_REGISTER_ENABLED=0 OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0 TORCH_BLAS_PREFER_HIPBLASLT=1
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=0 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH MOSS_TUNING_FILE MOSS_TUNING_SHA256
if [[ -z "${RUN_DIR:-}" ]]; then
  mkdir -p "$root/runs"
  RUN_DIR="$(mktemp -d "$root/runs/run.XXXXXXXX")"
  rmdir "$RUN_DIR"
  export RUN_DIR
fi
env -i HOME="${HOME:?HOME must be set}" PATH="$PATH" LANG=C.UTF-8 \
  OMNI_SOURCE="$OMNI_SOURCE" MODEL_DIR="$MODEL_DIR" FP8_OVERLAY_ROOT="$FP8_OVERLAY_ROOT" RUN_DIR="$RUN_DIR" MOSS_RUN_DEADLINE="$MOSS_RUN_DEADLINE" \
  "$root/engine/prepare-config.sh"
export CONFIG="$RUN_DIR/configuration.json"
env -i HOME="$HOME" PATH="$PATH" LANG=C.UTF-8 python -E -s "$root/verify_candidate_install.py" --configuration "$CONFIG"
# Carry only the controls used in the measured runtime into the inference
# process. Optional device selectors remain explicit for mixed-GPU hosts.
controls=(HOME PATH LD_LIBRARY_PATH OMNI_SOURCE MODEL_DIR FP8_OVERLAY_ROOT MOSS_RUN_DEADLINE CONFIG
  ROCM_PATH HIP_PATH ROCBLAS_TENSILE_LIBPATH HIPBLASLT_TENSILE_LIBPATH
  HSA_ENABLE_DXG_DETECTION OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
  MAX_JOBS TORCHINDUCTOR_COMPILE_THREADS VLLM_TARGET_DEVICE VLLM_OMNI_TARGET_DEVICE
  VLLM_ROCM_USE_AITER DIFFUSION_ATTENTION_BACKEND ROCPROFILER_REGISTER_ENABLED
  OMNI_DIFFUSION_PROMPT_EMBED_CACHE TORCH_BLAS_PREFER_HIPBLASLT
  PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING PYTORCH_TUNABLEOP_RECORD_UNTUNED
  HSA_OVERRIDE_CPU_AFFINITY_DEBUG HF_HUB_OFFLINE TRANSFORMERS_OFFLINE)
env_args=("LANG=C.UTF-8" "PYTHONPATH=$FP8_OVERLAY_ROOT")
for name in "${controls[@]}"; do env_args+=("$name=${!name}"); done
for name in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES INFERENCEKIT_ARM_OWNER; do
  if [[ -v "$name" ]]; then env_args+=("$name=${!name}"); fi
done
# Do not use -E here: the adapter adds its own PYTHONPATH for spawned workers,
# and Python propagates interpreter flags into multiprocessing children.
exec env -i "${env_args[@]}" python -s "$root/engine/adapter/moss_omni_adapter.py" --configuration "$CONFIG" --transport-root "$root/engine/transport" --port "${PORT:-8000}"

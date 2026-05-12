#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export DS_LOG_LEVEL=error
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
fi

## basic setup for the env
export CLUSTER_NAME=""
export HOME_PREFIX="${HOME_PREFIX:-${HOME}}"

# Resolve paths from the actual repo location instead of assuming an extra
# rl-reasoning/Tina directory layer exists locally.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PROJECT_PREFIX="${REPO_ROOT}"
export SCRATCH_PREFIX="${REPO_ROOT}/scratch"
mkdir -p "${HOME_PREFIX}" "${PROJECT_PREFIX}" "${SCRATCH_PREFIX}"

export TOPIC_NAME="Tina"
export CORE_POSTFIX="tina"
export PROJECT_DIR="${REPO_ROOT}"
export HOME_DIR="${REPO_ROOT}"
export PYTHONPATH="${HOME_DIR}:${HOME_DIR}/${CORE_POSTFIX}:${PYTHONPATH:-}"
mkdir -p "${PROJECT_DIR}"

export CKPT_DIR="${PROJECT_DIR}/ckpts"
export DATA_DIR="${PROJECT_DIR}/datasets"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"

if [[ -f "${SCRIPT_DIR}/local_vars.sh" ]]; then
  # Optional untracked local override for tokens and machine-specific settings.
  source "${SCRIPT_DIR}/local_vars.sh"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${TOPIC_NAME}"
export WANDB_DIR="${OUTPUT_DIR}"

if [[ -n "${WANDB_API_KEY}" ]] && command -v wandb >/dev/null 2>&1; then
  wandb login "${WANDB_API_KEY}"
fi

export CACHE_DIR="${PROJECT_DIR}/.cache"
export WANDB_CACHE_DIR="${CACHE_DIR}"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"
mkdir -p "${CACHE_DIR}" "${TRITON_CACHE_DIR}"

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
if [[ -n "${HF_TOKEN}" ]] && command -v hf >/dev/null 2>&1; then
  hf auth login --token "${HF_TOKEN}" --add-to-git-credential
fi

export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export HOME_DIR="${PROJECT_DIR}"

export TOPIC_NAME="Tina"
export CORE_POSTFIX="tina"
export PYTHONPATH="${HOME_DIR}:${HOME_DIR}/${CORE_POSTFIX}:${PYTHONPATH:-}"

export CKPT_DIR="${PROJECT_DIR}/ckpts"
export DATA_DIR="${PROJECT_DIR}/datasets"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"

export WANDB_API_KEY="${WANDB_API_KEY:-TODO}"
export WANDB_PROJECT="${TOPIC_NAME}"
export WANDB_DIR="${OUTPUT_DIR}"

if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != "TODO" ]]; then
    wandb login "${WANDB_API_KEY}"
fi

export CACHE_DIR="${PROJECT_DIR}/.cache"
export WANDB_CACHE_DIR="${CACHE_DIR}"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"
mkdir -p "${CACHE_DIR}" "${WANDB_CACHE_DIR}" "${TRITON_CACHE_DIR}"

export HF_TOKEN="${HF_TOKEN:-TODO}"
if [[ -n "${HF_TOKEN:-}" && "${HF_TOKEN}" != "TODO" ]]; then
    git config --global credential.helper store
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential
fi

export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

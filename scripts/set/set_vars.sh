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

## basic setup for the env
export CLUSTER_NAME=""
export HOME_PREFIX="/root"
export PROJECT_PREFIX="/root"
export SCRATCH_PREFIX="/root/Tina"
mkdir -p "${HOME_PREFIX}" "${PROJECT_PREFIX}" "${SCRATCH_PREFIX}"

export PROJECT_NAME=""
export TOPIC_NAME="Tina"
export CORE_POSTFIX="tina"
export PROJECT_POSTFIX="${TOPIC_NAME}"
export PROJECT_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}"
export HOME_DIR="${HOME_PREFIX}/${PROJECT_POSTFIX}"
export PYTHONPATH="${HOME_DIR}":$PYTHONPATH
export PYTHONPATH="${HOME_DIR}/${CORE_POSTFIX}":$PYTHONPATH
mkdir -p "${HOME_PREFIX}"

export CKPT_DIR="${PROJECT_DIR}/ckpts"
export DATA_DIR="${PROJECT_DIR}/datasets"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${TOPIC_NAME}"
export WANDB_DIR="${OUTPUT_DIR}"

if [ -n "${WANDB_API_KEY}" ]; then
    wandb login "${WANDB_API_KEY}"
fi

export CACHE_DIR="${PROJECT_DIR}/.cache"
export WANDB_CACHE_DIR="${CACHE_DIR}"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
git config --global credential.helper store
if [ -n "${HF_TOKEN}" ]; then
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential
fi

export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

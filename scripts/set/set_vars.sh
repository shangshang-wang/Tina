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
export HOME_PREFIX="TODO"
export PROJECT_PREFIX="TODO"
export SCRATCH_PREFIX="TODO"
mkdir -p "${HOME_PREFIX}" "${PROJECT_PREFIX}" "${SCRATCH_PREFIX}"

export PROJECT_NAME="rl-reasoning"
export TOPIC_NAME="Tina"
export CORE_POSTFIX="tina"
export PROJECT_POSTFIX="${PROJECT_NAME}/${TOPIC_NAME}"
export PROJECT_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}"
export HOME_DIR="${HOME_PREFIX}/${PROJECT_POSTFIX}"
export PYTHONPATH="${HOME_DIR}":$PYTHONPATH
export PYTHONPATH="${HOME_DIR}/${CORE_POSTFIX}":$PYTHONPATH
mkdir -p "${HOME_PREFIX}/${PROJECT_NAME}"

export CKPT_DIR="${PROJECT_DIR}/ckpts"
export DATA_DIR="${PROJECT_DIR}/datasets"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"

export WANDB_API_KEY="TODO"
export WANDB_PROJECT="${TOPIC_NAME}"
export WANDB_DIR="${OUTPUT_DIR}"

wandb login $WANDB_API_KEY

export CACHE_DIR="${PROJECT_DIR}/.cache"
export WANDB_CACHE_DIR="${CACHE_DIR}"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"

export HF_TOKEN="TODO"
git config --global credential.helper store
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
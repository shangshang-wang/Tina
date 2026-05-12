#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
unset CUDA_LAUNCH_BLOCKING
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_IB_DISABLE
export DS_LOG_LEVEL=error
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source "${REPO_ROOT}/scripts/set/set_vars.sh"
export WANDB_PROJECT=Tina_train_model

export WANDB_MODE=offline
export TINA_STOP_AT_STEP="${TINA_STOP_AT_STEP:-250}"

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
PY_SCRIPT="./tina/post_train_hf/grpo.py"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"
CONFIGS=(
  "open_rs3_plan25_ablation"
  "open_rs3_plan35_ablation"
)

for PT_CONFIG_NAME in "${CONFIGS[@]}"; do
  PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/train_model_${PT_CONFIG_NAME}.yaml"
  LOG_FILE="${LOGGING_DIR}/train_${PT_CONFIG_NAME}_gpu${CUDA_VISIBLE_DEVICES//,/}_$(date +%Y%m%d_%H%M%S).log"

  {
    echo "START train ${PT_CONFIG_NAME} on GPU 6,7 $(date)"
    echo "PYTHON ENV: $(conda run -n tina which python)"
    echo "PY_CONFIG: ${PY_CONFIG}"
    echo "ACCELERATE_DS_CONFIG: ${ACCELERATE_DS_CONFIG}"
    echo "TINA_STOP_AT_STEP: ${TINA_STOP_AT_STEP}"
    conda run -n tina accelerate launch \
      --config_file "${ACCELERATE_DS_CONFIG}" \
      --main_process_port=29567 \
      --num_processes=2 \
      "${PY_SCRIPT}" --config "${PY_CONFIG}"
    echo "END train ${PT_CONFIG_NAME} on GPU 6,7 $(date)"
  } 2>&1 | tee -a "${LOG_FILE}"
done

touch "${LOGGING_DIR}/planner_sparse_train_checkpoint${TINA_STOP_AT_STEP}.done"

#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

BASE_MODEL_NAME="${BASE_MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
PT_TYPE="grpo"
PT_CONFIG_NAME="${PT_CONFIG_NAME:-open_rs3_dd_filter}"

PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/train_model_${PT_CONFIG_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

echo ""
echo "Running ${PY_SCRIPT} on model ${BASE_MODEL_NAME} with dataset ${PT_CONFIG_NAME} via ${PT_TYPE}"
echo ""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port="${MAIN_PROCESS_PORT:-29500}" \
    --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}" --cosine_max_len 3584

echo "END TIME: $(date)"
echo "DONE"

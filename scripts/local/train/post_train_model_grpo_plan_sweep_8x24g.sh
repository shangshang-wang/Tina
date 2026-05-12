#!/bin/bash
# Local machine launch helper: assumes a 5+ GPU layout with 4 training GPUs and
# one vLLM GPU on this host.
set -euo pipefail

echo "START TIME: $(date)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-tina}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_NAME}"

echo "CONDA ENV: ${CONDA_DEFAULT_ENV:-unknown}"
echo "PYTHON ENV: $(which python)"
source "./scripts/set/set_vars.sh"
export WANDB_MODE="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
VISIBLE_GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
TRAIN_PROCESS_COUNT="${TRAIN_PROCESS_COUNT:-4}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:4}"

echo ""
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES} (${VISIBLE_GPU_COUNT})"
echo "GRPO train processes: ${TRAIN_PROCESS_COUNT}"
echo "vLLM device: ${VLLM_DEVICE}"
echo ""

if [[ "${VISIBLE_GPU_COUNT}" -lt 5 ]]; then
    echo "This 8x24G sweep expects at least 5 visible GPUs: 4 for training and 1 for vLLM."
    exit 1
fi

BASE_MODEL_NAME="${BASE_MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
PT_TYPE="grpo"
PT_CONFIG_NAME="${PT_CONFIG_NAME:-open_rs3_plan15_ablation}"

PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/train_model_${PT_CONFIG_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

echo ""
echo "Running ${PY_SCRIPT} on model ${BASE_MODEL_NAME} with config ${PT_CONFIG_NAME}"
echo ""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port="${MAIN_PROCESS_PORT:-29515}" \
    --num_processes="${TRAIN_PROCESS_COUNT}" \
    "${PY_SCRIPT}" \
    --config "${PY_CONFIG}" \
    --cosine_max_len 3584 \
    --per_device_train_batch_size 3 \
    --vllm_device "${VLLM_DEVICE}" \
    --vllm_gpu_memory_utilization 0.75

echo "END TIME: $(date)"
echo "DONE"

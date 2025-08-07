#!/bin/bash


MAMBA_ENV="tina"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

# Set the GPUs you want to use (set as 3 GPUs if possible, to align with previous 2 GPUs setting)
export CUDA_VISIBLE_DEVICES=0,1,2
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
## Long Completion Ablation
PT_CONFIG_NAME="open_rs3_small_completion_ablation" # 5k
# open_rs3_small_completion_ablation -> 5k
# open_rs3_medium_completion_ablation -> 7k
# open_rs3_large_completion_ablation -> 9k

PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/train_model_${PT_CONFIG_NAME}.yaml"

GRPO_GPU_COUNT=$(($GPU_COUNT - 1))
if [[ "${GRPO_GPU_COUNT}" == 2 ]]; then
    ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2_3gpus.yaml"
    echo "Using 2 GPUs for GRPO, 1 GPU for VLLM"
elif [[ "${GRPO_GPU_COUNT}" == 3 ]]; then
    ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2_4gpus.yaml"
    echo "Using 3 GPUs for GRPO, 1 GPU for VLLM"
else
    echo "Invalid GPU numbers."
    exit 1
fi

echo ""
echo "Running ${PY_SCRIPT} on model ${BASE_MODEL_NAME} with dataset ${PT_CONFIG_NAME} via ${PT_TYPE}"
echo ""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port=29500 \
    --num_processes="${GRPO_GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}" --cosine_max_len 3584

echo "END TIME: $(date)"
echo "DONE"
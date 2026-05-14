#!/bin/bash


echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}, make sure it is 2 GPUs"
echo ""

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B" # Qwen2.5-1.5B, Qwen2.5-Math-1.5B
PT_TYPE="grpo"
PT_CONFIG_NAME="still"
# main: still, deepscaler, 2thought, fastcurl, l1_exact, l1_max, open_rs3, open_rs2, open_rs1
# extra: limr, open_r1, thoughts
# ablation: limr_large_lr_ablation, limr_small_lr_ablation, limr_large_rank_ablation, limr_medium_rank_ablation, limr_small_rank_ablation, limr_tiny_rank_ablation
# open_rs3_drgrpo_ablation, open_rs3_format_ablation, open_rs3_long_completion_ablation

PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/train_model_${PT_CONFIG_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

echo ""
echo "Running ${PY_SCRIPT} on model ${BASE_MODEL_NAME} with dataset ${PT_CONFIG_NAME} via ${PT_TYPE}"
echo ""

if [[ "${PT_CONFIG_NAME}" == "thoughts" || "${PT_CONFIG_NAME}" == "open_r1" || "${PT_CONFIG_NAME}" == "open_rs3" || "${PT_CONFIG_NAME}" == "open_rs3_drgrpo_ablation" ]]; then
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${ACCELERATE_DS_CONFIG}" \
        --main_process_port=29500 \
        --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}" --cosine_max_len 3584
else
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${ACCELERATE_DS_CONFIG}" \
        --main_process_port=29500 \
        --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}"
fi

echo "END TIME: $(date)"
echo "DONE"
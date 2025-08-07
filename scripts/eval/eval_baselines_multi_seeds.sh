#!/bin/bash


MAMBA_ENV="tina_eval"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}, make sure using 2 GPUs."
echo ""

MODEL_LIST=(
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-Math-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "RUC-AIBOX/STILL-3-1.5B-preview"
    "Intelligent-Internet/II-Thought-1.5B-Preview"
    "agentica-org/DeepScaleR-1.5B-Preview"
    "Nickyang/FastCuRL-1.5B-Preview"
    "l3lab/L1-Qwen-1.5B-Exact"
    "l3lab/L1-Qwen-1.5B-Max"
    "knoveleng/Open-RS1"
    "knoveleng/Open-RS2"
    "knoveleng/Open-RS3"
)
SEED_LIST=(0 1 2 3 4 5 6 7 8 9)

for MODEL_NAME in "${MODEL_LIST[@]}"; do

    for SEED in "${SEED_LIST[@]}"; do

        if [ "${MODEL_NAME}" == "Qwen/Qwen2.5-Math-1.5B" ]; then
            MAX_MODEL_LENGTH=4096
            MAX_NEW_TOKENS=4096
        elif [ "${MODEL_NAME}" == "Qwen/Qwen2.5-1.5B" ]; then
            MAX_MODEL_LENGTH=32768 # 131072
            MAX_NEW_TOKENS=32768 # 131072
        else
            MAX_MODEL_LENGTH=32768
            MAX_NEW_TOKENS=32768
        fi

        tasks=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond") # extra task: olympiadbench

        for TASK in "${tasks[@]}"; do
            echo "Evaluating task: ${TASK} on model ${MODEL_NAME} with seed ${SEED}"
            python ./scripts/eval/run_eval_multi_seeds.py \
                --model "${MODEL_NAME}" \
                --task "${TASK}" \
                --temperature 0.6 \
                --top_p 0.95 \
                --seed "${SEED}" \
                --output_dir "${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_NAME}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --max_model_length "${MAX_MODEL_LENGTH}" \
                --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
                --use_chat_template
        done
    done
done

echo "END TIME: $(date)"
echo "DONE"
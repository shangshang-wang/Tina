#!/bin/bash
# Local machine launch helper: assumes this host has 8 visible 24GB GPUs and
# local checkpoints/caches under scripts/set/set_vars.sh paths.
set -euo pipefail

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_USAGE_STATS_ENABLED=0
export TMPDIR="${PROJECT_DIR}/.cache/ray_tmp"
export RAY_TMPDIR="${TMPDIR}"
mkdir -p "${TMPDIR}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "${GPU_COUNT}" -ne 8 ]; then
    echo "Expected 8 visible GPUs for this local 8x24G run, got ${GPU_COUNT}." >&2
    exit 1
fi

MODEL_PATH="${PROJECT_DIR}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
MODEL_NAME_FOR_OUTPUT="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# DeepSeek-R1 paper/eval alignment:
# - max generation length: 32768
# - temperature: 0.6
# - top_p: 0.95
# - no system prompt
# - multiple sampled runs, here seeds 0-4 to match the other-machine setup
TEMPERATURE=0.6
TOP_P=0.95
MAX_MODEL_LENGTH=131072
MAX_NEW_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.7
SEED_LIST=(0 1 2 3 4)
TASKS=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond")

echo "MODEL_PATH: ${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_COUNT: ${GPU_COUNT}"
echo "DATA_PARALLEL_SIZE: ${GPU_COUNT}"
echo "TEMPERATURE: ${TEMPERATURE}"
echo "TOP_P: ${TOP_P}"
echo "MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "SEEDS: ${SEED_LIST[*]}"
echo "TASKS: ${TASKS[*]}"

trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

for SEED in "${SEED_LIST[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Evaluating task: ${TASK} on ${MODEL_PATH} with seed ${SEED}"
        python ./scripts/eval/run_eval_multi_seeds.py \
            --model "${MODEL_PATH}" \
            --task "${TASK}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --seed "${SEED}" \
            --output_dir "${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_NAME_FOR_OUTPUT}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --max_model_length "${MAX_MODEL_LENGTH}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
            --data_parallel_size "${GPU_COUNT}" \
            --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
            --use_chat_template
    done
done

echo "END TIME: $(date)"
echo "DONE"

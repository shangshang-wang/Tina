#!/bin/bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/node/anaconda3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    if [ "${CONDA_DEFAULT_ENV:-}" != "tina_eval" ]; then
        conda activate tina_eval
    fi
fi

cd "${PROJECT_DIR}"

echo "START TIME: $(date)"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "PYTHON ENV: $(which python)"

# shellcheck disable=SC1091
source "./scripts/set/set_vars.sh"

export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"
if [ -n "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_COUNT="${DATA_PARALLEL_SIZE:-$(python -c "import torch; print(torch.cuda.device_count())")}"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TASKS="${TASKS:-aime24,aime25,amc23,math_500,minerva,gpqa:diamond}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"

case "${MODEL_NAME}" in
    "Qwen/Qwen2.5-Math-1.5B"|"Qwen2.5-Math-1.5B")
        MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-4096}"
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
        ;;
    *)
        MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
        ;;
esac

MODEL_ARGS="pretrained=${MODEL_NAME},dtype=${DTYPE},data_parallel_size=${GPU_COUNT},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:${TEMPERATURE},top_p:${TOP_P},seed:${SEED}}"

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"

echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_COUNT: ${GPU_COUNT}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "TASKS: ${TASKS}"
echo "SEED: ${SEED}"
echo "MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo ""

CHAT_TEMPLATE_ARGS=()
if [ "${USE_CHAT_TEMPLATE}" = "1" ]; then
    CHAT_TEMPLATE_ARGS+=(--use-chat-template)
fi

for TASK in "${TASK_ARRAY[@]}"; do
    TASK="$(echo "${TASK}" | xargs)"
    [ -n "${TASK}" ] || continue

    echo "Evaluating task: ${TASK} on model ${MODEL_NAME}"
    lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
        --custom-tasks ./scripts/eval/run_eval_custom_tasks.py \
        "${CHAT_TEMPLATE_ARGS[@]}" \
        --output-dir "${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_NAME}"
done

echo "END TIME: $(date)"
echo "DONE"

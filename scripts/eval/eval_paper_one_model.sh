#!/bin/bash


echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_COUNT=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")

MAX_MODEL_LENGTH=32768
MAX_NEW_TOKENS=32768

echo ""
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_COUNT: ${GPU_COUNT}, make sure using 2 GPUs."
echo "MODEL_NAME: ${MODEL_NAME}"
echo ""

MODEL_ARGS="pretrained=${MODEL_NAME},dtype=bfloat16,data_parallel_size=${GPU_COUNT},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:0.6,top_p:0.95}"

tasks=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond")

for TASK in "${tasks[@]}"; do
    echo "Evaluating task: ${TASK} on model ${MODEL_NAME}"
    lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
        --custom-tasks ./scripts/eval/run_eval_custom_tasks.py \
        --use-chat-template \
        --output-dir "${OUTPUT_DIR}/${TASK}/42/${MODEL_NAME}"
done

echo "END TIME: $(date)"
echo "DONE"

#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
PT_TYPE="grpo"
PT_CONFIG_NAME="${PT_CONFIG_NAME:-open_rs3_dd_filter}"
CKPT_LIST=("checkpoint-100" "checkpoint-200" "checkpoint-400" "checkpoint-600" "checkpoint-850")
TASKS=(${TASKS:-aime24 aime25 amc23 math_500 minerva})

for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running post-trained model ${PT_CONFIG_NAME} with ckpt ${CKPT}"
    python ./tina/post_train_hf/merge_post_trained_models.py \
      --model_name "${MODEL_NAME}" \
      --adapter_type "${PT_TYPE}_${PT_CONFIG_NAME}" \
      --ckpt "${CKPT}"

    if [ "${MODEL_NAME}" == "Qwen2.5-Math-1.5B" ]; then
        MAX_MODEL_LENGTH=4096
        MAX_NEW_TOKENS=4096
    else
        MAX_MODEL_LENGTH=32768
        MAX_NEW_TOKENS=32768
    fi

    MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${PT_CONFIG_NAME}/${CKPT}-merged"
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,data_parallel_size=${GPU_COUNT},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:0.6,top_p:0.95}"

    for TASK in "${TASKS[@]}"; do
        echo "Evaluating task: ${TASK} on model ${MODEL_NAME} post-trained with ${PT_CONFIG_NAME} (${CKPT})"
        lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
            --custom-tasks ./scripts/eval/run_eval_custom_tasks.py \
            --use-chat-template \
            --output-dir "${OUTPUT_DIR}/${TASK}/42/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}"
    done
done

echo "END TIME: $(date)"
echo "DONE"

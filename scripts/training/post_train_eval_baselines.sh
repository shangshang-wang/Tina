#!/bin/bash


MAMBA_ENV="tina_eval"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1 # make sure all evaluation run on 2 GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: $GPU_COUNT, make sure using 2 GPUs."
echo ""

MODEL_LIST=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "agentica-org/DeepScaleR-1.5B-Preview" "knoveleng/Open-RS1" "knoveleng/Open-RS2" "knoveleng/Open-RS3" "RUC-AIBOX/STILL-3-1.5B-preview")

for MODEL_NAME in "${MODEL_LIST[@]}"; do
    MODEL_ARGS="pretrained=$MODEL_NAME,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    # Define an array of tasks to evaluate
    tasks=("aime24" "math_500" "gpqa:diamond" "aime25" "amc23" "minerva")

    for TASK in "${tasks[@]}"; do
      echo "Evaluating task: $TASK"
      lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
          --custom-tasks ./scripts/training/run_post_train_eval.py \
          --use-chat-template \
          --output-dir "${OUTPUT_DIR}/${MODEL_NAME}/${TASK}"
    done
done

echo "END TIME: $(date)"
echo "DONE"

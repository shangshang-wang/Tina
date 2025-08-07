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

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
PT_CONFIG_NAME="deepscaler"
# main: deepscaler, still, 2thought, open_rs3, open_rs2, open_rs1
# extra: limr, open_r1, thoughts
# ablation: limr_large_lr_ablation, limr_small_lr_ablation, limr_large_rank_ablation, limr_medium_rank_ablation, limr_small_rank_ablation,
# limr_tiny_rank_ablation, open_rs3_drgrpo_ablation, open_rs3_format_ablation, open_rs3_long_completion_ablation

CKPT_LIST=("checkpoint-XXXX")
SEED_LIST=(0 1 2 3 4 5 6 7 8 9)

# loop over all the checkpoints in the list
for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running model post train merging base and adapter for checkpoint: ${CKPT}"
    python  ./tina/post_train_hf/merge_post_trained_models.py \
      --model_name "${MODEL_NAME}" \
      --adapter_type "${PT_TYPE}_${PT_CONFIG_NAME}" \
      --ckpt "${CKPT}"

    MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${PT_CONFIG_NAME}/${CKPT}-merged"

    for SEED in "${SEED_LIST[@]}"; do

        tasks=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond") # extra task: olympiadbench

        if [ "${MODEL_NAME}" == "Qwen2.5-Math-1.5B" ]; then
            MAX_MODEL_LENGTH=4096
            MAX_NEW_TOKENS=4096
        elif [ "${MODEL_NAME}" == "Qwen2.5-1.5B" ]; then
            MAX_MODEL_LENGTH=32768 # 131072
            MAX_NEW_TOKENS=32768 # 131072
        else
            MAX_MODEL_LENGTH=32768
            MAX_NEW_TOKENS=32768
        fi

        for TASK in "${tasks[@]}"; do
            echo "Evaluating task: ${TASK} on model ${MODEL_NAME} with seed ${SEED}"
            python ./scripts/eval/run_eval_multi_seeds.py \
                --model "${MODEL_PATH}" \
                --task "${TASK}" \
                --temperature 0.6 \
                --top_p 0.95 \
                --seed "${SEED}" \
                --output_dir "${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}" \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --max_model_length ${MAX_MODEL_LENGTH} \
                --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
                --use_chat_template
        done
    done
done

echo "END TIME: $(date)"
echo "DONE"
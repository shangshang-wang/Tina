#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
unset CUDA_LAUNCH_BLOCKING
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_IB_DISABLE
export DS_LOG_LEVEL=error
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source "${REPO_ROOT}/scripts/set/set_vars.sh"
export WANDB_PROJECT=Tina_eval_model

export WANDB_MODE=offline

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
CKPT="${CKPT:-checkpoint-250}"
SEED_LIST=(${SEED_LIST:-42})
TASKS=(${TASKS:-aime24 aime25 amc23 math_500 minerva gpqa:diamond})
CONFIGS=(${CONFIGS:-open_rs3_plan25_ablation open_rs3_plan35_ablation})

GPU_COUNT="$(conda run -n tina_eval python -c "import torch; print(torch.cuda.device_count())")"
MAX_MODEL_LENGTH=32768
MAX_NEW_TOKENS=32768

for PT_CONFIG_NAME in "${CONFIGS[@]}"; do
  LOG_FILE="${LOGGING_DIR}/eval_${PT_CONFIG_NAME}_${CKPT}_gpu${CUDA_VISIBLE_DEVICES//,/}_$(date +%Y%m%d_%H%M%S).log"
  {
    echo "START eval ${PT_CONFIG_NAME} ${CKPT} on GPU ${CUDA_VISIBLE_DEVICES} $(date)"
    echo "PYTHON ENV: $(conda run -n tina_eval which python)"
    echo "TASKS: ${TASKS[*]}"
    echo "SEEDS: ${SEED_LIST[*]}"

    conda run -n tina_eval python ./tina/post_train_hf/merge_post_trained_models.py \
      --model_name "${MODEL_NAME}" \
      --adapter_type "${PT_TYPE}_${PT_CONFIG_NAME}" \
      --ckpt "${CKPT}"

    MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${PT_CONFIG_NAME}/${CKPT}-merged"

    for SEED in "${SEED_LIST[@]}"; do
      for TASK in "${TASKS[@]}"; do
        echo "Evaluating task: ${TASK} seed ${SEED} on ${PT_CONFIG_NAME} ${CKPT}"
        conda run -n tina_eval python ./scripts/eval/run_eval_multi_seeds.py \
          --model "${MODEL_PATH}" \
          --task "${TASK}" \
          --temperature 0.6 \
          --top_p 0.95 \
          --seed "${SEED}" \
          --output_dir "${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --max_model_length "${MAX_MODEL_LENGTH}" \
          --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
          --use_chat_template
      done
    done
    echo "END eval ${PT_CONFIG_NAME} ${CKPT} on GPU ${CUDA_VISIBLE_DEVICES} $(date)"
  } 2>&1 | tee -a "${LOG_FILE}"
done

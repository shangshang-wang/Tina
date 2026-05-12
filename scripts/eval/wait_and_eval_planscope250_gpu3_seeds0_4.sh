#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/set/set_vars.sh"

GPU_INDEX="${GPU_INDEX:-3}"
MEMORY_FREE_THRESHOLD_MIB="${MEMORY_FREE_THRESHOLD_MIB:-2000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
PT_CONFIG_NAME="open_rs3_planscope_250"
CKPT="${CKPT:-checkpoint-250}"
SEED_LIST=(${SEED_LIST:-0 1 2 3 4})
TASKS=(${TASKS:-aime24 aime25 amc23 math_500 minerva gpqa:diamond})

LOG_FILE="${LOGGING_DIR}/eval_${PT_CONFIG_NAME}_${CKPT}_gpu${GPU_INDEX}_seeds0_4_$(date +%Y%m%d_%H%M%S).log"

gpu_memory_used_mib() {
  nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

gpu_compute_processes() {
  nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

{
  echo "START wait/eval ${PT_CONFIG_NAME} ${CKPT} on GPU ${GPU_INDEX} $(date)"
  echo "TASKS: ${TASKS[*]}"
  echo "SEEDS: ${SEED_LIST[*]}"

  while true; do
    used_mib="$(gpu_memory_used_mib)"
    procs="$(gpu_compute_processes)"
    if [[ "${used_mib}" -le "${MEMORY_FREE_THRESHOLD_MIB}" && -z "${procs}" ]]; then
      echo "GPU ${GPU_INDEX} is free: used=${used_mib} MiB $(date)"
      break
    fi

    echo "GPU ${GPU_INDEX} busy: used=${used_mib} MiB $(date)"
    if [[ -n "${procs}" ]]; then
      echo "${procs}"
    fi
    sleep "${SLEEP_SECONDS}"
  done

  export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
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

  export WANDB_PROJECT=Tina_eval_model
  export WANDB_MODE=offline

  echo "PYTHON ENV: $(conda run -n tina_eval which python)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  conda run -n tina_eval python ./tina/post_train_hf/merge_post_trained_models.py \
    --model_name "${MODEL_NAME}" \
    --adapter_type "${PT_TYPE}_${PT_CONFIG_NAME}" \
    --ckpt "${CKPT}"

  MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${PT_CONFIG_NAME}/${CKPT}-merged"
  MAX_MODEL_LENGTH=32768
  MAX_NEW_TOKENS=32768

  for SEED in "${SEED_LIST[@]}"; do
    for TASK in "${TASKS[@]}"; do
      echo "Evaluating task: ${TASK} seed ${SEED} on ${MODEL_PATH} $(date)"
      conda run -n tina_eval python ./scripts/eval/run_eval_multi_seeds.py \
        --model "${MODEL_PATH}" \
        --task "${TASK}" \
        --temperature 0.6 \
        --top_p 0.95 \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_model_length "${MAX_MODEL_LENGTH}" \
        --gpu_memory_utilization 0.5 \
        --data_parallel_size 1 \
        --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
        --use_chat_template
    done
  done

  echo "END eval ${PT_CONFIG_NAME} ${CKPT} on GPU ${GPU_INDEX} $(date)"
} 2>&1 | tee -a "${LOG_FILE}"

#!/bin/bash
set -u

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PROJECT_DIR
export HOME_DIR="${PROJECT_DIR}"
export PYTHONPATH="${HOME_DIR}:${HOME_DIR}/tina:${PYTHONPATH:-}"
export CKPT_DIR="${PROJECT_DIR}/ckpts"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logs"
export CACHE_DIR="${PROJECT_DIR}/.cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export RAY_TMPDIR="${RAY_TMPDIR:-${PROJECT_DIR}/scratch/ray}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
mkdir -p "${RAY_TMPDIR}" "${LOGGING_DIR}"

GPU_COUNT="${GPU_COUNT:-1}"

echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_COUNT: ${GPU_COUNT}, running DeepSeek base multi-seed baseline eval."
echo ""

MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
MODEL_OUTPUT_NAME="${MODEL_OUTPUT_NAME:-DeepSeek-R1-Distill-Qwen-1.5B_base}"
SEED_LIST=(${SEED_LIST:-0 1 2 3 4})
TASK_LIST=(${TASK_LIST:-aime24 aime25 amc23 math_500 minerva gpqa:diamond})
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
CONTINUE_ON_EVAL_ERROR="${CONTINUE_ON_EVAL_ERROR:-1}"

for SEED in "${SEED_LIST[@]}"; do
  for TASK in "${TASK_LIST[@]}"; do
    OUTPUT_PATH="${OUTPUT_DIR}/${TASK}/${SEED}/${MODEL_OUTPUT_NAME}"
    SUMMARY_PATH="${OUTPUT_PATH}/${MODEL_PATH//\//_}/${SEED}-0.6-0.95-${TASK}-${MAX_NEW_TOKENS}.json"

    if [[ -f "${SUMMARY_PATH}" ]]; then
      echo "Skipping existing result: seed=${SEED}, task=${TASK}, file=${SUMMARY_PATH}"
      continue
    fi

    echo "Evaluating task: ${TASK} on ${MODEL_PATH} with seed ${SEED}"
    if ! conda run -n tina_eval python ./scripts/eval/run_eval_multi_seeds.py \
      --model "${MODEL_PATH}" \
      --task "${TASK}" \
      --temperature 0.6 \
      --top_p 0.95 \
      --seed "${SEED}" \
      --output_dir "${OUTPUT_PATH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --max_model_length "${MAX_MODEL_LENGTH}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --data_parallel_size "${GPU_COUNT}" \
      --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
      --use_chat_template; then
        echo "Eval failed: seed=${SEED}, task=${TASK}"
        if [[ "${CONTINUE_ON_EVAL_ERROR}" != "1" ]]; then
          exit 1
        fi
    fi
  done
done

echo "END TIME: $(date)"
echo "DONE"

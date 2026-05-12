#!/bin/bash
# Local machine launch helper: queue defaults assume this host's 8-GPU layout and
# local plan15 checkpoint paths.
set -euo pipefail

echo "START TIME: $(date)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-tina}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_NAME}"

echo "CONDA ENV: ${CONDA_DEFAULT_ENV:-unknown}"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export RAY_USAGE_STATS_ENABLED=0
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR="${PROJECT_DIR}/.cache/ray_tmp"
export RAY_TMPDIR="${TMPDIR}"
mkdir -p "${TMPDIR}"

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_TYPE="grpo_open_rs3_plan15_ablation"
CKPT="checkpoint-250"
MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${CKPT}-merged"
MODEL_NAME_FOR_OUTPUT="${MODEL_NAME}_${ADAPTER_TYPE}_${CKPT}"
MODEL_FOLDER="${MODEL_PATH//\//_}"

TEMPERATURE=0.6
TOP_P=0.95
MAX_MODEL_LENGTH=32768
MAX_NEW_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.7
SEED_LIST=(0 1 2 3 4)
TASKS=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond")
GPU_LIST=(0 1 2 3 4 5 6 7)

QUEUE_DIR="${PROJECT_DIR}/.cache/eval_queue/${ADAPTER_TYPE}_${CKPT}_seeds0_4"
CLAIMS_DIR="${QUEUE_DIR}/claims"
LOG_DIR="${LOGGING_DIR}/gpu_queue_${ADAPTER_TYPE}_${CKPT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CLAIMS_DIR}" "${LOG_DIR}"

echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_NAME_FOR_OUTPUT: ${MODEL_NAME_FOR_OUTPUT}"
echo "MODEL_FOLDER: ${MODEL_FOLDER}"
echo "TEMPERATURE: ${TEMPERATURE}"
echo "TOP_P: ${TOP_P}"
echo "MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "SEEDS: ${SEED_LIST[*]}"
echo "TASKS: ${TASKS[*]}"
echo "GPU_LIST: ${GPU_LIST[*]}"
echo "QUEUE_DIR: ${QUEUE_DIR}"
echo "LOG_DIR: ${LOG_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Merged model not found. Merging ${ADAPTER_TYPE}/${CKPT} first."
    python ./tina/post_train_hf/merge_post_trained_models.py \
        --model_name "${MODEL_NAME}" \
        --adapter_type "${ADAPTER_TYPE}" \
        --ckpt "${CKPT}"
fi

task_output_file() {
    local task="$1"
    local seed="$2"
    printf "%s/%s/%s/%s/%s/%s-%s-%s-%s-%s.json" \
        "${OUTPUT_DIR}" "${task}" "${seed}" "${MODEL_NAME_FOR_OUTPUT}" "${MODEL_FOLDER}" \
        "${seed}" "${TEMPERATURE}" "${TOP_P}" "${task}" "${MAX_NEW_TOKENS}"
}

worker() {
    local gpu="$1"
    local worker_log="${LOG_DIR}/gpu${gpu}.log"
    echo "[gpu ${gpu}] worker start $(date)" | tee -a "${worker_log}"

    while true; do
        local claimed=""
        local task=""
        local seed=""

        for seed_candidate in "${SEED_LIST[@]}"; do
            for task_candidate in "${TASKS[@]}"; do
                local out_file
                out_file="$(task_output_file "${task_candidate}" "${seed_candidate}")"
                if [[ -f "${out_file}" ]]; then
                    continue
                fi

                local claim_name
                claim_name="${seed_candidate}__${task_candidate//[:\/]/_}"
                if mkdir "${CLAIMS_DIR}/${claim_name}" 2>/dev/null; then
                    printf "%s\n" \
                        "gpu=${gpu}" \
                        "pid=$$" \
                        "host=$(hostname)" \
                        "time=$(date -Iseconds)" \
                        "task=${task_candidate}" \
                        "seed=${seed_candidate}" > "${CLAIMS_DIR}/${claim_name}/claim.txt"
                    claimed="${claim_name}"
                    task="${task_candidate}"
                    seed="${seed_candidate}"
                    break 2
                fi
            done
        done

        if [[ -z "${claimed}" ]]; then
            echo "[gpu ${gpu}] no remaining unclaimed work $(date)" | tee -a "${worker_log}"
            return 0
        fi

        echo "[gpu ${gpu}] running task=${task} seed=${seed} $(date)" | tee -a "${worker_log}"
        set +e
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export TMPDIR="${PROJECT_DIR}/.cache/ray_tmp/${ADAPTER_TYPE}_${CKPT}_gpu${gpu}"
            export RAY_TMPDIR="${TMPDIR}"
            mkdir -p "${TMPDIR}"
            python ./scripts/eval/run_eval_multi_seeds.py \
                --model "${MODEL_PATH}" \
                --task "${task}" \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --seed "${seed}" \
                --output_dir "${OUTPUT_DIR}/${task}/${seed}/${MODEL_NAME_FOR_OUTPUT}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --max_model_length "${MAX_MODEL_LENGTH}" \
                --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
                --data_parallel_size 1 \
                --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
                --use_chat_template
        ) >> "${worker_log}" 2>&1
        local status=$?
        set -e
        if [[ "${status}" -eq 0 ]]; then
            echo "[gpu ${gpu}] done task=${task} seed=${seed} $(date)" | tee -a "${worker_log}"
            rm -rf "${CLAIMS_DIR:?}/${claimed}"
        else
            echo "[gpu ${gpu}] FAILED status=${status} task=${task} seed=${seed} $(date)" | tee -a "${worker_log}"
            mv "${CLAIMS_DIR}/${claimed}" "${CLAIMS_DIR}/${claimed}.failed.$(date +%s)" 2>/dev/null || true
            return "${status}"
        fi
    done
}

pids=()
for gpu in "${GPU_LIST[@]}"; do
    worker "${gpu}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo "END TIME: $(date)"
if [[ "${status}" -eq 0 ]]; then
    echo "DONE"
else
    echo "FAILED"
fi
exit "${status}"

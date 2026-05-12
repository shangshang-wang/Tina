#!/usr/bin/env bash
set -euo pipefail

# Transparent GPU-idle evaluation queue for Tina checkpoints.
# It waits for the configured GPU group to be idle, then evaluates queued
# checkpoints with normal process names and persistent marker files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/set/set_vars.sh"

export DS_LOG_LEVEL="${DS_LOG_LEVEL:-error}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
PT_TYPE="${PT_TYPE:-grpo}"
TASKS_TEXT="${TASKS_TEXT:-aime24 aime25 amc23 math_500 minerva gpqa:diamond}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"

# Paper Appendix C.2 uses 0.50. The older summary notes local evals used 0.70.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"

# Use all four cards by default. Override, for example:
#   EVAL_GPU_GROUP=0,1 ./scripts/eval/auto_eval_on_free_gpus.sh
EVAL_GPU_GROUP="${EVAL_GPU_GROUP:-0,1,2,3}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
REQUIRED_FREE_CHECKS="${REQUIRED_FREE_CHECKS:-3}"
MAX_GPU_MEM_USED_MB="${MAX_GPU_MEM_USED_MB:-1000}"
MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-10}"
DRY_RUN="${DRY_RUN:-0}"
CLEANUP_MERGED_AFTER_EVAL="${CLEANUP_MERGED_AFTER_EVAL:-0}"
CONTINUE_ON_EVAL_ERROR="${CONTINUE_ON_EVAL_ERROR:-0}"

LOCK_FILE="${LOCK_FILE:-/tmp/tina_auto_eval_on_free_gpus.lock}"
MARKER_DIR="${MARKER_DIR:-${LOGGING_DIR}/auto_eval_markers}"
mkdir -p "${MARKER_DIR}"

AUTO_EVAL_LOG="${AUTO_EVAL_LOG:-${LOGGING_DIR}/auto_eval_on_free_gpus_$(date -u +%Y%m%d_%H%M%S).log}"
trap 'rc=$?; echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] auto-eval exited with rc=${rc} at line ${LINENO}" | tee -a "${AUTO_EVAL_LOG}"; exit ${rc}' ERR
trap 'echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] auto-eval received SIGTERM" | tee -a "${AUTO_EVAL_LOG}"; exit 143' TERM
trap 'echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] auto-eval received SIGHUP" | tee -a "${AUTO_EVAL_LOG}"; exit 129' HUP

DEFAULT_JOBS=(
  # pt_config|checkpoint|seeds|why
  "limr|checkpoint-180|42|unreviewed-final-checkpoint"
  "open_rs3_repo_default|checkpoint-500|42|unreviewed-final-checkpoint"
  "open_rs2|checkpoint-450|0,1,2|repeat-previous-paper-aligned-eval"
  "open_rs3|checkpoint-500|0,1,2|repeat-previous-paper-aligned-eval"
)

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another auto-eval watcher is already running; lock=${LOCK_FILE}" >&2
  exit 1
fi

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "${AUTO_EVAL_LOG}"
}

gpu_count_from_group() {
  local group="$1"
  local commas="${group//[^,]/}"
  echo $((${#commas} + 1))
}

checkpoint_complete() {
  local ckpt_dir="$1"
  [[ -f "${ckpt_dir}/trainer_state.json" && -f "${ckpt_dir}/adapter_model.safetensors" && -f "${ckpt_dir}/adapter_config.json" ]]
}

merged_model_complete() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" && -f "${model_dir}/tokenizer_config.json" && -f "${model_dir}/model.safetensors" ]]
}

gpu_group_free() {
  local group="$1"
  local query_output

  if ! query_output="$(nvidia-smi -i "${group}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    log "nvidia-smi query failed for GPU group ${group}"
    return 1
  fi

  local line idx mem util
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    IFS=',' read -r idx mem util <<<"${line}"
    idx="${idx//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    util="${util//[[:space:]]/}"

    if (( mem > MAX_GPU_MEM_USED_MB )); then
      log "GPU ${idx} is busy: memory=${mem}MB > ${MAX_GPU_MEM_USED_MB}MB"
      return 1
    fi
    if (( util > MAX_GPU_UTIL_PERCENT )); then
      log "GPU ${idx} is busy: util=${util}% > ${MAX_GPU_UTIL_PERCENT}%"
      return 1
    fi
  done <<<"${query_output}"

  return 0
}

wait_for_gpus() {
  local group="$1"
  local free_count=0

  log "Waiting for GPUs ${group} to be idle for ${REQUIRED_FREE_CHECKS} consecutive checks"
  while (( free_count < REQUIRED_FREE_CHECKS )); do
    if gpu_group_free "${group}"; then
      free_count=$((free_count + 1))
      log "GPU group ${group} idle check ${free_count}/${REQUIRED_FREE_CHECKS}"
    else
      free_count=0
    fi

    if (( free_count < REQUIRED_FREE_CHECKS )); then
      sleep "${CHECK_INTERVAL_SECONDS}"
    fi
  done
}

marker_path() {
  local adapter_type="$1"
  local ckpt="$2"
  local seed="$3"
  local mem_tag
  mem_tag="${GPU_MEMORY_UTILIZATION/./p}"
  echo "${MARKER_DIR}/${MODEL_NAME}_${adapter_type}_${ckpt}_seed${seed}_mem${mem_tag}.done"
}

merge_checkpoint_if_needed() {
  local adapter_type="$1"
  local ckpt="$2"
  local adapter_ckpt_dir="${CKPT_DIR}/models/${MODEL_NAME}/${adapter_type}/${ckpt}"
  local merged_model_path="${adapter_ckpt_dir}-merged"

  if ! checkpoint_complete "${adapter_ckpt_dir}"; then
    log "Checkpoint is incomplete, skipping: ${adapter_ckpt_dir}"
    return 1
  fi

  if merged_model_complete "${merged_model_path}"; then
    log "Merged model already exists: ${merged_model_path}"
    return 0
  fi

  log "Merging ${adapter_type} ${ckpt}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1, would merge ${adapter_type} ${ckpt}"
    return 0
  fi

  conda run -n tina python ./tina/post_train_hf/merge_post_trained_models.py \
    --model_name "${MODEL_NAME}" \
    --adapter_type "${adapter_type}" \
    --ckpt "${ckpt}" 2>&1 | tee -a "${AUTO_EVAL_LOG}"
}

run_eval_seed() {
  local pt_config_name="$1"
  local ckpt="$2"
  local seed="$3"
  local why="$4"
  local adapter_type="${PT_TYPE}_${pt_config_name}"
  local adapter_ckpt_dir="${CKPT_DIR}/models/${MODEL_NAME}/${adapter_type}/${ckpt}"
  local merged_model_path="${adapter_ckpt_dir}-merged"
  local marker
  local gpu_count
  local merged_existed_before=0

  marker="$(marker_path "${adapter_type}" "${ckpt}" "${seed}")"
  if [[ -f "${marker}" ]]; then
    log "Marker exists, skipping ${adapter_type} ${ckpt} seed=${seed}: ${marker}"
    return 0
  fi

  if ! checkpoint_complete "${adapter_ckpt_dir}"; then
    log "Checkpoint not ready, skipping ${adapter_type} ${ckpt}: ${adapter_ckpt_dir}"
    return 0
  fi

  wait_for_gpus "${EVAL_GPU_GROUP}"
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU_GROUP}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_tina_eval_${EVAL_GPU_GROUP//,/}_$$}"
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
  unset CUDA_LAUNCH_BLOCKING
  unset NCCL_P2P_DISABLE
  unset NCCL_SHM_DISABLE
  unset NCCL_IB_DISABLE

  if merged_model_complete "${merged_model_path}"; then
    merged_existed_before=1
  fi
  merge_checkpoint_if_needed "${adapter_type}" "${ckpt}"
  gpu_count="$(gpu_count_from_group "${EVAL_GPU_GROUP}")"

  log "Starting eval: ${adapter_type} ${ckpt}, seed=${seed}, gpus=${EVAL_GPU_GROUP}, dp=${gpu_count}, mem=${GPU_MEMORY_UTILIZATION}, reason=${why}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1, would evaluate ${adapter_type} ${ckpt} seed=${seed}"
    return 0
  fi

  local task output_dir
  local eval_failed=0
  for task in ${TASKS_TEXT}; do
    output_dir="${OUTPUT_DIR}/${task}/${seed}/${MODEL_NAME}_${adapter_type}_${ckpt}"
    log "Evaluating task=${task}, output_dir=${output_dir}"
    if ! conda run -n tina_eval python ./scripts/eval/run_eval_multi_seeds.py \
      --model "${merged_model_path}" \
      --task "${task}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --seed "${seed}" \
      --output_dir "${output_dir}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --max_model_length "${MAX_MODEL_LENGTH}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --data_parallel_size "${gpu_count}" \
      --custom_tasks_directory ./scripts/eval/run_eval_custom_tasks.py \
      --use_chat_template 2>&1 | tee -a "${AUTO_EVAL_LOG}"; then
      eval_failed=1
      log "Eval failed: ${adapter_type} ${ckpt}, seed=${seed}, task=${task}"
      if [[ "${CONTINUE_ON_EVAL_ERROR}" != "1" ]]; then
        return 1
      fi
    fi
  done

  if [[ "${CLEANUP_MERGED_AFTER_EVAL}" == "1" && "${merged_existed_before}" == "0" ]]; then
    log "Cleaning temporary merged model: ${merged_model_path}"
    rm -rf "${merged_model_path}"
  fi

  if [[ "${eval_failed}" == "0" ]]; then
    touch "${marker}"
    log "Finished eval: ${adapter_type} ${ckpt}, seed=${seed}; marker=${marker}"
  else
    log "Finished with failed tasks: ${adapter_type} ${ckpt}, seed=${seed}; marker not written"
  fi
}

load_jobs() {
  if [[ -n "${AUTO_EVAL_JOBS_FILE:-}" && -f "${AUTO_EVAL_JOBS_FILE}" ]]; then
    mapfile -t JOBS < <(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "${AUTO_EVAL_JOBS_FILE}")
  else
    JOBS=("${DEFAULT_JOBS[@]}")
  fi
}

main() {
  local job pt_config_name ckpt seeds why seed

  load_jobs
  log "Auto-eval watcher started"
  log "GPU group: ${EVAL_GPU_GROUP}; tasks: ${TASKS_TEXT}"
  log "CLEANUP_MERGED_AFTER_EVAL=${CLEANUP_MERGED_AFTER_EVAL}"
  log "CONTINUE_ON_EVAL_ERROR=${CONTINUE_ON_EVAL_ERROR}"
  log "Jobs queued: ${#JOBS[@]}; log=${AUTO_EVAL_LOG}"

  for job in "${JOBS[@]}"; do
    IFS='|' read -r pt_config_name ckpt seeds why <<<"${job}"
    if [[ -z "${pt_config_name}" || -z "${ckpt}" || -z "${seeds}" ]]; then
      log "Malformed job, skipping: ${job}"
      continue
    fi

    IFS=',' read -ra seed_list <<<"${seeds}"
    for seed in "${seed_list[@]}"; do
      seed="${seed//[[:space:]]/}"
      [[ -z "${seed}" ]] && continue
      run_eval_seed "${pt_config_name}" "${ckpt}" "${seed}" "${why:-manual}"
    done
  done

  log "Auto-eval queue finished"
}

main "$@"

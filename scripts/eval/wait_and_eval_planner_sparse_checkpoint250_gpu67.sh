#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/set/set_vars.sh"

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
CKPT="${CKPT:-checkpoint-250}"
CONFIGS=(
  "open_rs3_plan25_ablation"
  "open_rs3_plan35_ablation"
)

echo "START wait for planner sparse ${CKPT} checkpoints $(date)"
while true; do
  missing=0
  for cfg in "${CONFIGS[@]}"; do
    path="${CKPT_DIR}/models/${MODEL_NAME}/grpo_${cfg}/${CKPT}"
    if [[ ! -d "${path}" ]]; then
      missing=1
      echo "Waiting for ${path} $(date)"
      break
    fi
  done

  if [[ "${missing}" -eq 0 ]] && ! tmux has-session -t planner_sparse_train 2>/dev/null; then
    break
  fi

  if [[ "${missing}" -eq 0 ]]; then
    echo "All checkpoints are present; waiting for planner_sparse_train tmux session to exit $(date)"
  fi

  sleep 600
done

echo "All ${CKPT} checkpoints found; starting eval $(date)"
exec "${REPO_ROOT}/scripts/eval/eval_planner_sparse_checkpoint250_gpu67.sh"

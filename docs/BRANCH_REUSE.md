# Branch Upload and Reuse Notes

This branch contains reusable offline diagnostics, PlanScope / plan-prefix GRPO
support, filtered-data utilities, recipes, evaluation helpers, and local
runbooks.

## 0. Current Sync Policy

Keep reusable source, recipes, and small documentation in Git. Keep machine
artifacts out of Git.

Reusable:

```text
tina/analysis/
tina/post_train_hf/
tina/utils/
recipes/
scripts/analysis/
scripts/data/
scripts/eval/
scripts/train/
scripts/set/
docs/
```

Local-only:

```text
ckpts/
datasets/
outputs/
logs/
.cache/
wandb/
scratch/
scripts/set/local_vars.sh
```

Tokens must not be committed. `scripts/set/set_vars.sh` reads `WANDB_API_KEY`
and `HF_TOKEN` from the shell, or from the optional untracked
`scripts/set/local_vars.sh`.

Example local secret file:

```bash
export WANDB_API_KEY="..."
export HF_TOKEN="..."
```

Run this before staging:

```bash
rg -n 'hf_[A-Za-z0-9_-]+|wandb_[A-Za-z0-9_-]+' \
  --glob '!ckpts/**' --glob '!datasets/**' --glob '!outputs/**' \
  --glob '!logs/**' --glob '!.cache/**'
```

Expected result: no real tokens.

## 1. Push This Branch to Your Fork

Use a fork remote when your SSH key authenticates as that fork owner. On this
machine, SSH authenticates as `seedsaw`, so push to the `seedsaw` fork rather
than `shangshang-wang/Tina` unless that account has write access there.

```bash
cd /path/to/Tina
git remote -v
git status --short
git push -u seedsaw analysis/offline-diagnostics
```

If the fork remote is missing:

```bash
git remote add seedsaw git@github.com:seedsaw/Tina.git
git push -u seedsaw analysis/offline-diagnostics
```

Before pushing, check that generated artifacts are not staged:

```bash
git diff --cached --name-only | rg '^(ckpts|datasets|outputs|logs|\.cache|wandb|scratch)/'
git diff --cached --name-only | rg 'scripts/set/local_vars.sh'
```

Both commands should print nothing. These paths are intentionally ignored:

```text
/ckpts/
/datasets/
/outputs/
/logs/
/.cache/
/wandb/
/scratch/
scripts/set/local_vars.sh
```

## 2. Reuse on Another Machine

Clone the fork and check out the branch:

```bash
git clone git@github.com:seedsaw/Tina.git
cd Tina
git checkout analysis/offline-diagnostics
```

Set up the environment using the target machine's own CUDA, Python, and package
versions. Do not copy local cache, checkpoint, dataset, output, or log
directories from one machine into Git.

Then configure local paths:

```bash
source ./scripts/set/set_vars.sh
```

The script derives `PROJECT_DIR` from the repository location and creates local
artifact directories:

```text
ckpts/
datasets/
outputs/
logs/
.cache/
```

These directories are for local runtime artifacts only.

Provide tokens from the local shell, not from committed files:

```bash
export HF_TOKEN="..."
export WANDB_API_KEY="..."
source ./scripts/set/set_vars.sh
```

or create the untracked file:

```text
scripts/set/local_vars.sh
```

## 3. What Is Portable

Reusable code and entry points:

```text
scripts/analysis/
scripts/data/
scripts/eval/
scripts/train/
tina/analysis/
tina/post_train_hf/
tina/utils/
recipes/
docs/
```

Typical commands:

```bash
python scripts/analysis/analyze_plan_blocks.py --help
python scripts/analysis/compare_rollouts.py --help
python scripts/analysis/analyze_token_signals.py --help
python scripts/data/probe_desirable_difficulty.py --help
bash scripts/train/post_train_model_grpo.sh
bash scripts/train/post_train_model_grpo_dd_filter.sh
bash scripts/eval/eval_dd_filter_checkpoints.sh
bash scripts/eval/auto_eval_on_free_gpus.sh
bash scripts/eval/eval_planscope250_single_gpu_seed.sh
bash scripts/eval/eval_planscope250_gpu67_seeds0_4.sh
```

The PlanScope / plan-prefix implementation is spread across:

```text
tina/post_train_hf/grpo_config.py
tina/post_train_hf/grpo_trainer.py
tina/post_train_hf/grpo.py
tina/post_train_hf/rewards.py
tina/utils/prompt.py
tina/utils/constant.py
recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/train_model_open_rs3_*ablation.yaml
recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/train_model_open_rs3_plan*.yaml
recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/train_model_open_rs3_planscope_250.yaml
```

Both config spellings are currently supported:

```yaml
# Newer filtered-data branch spelling
token_loss_mask_type: plan_prefix
plan_format_anchor_radius: 4
plan_answer_anchor_tokens: 32

# Backward-compatible local spelling
token_loss_mask: plan_prefix
format_anchor_width: 8
answer_anchor_width: 32
```

PlanScope-GRPO requires KL on all tokens:

```yaml
use_plan_scaffold: true
kl_all_tokens: true
rl_post_train_reward_funcs:
  - format
  - plan_format
  - cosine
```

Filtered-data recipes expect JSONL files under paths such as:

```text
datasets/desirable_difficulty/open_rs3/open_rs3_filtered.jsonl
datasets/desirable_difficulty/open_rs2/open_rs2_filtered.jsonl
datasets/desirable_difficulty/limr/limr_filtered.jsonl
```

Those JSONL files are data artifacts and should be created or copied locally,
not committed.

## 4. What Is Local-Machine Specific

Machine-specific scripts and environment notes live under:

```text
scripts/local/
```

They may assume this machine's GPU layout, local checkpoint paths, offline cache
settings, queue behavior, or package build details. Treat them as examples or
runbooks, not portable Tina APIs.

This repository also keeps machine observations under:

```text
docs/ENVIRONMENT_SETUP_NOTES.md
docs/EVAL_RESULTS_SUMMARY.md
docs/PRIORITY_MULTI_SEED_EVAL_SUMMARY.md
```

These documents are small and useful for comparing machines, but referenced
paths such as `/home/wangls/Tina` are examples from one machine.

## 5. Safe Commit Checklist

Before committing or pushing:

```bash
git status --short
git status --ignored --short | rg '^(!! )?(ckpts|datasets|outputs|logs|\.cache|wandb|scratch)/'
git diff --cached --name-only | rg '^(ckpts|datasets|outputs|logs|\.cache|wandb|scratch)/'
git diff --cached --name-only | rg 'scripts/set/local_vars.sh'
```

Expected result:

- `ckpts/`, `datasets/`, `outputs/`, `logs/`, `.cache/`, `wandb/`, and `scratch/`
  may appear as ignored.
- They must not appear as staged files.
- `scripts/set/local_vars.sh` must not be staged.
- Only code, configs, recipes, and small documentation files should be staged.

Optional syntax smoke test:

```bash
python -m py_compile \
  tina/config.py \
  tina/post_train_hf/callback.py \
  tina/post_train_hf/grpo.py \
  tina/post_train_hf/grpo_config.py \
  tina/post_train_hf/grpo_trainer.py \
  tina/post_train_hf/preprocess.py \
  tina/post_train_hf/rewards.py \
  tina/utils/constant.py \
  scripts/data/probe_desirable_difficulty.py \
  scripts/data/merge_desirable_difficulty_shards.py \
  scripts/eval/run_eval_custom_tasks.py \
  scripts/eval/run_eval_multi_seeds.py
```

## 6. Pulling the Other Machine's Code

Preferred workflow:

```bash
git status --short
git fetch --all --prune
git log --oneline --decorate --graph --all -n 30
git merge seedsaw/analysis/offline-diagnostics
```

Conflict policy:

- Keep `.gitignore` and `set_vars.sh` secret-safe.
- Prefer repo-location-derived paths over hardcoded absolute paths.
- Preserve both machines' recipes when names differ.
- If two recipes share a filename but differ in machine-specific GPU settings,
  keep the portable recipe in `recipes/` and put machine-specific launch details
  under `scripts/local/`.
- Do not resolve conflicts by deleting local checkpoint/output directories;
  those directories should be ignored and handled outside Git.

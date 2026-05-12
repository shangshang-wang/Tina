# Branch Upload and Reuse Notes

This branch contains reusable offline diagnostics, PlanScope / plan-prefix GRPO
support, filtered-data utilities, recipes, and local runbooks.

## 1. Push This Branch to Your Fork

Use the fork remote, not the official Tina remote:

```bash
cd /path/to/Tina
git remote -v
git status --short
git push seedsaw analysis/offline-diagnostics
```

If the fork remote is missing:

```bash
git remote add seedsaw git@github.com:seedsaw/Tina.git
git push -u seedsaw analysis/offline-diagnostics
```

Before pushing, check that generated artifacts are not staged:

```bash
git diff --cached --name-only | rg '^(ckpts|datasets|outputs|logs|\.cache|wandb)/'
```

That command should print nothing. These paths are intentionally ignored:

```text
/ckpts/
/datasets/
/outputs/
/logs/
/.cache/
/wandb/
```

## 2. Reuse on Another Machine

Clone your fork and check out the branch:

```bash
git clone git@github.com:seedsaw/Tina.git
cd Tina
git checkout analysis/offline-diagnostics
```

Set up the environment using the target machine's own CUDA, Python, and package
versions. Do not copy local cache, checkpoint, dataset, output, or log
directories from this machine into Git.

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

## 3. What Is Portable

Reusable code and entry points:

```text
scripts/analysis/
scripts/data/
scripts/eval/
scripts/train/
tina/analysis/
tina/post_train_hf/
recipes/
```

Typical commands:

```bash
python scripts/analysis/analyze_plan_blocks.py --help
python scripts/analysis/compare_rollouts.py --help
python scripts/analysis/analyze_token_signals.py --help
python scripts/data/probe_desirable_difficulty.py --help
bash scripts/train/post_train_model_grpo_dd_filter.sh
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

The local environment note is:

```text
scripts/local/ENVIRONMENT.md
```

## 5. Safe Commit Checklist

Before committing or pushing:

```bash
git status --short
git status --ignored --short | rg '^(!! )?(ckpts|datasets|outputs|logs|\.cache|wandb)/'
git diff --cached --name-only | rg '^(ckpts|datasets|outputs|logs|\.cache|wandb)/'
```

Expected result:

- `ckpts/`, `datasets/`, `outputs/`, `logs/`, `.cache/`, and `wandb/` may appear
  as ignored.
- They must not appear as staged files.
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


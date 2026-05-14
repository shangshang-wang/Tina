# Checkpoint Sentinel Eval

This document describes the lightweight checkpoint-selection eval set built from
the existing DeepSeek-R1-Distill-Qwen-1.5B base and Open-RS3 full benchmark
results.

The sentinel set is not a final benchmark. It is a fixed checkpoint-selection
set chosen from the target benchmarks by comparing per-item behavior between:

- base: `DeepSeek-R1-Distill-Qwen-1.5B_base`
- baseline: `grpo_open_rs3_checkpoint-500`

The most important items are `base_wrong_baseline_right`, because they are
questions where Open-RS3 improved over the base model. A new checkpoint that
cannot recover these items is unlikely to rank well in a full benchmark run.
`both_wrong` items are included to observe breakthrough potential, but their
share is limited because too many hard unsolved items increase variance.

## Tracked Sentinel Files

The generated sentinel artifacts are tracked under:

```text
outputs/ckpt_sentinel/
  sentinel_all.jsonl
  sentinel_by_benchmark/
    aime24.jsonl
    aime25.jsonl
    amc23.jsonl
    math500.jsonl
    gpqa.jsonl
    minerva.jsonl
  sentinel_weights.json
  sentinel_selection_report.md
  sentinel_selection_report.json
  candidate_pool.jsonl
```

Current size and allocation:

| benchmark | items |
| --- | ---: |
| aime24 | 6 |
| aime25 | 6 |
| amc23 | 12 |
| math500 | 56 |
| gpqa | 35 |
| minerva | 35 |
| total | 150 |

Current label distribution:

| label | items |
| --- | ---: |
| `base_wrong_baseline_right` | 53 |
| `both_wrong` | 59 |
| `base_right_baseline_wrong` | 21 |
| `both_right` | 17 |

The construction report is:

```text
outputs/ckpt_sentinel/sentinel_selection_report.md
```

## Build Or Rebuild The Sentinel

Use existing LightEval per-item details under `outputs/`:

```bash
python scripts/build_ckpt_sentinel_eval.py \
  --benchmark_dir outputs \
  --dataset_dir datasets \
  --base_model_name DeepSeek-R1-Distill-Qwen-1.5B_base \
  --baseline_model_name grpo_open_rs3_checkpoint-500 \
  --output_dir outputs/ckpt_sentinel \
  --seed 42 \
  --sentinel_size 150 \
  --max_new_tokens 32768
```

The builder supports:

- LightEval `details_*.parquet`
- JSONL per-item result files
- JSON files that contain per-item arrays under common keys

For LightEval parquet inputs, install `pandas` and `pyarrow` in the Python
environment used to run the script.

The builder is deterministic for a fixed input set and seed. Do not retune the
sentinel items against new method results. If a new sentinel is intentionally
created, keep it as a separate versioned artifact.

Useful debugging flags:

```bash
python scripts/build_ckpt_sentinel_eval.py \
  --benchmark_dir outputs \
  --dataset_dir datasets \
  --base_model_name DeepSeek-R1-Distill-Qwen-1.5B_base \
  --baseline_model_name grpo_open_rs3_checkpoint-500 \
  --output_dir outputs/ckpt_sentinel_debug \
  --seed 42 \
  --sentinel_size 24 \
  --dry_run \
  --limit_per_benchmark 4
```

## Evaluate Candidate Checkpoints

Evaluate explicit LoRA adapter checkpoints:

```bash
python scripts/eval_ckpt_sentinel.py \
  --base_model ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base \
  --adapter_root ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3_repo_default \
  --sentinel_file outputs/ckpt_sentinel/sentinel_all.jsonl \
  --weights_file outputs/ckpt_sentinel/sentinel_weights.json \
  --output_dir outputs/ckpt_selection \
  --seed 42 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_new_tokens 4096 \
  --batch_size 16 \
  --candidate_checkpoints checkpoint-100,checkpoint-200,checkpoint-300
```

The evaluator loads the base model once and then loads LoRA adapters for each
candidate checkpoint. Full model checkpoint directories are also supported when
passed via `--checkpoint_paths`.

Outputs:

```text
outputs/ckpt_selection/
  sentinel_eval_results/
    checkpoint-100.jsonl
    checkpoint-200.jsonl
    checkpoint-300.jsonl
  ckpt_sentinel_scores.csv
  ckpt_sentinel_scores.json
  ckpt_selection_report.md
  top_candidates.json
```

The primary score is:

1. compute item-weighted accuracy inside each benchmark;
2. average benchmark scores with equal benchmark weights.

The evaluator also reports unweighted overall accuracy, but that is not used
for main checkpoint ranking.

Important derived metrics:

- `base_wrong_baseline_right_recovery_rate`: does the checkpoint preserve the
  Open-RS3 improvement over base?
- `base_right_baseline_wrong_forgetting_rate`: does the checkpoint still fail
  items that base solved but Open-RS3 regressed on?
- `both_wrong_breakthrough_rate`: does the checkpoint solve items both base and
  Open-RS3 missed?
- `both_right_retention_rate`: does the checkpoint keep easy sanity items?

## Auto-Select Candidates From Trainer Logs

The evaluator can preselect candidate checkpoints from a trainer log:

```bash
python scripts/eval_ckpt_sentinel.py \
  --base_model ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base \
  --adapter_root ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3_repo_default \
  --sentinel_file outputs/ckpt_sentinel/sentinel_all.jsonl \
  --weights_file outputs/ckpt_sentinel/sentinel_weights.json \
  --output_dir outputs/ckpt_selection \
  --trainer_log ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3_repo_default/trainer_state.json \
  --auto_select_candidates true \
  --max_candidates 10 \
  --seed 42 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_new_tokens 4096 \
  --batch_size 16
```

The auto-selector:

- computes reward moving average;
- computes format reward moving average;
- drops low format-reward checkpoints;
- drops extreme completion-length checkpoints;
- drops top reward-std checkpoints;
- keeps high reward moving-average checkpoints;
- force-includes reward max, format-reward max, baseline-length-nearest,
  mid-training, and late non-abnormal checkpoints.

When `--adapter_root` is provided, log steps are mapped to the nearest existing
`checkpoint-N` directory so the selected checkpoints are actually evaluable.

The auto-selected checkpoint list is written to:

```text
outputs/ckpt_selection/candidate_checkpoints.json
```

## Dry Runs

Check candidate resolution and output wiring without loading a model:

```bash
python scripts/eval_ckpt_sentinel.py \
  --base_model ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base \
  --adapter_root ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3_repo_default \
  --sentinel_file outputs/ckpt_sentinel/sentinel_all.jsonl \
  --weights_file outputs/ckpt_sentinel/sentinel_weights.json \
  --output_dir outputs/ckpt_selection_dry_run \
  --candidate_checkpoints checkpoint-250 \
  --dry_run \
  --dry_run_no_model
```

## Eval Harness Compatibility

The current fast evaluator uses the project fallback verifier from
`tina.analysis.rollout_utils`, including math answer extraction and GPQA letter
extraction. This is suitable for quick checkpoint selection.

For final reporting, continue to use the full existing benchmark harness. If
the project LightEval harness is extended to accept a fixed item manifest,
`sentinel_all.jsonl` can be used as that manifest while preserving the same
fixed item IDs, prompts, gold answers, labels, and item weights.

## Promotion To Full Benchmarks

After sentinel evaluation, use:

```text
outputs/ckpt_selection/top_candidates.json
```

Run the full benchmark suite on the top 2-3 checkpoints. If the top sentinel
scores are close, run full benchmarks for all close candidates rather than
trusting a narrow sentinel margin.

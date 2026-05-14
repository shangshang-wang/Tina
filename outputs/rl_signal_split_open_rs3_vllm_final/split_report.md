# RL Signal Split Report

## Data Overview

- total samples: 7000
- valid samples: 7000
- mean pass_rate: 0.0836
- mean reward_mean: 0.0836
- mean reward_std: 0.0880
- mean length tokens: 1808.84
- mean format_valid_rate: 0.9998

## Split Statistics

| split | size | unique | pass_rate | reward_std | length | answer_diversity | format_valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 257 | 257 | 0.8575 | 0.2828 | 1106.43 | 0.2490 | 1.0000 |
| medium | 766 | 766 | 0.3854 | 0.4675 | 1476.81 | 0.5692 | 1.0000 |
| hard | 5977 | 5977 | 0.0117 | 0.0310 | 1881.59 | 0.7879 | 0.9997 |
| high_variance | 2100 | 2100 | 0.2454 | 0.2930 | 1629.30 | 0.7093 | 0.9999 |
| long_reasoning | 1750 | 1750 | 0.0103 | 0.0174 | 2048.00 | 0.8164 | 0.9997 |
| core_medium | 383 | 383 | 0.3776 | 0.4660 | 1700.31 | 0.5819 | 1.0000 |
| mixed_balanced | 2000 | 2000 | 0.2090 | 0.1700 | 1736.71 | 0.6739 | 0.9998 |

## Overlap With Other Splits

- easy: medium: 0, hard: 0, high_variance: 187, long_reasoning: 3, core_medium: 0, mixed_balanced: 257
- medium: easy: 0, hard: 0, high_variance: 766, long_reasoning: 29, core_medium: 383, mixed_balanced: 468
- hard: easy: 0, medium: 0, high_variance: 1147, long_reasoning: 1718, core_medium: 0, mixed_balanced: 1275
- high_variance: easy: 187, medium: 766, hard: 1147, long_reasoning: 481, core_medium: 383, mixed_balanced: 1007
- long_reasoning: easy: 3, medium: 29, hard: 1718, high_variance: 481, core_medium: 29, mixed_balanced: 561
- core_medium: easy: 0, medium: 383, hard: 0, high_variance: 383, long_reasoning: 29, mixed_balanced: 383
- mixed_balanced: easy: 257, medium: 468, hard: 1275, high_variance: 1007, long_reasoning: 561, core_medium: 383

## Distribution Analysis

- split_visualization.png

## Formulas

- difficulty_score: `1 - pass_rate`
- variance_score: `mean(robust_percentile_norm(reward_std), robust_percentile_norm(answer_diversity), robust_percentile_norm(correctness_entropy))`
- long_reasoning_score: `mean(robust_percentile_norm(length_mean_tokens), robust_percentile_norm(length_p90_tokens))`
- instability_score: `mean(variance_score, robust_percentile_norm(invalid_count), robust_percentile_norm(completion_length_cv))`
- robust_percentile_norm: `clip((x - dataset_p5) / max(dataset_p95 - dataset_p5, eps), 0, 1)`

## Warnings

- none

## Training Advice

- medium split is small; the dataset may be too easy or too hard for the reference model, reducing dense RL learning signal.
- hard split is large; expect sparse rewards and consider curriculum, stronger verifier feedback, or mixing easier samples.

## Reproducibility

- input: `/root/Tina/.cache/huggingface/hub/datasets--knoveleng--open-rs/snapshots/8de2b2d11162d97b45c7606902eae2bb1ff629b4/data/train-00000-of-00001.parquet`
- model: `/root/Tina/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- seed: `42`
- n_samples: `8`
- max_new_tokens: `2048`
- max_prompt_tokens: `512`
- temperature: `0.7`
- top_p: `0.95`
- finalized_from_shards: `['outputs/rl_signal_split_open_rs3_vllm_shard0', 'outputs/rl_signal_split_open_rs3_vllm_shard1']`
- timestamp_utc: `2026-05-14T09:30:28.686400+00:00`
- git_commit: `unavailable`

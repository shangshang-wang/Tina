# Checkpoint Sentinel Selection Report

This sentinel is not a final evaluation set. It is a fixed checkpoint-selection set chosen from final target benchmarks by comparing base and Open-RS3 per-item behavior.

The highest-value items are `base_wrong_baseline_right`: they are questions where Open-RS3 preserved or learned capability over the base model. New checkpoints that fail these items are unlikely to rank well on the full benchmark. `both_wrong` items are included to observe breakthroughs, but their share is limited to control variance.

## Inputs

- benchmark_dir: `outputs`
- dataset_dir: `datasets`
- base_model_name: `DeepSeek-R1-Distill-Qwen-1.5B_base`
- baseline_model_name: `grpo_open_rs3_checkpoint-500`
- seed: `42`
- sentinel_size: `150`
- base files loaded: `30`
- baseline files loaded: `30`

## Full Benchmark Scores From Loaded Candidate Pool

| benchmark | base | open-rs3 | candidate items | base_wrong_baseline_right | base_right_baseline_wrong | both_wrong | both_right | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aime24 | 0.2333 | 0.2333 | 30 | 1 | 1 | 22 | 6 | 6 |
| aime25 | 0.2000 | 0.2667 | 30 | 2 | 0 | 22 | 6 | 6 |
| amc23 | 0.7250 | 0.7750 | 40 | 4 | 2 | 7 | 27 | 12 |
| math500 | 0.8760 | 0.8680 | 500 | 13 | 17 | 49 | 421 | 56 |
| gpqa | 0.2677 | 0.2778 | 198 | 20 | 18 | 125 | 35 | 35 |
| minerva | 0.2831 | 0.3088 | 272 | 16 | 9 | 179 | 68 | 35 |

## Sentinel Selection

| benchmark | selected | B wrong/O right | B right/O wrong | both wrong | both right | selected IDs |
| --- | --- | --- | --- | --- | --- | --- |
| aime24 | 6 | 1 | 1 | 3 | 1 | 1, 14, 25, 27, 4, 8 |
| aime25 | 6 | 2 | 0 | 3 | 1 | 11, 14, 16, 18, 24, 3 |
| amc23 | 12 | 4 | 2 | 5 | 1 | 11, 13, 21, 3, 31, 32, 33, 35, 39, 7, 8, 9 |
| math500 | 56 | 13 | 8 | 29 | 6 | 10, 111, 122, 132, 138, 143, 151, 162, 163, 179, 187, 188, 201, 203, 225, 237, 238, 251, 253, 265, 291, 307, 325, 338, 358, 384, 387, 388, 390, 402, 405, 408, 41, 412, 416, 42, 426, 431, 435, 436 |
| gpqa | 35 | 17 | 5 | 9 | 4 | 1, 108, 113, 125, 134, 142, 145, 154, 157, 163, 167, 168, 17, 171, 180, 181, 184, 194, 20, 3, 32, 33, 35, 57, 60, 7, 73, 75, 8, 83, 87, 89, 92, 96, 99 |
| minerva | 35 | 16 | 5 | 10 | 4 | 102, 108, 109, 121, 130, 133, 135, 147, 15, 150, 156, 165, 190, 192, 213, 214, 219, 221, 223, 231, 233, 237, 24, 25, 254, 258, 26, 267, 39, 51, 65, 85, 89, 96, 99 |

## Small-Sample Risk

AIME24, AIME25, and AMC23 are deliberately capped. They are useful for catching reasoning regressions, but a checkpoint should not be selected or rejected from these small tasks alone.

## Selection Config

Benchmark allocation: `{"aime24": 6, "aime25": 6, "amc23": 12, "gpqa": 35, "math500": 56, "minerva": 35}`

Within each benchmark, deterministic stratified sampling targets 50% `base_wrong_baseline_right`, 15% `base_right_baseline_wrong`, 25% `both_wrong`, and 10% `both_right`, with shortfall fill priority `base_wrong_baseline_right -> both_wrong -> base_right_baseline_wrong -> both_right`.

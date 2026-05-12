# Tina 重点重跑多 Seed 评测汇总

更新时间：2026-05-10 15:45 UTC

本文档汇总 2026-05-09 至 2026-05-10 对重点 checkpoint 和 DeepSeek base baseline 重新评测的结果。指标为 `extractive_match`，表中数值均为百分比。

## 评测设置

| 项 | 值 |
| --- | --- |
| 评测 seeds | `0, 1, 2, 3, 4` |
| 任务 | `aime24`, `aime25`, `amc23`, `math_500`, `minerva`, `gpqa:diamond` |
| backend | `lighteval` + `vLLM` |
| dtype | `bfloat16` |
| generation | `max_new_tokens=32768`, `temperature=0.6`, `top_p=0.95` |
| context | `max_model_length=32768`, chat template enabled |
| GPU memory | 新重跑主要使用 `gpu_memory_utilization=0.45`；早先 seed42 旧结果多为 `0.70`，不混入均值 |

## 评测对象

| 简称 | checkpoint | 说明 |
| --- | --- | --- |
| DeepSeek base baseline | `ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base` | 未经过 Tina GRPO/LoRA 后训练的本地基线模型。 |
| 仓库当前配置重跑 Open-RS3 repo-default | `grpo_open_rs3_repo_default/checkpoint-500` | 当前仓库默认诊断配置：num_generations=6、per_device_train_batch_size=6。 |
| 论文 Open-RS3 best checkpoint | `grpo_open_rs3/checkpoint-500` | paper-aligned Open-RS3，论文建议最优点附近。 |
| 论文 Open-RS2 best checkpoint | `grpo_open_rs2/checkpoint-450` | paper-aligned Open-RS2，论文建议最优点附近。 |
| LIMR checkpoint-180 | `grpo_limr/checkpoint-180` | LIMR 当前完成 checkpoint；注意论文常报告的是 Tina-LIMR 主表最优结果，checkpoint 编号不与本地 ckpt-180 直接一一对应。 |

## 完成情况

| 模型 | 完整 seeds | 任务覆盖 |
| --- | --- | --- |
| DeepSeek base baseline | `[0, 1, 2, 3, 4]` | AIME 2024: 5/5; AIME 2025: 5/5; AMC 2023: 5/5; MATH-500: 5/5; Minerva Math: 5/5; GPQA Diamond: 5/5 |
| 仓库当前配置重跑 Open-RS3 repo-default | `[0, 1, 2, 3, 4]` | AIME 2024: 5/5; AIME 2025: 5/5; AMC 2023: 5/5; MATH-500: 5/5; Minerva Math: 5/5; GPQA Diamond: 5/5 |
| 论文 Open-RS3 best checkpoint | `[0, 1, 2, 3, 4]` | AIME 2024: 5/5; AIME 2025: 5/5; AMC 2023: 5/5; MATH-500: 5/5; Minerva Math: 5/5; GPQA Diamond: 5/5 |
| 论文 Open-RS2 best checkpoint | `[0, 1, 2, 3, 4]` | AIME 2024: 5/5; AIME 2025: 5/5; AMC 2023: 5/5; MATH-500: 5/5; Minerva Math: 5/5; GPQA Diamond: 5/5 |
| LIMR checkpoint-180 | `[0, 1, 2, 3, 4]` | AIME 2024: 5/5; AIME 2025: 5/5; AMC 2023: 5/5; MATH-500: 5/5; Minerva Math: 5/5; GPQA Diamond: 5/5 |

## 多 Seed 主结果（mean ± std）

标准差按 5 个 seed 的样本标准差计算。`6-task avg` 是先对每个 seed 的 6 个任务取平均，再对 seed 求均值和标准差。

| 模型 | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek base baseline | 31.33 ± 10.17 (n=5) | 21.33 ± 6.06 (n=5) | 72.50 ± 4.68 (n=5) | 84.84 ± 0.55 (n=5) | 28.16 ± 1.39 (n=5) | 33.33 ± 2.58 (n=5) | 45.25 ± 2.85 (n=5) |
| 仓库当前配置重跑 Open-RS3 repo-default | 32.67 ± 9.83 (n=5) | 22.67 ± 2.79 (n=5) | 75.00 ± 3.95 (n=5) | 84.00 ± 1.38 (n=5) | 30.07 ± 1.17 (n=5) | 36.87 ± 2.20 (n=5) | 46.88 ± 1.88 (n=5) |
| 论文 Open-RS3 best checkpoint | 34.67 ± 6.06 (n=5) | 26.00 ± 4.94 (n=5) | 76.00 ± 4.87 (n=5) | 84.48 ± 0.90 (n=5) | 30.15 ± 1.52 (n=5) | 33.74 ± 2.80 (n=5) | 47.51 ± 2.05 (n=5) |
| 论文 Open-RS2 best checkpoint | 28.67 ± 1.83 (n=5) | 25.33 ± 1.83 (n=5) | 67.50 ± 5.30 (n=5) | 84.92 ± 1.21 (n=5) | 29.63 ± 3.04 (n=5) | 35.86 ± 2.45 (n=5) | 45.32 ± 0.87 (n=5) |
| LIMR checkpoint-180 | 30.00 ± 7.82 (n=5) | 24.67 ± 4.47 (n=5) | 71.50 ± 2.85 (n=5) | 84.64 ± 0.33 (n=5) | 29.78 ± 1.72 (n=5) | 34.85 ± 2.65 (n=5) | 45.91 ± 1.30 (n=5) |

## 6-task 平均排序

| 排名 | 模型 | 6-task avg mean ± std | 相对本地 base 5-seed 均值 |
| ---: | --- | ---: | ---: |
| 1 | 论文 Open-RS3 best checkpoint | 47.51 ± 2.05 (n=5) | +2.26 |
| 2 | 仓库当前配置重跑 Open-RS3 repo-default | 46.88 ± 1.88 (n=5) | +1.63 |
| 3 | LIMR checkpoint-180 | 45.91 ± 1.30 (n=5) | +0.66 |
| 4 | 论文 Open-RS2 best checkpoint | 45.32 ± 0.87 (n=5) | +0.07 |
| 5 | DeepSeek base baseline | 45.25 ± 2.85 (n=5) | +0.00 |

说明：本地 base 的 5-seed 均值为 45.25 ± 2.85；此前 seed42 旧结果为 47.42，不再作为本节主参照。

## 论文结果与本地最好结果对比

论文数值来源：Tina ar5iv 版本 <https://ar5iv.labs.arxiv.org/html/2504.15777v1>。DeepSeek base 使用论文 Table 2 的 `DeepSeek-R1-Distilled-Qwen-1.5B` 行；Open-RS2/Open-RS3 使用论文主结果/附录 checkpoint 表中对应的 best 行；LIMR 使用论文 Table 4 的 Tina-LIMR 主结果。`repo-default` 是当前仓库默认配置诊断实验，论文没有直接对应行。

这里给出三种本地口径：

- `5-seed 均值`：最稳健，推荐作为主要结论。
- `本地最佳单 seed`：从 seed 0-4 中选择 6-task avg 最高的一个完整 seed，模拟“单次最好 run”。
- `逐任务 best-of-5`：每个任务各自取 seed 0-4 的最大值，属于最乐观的 cherry-pick 口径；只用于判断论文若报最优值时，本地上限能否接近。

| 模型 | 论文结果 avg | 本地 5-seed 均值 | 本地最佳单 seed | 最佳 seed | 逐任务 best-of-5 avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek base baseline | 41.18 | 45.25 | 49.66 | 1 | 50.59 |
| 仓库当前配置重跑 Open-RS3 repo-default | - | 46.88 | 50.02 | 3 | 51.56 |
| 论文 Open-RS3 best checkpoint | 49.45 | 47.51 | 49.43 | 1 | 51.92 |
| 论文 Open-RS2 best checkpoint | 50.60 | 45.32 | 46.48 | 4 | 48.27 |
| LIMR checkpoint-180 | 48.47 | 45.91 | 47.55 | 4 | 49.42 |

### 论文逐任务结果

| 模型 | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | Avg | 来源 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek base baseline | 23.33 | 16.67 | 62.50 | 82.60 | 30.15 | 31.82 | 41.18 | Table 2 DeepSeek-R1-Distilled-Qwen-1.5B |
| 论文 Open-RS3 best checkpoint | 36.67 | 23.33 | 82.50 | 85.20 | 31.62 | 37.37 | 49.45 | Table 3/4 Tina-Open-RS3 |
| 论文 Open-RS2 best checkpoint | 43.33 | 26.67 | 77.50 | 87.00 | 32.72 | 36.36 | 50.60 | Table 3/4 Tina-Open-RS2 |
| LIMR checkpoint-180 | 46.67 | 20.00 | 75.00 | 83.80 | 30.51 | 34.85 | 48.47 | Table 4 Tina-LIMR main result |

### 本地最佳单 seed 详情

| 模型 | 最佳 seed | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek base baseline | 1 | 40.00 | 30.00 | 80.00 | 84.60 | 29.04 | 34.34 | 49.66 |
| 仓库当前配置重跑 Open-RS3 repo-default | 3 | 46.67 | 23.33 | 77.50 | 81.60 | 31.62 | 39.39 | 50.02 |
| 论文 Open-RS3 best checkpoint | 1 | 40.00 | 33.33 | 77.50 | 84.40 | 29.04 | 32.32 | 49.43 |
| 论文 Open-RS2 best checkpoint | 4 | 26.67 | 26.67 | 70.00 | 84.20 | 33.46 | 37.88 | 46.48 |
| LIMR checkpoint-180 | 4 | 33.33 | 30.00 | 72.50 | 84.40 | 27.21 | 37.88 | 47.55 |

### 本地逐任务 best-of-5

| 模型 | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | best-of-5 avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek base baseline | 43.33 | 30.00 | 80.00 | 85.80 | 29.04 | 35.35 | 50.59 |
| 仓库当前配置重跑 Open-RS3 repo-default | 46.67 | 26.67 | 80.00 | 85.00 | 31.62 | 39.39 | 51.56 |
| 论文 Open-RS3 best checkpoint | 40.00 | 33.33 | 82.50 | 85.80 | 31.99 | 37.88 | 51.92 |
| 论文 Open-RS2 best checkpoint | 30.00 | 26.67 | 75.00 | 86.60 | 33.46 | 37.88 | 48.27 |
| LIMR checkpoint-180 | 36.67 | 30.00 | 75.00 | 85.00 | 31.99 | 37.88 | 49.42 |

### 与论文差距（百分点）

| 模型 | 5-seed 均值 - 论文 | 最佳单 seed - 论文 | 逐任务 best-of-5 - 论文 |
| --- | ---: | ---: | ---: |
| DeepSeek base baseline | +4.07 | +8.48 | +9.41 |
| 论文 Open-RS3 best checkpoint | -1.94 | -0.02 | +2.47 |
| 论文 Open-RS2 best checkpoint | -5.28 | -4.12 | -2.33 |
| LIMR checkpoint-180 | -2.56 | -0.92 | +0.95 |

解读：本地 DeepSeek base 5-seed 均值为 45.25，高于论文 Table 2 base 的 41.18，说明本地评测环境下的 base 明显更强。Open-RS3 的本地最佳单 seed 为 49.43，几乎贴近论文 49.45；但 5-seed 均值只有 47.51，说明单次结果存在明显运气成分。Open-RS2 即使用本地最佳单 seed 也只有 46.48，逐任务 best-of-5 为 48.27，仍低于论文 50.60，差距不是单 seed 偶然性能完全解释的。LIMR 的逐任务 best-of-5 可超过论文主表，但本地评的是 checkpoint-180，论文主表 Tina-LIMR 的 checkpoint 口径并不直接对应，因此只能作为参考。

## 与论文/旧 seed42 对照

| 模型 | 本次 5-seed avg | 旧 seed42 avg | 论文主表 avg | 备注 |
| --- | ---: | ---: | ---: | --- |
| DeepSeek base baseline | 45.25 | 47.42 | 41.18 | 论文值来自 Table 2 base re-evaluation |
| 仓库当前配置重跑 Open-RS3 repo-default | 46.88 | - | - | repo-default 为当前仓库配置诊断，论文无直接对应主表 |
| 论文 Open-RS3 best checkpoint | 47.51 | 46.78 | 49.45 |  |
| 论文 Open-RS2 best checkpoint | 45.32 | 44.71 | 50.60 |  |
| LIMR checkpoint-180 | 45.91 | - | 48.47 |  |

## 单 Seed 原始结果

### DeepSeek base baseline

| seed | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20.00 | 13.33 | 70.00 | 84.60 | 28.31 | 35.35 | 41.93 |
| 1 | 40.00 | 30.00 | 80.00 | 84.60 | 29.04 | 34.34 | 49.66 |
| 2 | 30.00 | 20.00 | 67.50 | 84.40 | 29.04 | 35.35 | 44.38 |
| 3 | 23.33 | 23.33 | 72.50 | 85.80 | 28.68 | 32.32 | 44.33 |
| 4 | 43.33 | 20.00 | 72.50 | 84.80 | 25.74 | 29.29 | 45.94 |

### 仓库当前配置重跑 Open-RS3 repo-default

| seed | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 36.67 | 20.00 | 70.00 | 84.20 | 30.15 | 34.34 | 45.89 |
| 1 | 30.00 | 23.33 | 75.00 | 84.40 | 30.15 | 37.37 | 46.71 |
| 2 | 20.00 | 26.67 | 80.00 | 85.00 | 30.15 | 38.38 | 46.70 |
| 3 | 46.67 | 23.33 | 77.50 | 81.60 | 31.62 | 39.39 | 50.02 |
| 4 | 30.00 | 20.00 | 72.50 | 84.80 | 28.31 | 34.85 | 45.08 |

### 论文 Open-RS3 best checkpoint

| seed | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 40.00 | 23.33 | 77.50 | 84.80 | 30.15 | 31.82 | 47.93 |
| 1 | 40.00 | 33.33 | 77.50 | 84.40 | 29.04 | 32.32 | 49.43 |
| 2 | 26.67 | 20.00 | 72.50 | 85.80 | 31.99 | 31.31 | 44.71 |
| 3 | 30.00 | 26.67 | 70.00 | 84.00 | 28.31 | 37.88 | 46.14 |
| 4 | 36.67 | 26.67 | 82.50 | 83.40 | 31.25 | 35.35 | 49.31 |

### 论文 Open-RS2 best checkpoint

| seed | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30.00 | 26.67 | 62.50 | 85.40 | 27.94 | 37.88 | 45.06 |
| 1 | 30.00 | 23.33 | 67.50 | 86.60 | 32.35 | 32.32 | 45.35 |
| 2 | 26.67 | 26.67 | 75.00 | 83.40 | 27.57 | 34.34 | 45.61 |
| 3 | 30.00 | 23.33 | 62.50 | 85.00 | 26.84 | 36.87 | 44.09 |
| 4 | 26.67 | 26.67 | 70.00 | 84.20 | 33.46 | 37.88 | 46.48 |

### LIMR checkpoint-180

| seed | AIME 2024 | AIME 2025 | AMC 2023 | MATH-500 | Minerva | GPQA Diamond | 6-task avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20.00 | 26.67 | 75.00 | 84.20 | 31.99 | 34.85 | 45.45 |
| 1 | 36.67 | 20.00 | 72.50 | 84.80 | 30.15 | 31.31 | 45.90 |
| 2 | 36.67 | 26.67 | 67.50 | 85.00 | 30.15 | 33.33 | 46.55 |
| 3 | 23.33 | 20.00 | 70.00 | 84.80 | 29.41 | 36.87 | 44.07 |
| 4 | 33.33 | 30.00 | 72.50 | 84.40 | 27.21 | 37.88 | 47.55 |

## 结论

1. 按稳健的 5-seed 均值，`Open-RS3 best checkpoint-500` 的 6-task avg 最高，为 47.51 ± 2.05。
2. DeepSeek base baseline 的 5-seed 均值为 45.25 ± 2.85，高于论文 Table 2 base 的 41.18，也低于此前单次 seed42 的 47.42。这说明 base 本身的单 seed 方差很大，之前用 seed42 作为唯一 baseline 会高估基线。
3. `repo-default checkpoint-500` 为 46.88 ± 1.88，相对本地 5-seed base 提升 +1.63 个百分点，低于 paper-aligned Open-RS3 best 约 0.63 个百分点，但高于 Open-RS2 best 和 LIMR checkpoint-180。
4. Open-RS3 best checkpoint 为 47.51 ± 2.05，相对本地 5-seed base 提升 +2.26 个百分点；这和之前“低于 seed42 base”的结论不同，原因是 seed42 base 本身偏高。
5. 若采用“最佳单 seed”口径，Open-RS3 可达到 49.43，几乎复现论文 49.45；repo-default 可达到 50.02；base 自身也可达到 49.66。但这些结果都明显高于各自 5-seed 均值，说明 AIME/AMC 的随机波动会显著影响单次报告。
6. Open-RS2 的本地 5-seed 均值为 45.32，几乎只比本地 5-seed base 高 +0.07 个百分点；最佳单 seed 为 46.48，逐任务 best-of-5 为 48.27，仍低于论文 50.60。它与论文的差距不太像纯 seed 偶然性。
7. LIMR checkpoint-180 为 45.91 ± 1.30，相对本地 5-seed base 提升 +0.66 个百分点。它和论文 Tina-LIMR 主表结果 48.47 接近，但 checkpoint 口径不直接对应，不能下严格复现结论。
8. AIME 类任务方差很大，尤其 AIME 2024：base 的 std 为 10.17，repo-default 为 9.83，LIMR 为 7.82。后续做实验选择时，应优先看 5-seed 均值和 best single-seed 两个口径，而不要只看一个 seed。
9. 本次重跑使用 `gpu_memory_utilization=0.45`，此前旧 seed42 结果多使用 `0.70`。生成参数一致，但严格比较时建议优先采用本次 seed 0-4 的同配置结果。

## 结果文件位置

原始结果保存在以下目录模式中：

```text
outputs/<task>/<seed>/<model_id>/
```

本次汇总使用的是每个目录下由 `run_eval_multi_seeds.py` 写出的：

```text
<seed>-0.6-0.95-<task>-32768.json
```

对应的 lighteval `results_*.json` 也保留在同一输出目录的 `results/` 子目录中。

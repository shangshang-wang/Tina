# Tina Evaluation Results Summary

更新时间：2026-05-10

本文档汇总本地已经完成评测的 Tina 复现实验结果。指标来自 lighteval 输出 JSON 中的 `results.all.extractive_match`，下表以百分比展示；括号内为相对本地 base 模型的百分点变化。旧主表使用 seed42 单次评测；更稳健的 seed0-4 多 seed 结果见 `PRIORITY_MULTI_SEED_EVAL_SUMMARY.md`。

## 已完成评测组

| 组别 | 模型/Checkpoint | 说明 |
| --- | --- | --- |
| Base | `DeepSeek-R1-Distill-Qwen-1.5B_base` | 未做 Tina GRPO/LoRA 后训练的本地基线 |
| Open-RS2 | `grpo_open_rs2_checkpoint-450` | paper-aligned 配方，训练到论文建议的 Open-RS2 最优附近 checkpoint |
| Open-RS3 | `grpo_open_rs3_checkpoint-500` | paper-aligned 配方，训练到论文建议的 Open-RS3 最优附近 checkpoint |

## 主结果

| 评测任务 | Base | Open-RS2 ckpt-450 | Open-RS3 ckpt-500 |
| --- | ---: | ---: | ---: |
| AIME 2024 | 33.33 | 23.33 (-10.00) | 30.00 (-3.33) |
| AIME 2025 | 26.67 | 20.00 (-6.67) | 16.67 (-10.00) |
| AMC 2023 | 70.00 | 75.00 (+5.00) | 80.00 (+10.00) |
| MATH-500 | 86.00 | 84.40 (-1.60) | 85.60 (-0.40) |
| Minerva Math | 30.15 | 30.15 (+0.00) | 30.51 (+0.37) |
| GPQA Diamond | 38.38 | 35.35 (-3.03) | 37.88 (-0.51) |
| 6-task unweighted avg | 47.42 | 44.71 (-2.72) | 46.78 (-0.65) |

## 原始分数

| 评测任务 | Base | Open-RS2 ckpt-450 | Open-RS3 ckpt-500 |
| --- | ---: | ---: | ---: |
| `aime24` | 0.3333333333 | 0.2333333333 | 0.3000000000 |
| `aime25` | 0.2666666667 | 0.2000000000 | 0.1666666667 |
| `amc23` | 0.7000000000 | 0.7500000000 | 0.8000000000 |
| `math_500` | 0.8600000000 | 0.8440000000 | 0.8560000000 |
| `minerva` | 0.3014705882 | 0.3014705882 | 0.3051470588 |
| `gpqa:diamond` | 0.3838383838 | 0.3535353535 | 0.3787878788 |

## 和论文结果对比

论文数值来源：

- 论文 Table 2: base model re-evaluation，`DeepSeek-R1-Distilled-Qwen-1.5B` 行。
- 论文 Table 3: main Tina results，`Tina-Open-RS2` 和 `Tina-Open-RS3` 行。
- ar5iv HTML 链接：<https://ar5iv.labs.arxiv.org/html/2504.15777v1>

### Base vs 论文 Base

| 评测任务 | 本地 Base | 论文 Base | 本地 - 论文 |
| --- | ---: | ---: | ---: |
| AIME 2024 | 33.33 | 23.33 | +10.00 |
| AIME 2025 | 26.67 | 16.67 | +10.00 |
| AMC 2023 | 70.00 | 62.50 | +7.50 |
| MATH-500 | 86.00 | 82.60 | +3.40 |
| Minerva Math | 30.15 | 30.15 | +0.00 |
| GPQA Diamond | 38.38 | 31.82 | +6.56 |
| 6-task unweighted avg | 47.42 | 41.18 | +6.24 |

本地 base 的 seed42 单次结果明显强于论文 base，尤其是 AIME 2024、AIME 2025、AMC 2023 和 GPQA Diamond。因此后训练模型即使绝对分数接近论文，也可能在“相对 base 提升”上表现得更差。

### Base 多 Seed 补充

2026-05-10 已补跑 DeepSeek base baseline 的 seed `0,1,2,3,4`，使用和重点重跑实验一致的 `lighteval + vLLM`、`temperature=0.6`、`top_p=0.95`、`max_new_tokens=32768`、`gpu_memory_utilization=0.45` 配置。

| 评测任务 | Base 5-seed mean ± std | 论文 Base | 本地均值 - 论文 |
| --- | ---: | ---: | ---: |
| AIME 2024 | 31.33 ± 10.17 | 23.33 | +8.00 |
| AIME 2025 | 21.33 ± 6.06 | 16.67 | +4.66 |
| AMC 2023 | 72.50 ± 4.68 | 62.50 | +10.00 |
| MATH-500 | 84.84 ± 0.55 | 82.60 | +2.24 |
| Minerva Math | 28.16 ± 1.39 | 30.15 | -1.99 |
| GPQA Diamond | 33.33 ± 2.58 | 31.82 | +1.51 |
| 6-task unweighted avg | 45.25 ± 2.85 | 41.18 | +4.07 |

Base 的最佳单 seed 是 seed `1`，6-task avg 为 `49.66`；逐任务 best-of-5 avg 为 `50.59`。这说明 base 自身也有明显 seed 波动，旧 seed42 的 `47.42` 比 5-seed 均值高 `2.17` 个百分点。因此后续比较 Tina 后训练收益时，应优先用 `45.25 ± 2.85` 作为本地 base 参照，而不是只用 seed42。

### Open-RS2 vs 论文 Tina-Open-RS2

| 评测任务 | 本地 Open-RS2 ckpt-450 | 论文 Tina-Open-RS2 | 本地 - 论文 |
| --- | ---: | ---: | ---: |
| AIME 2024 | 23.33 | 43.33 | -20.00 |
| AIME 2025 | 20.00 | 26.67 | -6.67 |
| AMC 2023 | 75.00 | 77.50 | -2.50 |
| MATH-500 | 84.40 | 87.00 | -2.60 |
| Minerva Math | 30.15 | 32.72 | -2.57 |
| GPQA Diamond | 35.35 | 36.36 | -1.01 |
| 6-task unweighted avg | 44.71 | 50.60 | -5.89 |

Open-RS2 和论文差距主要来自 AIME 2024；其余任务也普遍低于论文，但差距相对小一些。由于本地 base 已经更强，这组结果同时呈现出“低于论文 Tina-Open-RS2”和“低于本地 base”的双重问题。

### Open-RS3 vs 论文 Tina-Open-RS3

| 评测任务 | 本地 Open-RS3 ckpt-500 | 论文 Tina-Open-RS3 | 本地 - 论文 |
| --- | ---: | ---: | ---: |
| AIME 2024 | 30.00 | 36.67 | -6.67 |
| AIME 2025 | 16.67 | 23.33 | -6.66 |
| AMC 2023 | 80.00 | 82.50 | -2.50 |
| MATH-500 | 85.60 | 85.20 | +0.40 |
| Minerva Math | 30.51 | 31.62 | -1.11 |
| GPQA Diamond | 37.88 | 37.37 | +0.51 |
| 6-task unweighted avg | 46.78 | 49.45 | -2.67 |

Open-RS3 比 Open-RS2 更接近论文结果：MATH-500 和 GPQA Diamond 已经达到或略高于论文，主要缺口在 AIME 2024、AIME 2025 和 AMC 2023。

### 论文内相对提升 vs 本地相对提升

| 组别 | 论文模型 Avg | 论文对应 Base Avg | 论文提升 | 本地模型 Avg | 本地 Base Avg | 本地提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-RS2 | 50.60 | 41.60 | +9.00 | 44.71 | 47.42 | -2.72 |
| Open-RS3 | 49.45 | 46.06 | +3.39 | 46.78 | 47.42 | -0.65 |

注意：论文 Table 3 里的 “Baseline” 列是对应训练数据源/模型族的 baseline 平均分，Open-RS2 对应 41.60，Open-RS3 对应 46.06；本地这里统一用同一个本地 base 模型评测平均分 47.42 作为参照。

## 结论速记

1. Open-RS2 ckpt-450 在本地评测中只提升了 AMC 2023，对 AIME 2024、AIME 2025、MATH-500、GPQA Diamond 都下降，整体 6-task 平均比 base 低 2.72 个百分点。
2. Open-RS3 ckpt-500 比 Open-RS2 更稳，明显提升 AMC 2023，小幅提升 Minerva Math，但 AIME 2025 明显下降，整体 6-task 平均比 base 低 0.65 个百分点。
3. 和论文主结果相比，本地 Open-RS2 平均分低 5.89 个百分点，本地 Open-RS3 平均分低 2.67 个百分点；Open-RS3 的复现偏差明显更小。
4. 本地 base 多 seed 均值仍强于论文 Table 2 的 base，6-task 平均高 4.07 个百分点；但旧 seed42 比 5-seed 均值高 2.17 个百分点，说明单 seed baseline 不稳定。
5. 按多 seed 口径，Open-RS3 ckpt-500 的 5-seed 均值为 47.51 ± 2.05，相对本地 base 5-seed 均值提升 +2.26；Open-RS2 ckpt-450 为 45.32 ± 0.87，几乎只比本地 base 高 +0.07。
6. 这些旧 seed42 评测使用本地 lighteval/vLLM 配置，日志显示评测侧 `gpu_memory_utilization=0.70`；2026-05-10 的多 seed 重跑使用 `gpu_memory_utilization=0.45`。生成参数一致，但严格比较时应优先采用多 seed 同配置结果。

## 评测配置备注

训练侧 paper-aligned 配方主要参数：

| 参数 | Open-RS2 paper-aligned | Open-RS3 paper-aligned |
| --- | --- | --- |
| Reward | `format + accuracy` | `format + cosine` |
| LoRA | `r=32, alpha=128, dropout=0.05` | `r=32, alpha=128, dropout=0.05` |
| LR | `1e-6` | `1e-6` |
| Batch | `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, 2 GPUs, total batch 32 | 同左 |
| `num_generations` | 4 | 4 |
| `max_prompt_length` | 512 | 512 |
| `max_completion_length` | 3584 | 3584 |
| Stop checkpoint | 450 | 500 |

评测侧共同参数：

| 参数 | 值 |
| --- | --- |
| Backend | `lighteval vllm` |
| dtype | `bfloat16` |
| data parallel | 2 |
| max model length | 32768 |
| generation | `max_new_tokens=32768, temperature=0.6, top_p=0.95` |
| seed/output group | `42` |
| chat template | enabled |

## 结果文件索引

| 组别 | 任务 | JSON |
| --- | --- | --- |
| Base | `aime24` | `outputs/aime24/42/DeepSeek-R1-Distill-Qwen-1.5B_base/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_base/results_2026-05-06T13-40-49.011205.json` |
| Base | `aime25` | `outputs/aime25/42/DeepSeek-R1-Distill-Qwen-1.5B_base/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_base/results_2026-05-06T13-49-09.276945.json` |
| Base | `amc23` | `outputs/amc23/42/DeepSeek-R1-Distill-Qwen-1.5B_base/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_base/results_2026-05-06T13-59-27.169357.json` |
| Base | `math_500` | `outputs/math_500/42/DeepSeek-R1-Distill-Qwen-1.5B_base/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_base/results_2026-05-06T13-57-35.606379.json` |
| Base | `minerva` | `outputs/minerva/42/DeepSeek-R1-Distill-Qwen-1.5B_base/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_base/results_2026-05-06T14-11-05.044135.json` |
| Base | `gpqa:diamond` | `outputs/gpqa:diamond/42/DeepSeek-R1-Distill-Qwen-1.5B_base/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_base/results_2026-05-06T14-26-40.453418.json` |
| Open-RS2 ckpt-450 | `aime24` | `outputs/aime24/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450-merged/results_2026-05-07T09-09-06.881743.json` |
| Open-RS2 ckpt-450 | `aime25` | `outputs/aime25/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450-merged/results_2026-05-07T09-17-18.916278.json` |
| Open-RS2 ckpt-450 | `amc23` | `outputs/amc23/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450-merged/results_2026-05-07T09-25-11.394903.json` |
| Open-RS2 ckpt-450 | `math_500` | `outputs/math_500/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450-merged/results_2026-05-07T09-37-21.438327.json` |
| Open-RS2 ckpt-450 | `minerva` | `outputs/minerva/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450-merged/results_2026-05-07T09-48-36.138139.json` |
| Open-RS2 ckpt-450 | `gpqa:diamond` | `outputs/gpqa:diamond/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs2_checkpoint-450-merged/results_2026-05-07T10-02-13.143596.json` |
| Open-RS3 ckpt-500 | `aime24` | `outputs/aime24/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500-merged/results_2026-05-07T10-51-21.383884.json` |
| Open-RS3 ckpt-500 | `aime25` | `outputs/aime25/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500-merged/results_2026-05-07T10-59-40.528559.json` |
| Open-RS3 ckpt-500 | `amc23` | `outputs/amc23/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500-merged/results_2026-05-07T11-07-33.010539.json` |
| Open-RS3 ckpt-500 | `math_500` | `outputs/math_500/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500-merged/results_2026-05-07T11-20-20.199468.json` |
| Open-RS3 ckpt-500 | `minerva` | `outputs/minerva/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500-merged/results_2026-05-07T11-28-27.141334.json` |
| Open-RS3 ckpt-500 | `gpqa:diamond` | `outputs/gpqa:diamond/42/DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500/results/_home_wangls_Tina_ckpts_models_DeepSeek-R1-Distill-Qwen-1.5B_grpo_open_rs3_checkpoint-500-merged/results_2026-05-07T11-41-52.941658.json` |

# Open-RS3 Training Rollout Signal and Split Summary

本文档总结 `knoveleng/open-rs` 训练数据在基准模型上的 rollout 推理统计，以及当前 RL 信号拆分工具链使用的拆分依据和最终结果。

## 运行设置

| item | value |
|---|---|
| dataset | `knoveleng/open-rs` train parquet |
| rollout input | `/root/Tina/.cache/huggingface/hub/datasets--knoveleng--open-rs/snapshots/8de2b2d11162d97b45c7606902eae2bb1ff629b4/data/train-00000-of-00001.parquet` |
| reference model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| generation backend | `vllm` |
| shards | `outputs/rl_signal_split_open_rs3_vllm_shard0`, `outputs/rl_signal_split_open_rs3_vllm_shard1` |
| final output | `outputs/rl_signal_split_open_rs3_vllm_final` |
| samples per problem | 8 |
| temperature | 0.7 |
| top_p | 0.95 |
| max_new_tokens | 2048 |
| max_prompt_tokens | 512 |
| prompt_style | `open_rs` |
| user_template | `raw` |
| format_mode | `any` |

## Rollout 完整性

两个 shard 合并后共得到 7000 条训练样本，每条样本生成 8 个 completion，共 56000 个 completion。

| metric | value |
|---|---:|
| rollout rows | 7000 |
| unique sample ids | 7000 |
| duplicate ids | 0 |
| samples per row | all 8 |
| total completions | 56000 |
| rollout rows with error | 0 |
| missing problem rows | 0 |
| missing answer rows | 0 |

结论：rollout 数据完整，没有发现 shard 重叠、缺题、缺答案或生成错误记录。

## Completion 级别统计

| metric | value |
|---|---:|
| completion accuracy | 0.0836 |
| mean reward | 0.0836 |
| format valid rate | 0.9998 |
| invalid completions | 13 / 56000 |
| mean completion length | 1808.84 tokens |
| length p10 | 886.00 tokens |
| length p25 | 2048.00 tokens |
| length p50 | 2048.00 tokens |
| length p75 | 2048.00 tokens |
| length p90 | 2048.00 tokens |
| max length | 2048 tokens |

Completion 的停止原因如下：

| finish_reason | count | percent |
|---|---:|---:|
| `length` | 42308 | 75.55% |
| `stop` | 13692 | 24.45% |

Verifier 来源如下：

| verifier | count | percent |
|---|---:|---:|
| `project` | 47992 | 85.70% |
| `fallback` | 8008 | 14.30% |

主要观察：

- 单 completion 正确率只有 8.36%，说明该 reference model 对 Open-RS3 训练样本整体较弱。
- 75.55% completion 因触达 `max_new_tokens=2048` 结束，生成长度存在明显截断压力。
- 格式有效率接近 100%，无效格式不是当前 reward 稀疏的主要原因。

## 样本级别统计

每个训练样本有 8 个 completion，因此 `pass_rate = correct_count / 8`。

| metric | mean | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pass_rate | 0.0836 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 1.0000 |
| reward_mean | 0.0836 | -0.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 1.0000 |
| reward_std | 0.0880 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4330 | 0.5000 |
| format_valid_rate | 0.9998 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| length_mean_tokens | 1808.84 | 273.38 | 1097.73 | 1756.72 | 2048.00 | 2048.00 | 2048.00 | 2048.00 |
| length_p90_tokens | 1940.44 | 330.40 | 1606.71 | 2048.00 | 2048.00 | 2048.00 | 2048.00 | 2048.00 |
| answer_diversity | 0.7441 | 0.1250 | 0.3750 | 0.6250 | 0.7500 | 0.8750 | 1.0000 | 1.0000 |
| correctness_entropy | 0.1613 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8113 | 1.0000 |
| variance_score | 0.3375 | 0.0000 | 0.1667 | 0.2222 | 0.2778 | 0.3333 | 0.7222 | 1.0000 |
| long_reasoning_score | 0.8610 | 0.0000 | 0.4048 | 0.8844 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| instability_score | 0.1927 | 0.0000 | 0.0632 | 0.0926 | 0.1111 | 0.2926 | 0.4219 | 0.6667 |

Correct count 分布如下：

| correct_count / 8 | pass_rate | samples | percent |
|---:|---:|---:|---:|
| 0 | 0.000 | 5419 | 77.41% |
| 1 | 0.125 | 558 | 7.97% |
| 2 | 0.250 | 294 | 4.20% |
| 3 | 0.375 | 221 | 3.16% |
| 4 | 0.500 | 144 | 2.06% |
| 5 | 0.625 | 107 | 1.53% |
| 6 | 0.750 | 106 | 1.51% |
| 7 | 0.875 | 81 | 1.16% |
| 8 | 1.000 | 70 | 1.00% |

| derived group | samples | percent |
|---|---:|---:|
| all wrong | 5419 | 77.41% |
| best-of-8 correct | 1581 | 22.59% |
| all correct | 70 | 1.00% |

长度截断相关统计：

| condition | samples | percent |
|---|---:|---:|
| mean completion length equals 2048 | 3900 | 55.71% |
| p90 completion length equals 2048 | 6022 | 86.03% |

主要观察：

- 77.41% 样本 8 次采样全部错误，说明数据对当前 reference model 明显偏难。
- 只有 1.00% 样本 8 次采样全部正确，稳定 easy 样本占比很低。
- 超过一半样本的平均 completion 长度已经达到最大生成长度，说明许多题的推理被 token budget 截断。
- `answer_diversity` 平均 0.7441，表明错误样本中答案分散度较高，模型经常采样出不同最终答案。

## 拆分依据

当前拆分策略在 `rl_signal_splits/splitting.py` 中实现。目标是把样本按 reference model 的 rollout 信号拆成若干可用于 LoRA/RL 消融的训练集合。

### 基础难度桶

基础难度直接由 `pass_rate` 和格式有效率决定：

| split | rule |
|---|---|
| `easy` | `pass_rate >= 0.75` 且 `format_valid_rate >= 0.75` |
| `medium` | `0.25 <= pass_rate < 0.75` 且 `format_valid_rate >= 0.5` |
| `hard` | `pass_rate < 0.25` 且 `format_valid_rate >= 0.5` |

这些桶互斥，反映 reference model 在 8 次采样下的经验难度。

### 高方差与长推理桶

旧版规则使用若干原始指标阈值并以 `>=` 判断，在本数据集上会因为 `reward_std_top30 = 0`、`correctness_entropy_top30 = 0` 或 `length_p90_top25 = 2048` 等离散/截断值导致 split 退化成近似全量集合。

当前规则改为基于 composite score 的排序 top-k：

| split | rule |
|---|---|
| `high_variance` | 按 `variance_score` 排名前 30% |
| `long_reasoning` | 按 `long_reasoning_score` 排名前 25% |

Composite score 定义：

```text
variance_score =
  mean(
    robust_percentile_norm(reward_std),
    robust_percentile_norm(answer_diversity),
    robust_percentile_norm(correctness_entropy)
  )

long_reasoning_score =
  mean(
    robust_percentile_norm(length_mean_tokens),
    robust_percentile_norm(length_p90_tokens)
  )

instability_score =
  mean(
    variance_score,
    robust_percentile_norm(invalid_count),
    robust_percentile_norm(completion_length_cv)
  )
```

其中 `robust_percentile_norm` 使用数据集 p5/p95 做稳健归一化：

```text
clip((x - dataset_p5) / max(dataset_p95 - dataset_p5, eps), 0, 1)
```

### Core Medium 与 Mixed Balanced

| split | rule |
|---|---|
| `core_medium` | 在 `medium` 样本中按较低 `instability_score` 选取约前 50%；优先尝试避开 `high_variance` 和 `long_reasoning`，若不足则回填最低 instability 的 medium 样本 |
| `mixed_balanced` | 默认最多 2000 条，按 20% easy、40% core_medium/medium、20% hard、10% high_variance、10% long_reasoning 采样并去重补齐 |

注意：当前数据中 `medium` 样本基本都同时具有较高 correctness entropy 或 reward variance，因此 `core_medium` 仍与 `high_variance` 完全重叠。这是数据分布导致的语义重叠，不是导出错误。

## 最终拆分结果

最终目录：`outputs/rl_signal_split_open_rs3_vllm_final`

| split | size | unique | percent of 7000 | mean pass_rate | mean reward_std | mean length | mean answer_diversity | mean format_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 257 | 257 | 3.67% | 0.8575 | 0.2828 | 1106.43 | 0.2490 | 1.0000 |
| `medium` | 766 | 766 | 10.94% | 0.3854 | 0.4675 | 1476.81 | 0.5692 | 1.0000 |
| `hard` | 5977 | 5977 | 85.39% | 0.0117 | 0.0310 | 1881.59 | 0.7879 | 0.9997 |
| `high_variance` | 2100 | 2100 | 30.00% | 0.2454 | 0.2930 | 1629.30 | 0.7093 | 0.9999 |
| `long_reasoning` | 1750 | 1750 | 25.00% | 0.0103 | 0.0174 | 2048.00 | 0.8164 | 0.9997 |
| `core_medium` | 383 | 383 | 5.47% | 0.3776 | 0.4660 | 1700.31 | 0.5819 | 1.0000 |
| `mixed_balanced` | 2000 | 2000 | 28.57% | 0.2090 | 0.1700 | 1736.71 | 0.6739 | 0.9998 |

## Split 重叠关系

Split 默认不是互斥集合，除 `easy`/`medium`/`hard` 基础难度桶互斥外，`high_variance`、`long_reasoning`、`core_medium`、`mixed_balanced` 都可以与基础难度桶重叠。

| split | overlaps |
|---|---|
| `easy` | `high_variance`: 187, `long_reasoning`: 3, `mixed_balanced`: 257 |
| `medium` | `high_variance`: 766, `long_reasoning`: 29, `core_medium`: 383, `mixed_balanced`: 468 |
| `hard` | `high_variance`: 1147, `long_reasoning`: 1718, `mixed_balanced`: 1275 |
| `high_variance` | `easy`: 187, `medium`: 766, `hard`: 1147, `long_reasoning`: 481, `core_medium`: 383, `mixed_balanced`: 1007 |
| `long_reasoning` | `easy`: 3, `medium`: 29, `hard`: 1718, `high_variance`: 481, `core_medium`: 29, `mixed_balanced`: 561 |
| `core_medium` | `medium`: 383, `high_variance`: 383, `long_reasoning`: 29, `mixed_balanced`: 383 |
| `mixed_balanced` | `easy`: 257, `medium`: 468, `hard`: 1275, `high_variance`: 1007, `long_reasoning`: 561, `core_medium`: 383 |

## 结果解读

这批 Open-RS3 训练数据对 `DeepSeek-R1-Distill-Qwen-1.5B` reference model 来说明显偏难：

- `hard` 占 85.39%，`easy` 只有 3.67%。
- 单 completion accuracy 为 8.36%，best-of-8 accuracy 为 22.59%。
- `long_reasoning` 集合中 mean pass_rate 只有 1.03%，且平均长度为 2048 tokens，说明长推理样本很大程度上也伴随奖励稀疏和截断压力。
- `medium` 占 10.94%，是相对更可能提供 dense RL signal 的核心样本池，但其中 correctness entropy 和 reward variance 较高，因此会和 `high_variance` 大量重叠。
- `mixed_balanced` 将全量数据中极端 hard 的比例从 85.39% 降到 63.75% 左右，并提升 mean pass_rate 到 0.2090，可作为更温和的训练对照集。

## 可引用文件

| artifact | path |
|---|---|
| merged rollouts | `outputs/rl_signal_split_open_rs3_vllm_final/per_sample_rollouts.jsonl` |
| per-sample metrics | `outputs/rl_signal_split_open_rs3_vllm_final/per_sample_metrics.jsonl` |
| split report | `outputs/rl_signal_split_open_rs3_vllm_final/split_report.md` |
| split stats csv | `outputs/rl_signal_split_open_rs3_vllm_final/split_stats.csv` |
| split visualization | `outputs/rl_signal_split_open_rs3_vllm_final/split_visualization.png` |
| easy split | `outputs/rl_signal_split_open_rs3_vllm_final/split_easy.jsonl` |
| medium split | `outputs/rl_signal_split_open_rs3_vllm_final/split_medium.jsonl` |
| hard split | `outputs/rl_signal_split_open_rs3_vllm_final/split_hard.jsonl` |
| high variance split | `outputs/rl_signal_split_open_rs3_vllm_final/split_high_variance.jsonl` |
| long reasoning split | `outputs/rl_signal_split_open_rs3_vllm_final/split_long_reasoning.jsonl` |
| core medium split | `outputs/rl_signal_split_open_rs3_vllm_final/split_core_medium.jsonl` |
| mixed balanced split | `outputs/rl_signal_split_open_rs3_vllm_final/split_mixed_balanced.jsonl` |

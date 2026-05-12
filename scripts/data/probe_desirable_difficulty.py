#!/usr/bin/env python
"""Probe a Tina RL dataset with the base model and keep frontier-level prompts.

The output JSONL keeps the original training columns plus probe metadata. Tina's
GRPO entry point can train on the JSONL through `model_post_train_dataset_path`
without changing trainer logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tina.post_train_hf.rewards import accuracy_reward, format_reward, get_cosine_scaled_reward
from tina.utils.chat_template import REASON_CHAT_TEMPLATE
from tina.utils.constant import RL_POST_TRAIN_CONFIG_MAP
from tina.utils.prompt import OPEN_R1_SYSTEM_PROMPT, OPEN_RS_SYSTEM_PROMPT


DATASET_REWARD_SPECS = {
    "open_rs3": (["format", "cosine"], [1.0, 2.0]),
    "open_rs2": (["format", "accuracy"], [1.0, 2.0]),
    "limr": (["format", "accuracy"], [1.0, 2.0]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="open_rs3", choices=sorted(RL_POST_TRAIN_CONFIG_MAP))
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--max-completion-length", type=int, default=3584)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Probe only the first N shuffled examples.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=1)
    parser.add_argument("--max-correct", type=int, default=None)
    parser.add_argument(
        "--preferred-correct",
        default=None,
        help="Optional comma-separated correct_count values to tag as preferred, e.g. 2,3,4.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--write-completions", action="store_true")
    parser.add_argument("--cosine-max-len", type=int, default=3584)
    return parser.parse_args()


def normalize_dataset(dataset: Dataset, dataset_name: str) -> Dataset:
    if "solution" not in dataset.column_names and "answer" in dataset.column_names:
        dataset = dataset.rename_column("answer", "solution")

        def wrap_in_math(example: dict[str, Any]) -> dict[str, str]:
            return {"solution": f"${example['solution']}$"}

        dataset = dataset.map(wrap_in_math)
    if "problem" not in dataset.column_names and "question" in dataset.column_names:
        dataset = dataset.rename_column("question", "problem")
    if "problem" not in dataset.column_names and "prompt" in dataset.column_names:
        dataset = dataset.rename_column("prompt", "problem")
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns("messages")

    if "deepscaler" in dataset_name:
        dataset = dataset.rename_column("solution", "solution_archive")
        dataset = dataset.rename_column("answer", "solution")

        def wrap_in_math(example: dict[str, Any]) -> dict[str, str]:
            return {"solution": f"${example['solution']}$"}

        dataset = dataset.map(wrap_in_math)
    return dataset


def load_rl_dataset(dataset_name: str, dataset_config: str | None) -> tuple[Dataset, str]:
    dataset_path = RL_POST_TRAIN_CONFIG_MAP[dataset_name]
    if dataset_config is not None:
        dataset = load_dataset(dataset_path, split="train", name=dataset_config)
    else:
        dataset = load_dataset(dataset_path, split="train")
    return normalize_dataset(dataset, dataset_name), dataset_path


def build_reward_funcs(args: argparse.Namespace) -> tuple[list[Callable[..., list[float]]], list[float]]:
    reward_names, reward_weights = DATASET_REWARD_SPECS.get(args.dataset_name, (["format", "accuracy"], [1.0, 2.0]))
    reward_map = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=0.0,
            max_value_wrong=-0.5,
            min_value_correct=0.5,
            max_value_correct=1.0,
            max_len=args.cosine_max_len,
        ),
    }
    return [reward_map[name] for name in reward_names], reward_weights


def weighted_rewards(
    completions: list[list[dict[str, str]]],
    solutions: list[str],
    reward_funcs: list[Callable[..., list[float]]],
    reward_weights: list[float],
) -> tuple[list[float], list[list[float | None]]]:
    per_func: list[list[float | None]] = []
    for reward_func in reward_funcs:
        values = reward_func(completions=completions, solution=solutions)
        per_func.append(values)

    totals = []
    for idx in range(len(completions)):
        total = 0.0
        for values, weight in zip(per_func, reward_weights):
            value = values[idx]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                total += float(value) * float(weight)
        totals.append(total)
    return totals, per_func


def stable_json_dumps(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.max_correct is None:
        args.max_correct = args.num_generations - 1
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    project_dir = Path(os.environ.get("PROJECT_DIR", Path.cwd())).resolve()
    ckpt_dir = Path(os.environ.get("CKPT_DIR", project_dir / "ckpts")).resolve()
    model_path = args.model_name_or_path or str(ckpt_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B" / "base")
    output_dir = Path(args.output_dir or project_dir / "datasets" / "desirable_difficulty" / args.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    dataset, source_dataset_path = load_rl_dataset(args.dataset_name, args.dataset_config)
    dataset = dataset.shuffle(seed=args.seed)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.num_shards > 1:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    tokenizer.chat_template = REASON_CHAT_TEMPLATE
    system_prompt = OPEN_RS_SYSTEM_PROMPT if "open-rs" in source_dataset_path else OPEN_R1_SYSTEM_PROMPT

    prompts = []
    rows = []
    for idx, example in enumerate(dataset):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        rows.append(dict(example, _probe_source_index=idx))

    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_prompt_length + args.max_completion_length,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_length,
        n=args.num_generations,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)

    reward_funcs, reward_weights = build_reward_funcs(args)
    preferred_correct = None
    if args.preferred_correct:
        preferred_correct = {int(x) for x in args.preferred_correct.split(",") if x.strip()}

    shard_suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}-of-{args.num_shards}"
    metrics_path = output_dir / f"{args.dataset_name}_probe_metrics{shard_suffix}.jsonl"
    filtered_path = output_dir / f"{args.dataset_name}_filtered{shard_suffix}.jsonl"
    completions_path = output_dir / f"{args.dataset_name}_probe_completions{shard_suffix}.jsonl"

    kept = 0
    all_correct_counts = []
    with metrics_path.open("w", encoding="utf-8") as metrics_file, filtered_path.open("w", encoding="utf-8") as filtered_file:
        completions_file = completions_path.open("w", encoding="utf-8") if args.write_completions else None
        try:
            for row, output in zip(rows, outputs):
                texts = [item.text for item in output.outputs]
                completions = [[{"role": "assistant", "content": text}] for text in texts]
                solutions = [row["solution"]] * len(texts)
                accuracy_values = accuracy_reward(completions=completions, solution=solutions)
                correct_count = sum(1 for value in accuracy_values if value == 1.0)
                format_values = format_reward(completions=completions)
                total_rewards, per_func_rewards = weighted_rewards(completions, solutions, reward_funcs, reward_weights)
                avg_len = mean(len(text) for text in texts) if texts else 0.0
                reward_std = pstdev(total_rewards) if len(total_rewards) > 1 else 0.0
                format_rate = mean(float(value) for value in format_values) if format_values else 0.0
                keep = args.min_correct <= correct_count <= args.max_correct
                preferred = correct_count in preferred_correct if preferred_correct is not None else keep
                all_correct_counts.append(correct_count)

                probe = {
                    "dataset_name": args.dataset_name,
                    "num_generations": len(texts),
                    "correct_count": correct_count,
                    "pass_rate": correct_count / len(texts) if texts else 0.0,
                    "reward_std": reward_std,
                    "format_rate": format_rate,
                    "avg_len": avg_len,
                    "reward": mean(total_rewards) if total_rewards else 0.0,
                    "reward_values": total_rewards,
                    "accuracy_values": accuracy_values,
                    "format_values": format_values,
                    "per_func_rewards": per_func_rewards,
                    "kept": keep,
                    "preferred": preferred,
                }
                metrics_file.write(stable_json_dumps({**row, "probe": probe}) + "\n")
                if keep:
                    kept += 1
                    filtered_file.write(stable_json_dumps({**row, "probe": probe}) + "\n")
                if completions_file is not None:
                    completions_file.write(stable_json_dumps({**row, "probe": probe, "completions": texts}) + "\n")
        finally:
            if completions_file is not None:
                completions_file.close()

    histogram = {str(i): all_correct_counts.count(i) for i in range(args.num_generations + 1)}
    summary = {
        "dataset_name": args.dataset_name,
        "source_dataset_path": source_dataset_path,
        "model_name_or_path": model_path,
        "num_examples_probed": len(rows),
        "num_examples_kept": kept,
        "keep_rate": kept / len(rows) if rows else 0.0,
        "num_generations": args.num_generations,
        "min_correct": args.min_correct,
        "max_correct": args.max_correct,
        "correct_count_histogram": histogram,
        "metrics_path": str(metrics_path),
        "filtered_path": str(filtered_path),
    }
    summary_path = output_dir / f"{args.dataset_name}_probe_summary{shard_suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

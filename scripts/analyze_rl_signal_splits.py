#!/usr/bin/env python
"""Analyze RL signal rollouts and build LoRA training splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_signal_splits.generation import GenerationSettings, RolloutGenerator
from rl_signal_splits.io import (
    append_jsonl,
    iter_normalized_records,
    load_completed_rollout_ids,
    read_records,
    read_jsonl,
    write_jsonl,
)
from rl_signal_splits.metrics import aggregate_rollouts
from rl_signal_splits.reporting import write_reports
from rl_signal_splits.reward import evaluate_completion
from rl_signal_splits.splitting import SplitConfig, build_splits
from rl_signal_splits.visualization import make_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input training JSONL file.")
    parser.add_argument("--hf_dataset", help="Optional Hugging Face dataset name, e.g. knoveleng/open-rs.")
    parser.add_argument("--hf_config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--hf_split", default="train", help="Hugging Face split to export/analyze.")
    parser.add_argument("--model", required=True, help="Base/reference model path or HF model id.")
    parser.add_argument("--output_dir", default="outputs/rl_signal_split")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of source problems per generation batch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Only process this many input records.")
    parser.add_argument("--prompt_style", default="open_rs", choices=["open_rs", "open_r1", "none", "raw"])
    parser.add_argument("--user_template", default="raw", choices=["raw", "math"])
    parser.add_argument("--format_mode", default="any", choices=["any", "boxed", "answer_tag", "think_answer"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_project_verifier", action="store_true")
    parser.add_argument("--mock_generation", action="store_true", help="Use deterministic fake completions for plumbing tests.")
    parser.add_argument("--generation_backend", default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None, help="Optional vLLM max model length.")
    parser.add_argument("--skip_generation", action="store_true", help="Only rebuild metrics/splits/reports from rollouts.")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of input shards for multi-process generation.")
    parser.add_argument("--shard_index", type=int, default=0, help="Shard index for this process.")
    parser.add_argument("--max_split_size", type=int, default=None)
    parser.add_argument("--mixed_size", type=int, default=None)
    parser.add_argument("--easy_ratio", type=float, default=0.20)
    parser.add_argument("--medium_ratio", type=float, default=0.40)
    parser.add_argument("--hard_ratio", type=float, default=0.20)
    parser.add_argument("--high_variance_ratio", type=float, default=0.10)
    parser.add_argument("--long_reasoning_ratio", type=float, default=0.10)
    parser.add_argument("--make_disjoint", action="store_true")
    parser.add_argument("--no_visualization", action="store_true")
    parser.add_argument("--export_input_jsonl", default=None, help="Where to cache --hf_dataset as JSONL.")
    return parser.parse_args()


def require_tqdm():
    try:
        from tqdm import tqdm

        return tqdm
    except Exception:
        return lambda x, **kwargs: x


def export_hf_dataset(args: argparse.Namespace) -> str:
    if args.input:
        return args.input
    if not args.hf_dataset:
        raise ValueError("Provide --input or --hf_dataset.")
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for --hf_dataset.") from exc
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.export_input_jsonl) if args.export_input_jsonl else output_dir / "input_dataset.jsonl"
    if jsonl_path.exists() and not args.skip_generation:
        return str(jsonl_path)
    kwargs = {"split": args.hf_split}
    if args.hf_config:
        kwargs["name"] = args.hf_config
    ds = load_dataset(args.hf_dataset, **kwargs)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(normalize_open_rs_training_row(dict(row)), ensure_ascii=False) + "\n")
    return str(jsonl_path)


def normalize_open_rs_training_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "solution" not in row and "answer" in row:
        row = dict(row)
        row["solution"] = f"${row['answer']}$"
    if "problem" not in row and "question" in row:
        row = dict(row)
        row["problem"] = row["question"]
    if "problem" not in row and "prompt" in row:
        row = dict(row)
        row["problem"] = row["prompt"]
    return row


def maybe_export_input_file(input_path: str, output_dir: Path, limit: Optional[int]) -> str:
    suffix = Path(input_path).suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return input_path
    jsonl_path = output_dir / "input_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in read_records(input_path, limit=limit):
            f.write(json.dumps(normalize_open_rs_training_row(dict(row)), ensure_ascii=False) + "\n")
    return str(jsonl_path)


def batched(records: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_error_rollout(path: Path, record: Dict[str, Any], message: str, n_samples: int) -> None:
    append_jsonl(
        path,
        {
            "id": record["id"],
            "problem": record.get("problem", ""),
            "gold_answer": record.get("gold_answer", ""),
            "source": record.get("source", {}),
            "n_samples": n_samples,
            "samples": [],
            "error": message,
            "missing_problem": record.get("missing_problem", False),
            "missing_answer": record.get("missing_answer", False),
        },
    )


def run_generation(args: argparse.Namespace, input_path: str, rollout_path: Path) -> None:
    completed = load_completed_rollout_ids(rollout_path)
    generator = RolloutGenerator(
        GenerationSettings(
            model=args.model,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            seed=args.seed,
            prompt_style=args.prompt_style,
            user_template=args.user_template,
            trust_remote_code=args.trust_remote_code,
            mock_generation=args.mock_generation,
            backend=args.generation_backend,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    )
    tqdm = require_tqdm()
    records = (
        r
        for r in iter_normalized_records(input_path, limit=args.limit)
        if str(r["id"]) not in completed and int(r.get("index", 0)) % max(1, args.num_shards) == args.shard_index
    )
    for batch in tqdm(batched(records, max(1, args.batch_size)), desc="rollouts"):
        valid_batch = []
        for record in batch:
            if record.get("load_error"):
                write_error_rollout(rollout_path, record, record["load_error"], args.n_samples)
            elif record.get("missing_problem") or record.get("missing_answer"):
                write_error_rollout(rollout_path, record, "missing problem or gold answer field", args.n_samples)
            else:
                valid_batch.append(record)
        if not valid_batch:
            continue
        try:
            generated = generator.generate_for_records(valid_batch)
        except Exception as exc:
            for record in valid_batch:
                write_error_rollout(rollout_path, record, f"generation_error: {type(exc).__name__}: {exc}", args.n_samples)
            continue
        for record in valid_batch:
            samples = []
            for sample in generated.get(record["id"], []):
                completion = sample.get("completion", "")
                eval_result = evaluate_completion(
                    completion,
                    record["gold_answer"],
                    format_mode=args.format_mode,
                    use_project_verifier=not args.no_project_verifier,
                )
                samples.append({**sample, **eval_result})
            append_jsonl(
                rollout_path,
                {
                    "id": record["id"],
                    "problem": record["problem"],
                    "gold_answer": record["gold_answer"],
                    "source": record["source"],
                    "n_samples": args.n_samples,
                    "samples": samples,
                    "missing_problem": False,
                    "missing_answer": False,
                },
            )


def rebuild_outputs(args: argparse.Namespace, output_dir: Path, rollout_path: Path) -> None:
    rollout_rows = list(read_jsonl(rollout_path))
    metrics = aggregate_rollouts(rollout_rows)
    metrics_path = output_dir / "per_sample_metrics.jsonl"
    write_jsonl(metrics_path, metrics)

    split_result = build_splits(
        metrics,
        SplitConfig(
            max_split_size=args.max_split_size,
            mixed_size=args.mixed_size,
            easy_ratio=args.easy_ratio,
            medium_ratio=args.medium_ratio,
            hard_ratio=args.hard_ratio,
            high_variance_ratio=args.high_variance_ratio,
            long_reasoning_ratio=args.long_reasoning_ratio,
            make_disjoint=args.make_disjoint,
            seed=args.seed,
        ),
    )
    for name, rows in split_result["split_records"].items():
        write_jsonl(output_dir / f"split_{name}.jsonl", rows)
    visualizations = [] if args.no_visualization else make_visualizations(metrics, output_dir)
    write_reports(
        output_dir,
        metrics,
        split_result,
        {
            "input": args.input,
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "hf_split": args.hf_split,
            "model": args.model,
            "seed": args.seed,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "generation_backend": args.generation_backend,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "prompt_style": args.prompt_style,
            "user_template": args.user_template,
            "format_mode": args.format_mode,
            "make_disjoint": args.make_disjoint,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        visualizations=visualizations,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = export_hf_dataset(args)
    input_path = maybe_export_input_file(input_path, output_dir, args.limit)
    rollout_path = output_dir / "per_sample_rollouts.jsonl"
    if not args.skip_generation:
        run_generation(args, input_path, rollout_path)
    if not rollout_path.exists():
        raise FileNotFoundError(f"No rollout file found at {rollout_path}")
    rebuild_outputs(args, output_dir, rollout_path)
    print(f"wrote RL signal split outputs to {output_dir}")


if __name__ == "__main__":
    main()

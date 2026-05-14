#!/usr/bin/env python
"""Finalize RL signal split outputs from one or more rollout shard directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_signal_splits.io import read_jsonl, write_jsonl
from rl_signal_splits.metrics import aggregate_rollouts
from rl_signal_splits.reporting import write_reports
from rl_signal_splits.splitting import SplitConfig, build_splits
from rl_signal_splits.visualization import make_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_split_size", type=int, default=None)
    parser.add_argument("--mixed_size", type=int, default=None)
    parser.add_argument("--easy_ratio", type=float, default=0.20)
    parser.add_argument("--medium_ratio", type=float, default=0.40)
    parser.add_argument("--hard_ratio", type=float, default=0.20)
    parser.add_argument("--high_variance_ratio", type=float, default=0.10)
    parser.add_argument("--long_reasoning_ratio", type=float, default=0.10)
    parser.add_argument("--make_disjoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_visualization", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    rollouts = []
    for shard_dir in args.shard_dirs:
        path = Path(shard_dir) / "per_sample_rollouts.jsonl"
        for row in read_jsonl(path):
            sample_id = str(row.get("id"))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            rollouts.append(row)
    rollouts.sort(key=lambda row: str(row.get("id")))
    write_jsonl(output_dir / "per_sample_rollouts.jsonl", rollouts)

    metrics = aggregate_rollouts(rollouts)
    write_jsonl(output_dir / "per_sample_metrics.jsonl", metrics)
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
            "model": args.model,
            "seed": args.seed,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "finalized_from_shards": args.shard_dirs,
        },
        visualizations=visualizations,
    )
    print(f"finalized {len(rollouts)} rollouts into {output_dir}")


if __name__ == "__main__":
    main()

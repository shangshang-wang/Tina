#!/usr/bin/env python
"""Build balanced eval sets from RL signal metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_signal_splits.io import load_jsonl_by_id, merge_metadata, read_jsonl, write_jsonl
from rl_signal_splits.reporting import split_metric_rows
from rl_signal_splits.splitting import logical_split_ids, rl_signal_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="outputs/rl_signal_split/eval")
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_ratio", type=float, default=0.0)
    parser.add_argument("--exclude_train_split_files", nargs="*", default=[], help="Optional split JSONL files to avoid.")
    return parser.parse_args()


def sample_ids(pool: Sequence[str], size: int, rng: random.Random) -> tuple[List[str], bool]:
    pool = list(dict.fromkeys(pool))
    if not pool:
        return [], False
    rng.shuffle(pool)
    if len(pool) >= size:
        return pool[:size], False
    out = list(pool)
    replacement = True
    while len(out) < size:
        out.append(rng.choice(pool))
    return out, replacement


def load_excluded_ids(paths: Sequence[str]) -> set[str]:
    excluded = set()
    for path in paths:
        for row in read_jsonl(path):
            for key in ("id", "problem_id", "uid", "question_id"):
                if key in row:
                    excluded.add(str(row[key]))
                    break
    return excluded


def materialize(ids: Sequence[str], metrics_by_id: Dict[str, Dict[str, Any]], source_by_id: Dict[str, Dict[str, Any]], tags_by_id: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    rows = []
    for sample_id in ids:
        metric = metrics_by_id[sample_id]
        source = source_by_id.get(sample_id, {}).get("source") or metric.get("source") or {}
        rows.append(merge_metadata(source, rl_signal_metadata(metric, tags_by_id.get(sample_id, []))))
    return rows


def stats_markdown(name: str, ids: Sequence[str], metrics_by_id: Dict[str, Dict[str, Any]]) -> str:
    ms = [metrics_by_id[i] for i in dict.fromkeys(ids) if i in metrics_by_id]
    if not ms:
        return f"| {name} | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 |"
    avg = lambda key: mean(float(m.get(key, 0.0)) for m in ms)
    return (
        f"| {name} | {len(ids)} | {avg('pass_rate'):.4f} | {avg('reward_std'):.4f} | "
        f"{avg('length_mean_tokens'):.2f} | {avg('format_valid_rate'):.4f} |"
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [row for row in read_jsonl(args.metrics) if row.get("id") is not None]
    source_by_id = load_jsonl_by_id(args.input)
    metrics_by_id = {str(m["id"]): m for m in metrics}
    candidate_ids = list(metrics_by_id)
    if args.holdout_ratio > 0:
        rng.shuffle(candidate_ids)
        holdout_n = max(1, int(round(len(candidate_ids) * args.holdout_ratio)))
        holdout_ids = set(candidate_ids[:holdout_n])
        metrics_for_eval = [m for m in metrics if str(m["id"]) in holdout_ids]
    else:
        metrics_for_eval = list(metrics)

    excluded = load_excluded_ids(args.exclude_train_split_files)
    if excluded:
        metrics_for_eval = [m for m in metrics_for_eval if str(m["id"]) not in excluded]

    split_sets, thresholds = logical_split_ids(metrics_for_eval)
    split_sets["mixed"] = set()
    tags_by_id = {str(m["id"]): [] for m in metrics}
    for split, ids in split_sets.items():
        if split == "mixed":
            continue
        for sample_id in ids:
            tags_by_id.setdefault(sample_id, []).append(f"eval_{split}")

    per_split_size = args.eval_size
    eval_ids: Dict[str, List[str]] = {}
    warnings = []
    for split in ["easy", "medium", "hard", "high_variance", "long_reasoning"]:
        ids, replacement = sample_ids(sorted(split_sets.get(split, set())), per_split_size, rng)
        eval_ids[split] = ids
        if replacement:
            warnings.append(f"eval_{split} used sampling with replacement.")
        if len(set(ids)) < min(per_split_size, 10):
            warnings.append(f"eval_{split} has only {len(set(ids))} unique samples.")

    mixed_target = args.eval_size
    mixed_parts = []
    for split in ["easy", "medium", "hard", "high_variance", "long_reasoning"]:
        want = mixed_target // 5
        part, replacement = sample_ids(sorted(split_sets.get(split, set())), want, rng)
        mixed_parts.extend(part)
        if replacement:
            warnings.append(f"eval_mixed component {split} used replacement.")
    all_holdout = [str(m["id"]) for m in metrics_for_eval]
    while len(mixed_parts) < mixed_target and all_holdout:
        mixed_parts.append(rng.choice(all_holdout))
    eval_ids["mixed"] = mixed_parts[:mixed_target]

    for split, ids in eval_ids.items():
        for sample_id in ids:
            tags_by_id.setdefault(sample_id, []).append(f"eval_{split}")
        write_jsonl(output_dir / f"eval_{split}.jsonl", materialize(ids, metrics_by_id, source_by_id, tags_by_id))

    lines = [
        "# Split Eval Report",
        "",
        f"- metrics: `{args.metrics}`",
        f"- input: `{args.input}`",
        f"- seed: `{args.seed}`",
        f"- holdout_ratio: `{args.holdout_ratio}`",
        f"- thresholds: `{json.dumps(thresholds, ensure_ascii=False)}`",
        "",
        "| eval split | size | pass_rate | reward_std | length | format_valid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, ids in eval_ids.items():
        lines.append(stats_markdown(split, ids, metrics_by_id))
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {w}" for w in (warnings or ["none"]))
    (output_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote eval sets to {output_dir}")


if __name__ == "__main__":
    main()

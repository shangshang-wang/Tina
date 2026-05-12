#!/usr/bin/env python
"""Compute teacher-forcing token signals for rollout completions."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.rollout_utils import read_jsonl, write_csv, write_jsonl
from tina.analysis.token_signal_utils import aggregate_phase_stats, bucket_by_position, compute_token_logprobs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_jsonl", nargs="+", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--no_entropy", action="store_true")
    parser.add_argument("--top_k_high_entropy", type=int, default=20)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def _torch_dtype(dtype: str):
    import torch

    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("analyze_token_signals.py requires transformers and torch") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    rows = []
    for path in args.input_jsonl:
        rows.extend(read_jsonl(path))
    rows = rows[: args.max_samples]

    outputs = []
    aggregate_rows = []
    high_entropy_examples = []
    for row in rows:
        token_rows, truncated = compute_token_logprobs(
            model,
            tokenizer,
            row.get("prompt") or row.get("raw_prompt") or "",
            row.get("completion") or "",
            max_tokens=args.max_tokens,
            compute_entropy=not args.no_entropy,
        )
        phase_stats = aggregate_phase_stats(token_rows)
        if args.no_entropy:
            high = sorted(token_rows, key=lambda r: r.get("nll") or 0, reverse=True)[: args.top_k_high_entropy]
        else:
            high = sorted(token_rows, key=lambda r: r.get("entropy") or -1, reverse=True)[: args.top_k_high_entropy]
        high_compact = [
            {
                "position": item["position"],
                "token": item["token_text"],
                "entropy": item.get("entropy"),
                "nll": item.get("nll"),
                "phase": item.get("phase"),
            }
            for item in high
        ]
        high_entropy_examples.extend(
            {
                "task": row.get("task"),
                "seed": row.get("seed"),
                "index": row.get("index"),
                **item,
            }
            for item in high_compact[:5]
        )
        output = {
            "task": row.get("task"),
            "seed": row.get("seed"),
            "example_id": row.get("example_id"),
            "index": row.get("index"),
            "correct": bool(row.get("correct")),
            "num_tokens": len(token_rows),
            "truncated": truncated,
            "phase_stats": phase_stats,
            "high_entropy_tokens": high_compact,
            "entropy_by_position_bucket": bucket_by_position(token_rows),
        }
        outputs.append(output)
        for phase, stats in phase_stats.items():
            aggregate_rows.append(
                {
                    "task": row.get("task"),
                    "correct": bool(row.get("correct")),
                    "phase": phase,
                    **stats,
                }
            )

    grouped = defaultdict(list)
    for row in aggregate_rows:
        grouped[(row["task"], row["correct"], row["phase"])].append(row)
    csv_rows = []
    for (task, correct, phase), items in sorted(grouped.items()):
        csv_rows.append(
            {
                "task": task,
                "correct": correct,
                "phase": phase,
                "mean_entropy": _mean([r["mean_entropy"] for r in items]),
                "mean_nll": _mean([r["mean_nll"] for r in items]),
                "mean_top1_prob": _mean([r["mean_top1_prob"] for r in items]),
                "math_density": _mean([r["math_density"] for r in items]),
                "count": sum(int(r["count"]) for r in items),
            }
        )

    write_jsonl(args.output_jsonl, outputs)
    write_csv(args.output_csv, csv_rows)
    md = [
        "# Token Signal Summary",
        "",
        "## Plan vs Execute Entropy / NLL",
        _table(csv_rows, ["task", "correct", "phase", "mean_entropy", "mean_nll", "mean_top1_prob", "math_density", "count"]),
        "",
        "## High Entropy Token Examples",
        _table(high_entropy_examples[:50], ["task", "seed", "index", "position", "phase", "token", "entropy", "nll"]),
        "",
        "Cost controls used by this run:",
        "",
        f"- max_samples: `{args.max_samples}`",
        f"- max_tokens: `{args.max_tokens}`",
        f"- no_entropy: `{args.no_entropy}`",
        "",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

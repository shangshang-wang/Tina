#!/usr/bin/env python
"""Analyze <plan>...</plan> blocks in rollout JSONL files."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.plan_utils import analyze_plan_features
from tina.analysis.rollout_utils import parse_success, read_jsonl, write_csv


CSV_FIELDS = [
    "task",
    "seed",
    "example_id",
    "index",
    "correct",
    "has_plan_open",
    "has_plan_close",
    "num_plan_open",
    "num_plan_close",
    "plan_valid",
    "plan_format_valid",
    "plan_text",
    "plan_char_len",
    "plan_token_len",
    "plan_position_ratio",
    "execute_text",
    "answer_text",
    "has_final_answer_phrase",
    "has_boxed",
    "parse_success",
    "num_output_tokens",
    "plan_specificity",
    "problem_to_plan_overlap_count",
    "plan_math_density",
    "template_phrase_count",
    "template_score",
    "plan_computation_leakage",
    "leakage_score",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _avg(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else 0.0


def make_summary(rows):
    by_task = defaultdict(list)
    by_task_correct = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
        by_task_correct[(row["task"], bool(row["correct"]))].append(row)

    task_rows = []
    for task, task_items in sorted(by_task.items()):
        task_rows.append(
            {
                "task": task,
                "n": len(task_items),
                "valid_plan_rate": _avg(task_items, "plan_format_valid"),
                "avg_plan_token_len": _avg(task_items, "plan_token_len"),
                "avg_plan_specificity": _avg(task_items, "plan_specificity"),
                "avg_plan_math_density": _avg(task_items, "plan_math_density"),
                "avg_template_score": _avg(task_items, "template_score"),
                "avg_num_output_tokens": _avg(task_items, "num_output_tokens"),
            }
        )

    correct_rows = []
    for (task, correct), items in sorted(by_task_correct.items()):
        correct_rows.append(
            {
                "task": task,
                "correct": correct,
                "n": len(items),
                "valid_plan_rate": _avg(items, "plan_format_valid"),
                "avg_plan_token_len": _avg(items, "plan_token_len"),
                "avg_plan_specificity": _avg(items, "plan_specificity"),
                "avg_plan_math_density": _avg(items, "plan_math_density"),
                "avg_template_score": _avg(items, "template_score"),
                "avg_num_output_tokens": _avg(items, "num_output_tokens"),
            }
        )
    return task_rows, correct_rows


def _case_table(rows, sort_key, n=20, reverse=True):
    selected = sorted(rows, key=lambda r: (r.get(sort_key) is not None, r.get(sort_key) or 0), reverse=reverse)[:n]
    out = []
    for row in selected:
        plan = (row.get("plan_text") or "").replace("\n", " ")
        if len(plan) > 180:
            plan = plan[:177] + "..."
        out.append(
            {
                "task": row.get("task"),
                "seed": row.get("seed"),
                "index": row.get("index"),
                "correct": row.get("correct"),
                sort_key: row.get(sort_key),
                "specificity": row.get("plan_specificity"),
                "template": row.get("template_score"),
                "plan": plan,
            }
        )
    return _markdown_table(out, ["task", "seed", "index", "correct", sort_key, "specificity", "template", "plan"])


def main():
    args = parse_args()
    output_rows = []
    for path in args.input_jsonl:
        for row in read_jsonl(path):
            features = analyze_plan_features(
                row.get("problem") or "",
                row.get("completion") or "",
                row.get("num_output_tokens"),
            )
            output_rows.append(
                {
                    "task": row.get("task"),
                    "seed": row.get("seed"),
                    "example_id": row.get("example_id"),
                    "index": row.get("index"),
                    "correct": bool(row.get("correct")),
                    "parse_success": parse_success(row),
                    **features,
                }
            )

    write_csv(args.output_csv, output_rows, CSV_FIELDS)
    task_rows, correct_rows = make_summary(output_rows)
    correct_specific_low = [r for r in output_rows if r.get("correct")]
    wrong_specific_low = [r for r in output_rows if not r.get("correct")]
    md = [
        "# Plan Block Summary",
        "",
        "## Valid Plan Rate by Task",
        _markdown_table(
            task_rows,
            [
                "task",
                "n",
                "valid_plan_rate",
                "avg_plan_token_len",
                "avg_plan_specificity",
                "avg_plan_math_density",
                "avg_template_score",
                "avg_num_output_tokens",
            ],
        ),
        "",
        "## Correct vs Wrong Plan Features",
        _markdown_table(
            correct_rows,
            [
                "task",
                "correct",
                "n",
                "valid_plan_rate",
                "avg_plan_token_len",
                "avg_plan_specificity",
                "avg_plan_math_density",
                "avg_template_score",
                "avg_num_output_tokens",
            ],
        ),
        "",
        "## Highest Template Score Plans",
        _case_table(output_rows, "template_score", 20, True),
        "",
        "## Lowest Specificity Correct Plans",
        _case_table(correct_specific_low, "plan_specificity", 10, False),
        "",
        "## Lowest Specificity Wrong Plans",
        _case_table(wrong_specific_low, "plan_specificity", 10, False),
        "",
        "## Highest Computation Leakage Plans",
        _case_table(output_rows, "leakage_score", 20, True),
        "",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

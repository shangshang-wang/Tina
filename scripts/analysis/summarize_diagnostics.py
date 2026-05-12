#!/usr/bin/env python
"""Combine diagnostic outputs into one Markdown report."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.rollout_utils import read_csv_rows, read_jsonl


ERROR_TYPES = [
    "format_error",
    "answer_extraction_error",
    "shorter_overcompressed",
    "overlong_repetition",
    "plan_template_only",
    "plan_execute_mismatch",
    "arithmetic_or_execution_error",
    "knowledge_or_domain_error",
    "unknown",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--plan_stats", default=None)
    parser.add_argument("--token_stats", default=None)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


def _float(row, key):
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y"}:
            return 1.0
        if lowered in {"false", "no", "n", ""}:
            return 0.0
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(rows, headers):
    if not rows:
        return "No data."
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _avg(rows, key):
    vals = [_float(r, key) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    paired = read_csv_rows(analysis_dir / "paired_summary.csv") if (analysis_dir / "paired_summary.csv").exists() else []
    regressions = read_jsonl(analysis_dir / "regression_cases.jsonl") if (analysis_dir / "regression_cases.jsonl").exists() else []
    improvements = read_jsonl(analysis_dir / "improvement_cases.jsonl") if (analysis_dir / "improvement_cases.jsonl").exists() else []
    plan_rows = read_csv_rows(args.plan_stats) if args.plan_stats and Path(args.plan_stats).exists() else []
    token_rows = read_csv_rows(args.token_stats) if args.token_stats and Path(args.token_stats).exists() else []

    paired_by_task = defaultdict(list)
    for row in paired:
        paired_by_task[row["task"]].append(row)
    paired_summary = []
    for task, rows in sorted(paired_by_task.items()):
        paired_summary.append(
            {
                "task": task,
                "base_acc": _avg(rows, "base_acc"),
                "exp_acc": _avg(rows, "exp_acc"),
                "delta": _avg(rows, "delta"),
                "regression_count": sum(int(float(r.get("base_correct_exp_wrong") or 0)) for r in rows),
                "improvement_count": sum(int(float(r.get("base_wrong_exp_correct") or 0)) for r in rows),
            }
        )

    error_by_task = defaultdict(Counter)
    for case in regressions:
        error_by_task[case.get("task")][case.get("error_type", "unknown")] += 1
    error_rows = []
    for task, counter in sorted(error_by_task.items()):
        row = {"task": task}
        for error_type in ERROR_TYPES:
            row[error_type] = counter.get(error_type, 0)
        error_rows.append(row)

    plan_by_task = defaultdict(list)
    for row in plan_rows:
        plan_by_task[row.get("task")].append(row)
    plan_summary = []
    for task, rows in sorted(plan_by_task.items()):
        plan_summary.append(
            {
                "task": task,
                "valid_plan_rate": _avg(rows, "plan_format_valid"),
                "avg_plan_len": _avg(rows, "plan_token_len"),
                "specificity": _avg(rows, "plan_specificity"),
                "math_density": _avg(rows, "plan_math_density"),
                "template_score": _avg(rows, "template_score"),
                "computation_leakage_rate": _avg(rows, "plan_computation_leakage"),
            }
        )

    reps = []
    for case in (regressions[:10] + improvements[:5]):
        completion = case.get("exp_completion") or case.get("base_completion") or ""
        reps.append(
            {
                "task": case.get("task"),
                "seed": case.get("seed"),
                "index": case.get("index"),
                "error_type": case.get("error_type", "improvement"),
                "gold": case.get("gold_letter") or case.get("gold"),
                "base_pred": case.get("base_pred_letter") or case.get("base_pred"),
                "exp_pred": case.get("exp_pred_letter") or case.get("exp_pred"),
                "snippet": completion.replace("\n", " ")[:160],
            }
        )

    suggestions = []
    if plan_summary and any(r["valid_plan_rate"] < 0.8 for r in plan_summary):
        suggestions.append("Low valid_plan_rate: strengthen plan_format reward or prompt constraints.")
    if plan_summary and any(r["template_score"] > 0.5 for r in plan_summary):
        suggestions.append("High template_score: shorten plans, reward problem specificity, or penalize generic planning phrases.")
    if any(str(r.get("task", "")).startswith("gpqa") and int(r.get("format_error", 0)) > 0 for r in error_rows):
        suggestions.append("GPQA format errors: adjust GPQA prompt, answer anchor, or choice-letter extraction.")
    if any(r.get("answer_extraction_error", 0) or r.get("format_error", 0) for r in error_rows):
        suggestions.append("Parse failures: increase answer span weight or format reward.")
    if token_rows:
        execute_rows = [r for r in token_rows if r.get("phase") == "execute"]
        if execute_rows and _avg(execute_rows, "mean_entropy") > 4.0:
            suggestions.append("High execute entropy: consider stronger execute_rl_weight, full-token KL, or avoiding plan-prefix-only reward.")
    if plan_summary and any(r["specificity"] > 0.35 and r["template_score"] < 0.35 for r in plan_summary):
        suggestions.append("Specific plans with remaining errors suggest execute drift; inspect plan_execute_mismatch and execution_error cases.")

    md = [
        "# Diagnostic Report",
        "",
        "## 1. Paired Accuracy Changes",
        _table(paired_summary, ["task", "base_acc", "exp_acc", "delta", "regression_count", "improvement_count"]),
        "",
        "## 2. Regression Case Taxonomy",
        _table(error_rows, ["task"] + ERROR_TYPES),
        "",
        "## 3. Plan Block Quality",
        _table(plan_summary, ["task", "valid_plan_rate", "avg_plan_len", "specificity", "math_density", "template_score", "computation_leakage_rate"]),
        "",
        "## 4. Token Signal Analysis",
        _table(token_rows, ["task", "correct", "phase", "mean_entropy", "mean_nll", "mean_top1_prob", "math_density", "count"]),
        "",
        "## 5. Representative Cases",
        _table(reps, ["task", "seed", "index", "error_type", "gold", "base_pred", "exp_pred", "snippet"]),
        "",
        "## 6. Actionable Suggestions",
        "\n".join(f"- {item}" for item in suggestions) if suggestions else "- No automatic suggestions triggered.",
        "",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

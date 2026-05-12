#!/usr/bin/env python
"""Paired comparison for two rollout directories."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.error_taxonomy import classify_regression_case
from tina.analysis.rollout_utils import (
    expand_task_arg,
    has_boxed,
    load_rollout_dir,
    parse_int_csv_arg,
    parse_success,
    rollout_match_key,
    write_csv,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task list, or aliases: all, math, science.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Omit with --seedless.")
    parser.add_argument("--seedless", action="store_true")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _case(base, exp):
    case = {
        "task": base.get("task"),
        "seed": base.get("seed"),
        "example_id": base.get("example_id") or exp.get("example_id"),
        "index": base.get("index"),
        "problem": base.get("problem") or exp.get("problem"),
        "gold": base.get("gold") or exp.get("gold"),
        "gold_letter": base.get("gold_letter") or exp.get("gold_letter"),
        "base_completion": base.get("completion"),
        "base_pred": base.get("pred"),
        "base_pred_letter": base.get("pred_letter"),
        "exp_completion": exp.get("completion"),
        "exp_pred": exp.get("pred"),
        "exp_pred_letter": exp.get("pred_letter"),
        "base_num_output_tokens": base.get("num_output_tokens"),
        "exp_num_output_tokens": exp.get("num_output_tokens"),
        "base_has_boxed": has_boxed(base.get("completion") or ""),
        "exp_has_boxed": has_boxed(exp.get("completion") or ""),
        "base_parse_success": parse_success(base),
        "exp_parse_success": parse_success(exp),
    }
    return case


def _fmt(value):
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
    tasks = expand_task_arg(args.tasks)
    seeds = None if args.seedless else parse_int_csv_arg(args.seeds)
    if not args.seedless and not seeds:
        raise ValueError("Pass --seeds 0,1,... or use --seedless.")
    base_rows = load_rollout_dir(args.base_dir, tasks, seeds)
    exp_rows = load_rollout_dir(args.exp_dir, tasks, seeds)
    base_by_key = {rollout_match_key(row): row for row in base_rows}
    exp_by_key = {rollout_match_key(row): row for row in exp_rows}

    summary = []
    regressions = []
    improvements = []
    by_task_seed = defaultdict(list)
    for key in sorted(base_by_key.keys() & exp_by_key.keys()):
        base = base_by_key[key]
        exp = exp_by_key[key]
        seed_group = None if args.seedless else base.get("seed")
        by_task_seed[(base.get("task"), seed_group)].append((base, exp))

    for (task, seed), pairs in sorted(by_task_seed.items()):
        both_correct = both_wrong = base_correct_exp_wrong = base_wrong_exp_correct = 0
        for base, exp in pairs:
            b_ok = bool(base.get("correct"))
            e_ok = bool(exp.get("correct"))
            if b_ok and e_ok:
                both_correct += 1
            elif not b_ok and not e_ok:
                both_wrong += 1
            elif b_ok and not e_ok:
                base_correct_exp_wrong += 1
                case = _case(base, exp)
                case.update(classify_regression_case(case))
                regressions.append(case)
            elif not b_ok and e_ok:
                base_wrong_exp_correct += 1
                improvements.append(_case(base, exp))
        total = len(pairs)
        base_acc = (both_correct + base_correct_exp_wrong) / total if total else 0.0
        exp_acc = (both_correct + base_wrong_exp_correct) / total if total else 0.0
        summary.append(
            {
                "task": task,
                "seed": seed,
                "total": total,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "base_correct_exp_wrong": base_correct_exp_wrong,
                "base_wrong_exp_correct": base_wrong_exp_correct,
                "base_acc": base_acc,
                "exp_acc": exp_acc,
                "delta": exp_acc - base_acc,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "paired_summary.csv", summary)
    write_jsonl(out_dir / "regression_cases.jsonl", regressions)
    write_jsonl(out_dir / "improvement_cases.jsonl", improvements)

    by_task_errors = defaultdict(Counter)
    for case in regressions:
        by_task_errors[case["task"]][case["error_type"]] += 1
    error_rows = []
    for task, counter in sorted(by_task_errors.items()):
        for error_type, count in counter.most_common():
            error_rows.append({"task": task, "error_type": error_type, "count": count})

    reps = []
    for case in regressions[:20]:
        snippet = (case.get("exp_completion") or "").replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        reps.append(
            {
                "task": case.get("task"),
                "seed": case.get("seed"),
                "index": case.get("index"),
                "error_type": case.get("error_type"),
                "base_pred": case.get("base_pred") or case.get("base_pred_letter"),
                "exp_pred": case.get("exp_pred") or case.get("exp_pred_letter"),
                "exp_snippet": snippet,
            }
        )
    md = [
        "# Paired Rollout Comparison",
        "",
        "## Accuracy Changes",
        _table(summary, ["task", "seed", "total", "base_acc", "exp_acc", "delta", "base_correct_exp_wrong", "base_wrong_exp_correct"]),
        "",
        "## Regression Error Type Distribution",
        _table(error_rows, ["task", "error_type", "count"]) if error_rows else "No regressions.",
        "",
        "## Representative Regression Cases",
        _table(reps, ["task", "seed", "index", "error_type", "base_pred", "exp_pred", "exp_snippet"]) if reps else "No regressions.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

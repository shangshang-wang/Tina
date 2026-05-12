#!/usr/bin/env python
"""Select high-signal paired diagnostic cases from LightEval details parquet files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.rollout_utils import expand_task_arg, write_csv, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details_root", default="outputs")
    parser.add_argument("--tasks", default="amc23,math_500,minerva,gpqa:diamond,aime24,aime25,olympiadbench")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--base_pattern", default="DeepSeek-R1-Distill-Qwen-1.5B_base")
    parser.add_argument("--exp_pattern", default="plan15_ablation_checkpoint-250-merged")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_task", type=int, default=12)
    parser.add_argument("--max_hard_both_wrong_per_task", type=int, default=2)
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _scalar_metric(metrics: Any) -> float:
    if isinstance(metrics, dict):
        return float(metrics.get("extractive_match", 0.0) or 0.0)
    return 0.0


def _list_first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _stringify(value: Any) -> str:
    value = _list_first(value)
    return "" if value is None else str(value)


def _extract_specific_array(specifics: Any, key: str) -> str | None:
    if not isinstance(specifics, dict) or key not in specifics:
        return None
    value = specifics[key]
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            return str(value[0]) if value else None
        return str(value)
    except Exception:
        return str(value)


def _find_details_file(root: Path, task: str, seed: int, pattern: str) -> Path | None:
    candidates = [
        p
        for p in root.glob(f"{task}/{seed}/**/details_*.parquet")
        if pattern in str(p)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _load_details(path: Path, task: str, seed: int, model_label: str) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows = []
    for index, row in df.iterrows():
        metrics = row.get("metrics", {})
        completion = _stringify(row.get("predictions"))
        specifics = row.get("specifics", {})
        rows.append(
            {
                "task": task,
                "seed": seed,
                "example_id": str(index),
                "index": int(index),
                "problem": _stringify(row.get("example")),
                "prompt": _stringify(row.get("full_prompt")) or _stringify(row.get("example")),
                "gold": _stringify(row.get("gold")),
                "completion": completion,
                "pred": _extract_specific_array(specifics, "extracted_predictions"),
                "gold_extracted": _extract_specific_array(specifics, "extracted_golds"),
                "correct": _scalar_metric(metrics) > 0,
                "num_output_tokens": len(re.findall(r"\S+", completion)),
                "model_label": model_label,
                "details_path": str(path),
            }
        )
    return rows


def _length_ratio(base: dict[str, Any], exp: dict[str, Any]) -> float:
    return (exp.get("num_output_tokens") or 0) / max(1, base.get("num_output_tokens") or 0)


def _case_kind(base: dict[str, Any], exp: dict[str, Any]) -> str:
    b_ok = bool(base["correct"])
    e_ok = bool(exp["correct"])
    if b_ok and not e_ok:
        return "regression"
    if not b_ok and e_ok:
        return "improvement"
    if b_ok and e_ok:
        return "both_correct"
    if base.get("pred") != exp.get("pred") or abs(_length_ratio(base, exp) - 1.0) > 0.35:
        return "both_wrong_changed"
    return "both_wrong_same"


def _paired_case(base: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    kind = _case_kind(base, exp)
    return {
        "task": base["task"],
        "seed": base["seed"],
        "example_id": base["example_id"],
        "index": base["index"],
        "case_kind": kind,
        "problem": base["problem"],
        "gold": base["gold"],
        "gold_extracted": base.get("gold_extracted"),
        "base_correct": base["correct"],
        "exp_correct": exp["correct"],
        "base_pred": base.get("pred"),
        "exp_pred": exp.get("pred"),
        "base_num_output_tokens": base.get("num_output_tokens"),
        "exp_num_output_tokens": exp.get("num_output_tokens"),
        "length_ratio_exp_over_base": _length_ratio(base, exp),
        "base_completion": base.get("completion"),
        "exp_completion": exp.get("completion"),
        "base_details_path": base.get("details_path"),
        "exp_details_path": exp.get("details_path"),
    }


def _select_task_cases(cases: list[dict[str, Any]], per_task: int, max_hard: int) -> list[dict[str, Any]]:
    buckets = defaultdict(list)
    for case in cases:
        buckets[case["case_kind"]].append(case)
    for items in buckets.values():
        items.sort(key=lambda c: (c["seed"], c["index"]))

    selected: list[dict[str, Any]] = []
    priority = ["regression", "improvement", "both_correct", "both_wrong_changed"]
    limits = {"both_wrong_changed": max_hard}
    while len(selected) < per_task:
        made_progress = False
        for kind in priority:
            if len(selected) >= per_task:
                break
            if kind in limits and sum(c["case_kind"] == kind for c in selected) >= limits[kind]:
                continue
            if buckets[kind]:
                selected.append(buckets[kind].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def _md_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            vals.append(str(v).replace("\n", " ")[:220])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    root = Path(args.details_root)
    tasks = expand_task_arg(args.tasks)
    seeds = _parse_csv_ints(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = []
    missing = []
    for task in tasks:
        for seed in seeds:
            base_path = _find_details_file(root, task, seed, args.base_pattern)
            exp_path = _find_details_file(root, task, seed, args.exp_pattern)
            if base_path is None or exp_path is None:
                missing.append({"task": task, "seed": seed, "base_found": base_path is not None, "exp_found": exp_path is not None})
                continue
            base_rows = {r["index"]: r for r in _load_details(base_path, task, seed, "base")}
            exp_rows = {r["index"]: r for r in _load_details(exp_path, task, seed, "exp")}
            for index in sorted(base_rows.keys() & exp_rows.keys()):
                all_cases.append(_paired_case(base_rows[index], exp_rows[index]))

    selected = []
    summary_rows = []
    by_task = defaultdict(list)
    for case in all_cases:
        by_task[case["task"]].append(case)
    for task, cases in sorted(by_task.items()):
        selected_task = _select_task_cases(cases, args.per_task, args.max_hard_both_wrong_per_task)
        selected.extend(selected_task)
        counts = defaultdict(int)
        for case in cases:
            counts[case["case_kind"]] += 1
        summary_rows.append(
            {
                "task": task,
                "total_pairs": len(cases),
                "regression": counts["regression"],
                "improvement": counts["improvement"],
                "both_correct": counts["both_correct"],
                "both_wrong_changed": counts["both_wrong_changed"],
                "both_wrong_same": counts["both_wrong_same"],
                "selected": len(selected_task),
            }
        )

    write_jsonl(output_dir / "all_paired_cases.jsonl", all_cases)
    write_jsonl(output_dir / "selected_diagnostic_cases.jsonl", selected)
    write_csv(output_dir / "case_summary.csv", summary_rows)
    write_csv(output_dir / "missing_details.csv", missing)
    manifest_rows = [
        {
            "task": c["task"],
            "seed": c["seed"],
            "index": c["index"],
            "example_id": c["example_id"],
            "case_kind": c["case_kind"],
            "base_correct": c["base_correct"],
            "exp_correct": c["exp_correct"],
            "base_pred": c["base_pred"],
            "exp_pred": c["exp_pred"],
            "base_tokens": c["base_num_output_tokens"],
            "exp_tokens": c["exp_num_output_tokens"],
            "length_ratio": c["length_ratio_exp_over_base"],
        }
        for c in selected
    ]
    write_csv(output_dir / "selected_manifest.csv", manifest_rows)

    md = [
        "# Discriminative Diagnostic Case Selection",
        "",
        "## Pair Counts",
        _md_table(summary_rows, ["task", "total_pairs", "regression", "improvement", "both_correct", "both_wrong_changed", "both_wrong_same", "selected"]),
        "",
        "## Selected Cases",
        _md_table(manifest_rows, ["task", "seed", "index", "case_kind", "base_correct", "exp_correct", "base_pred", "exp_pred", "base_tokens", "exp_tokens", "length_ratio"]),
        "",
    ]
    (output_dir / "selection_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

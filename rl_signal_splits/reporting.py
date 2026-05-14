"""Markdown, JSON, and CSV reports for RL signal splits."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence


def avg(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(mean(vals)) if vals else 0.0


def split_metric_rows(metrics_by_id: Mapping[str, Dict[str, Any]], split_ids: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    rows = []
    for name, ids in split_ids.items():
        unique_ids = list(dict.fromkeys(ids))
        ms = [metrics_by_id[i] for i in unique_ids if i in metrics_by_id]
        rows.append(
            {
                "split": name,
                "size": len(ids),
                "unique_size": len(unique_ids),
                "mean_pass_rate": avg(ms, "pass_rate"),
                "mean_reward_mean": avg(ms, "reward_mean"),
                "mean_reward_std": avg(ms, "reward_std"),
                "mean_length_tokens": avg(ms, "length_mean_tokens"),
                "mean_answer_diversity": avg(ms, "answer_diversity"),
                "mean_format_valid_rate": avg(ms, "format_valid_rate"),
            }
        )
    return rows


def overlap_table(split_ids: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, int]]:
    sets = {name: set(ids) for name, ids in split_ids.items()}
    return {
        a: {b: len(ids_a & ids_b) for b, ids_b in sets.items() if b != a}
        for a, ids_a in sets.items()
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def training_advice(metrics: List[Dict[str, Any]], split_ids: Mapping[str, Sequence[str]]) -> List[str]:
    total = max(1, len(metrics))
    advice = []
    medium_frac = len(set(split_ids.get("medium", []))) / total
    hard_frac = len(set(split_ids.get("hard", []))) / total
    long_frac = len(set(split_ids.get("long_reasoning", []))) / total
    var_frac = len(set(split_ids.get("high_variance", []))) / total
    if medium_frac < 0.20:
        advice.append("medium split is small; the dataset may be too easy or too hard for the reference model, reducing dense RL learning signal.")
    if hard_frac > 0.50:
        advice.append("hard split is large; expect sparse rewards and consider curriculum, stronger verifier feedback, or mixing easier samples.")
    if long_frac > 0.35:
        advice.append("long_reasoning split is large; monitor completion length bias and token budget pressure during LoRA training.")
    if var_frac > 0.40:
        advice.append("high_variance split is large; reward variance may destabilize GRPO/RL updates, so use smaller LR or stronger clipping as needed.")
    if not advice:
        advice.append("split proportions look usable for comparing independent LoRA training runs.")
    return advice


def write_split_stats_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: str | Path,
    metrics: List[Dict[str, Any]],
    split_result: Dict[str, Any],
    run_config: Dict[str, Any],
    visualizations: Sequence[str] | None = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_id = {str(m["id"]): m for m in metrics}
    split_ids = split_result["split_ids"]
    stat_rows = split_metric_rows(metrics_by_id, split_ids)
    overlaps = overlap_table(split_ids)
    dataset = {
        "total_samples": len(metrics),
        "valid_samples": sum(1 for m in metrics if not m.get("anomaly")),
        "mean_pass_rate": avg(metrics, "pass_rate"),
        "mean_reward_mean": avg(metrics, "reward_mean"),
        "mean_reward_std": avg(metrics, "reward_std"),
        "mean_length_tokens": avg(metrics, "length_mean_tokens"),
        "mean_format_valid_rate": avg(metrics, "format_valid_rate"),
    }
    report = {
        "dataset": dataset,
        "splits": stat_rows,
        "overlaps": overlaps,
        "thresholds": split_result.get("thresholds", {}),
        "warnings": split_result.get("warnings", []),
        "training_advice": training_advice(metrics, split_ids),
        "score_formulas": {
            "difficulty_score": "1 - pass_rate",
            "variance_score": "mean(robust_percentile_norm(reward_std), robust_percentile_norm(answer_diversity), robust_percentile_norm(correctness_entropy))",
            "long_reasoning_score": "mean(robust_percentile_norm(length_mean_tokens), robust_percentile_norm(length_p90_tokens))",
            "instability_score": "mean(variance_score, robust_percentile_norm(invalid_count), robust_percentile_norm(completion_length_cv))",
            "robust_percentile_norm": "clip((x - dataset_p5) / max(dataset_p95 - dataset_p5, eps), 0, 1)",
        },
        "reproducibility": {
            **run_config,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        },
        "visualizations": list(visualizations or []),
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_split_stats_csv(output_dir / "split_stats.csv", stat_rows)
    write_markdown_report(output_dir / "split_report.md", report)
    return report


def write_markdown_report(path: str | Path, report: Dict[str, Any]) -> None:
    dataset = report["dataset"]
    lines = [
        "# RL Signal Split Report",
        "",
        "## Data Overview",
        "",
        f"- total samples: {dataset['total_samples']}",
        f"- valid samples: {dataset['valid_samples']}",
        f"- mean pass_rate: {dataset['mean_pass_rate']:.4f}",
        f"- mean reward_mean: {dataset['mean_reward_mean']:.4f}",
        f"- mean reward_std: {dataset['mean_reward_std']:.4f}",
        f"- mean length tokens: {dataset['mean_length_tokens']:.2f}",
        f"- mean format_valid_rate: {dataset['mean_format_valid_rate']:.4f}",
        "",
        "## Split Statistics",
        "",
        "| split | size | unique | pass_rate | reward_std | length | answer_diversity | format_valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["splits"]:
        lines.append(
            f"| {row['split']} | {row['size']} | {row['unique_size']} | {row['mean_pass_rate']:.4f} | "
            f"{row['mean_reward_std']:.4f} | {row['mean_length_tokens']:.2f} | "
            f"{row['mean_answer_diversity']:.4f} | {row['mean_format_valid_rate']:.4f} |"
        )
    lines.extend(["", "## Overlap With Other Splits", ""])
    for split, overlaps in report["overlaps"].items():
        overlap_text = ", ".join(f"{k}: {v}" for k, v in overlaps.items())
        lines.append(f"- {split}: {overlap_text}")
    lines.extend(["", "## Distribution Analysis", ""])
    if report.get("visualizations"):
        for viz in report["visualizations"]:
            lines.append(f"- {viz}")
    else:
        lines.append("- visualization generation was skipped or matplotlib was unavailable.")
    lines.extend(["", "## Formulas", ""])
    for name, formula in report["score_formulas"].items():
        lines.append(f"- {name}: `{formula}`")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings") or ["none"]
    lines.extend(f"- {w}" for w in warnings)
    lines.extend(["", "## Training Advice", ""])
    lines.extend(f"- {x}" for x in report["training_advice"])
    lines.extend(["", "## Reproducibility", ""])
    for key, value in report["reproducibility"].items():
        lines.append(f"- {key}: `{value}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

#!/usr/bin/env python
"""Build a fixed checkpoint-selection sentinel eval from paired full benchmark results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ckpt_sentinel_utils import (  # noqa: E402
    CANONICAL_BENCHMARKS,
    aggregate_runs,
    deterministic_select,
    discover_result_files,
    format_float,
    full_scores,
    make_candidate_pool,
    make_weights,
    normalize_benchmark,
    read_jsonl,
    scaled_allocation,
    summarize_counts,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", default="eval_results/full_benchmarks")
    parser.add_argument("--dataset_dir", default="data/eval_benchmarks")
    parser.add_argument("--base_model_name", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--baseline_model_name", default="open-rs3")
    parser.add_argument("--output_dir", default="outputs/ckpt_sentinel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sentinel_size", type=int, default=150)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--limit_per_benchmark", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--base_files",
        default=None,
        help="Optional comma-separated explicit base result files. Supports jsonl/json/parquet.",
    )
    parser.add_argument(
        "--baseline_files",
        default=None,
        help="Optional comma-separated explicit Open-RS3 result files. Supports jsonl/json/parquet.",
    )
    return parser.parse_args()


def parse_file_list(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def load_rows(files: list[Path], label: str) -> list[dict[str, Any]]:
    from scripts.ckpt_sentinel_utils import load_result_file

    rows = []
    failures = []
    for path in files:
        try:
            loaded = load_result_file(path)
            rows.extend(loaded)
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print(f"[warning] failed to load {label} files:", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  - {failure}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more", file=sys.stderr)
    return rows


def load_dataset_fallback(dataset_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_benchmark: dict[str, dict[str, dict[str, Any]]] = {}
    if not dataset_dir.exists():
        return by_benchmark
    for path in dataset_dir.rglob("*.jsonl"):
        bench = normalize_benchmark(path.as_posix())
        if not bench:
            continue
        for row in read_jsonl(path):
            item_id = row.get("id") or row.get("problem_id") or row.get("index")
            problem = row.get("problem") or row.get("question") or row.get("prompt")
            if item_id is None and problem:
                from scripts.ckpt_sentinel_utils import stable_problem_hash

                item_id = stable_problem_hash(bench, problem)
            if item_id is not None:
                by_benchmark.setdefault(bench, {})[str(item_id)] = row
    return by_benchmark


def apply_dataset_fallback(records: list[dict[str, Any]], dataset_rows: dict[str, dict[str, dict[str, Any]]]) -> None:
    for record in records:
        if record.get("problem") and record.get("gold_answer"):
            continue
        candidates = dataset_rows.get(record["benchmark"], {})
        row = candidates.get(str(record["id"]))
        if not row:
            continue
        record["problem"] = record.get("problem") or row.get("problem") or row.get("question") or row.get("prompt")
        record["gold_answer"] = record.get("gold_answer") or row.get("answer") or row.get("target") or row.get("solution")
        record["prompt"] = record.get("prompt") or row.get("prompt") or row.get("problem") or row.get("question")


def cap_per_benchmark(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    kept = []
    counts = Counter()
    for row in sorted(rows, key=lambda r: (r["benchmark"], str(r["id"]))):
        if counts[row["benchmark"]] >= limit:
            continue
        kept.append(row)
        counts[row["benchmark"]] += 1
    return kept


def selected_item(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "benchmark",
        "problem",
        "prompt",
        "gold_answer",
        "base_correct",
        "baseline_correct",
        "base_correct_rate",
        "baseline_correct_rate",
        "base_extracted_answer",
        "baseline_extracted_answer",
        "base_length",
        "baseline_length",
        "base_format_valid",
        "baseline_format_valid",
        "difficulty_tag",
        "improvement_tag",
        "auxiliary_tags",
        "selection_score",
        "selected",
        "selection_reason",
        "item_weight",
    ]
    return {key: row.get(key) for key in keys if key in row}


def md_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    candidate_pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    allocation: dict[str, int],
    warnings_list: list[str],
    base_files: list[Path],
    baseline_files: list[Path],
) -> None:
    selected_counts = summarize_counts(selected)
    candidate_counts = summarize_counts(candidate_pool)
    base_scores = full_scores(candidate_pool, "base")
    baseline_scores = full_scores(candidate_pool, "baseline")
    selected_ids = {
        bench: [str(r["id"]) for r in selected if r["benchmark"] == bench]
        for bench in CANONICAL_BENCHMARKS
    }
    report_json = {
        "config": {
            "benchmark_dir": args.benchmark_dir,
            "dataset_dir": args.dataset_dir,
            "base_model_name": args.base_model_name,
            "baseline_model_name": args.baseline_model_name,
            "seed": args.seed,
            "sentinel_size": args.sentinel_size,
            "allocation": allocation,
            "dry_run": args.dry_run,
            "limit_per_benchmark": args.limit_per_benchmark,
            "note": "Sentinel is a checkpoint-selection set built from base/Open-RS3 differences, not a final benchmark score.",
        },
        "input_files": {
            "base": [str(p) for p in base_files],
            "baseline": [str(p) for p in baseline_files],
        },
        "full_benchmark_scores_from_candidate_pool": {
            "base": base_scores,
            "baseline": baseline_scores,
        },
        "candidate_label_distribution": candidate_counts,
        "sentinel_label_distribution": selected_counts,
        "selected_ids": selected_ids,
        "warnings": warnings_list,
    }
    write_json(output_dir / "sentinel_selection_report.json", report_json)

    score_rows = [
        [
            bench,
            format_float(base_scores.get(bench)),
            format_float(baseline_scores.get(bench)),
            candidate_counts[bench]["total"],
            candidate_counts[bench]["base_wrong_baseline_right"],
            candidate_counts[bench]["base_right_baseline_wrong"],
            candidate_counts[bench]["both_wrong"],
            candidate_counts[bench]["both_right"],
            selected_counts[bench]["total"],
        ]
        for bench in CANONICAL_BENCHMARKS
    ]
    selection_rows = [
        [
            bench,
            selected_counts[bench]["total"],
            selected_counts[bench]["base_wrong_baseline_right"],
            selected_counts[bench]["base_right_baseline_wrong"],
            selected_counts[bench]["both_wrong"],
            selected_counts[bench]["both_right"],
            ", ".join(selected_ids[bench][:40]),
        ]
        for bench in CANONICAL_BENCHMARKS
    ]
    md = [
        "# Checkpoint Sentinel Selection Report",
        "",
        "This sentinel is not a final evaluation set. It is a fixed checkpoint-selection set chosen from final target benchmarks by comparing base and Open-RS3 per-item behavior.",
        "",
        "The highest-value items are `base_wrong_baseline_right`: they are questions where Open-RS3 preserved or learned capability over the base model. New checkpoints that fail these items are unlikely to rank well on the full benchmark. `both_wrong` items are included to observe breakthroughs, but their share is limited to control variance.",
        "",
        "## Inputs",
        "",
        f"- benchmark_dir: `{args.benchmark_dir}`",
        f"- dataset_dir: `{args.dataset_dir}`",
        f"- base_model_name: `{args.base_model_name}`",
        f"- baseline_model_name: `{args.baseline_model_name}`",
        f"- seed: `{args.seed}`",
        f"- sentinel_size: `{args.sentinel_size}`",
        f"- base files loaded: `{len(base_files)}`",
        f"- baseline files loaded: `{len(baseline_files)}`",
        "",
        "## Full Benchmark Scores From Loaded Candidate Pool",
        "",
        md_table(
            score_rows,
            [
                "benchmark",
                "base",
                "open-rs3",
                "candidate items",
                "base_wrong_baseline_right",
                "base_right_baseline_wrong",
                "both_wrong",
                "both_right",
                "selected",
            ],
        ),
        "",
        "## Sentinel Selection",
        "",
        md_table(
            selection_rows,
            ["benchmark", "selected", "B wrong/O right", "B right/O wrong", "both wrong", "both right", "selected IDs"],
        ),
        "",
        "## Small-Sample Risk",
        "",
        "AIME24, AIME25, and AMC23 are deliberately capped. They are useful for catching reasoning regressions, but a checkpoint should not be selected or rejected from these small tasks alone.",
        "",
        "## Selection Config",
        "",
        "Benchmark allocation: `" + json.dumps(allocation, ensure_ascii=False, sort_keys=True) + "`",
        "",
        "Within each benchmark, deterministic stratified sampling targets 50% `base_wrong_baseline_right`, 15% `base_right_baseline_wrong`, 25% `both_wrong`, and 10% `both_right`, with shortfall fill priority `base_wrong_baseline_right -> both_wrong -> base_right_baseline_wrong -> both_right`.",
    ]
    if warnings_list:
        md.extend(["", "## Warnings", ""])
        md.extend([f"- {warning}" for warning in warnings_list[:100]])
        if len(warnings_list) > 100:
            md.append(f"- ... {len(warnings_list) - 100} more warnings in JSON report")
    (output_dir / "sentinel_selection_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_files = parse_file_list(args.base_files)
    baseline_files = parse_file_list(args.baseline_files)
    benchmark_dir = Path(args.benchmark_dir)
    if not base_files:
        base_files = discover_result_files(benchmark_dir, args.base_model_name, "base")
    if not baseline_files:
        baseline_files = discover_result_files(benchmark_dir, args.baseline_model_name, "baseline")
    if not base_files or not baseline_files:
        raise SystemExit(
            "No paired result files discovered. Pass --benchmark_dir pointing at full eval details/results, "
            "or pass --base_files and --baseline_files explicitly."
        )

    base_rows = load_rows(base_files, "base")
    baseline_rows = load_rows(baseline_files, "baseline")
    if not base_rows or not baseline_rows:
        raise SystemExit(
            "Loaded no per-item rows. For LightEval JSON result summaries, use the sibling details parquet files; "
            "parquet requires pandas and pyarrow."
        )
    base_agg = aggregate_runs(base_rows)
    baseline_agg = aggregate_runs(baseline_rows)
    candidate_pool, warnings_list = make_candidate_pool(base_agg, baseline_agg, args.max_new_tokens)
    candidate_pool = cap_per_benchmark(candidate_pool, 12 if args.dry_run and args.limit_per_benchmark is None else args.limit_per_benchmark)
    dataset_rows = load_dataset_fallback(Path(args.dataset_dir))
    apply_dataset_fallback(candidate_pool, dataset_rows)

    sentinel_size = min(args.sentinel_size, 24) if args.dry_run else args.sentinel_size
    allocation = scaled_allocation(sentinel_size)
    selected = deterministic_select(candidate_pool, allocation, args.seed)
    weights = make_weights(selected)

    write_jsonl(output_dir / "candidate_pool.jsonl", candidate_pool)
    write_jsonl(output_dir / "sentinel_all.jsonl", [selected_item(row) for row in selected])
    by_dir = output_dir / "sentinel_by_benchmark"
    for benchmark in CANONICAL_BENCHMARKS:
        write_jsonl(by_dir / f"{benchmark}.jsonl", [selected_item(row) for row in selected if row["benchmark"] == benchmark])
    write_json(output_dir / "sentinel_weights.json", weights)
    write_report(output_dir, args, candidate_pool, selected, allocation, warnings_list, base_files, baseline_files)

    print(f"Wrote {len(selected)} sentinel items from {len(candidate_pool)} paired candidates to {output_dir}")
    if warnings_list:
        print(f"Warnings: {len(warnings_list)}; see sentinel_selection_report.json")


if __name__ == "__main__":
    main()

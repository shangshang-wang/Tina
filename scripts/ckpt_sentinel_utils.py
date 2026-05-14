#!/usr/bin/env python
"""Utilities for checkpoint-selection sentinel construction and scoring."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CANONICAL_BENCHMARKS = ["aime24", "aime25", "amc23", "math500", "gpqa", "minerva"]
TASK_TO_SENTINEL = {
    "aime24": "aime24",
    "aime25": "aime25",
    "amc23": "amc23",
    "math_500": "math500",
    "math500": "math500",
    "gpqa:diamond": "gpqa",
    "gpqa_diamond": "gpqa",
    "gpqa": "gpqa",
    "minerva": "minerva",
    "minerva_math": "minerva",
}
SENTINEL_TO_TASK = {
    "aime24": "aime24",
    "aime25": "aime25",
    "amc23": "amc23",
    "math500": "math_500",
    "gpqa": "gpqa:diamond",
    "minerva": "minerva",
}
DEFAULT_SENTINEL_ALLOCATION = {
    "aime24": 6,
    "aime25": 6,
    "amc23": 12,
    "math500": 56,
    "gpqa": 35,
    "minerva": 35,
}
TAG_TARGET_RATIOS = {
    "base_wrong_baseline_right": 0.50,
    "base_right_baseline_wrong": 0.15,
    "both_wrong": 0.25,
    "both_right": 0.10,
}
FILL_PRIORITY = ["base_wrong_baseline_right", "both_wrong", "base_right_baseline_wrong", "both_right"]
ITEM_WEIGHT_BY_TAG = {
    "base_wrong_baseline_right": 1.2,
    "base_right_baseline_wrong": 1.0,
    "both_wrong": 1.0,
    "both_right": 0.5,
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_selector(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def normalize_benchmark(value: Any) -> str | None:
    aliases = {
        "aime_2024": "aime24",
        "aime2024": "aime24",
        "aime24": "aime24",
        "aime_2025": "aime25",
        "aime2025": "aime25",
        "aime25": "aime25",
        "amc_2023": "amc23",
        "amc2023": "amc23",
        "amc23": "amc23",
        "math_500": "math500",
        "math500": "math500",
        "math-500": "math500",
        "gpqa": "gpqa",
        "gpqa_diamond": "gpqa",
        "gpqa:diamond": "gpqa",
        "minerva": "minerva",
        "minerva_math": "minerva",
    }
    raw_text = str(value or "").strip().lower()
    scan_text = re.sub(r"[^a-z0-9:_-]+", "_", raw_text).replace("-", "_")
    for key, canonical in aliases.items():
        if key in scan_text:
            return canonical
    text = raw_text.replace("custom|", "")
    if "|" in text:
        parts = [p for p in text.split("|") if p and p != "0"]
        text = parts[-1] if parts else text
    text = re.sub(r"[^a-z0-9:_-]+", "_", text)
    text = text.replace("-", "_")
    if text in aliases:
        return aliases[text]
    for key, canonical in aliases.items():
        if key in text:
            return canonical
    return None


def benchmark_to_task(benchmark: str) -> str:
    return SENTINEL_TO_TASK.get(normalize_benchmark(benchmark) or benchmark, benchmark)


def normalize_problem_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def stable_problem_hash(benchmark: str, problem: Any) -> str:
    digest = hashlib.sha1(normalize_problem_text(problem).encode("utf-8")).hexdigest()[:16]
    return f"{benchmark}-hash-{digest}"


def coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "correct"}:
        return True
    if text in {"false", "no", "n", "0", "incorrect"}:
        return False
    return None


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def list_first(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value[0] if value else None
    return value


def stringify(value: Any) -> str | None:
    value = list_first(value)
    if value is None:
        return None
    return str(value)


def scalar_metric(metrics: Any) -> float | None:
    metrics = list_first(metrics)
    if isinstance(metrics, dict):
        for key in ["extractive_match", "accuracy", "acc", "score"]:
            if key in metrics:
                try:
                    return float(list_first(metrics[key]) or 0.0)
                except Exception:
                    return None
    if isinstance(metrics, (int, float)):
        return float(metrics)
    return None


def extract_specific(specifics: Any, key: str) -> Any:
    if hasattr(specifics, "as_py"):
        specifics = specifics.as_py()
    if not isinstance(specifics, dict) or key not in specifics:
        return None
    return list_first(specifics[key])


def count_text_tokens(text: Any) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def infer_format_valid(benchmark: str, completion: Any, extracted_answer: Any = None, row_format: Any = None) -> bool | None:
    coerced = coerce_bool(row_format)
    if coerced is not None:
        return coerced
    if extracted_answer not in (None, ""):
        return True
    text = str(completion or "")
    if benchmark == "gpqa":
        return bool(re.search(r"\bAnswer\s*:\s*\$?\s*[A-D]\b", text, flags=re.IGNORECASE))
    return bool(re.search(r"\\boxed\s*\{", text) or re.search(r"final answer is", text, flags=re.IGNORECASE))


def record_key(row: dict[str, Any]) -> str:
    benchmark = row["benchmark"]
    stable_id = first_present(row, ["id", "problem_id", "example_id", "index"])
    if stable_id is not None:
        return f"{benchmark}:id:{stable_id}"
    problem = first_present(row, ["problem", "prompt", "question"])
    return f"{benchmark}:hash:{stable_problem_hash(benchmark, problem)}"


def standardize_result_row(row: dict[str, Any], source_path: Path, default_benchmark: str | None = None) -> dict[str, Any] | None:
    benchmark = normalize_benchmark(first_present(row, ["benchmark", "task", "task_name"]) or default_benchmark or source_path.as_posix())
    if benchmark not in CANONICAL_BENCHMARKS:
        return None
    stable_id = first_present(row, ["id", "problem_id", "example_id", "index"])
    problem = first_present(row, ["problem", "question", "prompt", "example"])
    prompt = first_present(row, ["prompt", "full_prompt", "raw_prompt", "question", "problem", "example"])
    gold = first_present(row, ["gold_answer", "answer", "target", "gold", "solution", "gold_extracted"])
    completion = first_present(row, ["prediction", "completion", "response", "predictions"])
    extracted = first_present(row, ["extracted_answer", "pred", "prediction_extracted", "pred_letter", "extracted_prediction"])
    correct = coerce_bool(first_present(row, ["correct", "is_correct", "score", "extractive_match", "accuracy"]))
    if correct is None:
        metric = scalar_metric(row.get("metrics"))
        if metric is not None:
            correct = metric > 0
    length = first_present(row, ["completion_length", "output_tokens", "response_tokens", "num_output_tokens"])
    try:
        length = int(length) if length is not None else count_text_tokens(completion)
    except Exception:
        length = count_text_tokens(completion)
    return {
        "id": str(stable_id) if stable_id is not None else None,
        "benchmark": benchmark,
        "problem": stringify(problem),
        "prompt": stringify(prompt),
        "gold_answer": stringify(gold),
        "prediction": stringify(completion),
        "extracted_answer": stringify(extracted),
        "correct": bool(correct) if correct is not None else False,
        "completion_length": length,
        "format_valid": infer_format_valid(benchmark, completion, extracted, first_present(row, ["format_valid", "parse_success"])),
        "reward": first_present(row, ["reward"]),
        "source_path": str(source_path),
        "seed": first_present(row, ["seed"]),
        "index": first_present(row, ["index"]),
    }


def load_lighteval_parquet(path: Path, default_benchmark: str | None = None) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Reading LightEval parquet details requires pandas plus a parquet engine such as pyarrow. "
            "Install them or export details as JSONL."
        ) from exc
    df = pd.read_parquet(path)
    rows = []
    for index, row_obj in df.iterrows():
        row = row_obj.to_dict()
        specifics = row.get("specifics", {})
        metrics = row.get("metrics", {})
        parsed = {
            "task": default_benchmark or row.get("task_name"),
            "index": int(index) if str(index).isdigit() else index,
            "example_id": row.get("example_id"),
            "problem": stringify(row.get("example")),
            "prompt": stringify(row.get("full_prompt")) or stringify(row.get("example")),
            "gold_answer": stringify(row.get("gold")) or stringify(extract_specific(specifics, "extracted_golds")),
            "completion": stringify(row.get("predictions")),
            "extracted_answer": stringify(extract_specific(specifics, "extracted_predictions")),
            "correct": (scalar_metric(metrics) or 0.0) > 0,
            "metrics": metrics,
        }
        standardized = standardize_result_row(parsed, path, default_benchmark)
        if standardized:
            rows.append(standardized)
    return rows


def load_result_file(path: Path) -> list[dict[str, Any]]:
    default_benchmark = normalize_benchmark(path.as_posix())
    if path.suffix == ".parquet":
        return load_lighteval_parquet(path, default_benchmark)
    if path.suffix == ".jsonl":
        return [r for r in (standardize_result_row(row, path, default_benchmark) for row in read_jsonl(path)) if r]
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [r for r in (standardize_result_row(row, path, default_benchmark) for row in obj) if r]
        if isinstance(obj, dict):
            for key in ["records", "items", "examples", "details", "rows"]:
                if isinstance(obj.get(key), list):
                    return [r for r in (standardize_result_row(row, path, default_benchmark) for row in obj[key]) if r]
        return []
    return []


def model_path_score(path: Path, selector: str, label: str) -> int:
    norm_path = normalize_selector(path.as_posix())
    norm_selector = normalize_selector(selector)
    score = 0
    if norm_selector and norm_selector in norm_path:
        score += 20
    if label == "base":
        if re.search(r"(^|[/_-])base($|[/_-])", path.as_posix(), flags=re.IGNORECASE):
            score += 100
        if any(token in norm_path for token in ["openrs", "grpo", "checkpoint", "merged"]):
            score -= 30
    else:
        if norm_selector and norm_selector in norm_path:
            score += 100
        elif norm_selector in {"openrs3", "tinaopenrs3"} and any(
            alias in norm_path for alias in ["openrs3", "tinaopenrs3", "grpoopenrs3"]
        ):
            score += 100
    return score


def discover_result_files(root: Path, selector: str, label: str) -> list[Path]:
    suffixes = {".jsonl", ".json", ".parquet"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes]
    scored = [(model_path_score(path, selector, label), path) for path in files if normalize_benchmark(path.as_posix())]
    selected = [path for score, path in scored if score > 0]
    detail_parquets = [path for path in selected if path.suffix == ".parquet" and "details" in path.parts]
    if detail_parquets:
        selected = detail_parquets
    selected.sort(key=lambda p: (normalize_benchmark(p.as_posix()) or "", p.as_posix()))
    return selected


def aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[record_key(row)].append(row)
    aggregated = {}
    for key, items in grouped.items():
        first = items[0]
        correctness = [bool(item.get("correct")) for item in items]
        formats = [item.get("format_valid") for item in items if item.get("format_valid") is not None]
        lengths = [int(item.get("completion_length") or 0) for item in items]
        extracted_values = [item.get("extracted_answer") for item in items if item.get("extracted_answer") not in (None, "")]
        aggregated[key] = {
            "id": first.get("id") or first.get("index") or stable_problem_hash(first["benchmark"], first.get("problem")),
            "benchmark": first["benchmark"],
            "problem": first.get("problem"),
            "prompt": first.get("prompt"),
            "gold_answer": first.get("gold_answer"),
            "correct": sum(correctness) >= (len(correctness) / 2),
            "correct_rate": sum(correctness) / max(1, len(correctness)),
            "extracted_answer": Counter(extracted_values).most_common(1)[0][0] if extracted_values else None,
            "completion_length": round(sum(lengths) / max(1, len(lengths))),
            "format_valid": (sum(bool(x) for x in formats) >= (len(formats) / 2)) if formats else None,
            "format_valid_rate": sum(bool(x) for x in formats) / max(1, len(formats)) if formats else None,
            "source_runs": len(items),
            "source_paths": sorted({item.get("source_path") for item in items if item.get("source_path")}),
            "source_seeds": sorted({str(item.get("seed")) for item in items if item.get("seed") is not None}),
        }
    return aggregated


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def transition_tag(base_correct: bool, baseline_correct: bool) -> str:
    if not base_correct and not baseline_correct:
        return "both_wrong"
    if not base_correct and baseline_correct:
        return "base_wrong_baseline_right"
    if base_correct and not baseline_correct:
        return "base_right_baseline_wrong"
    return "both_right"


def length_tags(base_length: int, baseline_length: int, p95: float | None = None, max_new_tokens: int | None = None) -> tuple[list[str], bool, bool]:
    tags = []
    expansion = baseline_length > base_length * 1.5 or (baseline_length - base_length) > 512
    reduction = baseline_length < base_length * 0.67
    extreme_expansion = expansion and ((p95 is not None and baseline_length >= p95) or (max_new_tokens and baseline_length >= max_new_tokens * 0.95))
    extreme_reduction = reduction and baseline_length <= max(64, base_length * 0.33)
    if expansion:
        tags.append("length_expansion")
    if reduction:
        tags.append("length_reduction")
    if extreme_expansion:
        tags.append("length_expansion_extreme")
    if extreme_reduction:
        tags.append("length_reduction_extreme")
    return tags, extreme_expansion, extreme_reduction


def selection_score(record: dict[str, Any]) -> float:
    score = {
        "base_wrong_baseline_right": 3.0,
        "both_wrong": 1.5,
        "base_right_baseline_wrong": 1.2,
        "both_right": 0.3,
    }.get(record["improvement_tag"], 0.0)
    tags = set(record.get("auxiliary_tags") or [])
    if "format_improvement" in tags:
        score += 0.5
    if "format_regression" in tags:
        score -= 0.8
    if "length_expansion_extreme" in tags:
        score -= 0.5
    if "length_reduction_extreme" in tags:
        score -= 0.3
    return round(score, 6)


def make_candidate_pool(
    base_rows: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    max_new_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings_list = []
    baseline_lengths_by_bench: dict[str, list[int]] = defaultdict(list)
    for row in baseline_rows.values():
        baseline_lengths_by_bench[row["benchmark"]].append(int(row.get("completion_length") or 0))
    p95_by_bench = {bench: percentile(values, 0.95) for bench, values in baseline_lengths_by_bench.items()}

    records = []
    for key in sorted(set(base_rows) | set(baseline_rows)):
        base = base_rows.get(key)
        baseline = baseline_rows.get(key)
        if not base or not baseline:
            warnings_list.append(f"unmatched_result key={key} base_found={base is not None} baseline_found={baseline is not None}")
            continue
        benchmark = base["benchmark"]
        base_correct = bool(base.get("correct"))
        baseline_correct = bool(baseline.get("correct"))
        base_length = int(base.get("completion_length") or 0)
        baseline_length = int(baseline.get("completion_length") or 0)
        auxiliary_tags, extreme_expansion, extreme_reduction = length_tags(
            base_length, baseline_length, p95_by_bench.get(benchmark), max_new_tokens
        )
        base_format = base.get("format_valid")
        baseline_format = baseline.get("format_valid")
        if base_format is True and baseline_format is False:
            auxiliary_tags.append("format_regression")
        if base_format is False and baseline_format is True:
            auxiliary_tags.append("format_improvement")
        tag = transition_tag(base_correct, baseline_correct)
        if tag == "base_wrong_baseline_right":
            if extreme_expansion or "format_regression" in auxiliary_tags:
                auxiliary_tags.append("risky_transition")
                difficulty = "risky_transition"
            else:
                auxiliary_tags.append("useful_transition")
                difficulty = "useful_transition"
        elif tag == "both_wrong":
            difficulty = "hard_both_wrong"
        elif tag == "base_right_baseline_wrong":
            difficulty = "regression_monitor"
        else:
            difficulty = "easy_sanity"
        record = {
            "id": str(base.get("id") or baseline.get("id") or key),
            "benchmark": benchmark,
            "problem": base.get("problem") or baseline.get("problem"),
            "prompt": base.get("prompt") or baseline.get("prompt"),
            "gold_answer": base.get("gold_answer") or baseline.get("gold_answer"),
            "base_correct": base_correct,
            "baseline_correct": baseline_correct,
            "base_correct_rate": base.get("correct_rate"),
            "baseline_correct_rate": baseline.get("correct_rate"),
            "base_extracted_answer": base.get("extracted_answer"),
            "baseline_extracted_answer": baseline.get("extracted_answer"),
            "base_length": base_length,
            "baseline_length": baseline_length,
            "base_format_valid": base_format,
            "baseline_format_valid": baseline_format,
            "base_format_valid_rate": base.get("format_valid_rate"),
            "baseline_format_valid_rate": baseline.get("format_valid_rate"),
            "difficulty_tag": difficulty,
            "improvement_tag": tag,
            "auxiliary_tags": sorted(set(auxiliary_tags)),
            "selected": False,
            "selection_reason": [],
            "source": {
                "base_runs": base.get("source_runs"),
                "baseline_runs": baseline.get("source_runs"),
                "base_paths": base.get("source_paths", [])[:5],
                "baseline_paths": baseline.get("source_paths", [])[:5],
                "base_seeds": base.get("source_seeds", []),
                "baseline_seeds": baseline.get("source_seeds", []),
            },
        }
        record["selection_score"] = selection_score(record)
        records.append(record)
    return records, warnings_list


def scaled_allocation(total_size: int) -> dict[str, int]:
    if total_size == 150:
        return dict(DEFAULT_SENTINEL_ALLOCATION)
    scale = total_size / 150.0
    allocation = {bench: int(round(count * scale)) for bench, count in DEFAULT_SENTINEL_ALLOCATION.items()}
    allocation["aime24"] = min(8, max(4, allocation.get("aime24", 0)))
    allocation["aime25"] = min(8, max(4, allocation.get("aime25", 0)))
    allocation["amc23"] = min(16, max(8, allocation.get("amc23", 0)))
    for bench in CANONICAL_BENCHMARKS:
        allocation.setdefault(bench, 0)
    diff = total_size - sum(allocation.values())
    adjustable = ["math500", "gpqa", "minerva"]
    while diff != 0:
        if diff > 0:
            target = max(adjustable, key=lambda b: DEFAULT_SENTINEL_ALLOCATION[b] - allocation[b] / max(scale, 1e-9))
            allocation[target] += 1
            diff -= 1
        else:
            candidates = [b for b in adjustable if allocation[b] > 1]
            if not candidates:
                candidates = [b for b in CANONICAL_BENCHMARKS if allocation[b] > 1]
            target = max(candidates, key=lambda b: allocation[b])
            allocation[target] -= 1
            diff += 1
    return allocation


def tag_quota(total: int) -> dict[str, int]:
    quotas = {tag: int(round(total * ratio)) for tag, ratio in TAG_TARGET_RATIOS.items()}
    diff = total - sum(quotas.values())
    order = ["base_wrong_baseline_right", "both_wrong", "base_right_baseline_wrong", "both_right"]
    i = 0
    while diff != 0:
        tag = order[i % len(order)]
        if diff > 0:
            quotas[tag] += 1
            diff -= 1
        elif quotas[tag] > 0:
            quotas[tag] -= 1
            diff += 1
        i += 1
    return quotas


def deterministic_select(records: list[dict[str, Any]], allocation: dict[str, int], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected_ids = set()
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_bench[record["benchmark"]].append(record)
    selected = []
    for benchmark in CANONICAL_BENCHMARKS:
        target = min(allocation.get(benchmark, 0), len(by_bench.get(benchmark, [])))
        if target <= 0:
            continue
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in by_bench[benchmark]:
            record["_sample_noise"] = rng.random() * 0.01
            buckets[record["improvement_tag"]].append(record)
        for items in buckets.values():
            items.sort(key=lambda r: (r["selection_score"] + r["_sample_noise"], -int(r.get("base_length") or 0)), reverse=True)
        quotas = tag_quota(target)
        bench_selected: list[dict[str, Any]] = []
        for tag, quota in quotas.items():
            for record in buckets[tag][:quota]:
                selected_key = f"{record['benchmark']}:{record['id']}"
                if selected_key not in selected_ids:
                    record["selection_reason"].append(f"stratified_quota:{tag}")
                    bench_selected.append(record)
                    selected_ids.add(selected_key)
        if len(bench_selected) < target:
            for tag in FILL_PRIORITY:
                for record in buckets[tag]:
                    if len(bench_selected) >= target:
                        break
                    selected_key = f"{record['benchmark']}:{record['id']}"
                    if selected_key in selected_ids:
                        continue
                    record["selection_reason"].append(f"fill_shortfall:{tag}")
                    bench_selected.append(record)
                    selected_ids.add(selected_key)
                if len(bench_selected) >= target:
                    break
        for record in bench_selected:
            record["selected"] = True
            record["selection_reason"].extend(
                [
                    f"benchmark_target:{target}",
                    f"selection_score:{record['selection_score']}",
                    "sentinel_is_checkpoint_selection_not_final_score",
                ]
            )
        selected.extend(sorted(bench_selected, key=lambda r: (r["benchmark"], str(r["id"]))))
    for record in records:
        record.pop("_sample_noise", None)
    return selected


def summarize_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for benchmark in CANONICAL_BENCHMARKS:
        items = [r for r in records if r.get("benchmark") == benchmark]
        tags = Counter(r.get("improvement_tag") for r in items)
        out[benchmark] = {"total": len(items), **{tag: tags.get(tag, 0) for tag in TAG_TARGET_RATIOS}}
    return out


def full_scores(records: list[dict[str, Any]], prefix: str) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for benchmark in CANONICAL_BENCHMARKS:
        items = [r for r in records if r.get("benchmark") == benchmark]
        if not items:
            scores[benchmark] = None
            continue
        scores[benchmark] = sum(bool(r.get(f"{prefix}_correct")) for r in items) / len(items)
    return scores


def make_weights(selected: list[dict[str, Any]]) -> dict[str, Any]:
    item_weights = {}
    for row in selected:
        weight = ITEM_WEIGHT_BY_TAG.get(row.get("improvement_tag"), 1.0)
        item_weights[f"{row['benchmark']}:{row['id']}"] = {
            "weight": weight,
            "benchmark": row["benchmark"],
            "improvement_tag": row.get("improvement_tag"),
        }
        row["item_weight"] = weight
    return {
        "benchmark_weights": {bench: 1.0 for bench in CANONICAL_BENCHMARKS},
        "aggregation": "mean_of_benchmark_scores",
        "item_weight_policy": ITEM_WEIGHT_BY_TAG,
        "item_weights": item_weights,
    }


def weighted_mean(values: list[tuple[float, float]]) -> float | None:
    denom = sum(weight for _, weight in values)
    if denom <= 0:
        return None
    return sum(value * weight for value, weight in values) / denom


def score_sentinel_results(rows: list[dict[str, Any]], weights: dict[str, Any]) -> dict[str, Any]:
    item_weights = weights.get("item_weights", {})
    bench_weights = weights.get("benchmark_weights", {bench: 1.0 for bench in CANONICAL_BENCHMARKS})
    by_bench: dict[str, list[tuple[float, float]]] = defaultdict(list)
    tag_totals: dict[str, list[bool]] = defaultdict(list)
    lengths = []
    format_valid = []
    for row in rows:
        benchmark = normalize_benchmark(row.get("benchmark") or row.get("task"))
        item_id = str(row.get("id") or row.get("example_id") or row.get("index"))
        correct = bool(row.get("correct"))
        key = f"{benchmark}:{item_id}"
        weight = item_weights.get(key, {}).get("weight", row.get("item_weight", 1.0))
        by_bench[benchmark].append((1.0 if correct else 0.0, float(weight)))
        tag = row.get("improvement_tag")
        if tag:
            tag_totals[tag].append(correct)
        if row.get("completion_length") is not None:
            lengths.append(float(row["completion_length"]))
        if row.get("format_valid") is not None:
            format_valid.append(bool(row["format_valid"]))
    benchmark_scores = {bench: weighted_mean(by_bench.get(bench, [])) for bench in CANONICAL_BENCHMARKS if by_bench.get(bench)}
    weighted_bench_values = [
        (score, float(bench_weights.get(bench, 1.0))) for bench, score in benchmark_scores.items() if score is not None
    ]
    sentinel_score = weighted_mean(weighted_bench_values)
    all_correct = [bool(row.get("correct")) for row in rows]
    tag_rate = lambda tag: (sum(tag_totals[tag]) / len(tag_totals[tag])) if tag_totals.get(tag) else None
    return {
        "sentinel_score": sentinel_score,
        "benchmark_scores": benchmark_scores,
        "overall_accuracy_unweighted": sum(all_correct) / len(all_correct) if all_correct else None,
        "format_valid_rate": sum(format_valid) / len(format_valid) if format_valid else None,
        "avg_completion_length": statistics.mean(lengths) if lengths else None,
        "length_by_benchmark": {
            bench: statistics.mean([float(r["completion_length"]) for r in rows if normalize_benchmark(r.get("benchmark")) == bench and r.get("completion_length") is not None])
            for bench in CANONICAL_BENCHMARKS
            if any(normalize_benchmark(r.get("benchmark")) == bench and r.get("completion_length") is not None for r in rows)
        },
        "base_wrong_baseline_right_recovery_rate": tag_rate("base_wrong_baseline_right"),
        "base_right_baseline_wrong_forgetting_rate": (1.0 - tag_rate("base_right_baseline_wrong")) if tag_rate("base_right_baseline_wrong") is not None else None,
        "both_wrong_breakthrough_rate": tag_rate("both_wrong"),
        "both_right_retention_rate": tag_rate("both_right"),
    }


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)

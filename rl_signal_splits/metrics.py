"""Per-sample RL signal aggregation."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

from .answer_extraction import normalize_answer


def safe_mean(values: List[float], default: float = 0.0) -> float:
    return float(mean(values)) if values else default


def safe_std(values: List[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def robust_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    p5 = percentile(values, 0.05)
    p95 = percentile(values, 0.95)
    if abs(p95 - p5) < 1e-12:
        return [0.0 for _ in values]
    return [float(min(1.0, max(0.0, (v - p5) / (p95 - p5)))) for v in values]


def aggregate_rollout_record(row: Dict[str, Any]) -> Dict[str, Any]:
    samples = row.get("samples") or []
    n_samples = int(row.get("n_samples") or len(samples) or 0)
    rewards = [float(s.get("reward", 0.0)) for s in samples if s.get("reward") is not None]
    correct = [bool(s.get("is_correct")) for s in samples]
    valid = [bool(s.get("format_valid")) for s in samples]
    lengths = [float(s.get("completion_length_tokens") or 0.0) for s in samples]
    char_lengths = [float(s.get("completion_length_chars") or 0.0) for s in samples]
    extracted = [normalize_answer(s.get("extracted_final_answer")) for s in samples if s.get("extracted_final_answer")]
    correct_count = sum(correct)
    invalid_count = len(valid) - sum(valid)
    pass_rate = correct_count / n_samples if n_samples else 0.0
    length_mean = safe_mean(lengths)
    length_std = safe_std(lengths)
    reward_mean = safe_mean(rewards)
    reward_std = safe_std(rewards)
    answer_diversity = len(set(extracted)) / n_samples if n_samples else 0.0
    entropy = binary_entropy(pass_rate)
    return {
        "id": row.get("id"),
        "problem": row.get("problem", ""),
        "gold_answer": row.get("gold_answer", ""),
        "source": row.get("source", {}),
        "n_samples": n_samples,
        "error": row.get("error"),
        "missing_problem": bool(row.get("missing_problem")),
        "missing_answer": bool(row.get("missing_answer")),
        "correct_count": int(correct_count),
        "pass_rate": float(pass_rate),
        "best_of_n_correct": bool(correct_count > 0),
        "all_wrong": bool(n_samples > 0 and correct_count == 0),
        "all_correct": bool(n_samples > 0 and correct_count == n_samples),
        "reward_mean": float(reward_mean),
        "reward_std": float(reward_std),
        "reward_min": float(min(rewards)) if rewards else 0.0,
        "reward_max": float(max(rewards)) if rewards else 0.0,
        "reward_range": float(max(rewards) - min(rewards)) if rewards else 0.0,
        "format_valid_rate": float(sum(valid) / n_samples) if n_samples else 0.0,
        "invalid_count": int(invalid_count),
        "length_mean_tokens": float(length_mean),
        "length_std_tokens": float(length_std),
        "length_min_tokens": float(min(lengths)) if lengths else 0.0,
        "length_max_tokens": float(max(lengths)) if lengths else 0.0,
        "length_p90_tokens": float(percentile(lengths, 0.90)),
        "length_mean_chars": float(safe_mean(char_lengths)),
        "answer_diversity": float(answer_diversity),
        "completion_length_cv": float(length_std / max(length_mean, 1.0)),
        "correctness_entropy": float(entropy),
        "difficulty_score": float(1.0 - pass_rate),
        "anomaly": bool(row.get("error") or row.get("missing_problem") or row.get("missing_answer") or n_samples == 0),
    }


def add_composite_scores(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reward_std_n = robust_normalize([float(m.get("reward_std", 0.0)) for m in metrics])
    diversity_n = robust_normalize([float(m.get("answer_diversity", 0.0)) for m in metrics])
    entropy_n = robust_normalize([float(m.get("correctness_entropy", 0.0)) for m in metrics])
    length_mean_n = robust_normalize([float(m.get("length_mean_tokens", 0.0)) for m in metrics])
    length_p90_n = robust_normalize([float(m.get("length_p90_tokens", 0.0)) for m in metrics])
    invalid_n = robust_normalize([float(m.get("invalid_count", 0.0)) for m in metrics])
    length_cv_n = robust_normalize([float(m.get("completion_length_cv", 0.0)) for m in metrics])
    for i, metric in enumerate(metrics):
        variance_score = (reward_std_n[i] + diversity_n[i] + entropy_n[i]) / 3.0
        long_reasoning_score = (length_mean_n[i] + length_p90_n[i]) / 2.0
        instability_score = (variance_score + invalid_n[i] + length_cv_n[i]) / 3.0
        metric["variance_score"] = float(variance_score)
        metric["long_reasoning_score"] = float(long_reasoning_score)
        metric["instability_score"] = float(instability_score)
        metric["score_formulas"] = {
            "difficulty_score": "1 - pass_rate",
            "variance_score": "mean(robust_percentile_norm(reward_std), robust_percentile_norm(answer_diversity), robust_percentile_norm(correctness_entropy))",
            "long_reasoning_score": "mean(robust_percentile_norm(length_mean_tokens), robust_percentile_norm(length_p90_tokens))",
            "instability_score": "mean(variance_score, robust_percentile_norm(invalid_count), robust_percentile_norm(completion_length_cv))",
            "robust_percentile_norm": "clip((x - dataset_p5) / max(dataset_p95 - dataset_p5, eps), 0, 1)",
        }
    return metrics


def aggregate_rollouts(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [aggregate_rollout_record(row) for row in rows]
    return add_composite_scores(metrics)

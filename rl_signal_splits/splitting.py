"""Split construction rules for RL-signal-aware LoRA training sets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .io import merge_metadata
from .metrics import percentile


SPLIT_NAMES = [
    "easy",
    "medium",
    "hard",
    "high_variance",
    "long_reasoning",
    "core_medium",
    "mixed_balanced",
]


@dataclass
class SplitConfig:
    max_split_size: int | None = None
    mixed_size: int | None = None
    default_mixed_size: int = 2000
    easy_ratio: float = 0.20
    medium_ratio: float = 0.40
    hard_ratio: float = 0.20
    high_variance_ratio: float = 0.10
    long_reasoning_ratio: float = 0.10
    make_disjoint: bool = False
    seed: int = 42
    min_warning_size: int = 10


def difficulty_bucket(pass_rate: float) -> str:
    if pass_rate >= 0.75:
        return "easy"
    if pass_rate >= 0.25:
        return "medium"
    return "hard"


def threshold(metrics: Sequence[Dict[str, Any]], key: str, top_fraction: float) -> float:
    values = [float(m.get(key, 0.0)) for m in metrics]
    return percentile(values, 1.0 - top_fraction)


def top_fraction_ids(
    metrics: Sequence[Dict[str, Any]],
    key: str,
    top_fraction: float,
    *,
    min_value: float | None = None,
) -> set[str]:
    if not metrics or top_fraction <= 0.0:
        return set()
    target = max(1, int(round(len(metrics) * top_fraction)))
    ranked = sorted(
        metrics,
        key=lambda m: (float(m.get(key, 0.0)), str(m.get("id"))),
        reverse=True,
    )
    out = set()
    for metric in ranked:
        if len(out) >= target:
            break
        value = float(metric.get(key, 0.0))
        if min_value is not None and value <= min_value:
            continue
        out.add(str(metric["id"]))
    return out


def core_medium_ids(
    medium_metrics: Sequence[Dict[str, Any]],
    excluded_ids: set[str],
    fraction: float,
) -> set[str]:
    if not medium_metrics or fraction <= 0.0:
        return set()
    target = max(1, int(round(len(medium_metrics) * fraction)))
    ranked = sorted(medium_metrics, key=lambda m: (float(m.get("instability_score", 0.0)), str(m.get("id"))))
    selected = [str(m["id"]) for m in ranked if str(m["id"]) not in excluded_ids][:target]
    if len(selected) < target:
        seen = set(selected)
        for metric in ranked:
            sample_id = str(metric["id"])
            if sample_id in seen:
                continue
            selected.append(sample_id)
            seen.add(sample_id)
            if len(selected) >= target:
                break
    return set(selected)


def logical_split_ids(metrics: Sequence[Dict[str, Any]]) -> tuple[Dict[str, set[str]], Dict[str, float]]:
    reward_std_top30 = threshold(metrics, "reward_std", 0.30)
    entropy_top30 = threshold(metrics, "correctness_entropy", 0.30)
    diversity_top30 = threshold(metrics, "answer_diversity", 0.30)
    length_p90_top25 = threshold(metrics, "length_p90_tokens", 0.25)
    length_mean_top25 = threshold(metrics, "length_mean_tokens", 0.25)
    reward_std_top20 = threshold(metrics, "reward_std", 0.20)
    length_mean_top10 = threshold(metrics, "length_mean_tokens", 0.10)
    variance_score_top30 = threshold(metrics, "variance_score", 0.30)
    long_reasoning_score_top25 = threshold(metrics, "long_reasoning_score", 0.25)
    splits = {name: set() for name in SPLIT_NAMES if name != "mixed_balanced"}
    high_variance_ids = top_fraction_ids(metrics, "variance_score", 0.30, min_value=0.0)
    long_reasoning_ids = top_fraction_ids(metrics, "long_reasoning_score", 0.25, min_value=0.0)
    for m in metrics:
        sample_id = str(m["id"])
        pass_rate = float(m.get("pass_rate", 0.0))
        format_rate = float(m.get("format_valid_rate", 0.0))
        if pass_rate >= 0.75 and format_rate >= 0.75:
            splits["easy"].add(sample_id)
        if 0.25 <= pass_rate < 0.75 and format_rate >= 0.5:
            splits["medium"].add(sample_id)
        if pass_rate < 0.25 and format_rate >= 0.5:
            splits["hard"].add(sample_id)
        if sample_id in high_variance_ids and format_rate >= 0.5:
            splits["high_variance"].add(sample_id)
        if sample_id in long_reasoning_ids and format_rate >= 0.5:
            splits["long_reasoning"].add(sample_id)
    medium_metrics = [m for m in metrics if str(m["id"]) in splits["medium"]]
    splits["core_medium"] = core_medium_ids(
        medium_metrics,
        splits["high_variance"] | splits["long_reasoning"],
        0.50,
    )
    thresholds = {
        "reward_std_top30": reward_std_top30,
        "correctness_entropy_top30": entropy_top30,
        "answer_diversity_top30": diversity_top30,
        "length_p90_top25": length_p90_top25,
        "length_mean_top25": length_mean_top25,
        "reward_std_top20": reward_std_top20,
        "length_mean_top10": length_mean_top10,
        "variance_score_top30": variance_score_top30,
        "long_reasoning_score_top25": long_reasoning_score_top25,
    }
    return splits, thresholds


def apply_disjoint(splits: Dict[str, set[str]]) -> Dict[str, set[str]]:
    priority = ["core_medium", "high_variance", "long_reasoning", "hard", "easy"]
    assigned: set[str] = set()
    out = {name: set(ids) for name, ids in splits.items()}
    for name in priority:
        out[name] = out.get(name, set()) - assigned
        assigned |= out[name]
    if "medium" in out:
        out["medium"] = out["medium"] - assigned
    return out


def trim_split(ids: Iterable[str], metrics_by_id: Mapping[str, Dict[str, Any]], max_size: int | None, seed: int) -> List[str]:
    ordered = sorted(set(ids), key=lambda x: (float(metrics_by_id[x].get("instability_score", 0.0)), x))
    if max_size is None or len(ordered) <= max_size:
        return ordered
    rng = random.Random(seed)
    sampled = rng.sample(ordered, max_size)
    return sorted(sampled)


def build_mixed_ids(
    splits: Mapping[str, Sequence[str]],
    target_size: int,
    config: SplitConfig,
) -> tuple[List[str], bool]:
    pools = {
        "easy": list(splits.get("easy", [])),
        "core_medium": list(splits.get("core_medium", [])) or list(splits.get("medium", [])),
        "hard": list(splits.get("hard", [])),
        "high_variance": list(splits.get("high_variance", [])),
        "long_reasoning": list(splits.get("long_reasoning", [])),
    }
    ratios = {
        "easy": config.easy_ratio,
        "core_medium": config.medium_ratio,
        "hard": config.hard_ratio,
        "high_variance": config.high_variance_ratio,
        "long_reasoning": config.long_reasoning_ratio,
    }
    rng = random.Random(config.seed)
    selected: List[str] = []
    used: set[str] = set()
    replacement_used = False
    for name, ratio in ratios.items():
        want = int(round(target_size * ratio))
        pool = list(dict.fromkeys(pools[name]))
        rng.shuffle(pool)
        take = [x for x in pool if x not in used][:want]
        selected.extend(take)
        used.update(take)
    all_unique = list(dict.fromkeys(x for pool in pools.values() for x in pool))
    rng.shuffle(all_unique)
    for sample_id in all_unique:
        if len(selected) >= target_size:
            break
        if sample_id not in used:
            selected.append(sample_id)
            used.add(sample_id)
    if len(selected) < target_size and all_unique:
        replacement_used = True
        while len(selected) < target_size:
            selected.append(rng.choice(all_unique))
    return selected[:target_size], replacement_used


def rl_signal_metadata(metric: Dict[str, Any], split_tags: Sequence[str]) -> Dict[str, Any]:
    pass_rate = float(metric.get("pass_rate", 0.0))
    return {
        "id": str(metric.get("id")),
        "pass_rate": pass_rate,
        "reward_mean": float(metric.get("reward_mean", 0.0)),
        "reward_std": float(metric.get("reward_std", 0.0)),
        "format_valid_rate": float(metric.get("format_valid_rate", 0.0)),
        "length_mean_tokens": float(metric.get("length_mean_tokens", 0.0)),
        "length_p90_tokens": float(metric.get("length_p90_tokens", 0.0)),
        "answer_diversity": float(metric.get("answer_diversity", 0.0)),
        "correctness_entropy": float(metric.get("correctness_entropy", 0.0)),
        "variance_score": float(metric.get("variance_score", 0.0)),
        "long_reasoning_score": float(metric.get("long_reasoning_score", 0.0)),
        "instability_score": float(metric.get("instability_score", 0.0)),
        "difficulty_bucket": difficulty_bucket(pass_rate),
        "split_tags": list(split_tags),
    }


def materialize_split_records(
    split_ids: Sequence[str],
    metrics_by_id: Mapping[str, Dict[str, Any]],
    all_tags_by_id: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    rows = []
    for sample_id in split_ids:
        metric = metrics_by_id[sample_id]
        rows.append(merge_metadata(metric.get("source") or {}, rl_signal_metadata(metric, all_tags_by_id.get(sample_id, []))))
    return rows


def build_splits(metrics: List[Dict[str, Any]], config: SplitConfig) -> Dict[str, Any]:
    metrics_by_id = {str(m["id"]): m for m in metrics}
    logical, thresholds = logical_split_ids(metrics)
    warnings: List[str] = []
    if config.make_disjoint:
        split_sets = apply_disjoint(logical)
    else:
        split_sets = {name: set(ids) for name, ids in logical.items()}
    split_ids: Dict[str, List[str]] = {
        name: trim_split(ids, metrics_by_id, config.max_split_size, config.seed) for name, ids in split_sets.items()
    }
    mixed_target = config.mixed_size if config.mixed_size is not None else min(len(metrics), config.default_mixed_size)
    mixed_ids, replacement_used = build_mixed_ids(split_ids, mixed_target, config)
    split_ids["mixed_balanced"] = mixed_ids
    all_tags: MutableMapping[str, List[str]] = {str(m["id"]): [] for m in metrics}
    for name, ids in split_ids.items():
        for sample_id in ids:
            all_tags.setdefault(sample_id, [])
            if name not in all_tags[sample_id]:
                all_tags[sample_id].append(name)
    for name, ids in split_ids.items():
        unique_size = len(set(ids))
        if unique_size < config.min_warning_size:
            warnings.append(f"{name} split is small: {unique_size} unique samples.")
    if replacement_used:
        warnings.append("mixed_balanced used sampling with replacement because source pools were smaller than mixed_size.")
    records = {
        name: materialize_split_records(ids, metrics_by_id, all_tags) for name, ids in split_ids.items()
    }
    return {
        "split_ids": split_ids,
        "split_records": records,
        "thresholds": thresholds,
        "warnings": warnings,
        "make_disjoint": config.make_disjoint,
    }

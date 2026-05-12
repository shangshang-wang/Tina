"""Heuristic error taxonomy for paired rollout regressions."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tina.analysis.plan_utils import analyze_plan_features, compute_plan_specificity, extract_plan_block
from tina.analysis.rollout_utils import extract_final_answer_text, has_boxed, parse_success


def repetition_score(text: str, ngram: int = 4) -> float:
    tokens = re.findall(r"\S+", text or "")
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def detect_answer_extraction_error(case: dict[str, Any]) -> bool:
    exp_completion = case.get("exp_completion") or ""
    if parse_success({"pred": case.get("exp_pred")}):
        return False
    if has_boxed(exp_completion):
        return True
    return extract_final_answer_text(exp_completion) is not None


def detect_plan_execute_mismatch(problem: str, completion: str) -> dict[str, Any]:
    split = extract_plan_block(completion or "")
    plan_text = split.get("plan_text") or ""
    execute_text = split.get("execute_text") or ""
    plan_keywords = set(re.findall(r"[A-Za-z]{4,}|\d+", plan_text.lower()))
    execute_keywords = set(re.findall(r"[A-Za-z]{4,}|\d+", execute_text.lower()))
    if not plan_keywords:
        return {"plan_execute_mismatch": False, "plan_execute_overlap": 0.0}
    overlap = len(plan_keywords & execute_keywords) / max(1, len(plan_keywords))
    return {"plan_execute_mismatch": overlap < 0.2, "plan_execute_overlap": overlap}


def classify_regression_case(case: dict[str, Any]) -> dict[str, Any]:
    task = str(case.get("task") or "")
    exp_completion = case.get("exp_completion") or ""
    base_len = int(case.get("base_num_output_tokens") or 0)
    exp_len = int(case.get("exp_num_output_tokens") or 0)
    tags: list[str] = []

    exp_parse_success = parse_success({"pred": case.get("exp_pred") or case.get("exp_pred_letter")})
    exp_has_boxed = has_boxed(exp_completion)
    exp_pred_letter = case.get("exp_pred_letter")

    if (task.startswith("gpqa") and not exp_pred_letter) or (not task.startswith("gpqa") and (not exp_has_boxed or not exp_parse_success)):
        tags.append("format_error")
    if detect_answer_extraction_error(case):
        tags.append("answer_extraction_error")
    if base_len > 0 and exp_len < 0.7 * base_len:
        tags.append("shorter_overcompressed")
    if base_len > 0 and exp_len > 1.3 * base_len and repetition_score(exp_completion) >= 0.08:
        tags.append("overlong_repetition")

    plan_features = analyze_plan_features(case.get("problem") or "", exp_completion, exp_len)
    if plan_features["has_plan_open"] and plan_features["template_score"] >= 0.65 and plan_features["plan_specificity"] <= 0.2:
        tags.append("plan_template_only")

    mismatch = detect_plan_execute_mismatch(case.get("problem") or "", exp_completion)
    if plan_features["has_plan_open"] and mismatch["plan_execute_mismatch"]:
        tags.append("plan_execute_mismatch")

    if not tags and task.startswith("gpqa") and exp_parse_success:
        tags.append("knowledge_or_domain_error")
    if not tags and exp_parse_success:
        tags.append("arithmetic_or_execution_error")
    if not tags:
        tags.append("unknown")

    return {
        "error_type": tags[0],
        "error_tags": tags,
        "repetition_score": repetition_score(exp_completion),
        "plan_template_score": plan_features["template_score"],
        "plan_specificity": plan_features["plan_specificity"],
        **mismatch,
    }


__all__ = [
    "classify_regression_case",
    "detect_answer_extraction_error",
    "detect_plan_execute_mismatch",
    "repetition_score",
]


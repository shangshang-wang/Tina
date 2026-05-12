"""Utilities for inspecting <plan>...</plan> blocks in rollouts."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


PLAN_OPEN = "<plan>"
PLAN_CLOSE = "</plan>"

TEMPLATE_PHRASES = [
    "we need to",
    "the key is",
    "first",
    "then",
    "use the given",
    "find the relationship",
    "solve step by step",
    "analyze the problem",
    "set up equations",
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "into",
    "onto",
    "can",
    "will",
    "would",
    "should",
    "could",
    "there",
    "their",
    "then",
    "than",
    "when",
    "what",
    "which",
    "where",
    "why",
    "how",
    "use",
    "using",
    "solve",
    "find",
    "answer",
    "problem",
    "given",
    "need",
    "step",
    "first",
    "next",
    "finally",
}

MATH_PATTERN = re.compile(
    r"(\\frac|\\sqrt|\\sum|\\binom|\d|[=<>+\-*/^%$()]|\[[^\]]*\]|\{[^}]*\})"
)
WORD_PATTERN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[a-zA-Z]\d+|\d+[a-zA-Z]")
FINAL_ANSWER_PATTERN = re.compile(
    r"(therefore,\s*the\s*final\s*answer\s*is|final\s*answer|answer\s*:|答案)",
    re.IGNORECASE,
)
BOXED_PATTERN = re.compile(r"\\boxed\s*\{")


def _word_tokens(text: str) -> list[str]:
    tokens = []
    for match in WORD_PATTERN.finditer((text or "").lower()):
        token = match.group(0)
        if token in STOPWORDS:
            continue
        if token.isdigit() or len(token) >= 3 or re.search(r"\d", token):
            tokens.append(token)
    return tokens


def approx_token_len(text: str) -> int:
    """Cheap tokenizer-free length approximation for analysis summaries."""
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def extract_plan_block(text: str, min_tokens: int = 8, max_tokens: int = 128) -> dict[str, Any]:
    text = text or ""
    open_spans = list(re.finditer(re.escape(PLAN_OPEN), text))
    close_spans = list(re.finditer(re.escape(PLAN_CLOSE), text))
    has_plan_open = bool(open_spans)
    has_plan_close = bool(close_spans)
    plan_text = ""
    execute_text = text
    plan_char_start = None
    plan_char_end = None
    close_after_open = False

    if has_plan_open and has_plan_close:
        first_open = open_spans[0]
        first_close = close_spans[0]
        close_after_open = first_close.start() >= first_open.end()
        if close_after_open:
            plan_char_start = first_open.end()
            plan_char_end = first_close.start()
            plan_text = text[plan_char_start:plan_char_end].strip()
            execute_text = text[first_close.end() :].strip()

    plan_token_len = approx_token_len(plan_text)
    plan_format_valid = (
        len(open_spans) == 1
        and len(close_spans) == 1
        and close_after_open
        and min_tokens <= plan_token_len <= max_tokens
    )
    position_ratio = None
    if plan_char_start is not None and len(text) > 0:
        position_ratio = plan_char_start / len(text)

    return {
        "has_plan_open": has_plan_open,
        "has_plan_close": has_plan_close,
        "num_plan_open": len(open_spans),
        "num_plan_close": len(close_spans),
        "plan_valid": plan_format_valid,
        "plan_format_valid": plan_format_valid,
        "plan_text": plan_text,
        "execute_text": execute_text,
        "plan_char_len": len(plan_text),
        "plan_token_len": plan_token_len,
        "plan_position_ratio": position_ratio,
        "plan_char_start": plan_char_start,
        "plan_char_end": plan_char_end,
        "plan_close_end": close_spans[0].end() if close_after_open else None,
    }


def split_plan_execute_answer(completion: str) -> dict[str, Any]:
    plan = extract_plan_block(completion)
    execute_text = plan["execute_text"]
    answer_match = None
    for pattern in [r"Therefore,\s*the\s*final\s*answer\s*is.*", r"Answer\s*:.*", r"\\boxed\s*\{.*"]:
        matches = list(re.finditer(pattern, execute_text or "", re.IGNORECASE | re.DOTALL))
        if matches:
            answer_match = matches[-1]
            break
    answer_text = answer_match.group(0).strip() if answer_match else ""
    return {**plan, "answer_text": answer_text}


def compute_plan_specificity(problem: str, plan_text: str) -> dict[str, Any]:
    problem_tokens = set(_word_tokens(problem or ""))
    plan_tokens = _word_tokens(plan_text or "")
    plan_counts = Counter(plan_tokens)
    overlap = sum(1 for token in plan_counts if token in problem_tokens)
    specificity = overlap / max(1, len(plan_counts))
    return {
        "plan_specificity": specificity,
        "problem_to_plan_overlap_count": overlap,
        "plan_keyword_count": len(plan_counts),
    }


def compute_math_density(text: str) -> dict[str, Any]:
    text = text or ""
    math_symbol_count = len(MATH_PATTERN.findall(text))
    token_len = max(1, approx_token_len(text))
    equals_count = text.count("=")
    long_formula_count = len(re.findall(r"[\dA-Za-z\\{}^_+\-*/=()]{18,}", text))
    return {
        "math_symbol_count": math_symbol_count,
        "math_density": math_symbol_count / token_len,
        "equals_count": equals_count,
        "long_formula_count": long_formula_count,
    }


def compute_template_score(plan_text: str, problem: str | None = None) -> dict[str, Any]:
    lowered = (plan_text or "").lower()
    phrase_count = sum(lowered.count(phrase) for phrase in TEMPLATE_PHRASES)
    specificity = compute_plan_specificity(problem or "", plan_text or "")["plan_specificity"] if problem else 0.0
    length_penalty = 1.0 if approx_token_len(plan_text or "") <= 32 else 0.6
    template_score = min(1.0, (phrase_count / 3.0) * length_penalty + max(0.0, 0.35 - specificity))
    return {
        "template_phrase_count": phrase_count,
        "template_score": template_score,
    }


def detect_computation_leakage(plan_text: str) -> dict[str, Any]:
    density = compute_math_density(plan_text)
    leakage_score = density["math_density"] + 0.25 * density["equals_count"] + 0.5 * density["long_formula_count"]
    return {
        "plan_computation_leakage": bool(
            density["math_density"] >= 0.45 or density["equals_count"] >= 2 or density["long_formula_count"] >= 2
        ),
        "leakage_score": leakage_score,
    }


def has_final_answer_phrase(text: str) -> bool:
    return bool(FINAL_ANSWER_PATTERN.search(text or ""))


def has_boxed(text: str) -> bool:
    return bool(BOXED_PATTERN.search(text or ""))


def infer_phase_for_char_position(completion: str, char_pos: int) -> str:
    completion = completion or ""
    plan = extract_plan_block(completion, min_tokens=0, max_tokens=10**9)
    if plan["plan_char_start"] is not None and plan["plan_char_end"] is not None:
        if plan["plan_char_start"] <= char_pos < plan["plan_char_end"]:
            return "plan"
        if char_pos < (plan["plan_close_end"] or 0):
            return "unknown"
    tail = completion[max(0, char_pos - 80) : char_pos + 80]
    if has_final_answer_phrase(tail) or has_boxed(tail):
        return "answer"
    return "execute"


def infer_phase_for_token_offsets(completion: str, offsets: list[tuple[int, int]]) -> list[str]:
    phases = []
    for start, end in offsets:
        pos = start if start is not None else end
        phases.append(infer_phase_for_char_position(completion, int(pos or 0)))
    return phases


def analyze_plan_features(problem: str, completion: str, num_output_tokens: int | None = None) -> dict[str, Any]:
    split = split_plan_execute_answer(completion or "")
    specificity = compute_plan_specificity(problem or "", split["plan_text"])
    math_density = compute_math_density(split["plan_text"])
    template = compute_template_score(split["plan_text"], problem or "")
    leakage = detect_computation_leakage(split["plan_text"])
    return {
        **split,
        "has_final_answer_phrase": has_final_answer_phrase(completion or ""),
        "has_boxed": has_boxed(completion or ""),
        "num_output_tokens": num_output_tokens,
        **specificity,
        "plan_math_density": math_density["math_density"],
        "plan_math_symbol_count": math_density["math_symbol_count"],
        **template,
        **leakage,
    }

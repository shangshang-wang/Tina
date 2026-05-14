"""Lightweight math answer extraction and equivalence checking."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Optional


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/[+-]?\d[\d,]*)?(?:e[-+]?\d+)?", re.I)


def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    depth = 0
    content_start = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if content_start is None:
                content_start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and content_start is not None:
                return text[content_start:i]
    return None


def extract_boxed(text: str) -> Optional[str]:
    for match in re.finditer(r"\\boxed\s*\{", text):
        inner = _extract_balanced_braces(text, match.end() - 1)
        if inner:
            return inner.strip()
    return None


def extract_answer_tag(text: str) -> Optional[str]:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if matches:
        return matches[-1].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract a final answer using boxed, answer anchors, and numeric fallbacks."""
    if not text:
        return None
    for extractor in (extract_boxed, extract_answer_tag):
        value = extractor(text)
        if value:
            boxed = extract_boxed(value)
            return boxed or value.strip()

    anchored = re.findall(
        r"(?:final\s+answer|answer|therefore,\s*the\s*final\s*answer\s*is)\s*[:：]\s*(.+)",
        text,
        flags=re.I,
    )
    if anchored:
        candidate = anchored[-1].strip()
        boxed = extract_boxed(candidate)
        if boxed:
            return boxed
        return candidate.splitlines()[0].strip().rstrip(".。")

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        boxed = extract_boxed(last)
        if boxed:
            return boxed
        numbers = NUMBER_RE.findall(last)
        if numbers:
            return numbers[-1]
        if len(last) <= 120:
            return last.rstrip(".。")

    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(ans: object) -> str:
    if ans is None:
        return ""
    text = str(ans).strip()
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    text = re.sub(r"^\$+|\$+$", "", text.strip())
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\,", "").replace("\\!", "").replace("\\%", "%")
    text = text.replace(",", "")
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("{}[]()")
    text = text.lower()
    if text.endswith("."):
        text = text[:-1]
    numeric = parse_numeric(text)
    if numeric is not None and math.isfinite(numeric):
        if abs(numeric - round(numeric)) < 1e-12:
            return str(int(round(numeric)))
        return f"{numeric:.12g}"
    return text


def parse_numeric(ans: object) -> Optional[float]:
    if ans is None:
        return None
    text = str(ans).strip()
    if not text:
        return None
    text = normalize_latex_for_numeric(text)
    try:
        return float(Fraction(text))
    except Exception:
        pass
    try:
        return float(text)
    except Exception:
        return None


def normalize_latex_for_numeric(text: str) -> str:
    text = text.strip()
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    text = re.sub(r"^\$+|\$+$", "", text.strip())
    text = text.replace(",", "").replace(" ", "")
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = text.strip("{}")
    return text


def sympy_equiv(pred: str, gold: str) -> Optional[bool]:
    try:
        import sympy as sp
    except Exception:
        return None
    try:
        p = sp.sympify(normalize_latex_for_sympy(pred))
        g = sp.sympify(normalize_latex_for_sympy(gold))
        return bool(sp.simplify(p - g) == 0)
    except Exception:
        return None


def normalize_latex_for_sympy(text: str) -> str:
    text = normalize_latex_for_numeric(text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("^", "**")
    text = text.replace("\\pi", "pi")
    return text


def is_equiv(pred: object, gold: object, tolerance: float = 1e-6) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    pred_num = parse_numeric(pred_norm)
    gold_num = parse_numeric(gold_norm)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) <= tolerance
    sympy_result = sympy_equiv(pred_norm, gold_norm)
    if sympy_result is not None:
        return sympy_result
    return False


def format_valid(text: str, mode: str = "any") -> bool:
    if mode == "boxed":
        return extract_boxed(text) is not None
    if mode == "answer_tag":
        return extract_answer_tag(text) is not None
    if mode == "think_answer":
        return "</think>" in text and extract_answer_tag(text) is not None
    return extract_final_answer(text) is not None

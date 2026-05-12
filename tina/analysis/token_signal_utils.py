"""Teacher-forcing token signal helpers."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from tina.analysis.plan_utils import infer_phase_for_token_offsets


def is_digit(token_text: str) -> bool:
    return bool(re.search(r"\d", token_text or ""))


def is_symbol(token_text: str) -> bool:
    return bool(re.search(r"[=<>+\-*/^%$(){}\[\]\\]", token_text or ""))


def is_math_token(token_text: str) -> bool:
    return is_digit(token_text) or is_symbol(token_text) or any(
        marker in (token_text or "") for marker in ["frac", "sqrt", "sum", "binom"]
    )


def is_newline_or_step_boundary(token_text: str) -> bool:
    return "\n" in (token_text or "") or bool(re.search(r"\b(step|first|then|next|finally)\b", token_text or "", re.I))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_phase_stats(token_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped[row.get("phase") or "unknown"].append(row)
    stats = {}
    for phase in ["plan", "execute", "answer", "unknown"]:
        rows = grouped.get(phase, [])
        stats[phase] = {
            "count": len(rows),
            "mean_entropy": _mean([r["entropy"] for r in rows if r.get("entropy") is not None]),
            "mean_nll": _mean([r["nll"] for r in rows if r.get("nll") is not None]),
            "mean_top1_prob": _mean([r["top1_prob"] for r in rows if r.get("top1_prob") is not None]),
            "math_density": sum(1 for r in rows if r.get("is_math_token")) / max(1, len(rows)),
        }
    return stats


def bucket_by_position(token_rows: list[dict[str, Any]], buckets: int = 10) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        norm = row.get("normalized_position") or 0.0
        idx = min(buckets - 1, max(0, int(norm * buckets)))
        grouped[idx].append(row)
    output = []
    for idx in range(buckets):
        rows = grouped.get(idx, [])
        output.append(
            {
                "bucket": f"{idx / buckets:.1f}-{(idx + 1) / buckets:.1f}",
                "mean_entropy": _mean([r["entropy"] for r in rows if r.get("entropy") is not None]),
                "mean_nll": _mean([r["nll"] for r in rows if r.get("nll") is not None]),
                "count": len(rows),
            }
        )
    return output


def compute_token_logprobs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_tokens: int = 4096,
    compute_entropy: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_encoding = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    completion_ids = completion_encoding.input_ids
    offsets = completion_encoding.offset_mapping
    truncated = len(completion_ids) > max_tokens
    completion_ids = completion_ids[:max_tokens]
    offsets = offsets[:max_tokens]
    if not completion_ids:
        return [], truncated

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    prompt_len = len(prompt_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    token_rows = []
    phases = infer_phase_for_token_offsets(completion, offsets)
    denom = max(1, len(completion_ids) - 1)
    for pos, token_id in enumerate(completion_ids):
        logits_pos = prompt_len + pos - 1
        if logits_pos < 0:
            continue
        dist_logits = logits[logits_pos].float()
        log_probs = F.log_softmax(dist_logits, dim=-1)
        probs = torch.exp(log_probs)
        logprob = float(log_probs[token_id].item())
        top1_prob = float(torch.max(probs).item())
        entropy = None
        if compute_entropy:
            entropy = float(-(probs * log_probs).sum().item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_rows.append(
            {
                "token_id": int(token_id),
                "token_text": token_text,
                "position": pos,
                "normalized_position": pos / denom,
                "logprob_of_actual_token": logprob,
                "nll": -logprob if math.isfinite(logprob) else None,
                "entropy": entropy,
                "top1_prob": top1_prob,
                "phase": phases[pos] if pos < len(phases) else "unknown",
                "is_math_token": is_math_token(token_text),
                "is_digit": is_digit(token_text),
                "is_symbol": is_symbol(token_text),
                "is_newline_or_step_boundary": is_newline_or_step_boundary(token_text),
            }
        )
    return token_rows, truncated


__all__ = [
    "aggregate_phase_stats",
    "bucket_by_position",
    "compute_token_logprobs",
    "is_digit",
    "is_math_token",
    "is_newline_or_step_boundary",
    "is_symbol",
]

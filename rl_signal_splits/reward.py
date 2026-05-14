"""Reward evaluation wrappers.

`evaluate_completion` is the integration point for project-specific verifiers.
Replace or extend `project_accuracy` when plugging in a stricter Open-RS/Tina
reward pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .answer_extraction import extract_final_answer, format_valid, is_equiv, normalize_answer


def project_accuracy(text: str, gold: str) -> Optional[bool]:
    """Try Tina's math_verify-backed accuracy reward when dependencies exist."""
    try:
        from tina.post_train_hf.rewards import accuracy_reward

        completions = [[{"role": "assistant", "content": text}]]
        rewards = accuracy_reward(completions=completions, solution=[gold])
        if rewards and rewards[0] is not None:
            return bool(float(rewards[0]) > 0.5)
    except Exception:
        return None
    return None


def evaluate_completion(
    text: str,
    gold: str,
    format_mode: str = "any",
    use_project_verifier: bool = True,
) -> Dict[str, Any]:
    extracted = extract_final_answer(text)
    valid = format_valid(text, mode=format_mode)
    project_result = project_accuracy(text, gold) if use_project_verifier else None
    correct = project_result if project_result is not None else is_equiv(extracted, gold)
    if correct:
        reward = 1.0
    elif valid:
        reward = 0.0
    else:
        reward = -0.2
    return {
        "extracted_final_answer": extracted,
        "normalized_extracted_answer": normalize_answer(extracted),
        "is_correct": bool(correct),
        "format_valid": bool(valid),
        "reward": float(reward),
        "verifier": "project" if project_result is not None else "fallback",
    }

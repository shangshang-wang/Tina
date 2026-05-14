"""Prompt construction for RL signal analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional


FALLBACK_OPEN_RS_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{} .
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
Note that respond by English, NOT use other languages."""

FALLBACK_OPEN_R1_SYSTEM_PROMPT = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format: <think>
...
</think>
<answer>
...
</answer>"""

MATH_QUERY_TEMPLATE = """Solve the following math problem efficiently and clearly. The last line of your response should contain the final answer in \\boxed{{ANSWER}}. Think step by step before answering.

{question}"""


def get_system_prompt(style: str) -> str:
    try:
        from tina.utils.prompt import OPEN_R1_SYSTEM_PROMPT, OPEN_RS_SYSTEM_PROMPT
    except Exception:
        OPEN_RS_SYSTEM_PROMPT = FALLBACK_OPEN_RS_SYSTEM_PROMPT
        OPEN_R1_SYSTEM_PROMPT = FALLBACK_OPEN_R1_SYSTEM_PROMPT
    if style == "open_r1":
        return OPEN_R1_SYSTEM_PROMPT.strip()
    if style in {"none", "raw"}:
        return ""
    return OPEN_RS_SYSTEM_PROMPT.strip()


def build_user_prompt(problem: str, template: str = "math") -> str:
    if template == "raw":
        return problem
    return MATH_QUERY_TEMPLATE.format(question=problem)


def build_prompt(
    record: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    prompt_style: str = "open_rs",
    user_template: str = "raw",
) -> str:
    problem = record.get("problem") or ""
    user_prompt = build_user_prompt(problem, template=user_template)
    system_prompt = get_system_prompt(prompt_style)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    if system_prompt:
        return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
    return user_prompt

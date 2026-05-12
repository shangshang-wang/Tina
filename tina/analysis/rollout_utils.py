"""Shared I/O, matching, prompt, dataset, and answer helpers for diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from tina.analysis.plan_utils import has_boxed, has_final_answer_phrase


MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

PLAN_SCAFFOLD_PREFIX = (
    "Before solving, write a concise high-level strategy inside exactly one <plan>...</plan> block. "
    "The plan should describe the approach, not carry out the full computation. "
    "After </plan>, continue the solution and give the final answer in the required format.\n\n"
)

GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

TASK_SPECS = {
    "aime24": {
        "repo": "HuggingFaceH4/aime_2024",
        "subset": "default",
        "split": "train",
        "problem_field": "problem",
        "gold_field": "answer",
        "kind": "math",
    },
    "aime25": {
        "repo": "yentinglin/aime_2025",
        "subset": "default",
        "split": "train",
        "problem_field": "problem",
        "gold_field": "answer",
        "kind": "math",
    },
    "amc23": {
        "repo": "knoveleng/AMC-23",
        "subset": "default",
        "split": "train",
        "problem_field": "problem",
        "gold_field": "answer",
        "kind": "math",
    },
    "math_500": {
        "repo": "HuggingFaceH4/MATH-500",
        "subset": "default",
        "split": "test",
        "problem_field": "problem",
        "gold_field": "solution",
        "kind": "math",
    },
    "minerva": {
        "repo": "knoveleng/Minerva-Math",
        "subset": "default",
        "split": "train",
        "problem_field": "problem",
        "gold_field": "solution",
        "kind": "math",
    },
    "olympiadbench": {
        "repo": "knoveleng/OlympiadBench",
        "subset": "default",
        "split": "train",
        "problem_field": "question",
        "gold_field": "answer",
        "kind": "math",
    },
    "gpqa:diamond": {
        "repo": "Idavidrein/gpqa",
        "subset": "gpqa_diamond",
        "split": "train",
        "problem_field": "Question",
        "gold_field": "Correct Answer",
        "kind": "gpqa",
    },
}

TASK_ALIASES = {
    "all": ["aime24", "aime25", "amc23", "math_500", "minerva", "olympiadbench", "gpqa:diamond"],
    "math": ["aime24", "aime25", "amc23", "math_500", "minerva", "olympiadbench"],
    "science": ["gpqa:diamond"],
}


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_arg(value: str | None) -> list[int]:
    return [int(item) for item in parse_csv_arg(value)]


def expand_task_arg(value: str | None) -> list[str]:
    tasks: list[str] = []
    for item in parse_csv_arg(value):
        expanded = TASK_ALIASES.get(item, [item])
        for task in expanded:
            if task not in tasks:
                tasks.append(task)
    return tasks


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stable_example_id(task: str, index: int, problem: str) -> str:
    digest = hashlib.sha1(f"{task}\n{index}\n{problem}".encode("utf-8")).hexdigest()[:16]
    return f"{task}-{index}-{digest}"


def rollout_match_key(row: dict[str, Any]) -> tuple[str, int | str | None, str]:
    task = row.get("task")
    seed = row.get("seed")
    example_id = row.get("example_id")
    if example_id is not None and str(example_id) != "":
        return (str(task), seed, f"id:{example_id}")
    return (str(task), seed, f"index:{row.get('index')}")


def parse_success(row: dict[str, Any]) -> bool:
    if "parse_success" in row:
        return bool(row["parse_success"])
    pred = row.get("pred") or row.get("pred_letter")
    return pred is not None and str(pred).strip() != ""


def count_output_tokens(tokenizer: Any, text: str) -> int:
    if tokenizer is not None:
        return len(tokenizer(text or "", add_special_tokens=False).input_ids)
    return len(re.findall(r"\S+", text or ""))


def extract_boxed_answer(text: str) -> str | None:
    text = text or ""
    last_start = text.rfind(r"\boxed{")
    if last_start < 0:
        return None
    i = last_start + len(r"\boxed{")
    depth = 1
    chars = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        i += 1
    return None


def extract_final_answer_text(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    patterns = [
        r"Answer\s*:\s*\$?([A-D])\b",
        r"final answer is\s*:?\s*\$?\\?boxed\{?([^}\n]+)",
        r"final answer is\s*:?\s*\$?([^\n$.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).strip().strip("$.")
    return None


def extract_gpqa_letter(text: str) -> str | None:
    text = text or ""
    patterns = [
        r"Answer\s*:\s*\$?\s*([A-D])\b",
        r"answer[\"']?\s*[:=]\s*[\"']?([A-D])\b",
        r"final answer is\s*:?\s*\$?\s*([A-D])\b",
        r"\b([A-D])\s*\)?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return None


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    return text


def evaluate_math_answer(completion: str, gold: str) -> tuple[str | None, bool]:
    pred = extract_final_answer_text(completion)
    if pred is None:
        return None, False
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        gold_parsed = parse(str(gold), extraction_mode="first_match")
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if gold_parsed:
            return pred, bool(verify(gold_parsed, answer_parsed))
    except Exception:
        pass
    return pred, normalize_answer(pred) == normalize_answer(gold)


def evaluate_gpqa_answer(completion: str, gold_letter: str) -> tuple[str | None, bool]:
    pred_letter = extract_gpqa_letter(completion)
    return pred_letter, pred_letter == gold_letter


def make_prompt(base_prompt: str, prompt_style: str) -> str:
    if prompt_style == "tina_original":
        return base_prompt
    if prompt_style == "plan_scaffold":
        return PLAN_SCAFFOLD_PREFIX + base_prompt
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def _seedless_gpqa_order(index: int) -> list[int]:
    # Cycle the correct option through A/B/C/D without using a random seed.
    return [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
    ][index % 4]


def build_gpqa_prompt(line: dict[str, Any], seed: int | None, index: int, prompt_style: str) -> dict[str, Any]:
    choices_with_original = [
        (0, line["Correct Answer"], True),
        (1, line["Incorrect Answer 1"], False),
        (2, line["Incorrect Answer 2"], False),
        (3, line["Incorrect Answer 3"], False),
    ]
    if seed is None:
        order = _seedless_gpqa_order(index)
        choices_with_original = [choices_with_original[i] for i in order]
    else:
        rng = random.Random(seed * 1_000_003 + index)
        rng.shuffle(choices_with_original)
    choices = [choice for _, choice, _ in choices_with_original]
    gold_index = next(i for i, (_, _, is_gold) in enumerate(choices_with_original) if is_gold)
    gold_letter = "ABCD"[gold_index]
    choice_order = [original_idx for original_idx, _, _ in choices_with_original]
    prompt = GPQA_QUERY_TEMPLATE.format(
        Question=line["Question"], A=choices[0], B=choices[1], C=choices[2], D=choices[3]
    )
    return {
        "problem": line["Question"],
        "gold": line["Correct Answer"],
        "gold_letter": gold_letter,
        "prompt": make_prompt(prompt, prompt_style),
        "choices": choices,
        "choice_order": choice_order,
    }


def build_math_prompt(task: str, line: dict[str, Any], prompt_style: str) -> dict[str, Any]:
    spec = TASK_SPECS[task]
    problem = line[spec["problem_field"]]
    gold = line[spec["gold_field"]]
    prompt = MATH_QUERY_TEMPLATE.format(Question=problem)
    return {
        "problem": problem,
        "gold": gold,
        "gold_letter": None,
        "prompt": make_prompt(prompt, prompt_style),
        "choices": None,
        "choice_order": None,
    }


def _stratum_for_line(task: str, line: dict[str, Any], index: int) -> str:
    fields = ["level", "type", "subject", "category", "subfield", "difficulty", "source", "domain"]
    parts = [f"{field}={line[field]}" for field in fields if field in line and line[field] not in (None, "")]
    if parts:
        return "|".join(parts[:3])
    spec = TASK_SPECS[task]
    problem = str(line.get(spec["problem_field"], ""))
    length_bucket = min(4, len(problem.split()) // 80)
    return f"length_bucket={length_bucket}"


def select_coverage_indices(dataset: Any, task: str, sample_count: int | None, strategy: str) -> list[int]:
    total = len(dataset)
    if sample_count is None or sample_count >= total:
        return list(range(total))
    sample_count = max(0, sample_count)
    if sample_count == 0:
        return []
    if strategy == "first":
        return list(range(sample_count))
    if strategy != "coverage":
        raise ValueError(f"Unknown sampling_strategy: {strategy}")

    buckets: dict[str, list[int]] = {}
    for index, line in enumerate(dataset):
        buckets.setdefault(_stratum_for_line(task, line, index), []).append(index)
    selected: list[int] = []
    bucket_items = sorted(buckets.items(), key=lambda item: (item[0], len(item[1])))
    cursor = 0
    while len(selected) < sample_count and bucket_items:
        made_progress = False
        for _, indices in bucket_items:
            if cursor < len(indices):
                selected.append(indices[cursor])
                made_progress = True
                if len(selected) >= sample_count:
                    break
        cursor += 1
        if not made_progress:
            break
    return sorted(selected)


def load_task_examples(
    task: str,
    seed: int | None,
    prompt_style: str,
    max_samples: int | None = None,
    sampling_strategy: str = "first",
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("collect_rollouts.py requires the datasets package") from exc
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}. Known tasks: {sorted(TASK_SPECS)}")
    spec = TASK_SPECS[task]
    dataset = load_dataset(spec["repo"], spec["subset"], split=spec["split"])
    indices = select_coverage_indices(dataset, task, max_samples, sampling_strategy)
    dataset = dataset.select(indices)
    examples = []
    for original_index, line in zip(indices, dataset):
        built = (
            build_gpqa_prompt(line, seed, original_index, prompt_style)
            if spec["kind"] == "gpqa"
            else build_math_prompt(task, line, prompt_style)
        )
        example_id = str(line.get("id") or line.get("example_id") or stable_example_id(task, original_index, built["problem"]))
        examples.append(
            {
                "task": task,
                "seed": seed,
                "seedless": seed is None,
                "example_id": example_id,
                "index": original_index,
                "sample_strategy": sampling_strategy,
                "sample_stratum": _stratum_for_line(task, line, original_index),
                **built,
            }
        )
    return examples


def load_rollout_dir(dir_path: str | Path, tasks: list[str] | None = None, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    dir_path = Path(dir_path)
    rows = []
    task_set = set(tasks or [])
    seed_set = set(seeds or [])
    for path in sorted(dir_path.glob("*.jsonl")):
        for row in iter_jsonl(path):
            if task_set and row.get("task") not in task_set:
                continue
            row_seed = row.get("seed")
            if seed_set and (row_seed is None or int(row_seed) not in seed_set):
                continue
            rows.append(row)
    return rows


def row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


__all__ = [
    "TASK_SPECS",
    "expand_task_arg",
    "evaluate_gpqa_answer",
    "evaluate_math_answer",
    "extract_gpqa_letter",
    "has_boxed",
    "has_final_answer_phrase",
    "load_rollout_dir",
    "load_task_examples",
    "parse_success",
    "read_csv_rows",
    "read_jsonl",
    "rollout_match_key",
    "write_csv",
    "write_jsonl",
]

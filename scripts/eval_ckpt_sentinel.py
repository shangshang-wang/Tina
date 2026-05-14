#!/usr/bin/env python
"""Evaluate and rank candidate checkpoints on a fixed sentinel eval set."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ckpt_sentinel_utils import (  # noqa: E402
    CANONICAL_BENCHMARKS,
    benchmark_to_task,
    format_float,
    normalize_benchmark,
    parse_bool_arg,
    read_jsonl,
    score_sentinel_results,
    write_csv,
    write_json,
    write_jsonl,
)
from tina.analysis.rollout_utils import evaluate_gpqa_answer, evaluate_math_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_root", default=None)
    parser.add_argument("--sentinel_file", required=True)
    parser.add_argument("--weights_file", required=True)
    parser.add_argument("--output_dir", default="outputs/ckpt_selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--candidate_checkpoints", default=None)
    parser.add_argument("--checkpoint_paths", default=None, help="Optional comma-separated full checkpoint paths.")
    parser.add_argument("--trainer_log", default=None)
    parser.add_argument("--auto_select_candidates", default=False, type=parse_bool_arg)
    parser.add_argument("--max_candidates", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit_per_benchmark", type=int, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument(
        "--dry_run_no_model",
        action="store_true",
        help="Write manifests and auto-selection outputs without loading a model.",
    )
    return parser.parse_args()


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def moving_average(values: list[float | None], window: int = 5) -> list[float | None]:
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1) : i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def numeric(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except Exception:
                continue
    return None


def row_step(row: dict[str, Any]) -> int | None:
    step = numeric(row, ["step", "global_step", "train/global_step"])
    return int(step) if step is not None else None


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


def read_trainer_rows(path: Path) -> list[dict[str, Any]]:
    obj = load_json_any(path)
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ["log_history", "logs", "history"]:
            if isinstance(obj.get(key), list):
                return [row for row in obj[key] if isinstance(row, dict)]
    return []


def nearest_checkpoint_name(step: int | None) -> str | None:
    if step is None:
        return None
    return f"checkpoint-{int(step)}"


def checkpoint_step_from_name(name: str) -> int | None:
    import re

    match = re.search(r"checkpoint-(\d+)$", name)
    return int(match.group(1)) if match else None


def available_checkpoint_names(adapter_root: str | None) -> list[str]:
    if not adapter_root:
        return []
    root = Path(adapter_root)
    if not root.exists():
        return []
    names = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        step = checkpoint_step_from_name(path.name)
        if step is not None:
            names.append(path.name)
    return sorted(names, key=lambda name: checkpoint_step_from_name(name) or 0)


def map_to_available_checkpoint(name: str | None, available: list[str]) -> str | None:
    if not name:
        return None
    if not available or name in available:
        return name
    target_step = checkpoint_step_from_name(name)
    if target_step is None:
        return name
    return min(available, key=lambda candidate: abs((checkpoint_step_from_name(candidate) or 0) - target_step))


def auto_select_candidates(
    trainer_log: Path,
    max_candidates: int,
    sentinel_rows: list[dict[str, Any]],
    output_dir: Path,
    adapter_root: str | None = None,
) -> list[str]:
    rows = [row for row in read_trainer_rows(trainer_log) if row_step(row) is not None]
    if not rows:
        return []
    rows.sort(key=lambda row: row_step(row) or 0)
    rewards = [numeric(row, ["reward", "train/reward", "rewards/cosine_scaled_reward", "cosine_scaled_reward"]) for row in rows]
    format_rewards = [numeric(row, ["rewards/format_reward", "format_reward", "train/format_reward"]) for row in rows]
    lengths = [numeric(row, ["completion_length", "train/completion_length", "response_length", "num_completion_tokens"]) for row in rows]
    reward_stds = [numeric(row, ["reward_std", "train/reward_std", "rewards/reward_std"]) for row in rows]
    reward_ma = moving_average(rewards)
    format_ma = moving_average(format_rewards)
    length_values = [v for v in lengths if v is not None]
    std_values = [v for v in reward_stds if v is not None]
    format_values = [v for v in format_ma if v is not None]
    length_low = percentile(length_values, 0.10)
    length_high = percentile(length_values, 0.90)
    std_high = percentile(std_values, 0.80)
    format_low = percentile(format_values, 0.20)
    target_baseline_length_values = [float(row["baseline_length"]) for row in sentinel_rows if row.get("baseline_length") is not None]
    target_baseline_length = statistics.mean(target_baseline_length_values) if target_baseline_length_values else None

    enriched = []
    for i, row in enumerate(rows):
        step = row_step(row)
        length = lengths[i]
        reward_std = reward_stds[i]
        abnormal_length = (
            length is not None
            and length_low is not None
            and length_high is not None
            and (length < length_low or length > length_high)
        )
        abnormal_std = reward_std is not None and std_high is not None and reward_std > std_high
        low_format = format_ma[i] is not None and format_low is not None and format_ma[i] < format_low
        enriched.append(
            {
                "step": step,
                "checkpoint": nearest_checkpoint_name(step),
                "reward": rewards[i],
                "reward_ma": reward_ma[i],
                "format_reward": format_rewards[i],
                "format_reward_ma": format_ma[i],
                "completion_length": length,
                "reward_std": reward_std,
                "abnormal_length": abnormal_length,
                "abnormal_reward_std": abnormal_std,
                "low_format_reward": low_format,
                "eligible": not abnormal_length and not abnormal_std and not low_format,
            }
        )
    eligible = [row for row in enriched if row["eligible"] and row["reward_ma"] is not None]
    top_reward_cutoff = percentile([row["reward_ma"] for row in eligible if row["reward_ma"] is not None], 0.70)
    pool = [row for row in eligible if top_reward_cutoff is None or row["reward_ma"] >= top_reward_cutoff]
    pool.sort(key=lambda row: row["reward_ma"] or float("-inf"), reverse=True)

    forced = []
    with_reward = [row for row in enriched if row["reward"] is not None]
    with_format = [row for row in enriched if row["format_reward"] is not None]
    with_length = [row for row in enriched if row["completion_length"] is not None]
    if with_reward:
        forced.append(max(with_reward, key=lambda row: row["reward"] or float("-inf")))
    if with_format:
        forced.append(max(with_format, key=lambda row: row["format_reward"] or float("-inf")))
    if with_length and target_baseline_length is not None:
        forced.append(min(with_length, key=lambda row: abs((row["completion_length"] or 0) - target_baseline_length)))
    mid = enriched[len(enriched) // 2]
    forced.append(mid)
    late = [row for row in enriched[int(len(enriched) * 0.70) :] if row["eligible"]]
    if late:
        forced.append(late[-1])

    available = available_checkpoint_names(adapter_root)
    selected = []
    for row in forced + pool:
        ckpt = map_to_available_checkpoint(row.get("checkpoint"), available)
        if ckpt and ckpt not in selected:
            selected.append(ckpt)
        if len(selected) >= max_candidates:
            break

    write_json(
        output_dir / "candidate_checkpoints.json",
        {
            "trainer_log": str(trainer_log),
            "adapter_root": adapter_root,
            "available_checkpoints": available,
            "selected_checkpoints": selected,
            "rules": [
                "reward moving average top 30% after filtering",
                "drop bottom/top 10% completion length",
                "drop top 20% reward_std",
                "drop bottom 20% format_reward moving average",
                "force include reward max, format_reward max, baseline-length-nearest, mid, late eligible",
            ],
            "thresholds": {
                "length_low_p10": length_low,
                "length_high_p90": length_high,
                "reward_std_p80": std_high,
                "format_reward_ma_p20": format_low,
                "target_open_rs3_baseline_length": target_baseline_length,
            },
            "rows": enriched,
        },
    )
    return selected


def resolve_checkpoint_paths(args: argparse.Namespace, sentinel_rows: list[dict[str, Any]], output_dir: Path) -> list[tuple[str, Path]]:
    names = parse_csv(args.candidate_checkpoints)
    if args.auto_select_candidates and args.trainer_log:
        names = auto_select_candidates(Path(args.trainer_log), args.max_candidates, sentinel_rows, output_dir, args.adapter_root)
    explicit_paths = parse_csv(args.checkpoint_paths)
    resolved: list[tuple[str, Path]] = []
    for item in explicit_paths:
        path = Path(item)
        resolved.append((path.name, path))
    root = Path(args.adapter_root) if args.adapter_root else None
    for name in names:
        path = Path(name)
        if not path.is_absolute() and root is not None:
            path = root / name
        resolved.append((name, path))
    seen = set()
    unique = []
    for name, path in resolved:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, path))
    if args.dry_run:
        return unique[: min(2, len(unique))]
    return unique


def filter_sentinel(rows: list[dict[str, Any]], limit_per_benchmark: int | None, dry_run: bool) -> list[dict[str, Any]]:
    limit = 2 if dry_run and limit_per_benchmark is None else limit_per_benchmark
    if limit is None:
        return rows
    out = []
    counts = defaultdict(int)
    for row in rows:
        bench = normalize_benchmark(row.get("benchmark"))
        if counts[bench] >= limit:
            continue
        out.append(row)
        counts[bench] += 1
    return out


def torch_dtype(dtype: str):
    import torch

    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)


def render_prompt(tokenizer: Any, row: dict[str, Any], use_chat_template: bool) -> str:
    prompt = row.get("prompt") or row.get("problem") or ""
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def detect_checkpoint_type(path: Path) -> str:
    if (path / "adapter_config.json").exists():
        return "lora_adapter"
    if (path / "config.json").exists() or any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")):
        return "full_model"
    return "missing_or_unknown"


def load_base_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if args.max_model_length:
        tokenizer.model_max_length = args.max_model_length
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def activate_lora_adapter(base_or_peft_model: Any, path: Path, adapter_name: str) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("LoRA adapter evaluation requires the peft package") from exc
    if not hasattr(base_or_peft_model, "peft_config"):
        model = PeftModel.from_pretrained(base_or_peft_model, str(path), adapter_name=adapter_name)
        model.eval()
        return model
    if adapter_name not in base_or_peft_model.peft_config:
        base_or_peft_model.load_adapter(str(path), adapter_name=adapter_name)
    base_or_peft_model.set_adapter(adapter_name)
    base_or_peft_model.eval()
    return base_or_peft_model


def load_full_checkpoint(args: argparse.Namespace, path: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model


def batch_iter(rows: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_name: str,
    checkpoint_path: Path,
    checkpoint_type: str,
    tokenizer: Any,
    base_model: Any,
    peft_model_holder: dict[str, Any],
    sentinel_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Any]:
    import torch
    from transformers import set_seed

    set_seed(args.seed)
    if checkpoint_type == "lora_adapter":
        model = activate_lora_adapter(peft_model_holder.get("model") or base_model, checkpoint_path, checkpoint_name)
        peft_model_holder["model"] = model
    elif checkpoint_type == "full_model":
        model = load_full_checkpoint(args, checkpoint_path)
    else:
        raise RuntimeError(f"Cannot evaluate {checkpoint_name}: checkpoint path is missing or unknown: {checkpoint_path}")
    model.eval()
    device = next(model.parameters()).device

    rows_out = []
    for batch in batch_iter(sentinel_rows, max(1, args.batch_size)):
        prompts = [render_prompt(tokenizer, row, args.use_chat_template) for row in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=bool(args.max_model_length),
            max_length=args.max_model_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs = {
            "do_sample": args.temperature > 0,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for row, sequence, input_len, prompt in zip(batch, output_ids, input_lengths, prompts):
            completion_ids = sequence[int(input_len) :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            benchmark = normalize_benchmark(row.get("benchmark"))
            if benchmark == "gpqa":
                pred, correct = evaluate_gpqa_answer(completion, row.get("gold_letter") or row.get("gold_answer"))
                pred_letter = pred
            else:
                pred, correct = evaluate_math_answer(completion, row.get("gold_answer"))
                pred_letter = None
            format_valid = pred is not None
            rows_out.append(
                {
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_type": checkpoint_type,
                    "id": str(row.get("id")),
                    "benchmark": benchmark,
                    "task": benchmark_to_task(benchmark),
                    "problem": row.get("problem"),
                    "gold_answer": row.get("gold_answer"),
                    "prompt": prompt,
                    "completion": completion,
                    "extracted_answer": pred,
                    "pred_letter": pred_letter,
                    "correct": bool(correct),
                    "format_valid": bool(format_valid),
                    "completion_length": int(completion_ids.numel()),
                    "improvement_tag": row.get("improvement_tag"),
                    "difficulty_tag": row.get("difficulty_tag"),
                    "item_weight": row.get("item_weight"),
                    "base_correct": row.get("base_correct"),
                    "baseline_correct": row.get("baseline_correct"),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                    "eval_protocol": "sentinel fallback generator using tina.analysis.rollout_utils verifier",
                }
            )
    if checkpoint_type == "full_model":
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows_out, peft_model_holder.get("model")


def summary_row(checkpoint_name: str, checkpoint_path: Path, checkpoint_type: str, result_rows: list[dict[str, Any]], weights: dict[str, Any]) -> dict[str, Any]:
    summary = score_sentinel_results(result_rows, weights)
    return {
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_type": checkpoint_type,
        "sentinel_score": summary["sentinel_score"],
        "benchmark_scores": summary["benchmark_scores"],
        "overall_accuracy_unweighted": summary["overall_accuracy_unweighted"],
        "format_valid_rate": summary["format_valid_rate"],
        "avg_completion_length": summary["avg_completion_length"],
        "length_by_benchmark": summary["length_by_benchmark"],
        "base_wrong_baseline_right_recovery_rate": summary["base_wrong_baseline_right_recovery_rate"],
        "base_right_baseline_wrong_forgetting_rate": summary["base_right_baseline_wrong_forgetting_rate"],
        "both_wrong_breakthrough_rate": summary["both_wrong_breakthrough_rate"],
        "both_right_retention_rate": summary["both_right_retention_rate"],
    }


def write_selection_report(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    checkpoints: list[tuple[str, Path, str]],
    auto_selected: bool,
) -> None:
    ranked = sorted(summaries, key=lambda row: row.get("sentinel_score") if row.get("sentinel_score") is not None else -1, reverse=True)
    top = ranked[:3]
    close_note = ""
    if len(ranked) >= 2 and ranked[0].get("sentinel_score") is not None and ranked[1].get("sentinel_score") is not None:
        if abs(ranked[0]["sentinel_score"] - ranked[1]["sentinel_score"]) < 0.02:
            close_note = "Top checkpoints are within 0.02 sentinel score; run full benchmarks for all close candidates."
    lines = [
        "# Checkpoint Sentinel Evaluation Report",
        "",
        "Sentinel score is for checkpoint selection only. It is a benchmark-weighted sentinel score, not a final paper/report score.",
        "",
        f"Auto-selected from trainer log: `{auto_selected}`",
        "",
        "## Evaluated Checkpoints",
        "",
    ]
    for name, path, ckpt_type in checkpoints:
        lines.append(f"- `{name}` ({ckpt_type}): `{path}`")
    lines.extend(["", "## Scores", ""])
    headers = [
        "checkpoint",
        "sentinel",
        "overall",
        "format",
        "avg len",
        "recovery",
        "forgetting",
        "breakthrough",
        "retention",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["checkpoint"]),
                    format_float(row.get("sentinel_score")),
                    format_float(row.get("overall_accuracy_unweighted")),
                    format_float(row.get("format_valid_rate")),
                    format_float(row.get("avg_completion_length"), 1),
                    format_float(row.get("base_wrong_baseline_right_recovery_rate")),
                    format_float(row.get("base_right_baseline_wrong_forgetting_rate")),
                    format_float(row.get("both_wrong_breakthrough_rate")),
                    format_float(row.get("both_right_retention_rate")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Benchmark Subscores", ""])
    headers = ["checkpoint"] + CANONICAL_BENCHMARKS
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in ranked:
        lines.append(
            "| "
            + " | ".join([str(row["checkpoint"])] + [format_float(row.get("benchmark_scores", {}).get(bench)) for bench in CANONICAL_BENCHMARKS])
            + " |"
        )
    lines.extend(["", "## Recommended Full-Benchmark Candidates", ""])
    for row in top:
        lines.append(f"- `{row['checkpoint']}`: sentinel_score={format_float(row.get('sentinel_score'))}")
    if close_note:
        lines.extend(["", close_note])
    lines.extend(
        [
            "",
            "Metrics:",
            "- recovery: accuracy on `base_wrong_baseline_right` items, where Open-RS3 improved over base.",
            "- forgetting: error rate on `base_right_baseline_wrong` items, where Open-RS3 regressed from base.",
            "- breakthrough: accuracy on `both_wrong` items.",
            "- retention: accuracy on `both_right` sanity items.",
        ]
    )
    (output_dir / "ckpt_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "top_candidates.json", {"top_candidates": top, "close_score_note": close_note})


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "sentinel_eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sentinel_rows = filter_sentinel(read_jsonl(args.sentinel_file), args.limit_per_benchmark, args.dry_run)
    with Path(args.weights_file).open("r", encoding="utf-8") as f:
        weights = json.load(f)
    checkpoints_raw = resolve_checkpoint_paths(args, sentinel_rows, output_dir)
    checkpoints = [(name, path, detect_checkpoint_type(path)) for name, path in checkpoints_raw]
    if not checkpoints:
        raise SystemExit("No candidate checkpoints. Pass --candidate_checkpoints, --checkpoint_paths, or enable --auto_select_candidates with --trainer_log.")

    if args.dry_run_no_model:
        manifest = [
            {"checkpoint": name, "path": str(path), "checkpoint_type": ckpt_type}
            for name, path, ckpt_type in checkpoints
        ]
        write_json(output_dir / "dry_run_manifest.json", {"checkpoints": manifest, "sentinel_items": len(sentinel_rows)})
        write_selection_report(output_dir, [], checkpoints, args.auto_select_candidates)
        print(f"Wrote dry-run manifest to {output_dir}")
        return

    tokenizer, base_model = load_base_model(args)
    peft_model_holder: dict[str, Any] = {}
    summaries = []
    for checkpoint_name, checkpoint_path, checkpoint_type in checkpoints:
        result_rows, active_peft = evaluate_checkpoint(
            args,
            checkpoint_name,
            checkpoint_path,
            checkpoint_type,
            tokenizer,
            base_model,
            peft_model_holder,
            sentinel_rows,
        )
        peft_model_holder["model"] = active_peft
        write_jsonl(results_dir / f"{checkpoint_name}.jsonl", result_rows)
        summaries.append(summary_row(checkpoint_name, checkpoint_path, checkpoint_type, result_rows, weights))

    ranked = sorted(summaries, key=lambda row: row.get("sentinel_score") if row.get("sentinel_score") is not None else -1, reverse=True)
    write_json(output_dir / "ckpt_sentinel_scores.json", ranked)
    csv_rows = []
    for row in ranked:
        flat = {
            "checkpoint": row["checkpoint"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_type": row["checkpoint_type"],
            "sentinel_score": row["sentinel_score"],
            "overall_accuracy_unweighted": row["overall_accuracy_unweighted"],
            "format_valid_rate": row["format_valid_rate"],
            "avg_completion_length": row["avg_completion_length"],
            "base_wrong_baseline_right_recovery_rate": row["base_wrong_baseline_right_recovery_rate"],
            "base_right_baseline_wrong_forgetting_rate": row["base_right_baseline_wrong_forgetting_rate"],
            "both_wrong_breakthrough_rate": row["both_wrong_breakthrough_rate"],
            "both_right_retention_rate": row["both_right_retention_rate"],
        }
        for bench in CANONICAL_BENCHMARKS:
            flat[f"{bench}_score"] = row.get("benchmark_scores", {}).get(bench)
        csv_rows.append(flat)
    write_csv(output_dir / "ckpt_sentinel_scores.csv", csv_rows)
    write_selection_report(output_dir, ranked, checkpoints, args.auto_select_candidates)
    print(f"Evaluated {len(checkpoints)} checkpoints on {len(sentinel_rows)} sentinel items. Results: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate rollout JSONL files for offline diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.rollout_utils import (
    evaluate_gpqa_answer,
    evaluate_math_answer,
    expand_task_arg,
    load_task_examples,
    parse_int_csv_arg,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task list, or aliases: all, math, science.")
    parser.add_argument("--seeds", default=None, help="Comma-separated generation seeds. Omit with --seedless.")
    parser.add_argument("--seedless", action="store_true", help="Do not set generation seeds or include seed as a comparison dimension.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--prompt_style", choices=["tina_original", "plan_scaffold"], default="tina_original")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Backward-compatible alias for --sample_per_task.")
    parser.add_argument("--sample_per_task", type=int, default=None)
    parser.add_argument("--sampling_strategy", choices=["first", "coverage"], default="coverage")
    parser.add_argument("--dry_run_manifest", action="store_true", help="Only write selected prompts/examples; do not load model or generate.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def _torch_dtype(dtype: str):
    import torch

    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)


def _render_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    sample_per_task = args.sample_per_task if args.sample_per_task is not None else args.max_samples
    seeds = [None] if args.seedless else parse_int_csv_arg(args.seeds)
    if not seeds:
        raise ValueError("Pass --seeds 0,1,... or use --seedless for cheap coverage diagnostics.")
    tasks = expand_task_arg(args.tasks)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run_manifest:
        for task in tasks:
            for seed in seeds:
                examples = load_task_examples(task, seed, args.prompt_style, sample_per_task, args.sampling_strategy)
                manifest_rows = [
                    {
                        "task": example["task"],
                        "seed": example["seed"],
                        "seedless": example["seedless"],
                        "example_id": example["example_id"],
                        "index": example["index"],
                        "sample_strategy": example.get("sample_strategy"),
                        "sample_stratum": example.get("sample_stratum"),
                        "problem": example["problem"],
                        "gold": example["gold"],
                        "gold_letter": example.get("gold_letter"),
                        "prompt": example["prompt"],
                        "choices": example.get("choices"),
                        "choice_order": example.get("choice_order"),
                    }
                    for example in examples
                ]
                suffix = "seedless" if seed is None else f"seed{seed}"
                write_jsonl(out_dir / f"{task}_{suffix}.manifest.jsonl", manifest_rows)
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError("collect_rollouts.py requires torch and transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if args.max_model_length:
        tokenizer.model_max_length = args.max_model_length
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for task in tasks:
        for seed in seeds:
            if seed is not None:
                set_seed(seed)
            examples = load_task_examples(task, seed, args.prompt_style, sample_per_task, args.sampling_strategy)
            rows = []
            for example in examples:
                raw_prompt = example["prompt"]
                prompt = _render_prompt(tokenizer, raw_prompt, args.use_chat_template)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=bool(args.max_model_length), max_length=args.max_model_length)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                if seed is not None:
                    set_seed(seed * 1_000_003 + int(example["index"]))
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
                completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                if task.startswith("gpqa"):
                    pred_letter, correct = evaluate_gpqa_answer(completion, example["gold_letter"])
                    pred = pred_letter
                else:
                    pred, correct = evaluate_math_answer(completion, example["gold"])
                    pred_letter = None
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "seedless": args.seedless,
                        "example_id": example["example_id"],
                        "index": example["index"],
                        "sample_strategy": example.get("sample_strategy"),
                        "sample_stratum": example.get("sample_stratum"),
                        "problem": example["problem"],
                        "gold": example["gold"],
                        "gold_letter": example["gold_letter"],
                        "prompt": prompt,
                        "raw_prompt": raw_prompt,
                        "completion": completion,
                        "pred": pred,
                        "pred_letter": pred_letter,
                        "correct": bool(correct),
                        "finish_reason": "eos" if completion_ids.numel() and completion_ids[-1].item() == tokenizer.eos_token_id else "length",
                        "num_output_tokens": int(completion_ids.numel()),
                        "model": args.model,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "max_model_length": args.max_model_length,
                        "prompt_style": args.prompt_style,
                        "choices": example.get("choices"),
                        "choice_order": example.get("choice_order"),
                    }
                )
            suffix = "seedless" if seed is None else f"seed{seed}"
            write_jsonl(out_dir / f"{task}_{suffix}.jsonl", rows)


if __name__ == "__main__":
    main()

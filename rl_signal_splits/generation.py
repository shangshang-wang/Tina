"""Transformers generation helpers for repeated per-sample rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompting import build_prompt


@dataclass
class GenerationSettings:
    model: str
    n_samples: int = 8
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 8
    seed: int = 42
    prompt_style: str = "open_rs"
    user_template: str = "raw"
    trust_remote_code: bool = False
    mock_generation: bool = False
    backend: str = "transformers"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None
    max_prompt_tokens: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class RolloutGenerator:
    def __init__(self, settings: GenerationSettings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.llm = None
        set_seed(settings.seed)
        if settings.mock_generation:
            return
        if settings.backend == "vllm":
            self._load_vllm()
        else:
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers and torch are required for real generation. "
                "Use --mock_generation for plumbing tests."
            ) from exc

        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model,
            trust_remote_code=self.settings.trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=self.settings.trust_remote_code,
        )
        self.model.eval()

    def _load_vllm(self) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except Exception as exc:
            raise RuntimeError("vllm and transformers are required for --generation_backend vllm.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model,
            trust_remote_code=self.settings.trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        llm_kwargs = {
            "model": self.settings.model,
            "tokenizer": self.settings.model,
            "tensor_parallel_size": self.settings.tensor_parallel_size,
            "dtype": "bfloat16",
            "trust_remote_code": self.settings.trust_remote_code,
            "gpu_memory_utilization": self.settings.gpu_memory_utilization,
            "seed": self.settings.seed,
        }
        if self.settings.max_model_len:
            llm_kwargs["max_model_len"] = self.settings.max_model_len
        self.llm = LLM(**llm_kwargs)

    def completion_token_length(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text.split()))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_prompt(self, prompt: str) -> str:
        if self.tokenizer is None or not self.settings.max_prompt_tokens:
            return prompt
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.settings.max_prompt_tokens:
            return prompt
        ids = ids[-self.settings.max_prompt_tokens :]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def generate_for_records(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        if self.settings.mock_generation:
            return self._mock_generate_for_records(records)
        if self.settings.backend == "vllm":
            return self._vllm_generate_for_records(records)
        assert self.model is not None and self.tokenizer is not None
        prompts: List[str] = []
        owners: List[str] = []
        for record in records:
            prompt = build_prompt(
                record,
                tokenizer=self.tokenizer,
                prompt_style=self.settings.prompt_style,
                user_template=self.settings.user_template,
            )
            prompt = self.truncate_prompt(prompt)
            for _ in range(self.settings.n_samples):
                prompts.append(prompt)
                owners.append(record["id"])
        return self._generate_prompts(prompts, owners)

    def _vllm_generate_for_records(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        from vllm import SamplingParams

        assert self.llm is not None and self.tokenizer is not None
        prompts = [
            self.truncate_prompt(
                build_prompt(
                    record,
                    tokenizer=self.tokenizer,
                    prompt_style=self.settings.prompt_style,
                    user_template=self.settings.user_template,
                )
            )
            for record in records
        ]
        params = SamplingParams(
            n=self.settings.n_samples,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            max_tokens=self.settings.max_new_tokens,
            seed=self.settings.seed,
        )
        results = self.llm.generate(prompts, params, use_tqdm=False)
        outputs_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for record, result in zip(records, results):
            row = []
            for out in result.outputs:
                text = out.text
                token_ids = getattr(out, "token_ids", None) or []
                finish_reason = getattr(out, "finish_reason", None)
                row.append(
                    {
                        "completion": text,
                        "completion_length_tokens": int(len(token_ids)) if token_ids else self.completion_token_length(text),
                        "completion_length_chars": int(len(text)),
                        "finish_reason": finish_reason or "unknown",
                    }
                )
            outputs_by_id[record["id"]] = row
        return outputs_by_id

    def _generate_prompts(self, prompts: List[str], owners: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        import torch

        outputs_by_id: Dict[str, List[Dict[str, Any]]] = {owner: [] for owner in owners}
        mini_batch_size = max(1, self.settings.batch_size * self.settings.n_samples)
        for start in range(0, len(prompts), mini_batch_size):
            batch_prompts = prompts[start : start + mini_batch_size]
            batch_owners = owners[start : start + mini_batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    max_new_tokens=self.settings.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            for seq, input_len, owner in zip(generated, input_lengths, batch_owners):
                completion_ids = seq[int(input_len) :]
                text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                finish_reason = "eos_or_max_length"
                if self.tokenizer.eos_token_id is not None and len(completion_ids) > 0:
                    finish_reason = "eos" if int(completion_ids[-1]) == self.tokenizer.eos_token_id else "max_length"
                outputs_by_id.setdefault(owner, []).append(
                    {
                        "completion": text,
                        "completion_length_tokens": int(len(completion_ids)),
                        "completion_length_chars": int(len(text)),
                        "finish_reason": finish_reason,
                    }
                )
        return outputs_by_id

    def _mock_generate_for_records(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        outputs: Dict[str, List[Dict[str, Any]]] = {}
        rng = random.Random(self.settings.seed)
        for record in records:
            gold = str(record.get("gold_answer") or "0")
            row = []
            for sample_idx in range(self.settings.n_samples):
                if sample_idx % 3 == 0:
                    answer = gold
                elif sample_idx % 3 == 1:
                    answer = "0"
                else:
                    answer = f"{rng.randint(1, 9)}"
                reasoning = " ".join(["reasoning"] * (5 + sample_idx * 3))
                text = f"<think>\n{reasoning}\n</think>\n<answer>\n\\boxed{{{answer}}}\n</answer>"
                row.append(
                    {
                        "completion": text,
                        "completion_length_tokens": self.completion_token_length(text),
                        "completion_length_chars": len(text),
                        "finish_reason": "mock",
                    }
                )
            outputs[record["id"]] = row
        return outputs

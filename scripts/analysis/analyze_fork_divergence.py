#!/usr/bin/env python
"""Analyze multi-seed rollout divergence by task/example."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tina.analysis.plan_utils import infer_phase_for_char_position
from tina.analysis.rollout_utils import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", nargs="+", required=True)
    parser.add_argument("--output_jsonl", required=True)
    return parser.parse_args()


def chunks(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [part.strip() for part in parts if part.strip()]


def first_divergence(items):
    chunk_lists = [chunks(row.get("completion") or "") for row in items]
    min_len = min((len(c) for c in chunk_lists), default=0)
    idx = 0
    while idx < min_len and len({c[idx] for c in chunk_lists}) == 1:
        idx += 1
    ratios = []
    snippets = []
    phases = []
    for row, chunk_list in zip(items, chunk_lists):
        text = row.get("completion") or ""
        snippet = chunk_list[idx] if idx < len(chunk_list) else ""
        snippets.append(snippet[:240])
        char_pos = text.find(snippet) if snippet else len(text)
        ratios.append(char_pos / max(1, len(text)))
        phases.append(infer_phase_for_char_position(text, char_pos))
    return {
        "first_divergence_ratio": sum(ratios) / max(1, len(ratios)),
        "divergence_in_plan": any(phase == "plan" for phase in phases),
        "snippets": snippets,
    }


def main():
    args = parse_args()
    groups = defaultdict(list)
    for path in args.input_jsonl:
        for row in read_jsonl(path):
            key = (row.get("task"), row.get("example_id") or f"index:{row.get('index')}")
            groups[key].append(row)
    outputs = []
    for (task, example_id), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        div = first_divergence(items)
        outputs.append(
            {
                "task": task,
                "example_id": example_id,
                "num_rollouts": len(items),
                "correct_pattern": [bool(row.get("correct")) for row in items],
                **div,
            }
        )
    write_jsonl(args.output_jsonl, outputs)


if __name__ == "__main__":
    main()

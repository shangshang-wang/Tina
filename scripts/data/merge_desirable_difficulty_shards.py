#!/usr/bin/env python
"""Merge probe shard JSONL files and sort kept prompts by desirability."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preferred-correct", default="2,3,4")
    return parser.parse_args()


def desirability(row: dict, preferred: set[int]) -> tuple[int, float, float]:
    probe = row.get("probe", {})
    correct_count = int(probe.get("correct_count", 0))
    preferred_rank = 0 if correct_count in preferred else 1
    return (preferred_rank, -float(probe.get("reward_std", 0.0)), -float(probe.get("format_rate", 0.0)))


def main() -> None:
    args = parse_args()
    preferred = {int(x) for x in args.preferred_correct.split(",") if x.strip()}
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_glob!r}")

    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: desirability(row, preferred))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "num_rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

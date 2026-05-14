"""Streaming JSONL helpers and dataset field normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


ID_FIELDS = ("id", "problem_id", "uid", "question_id")
PROBLEM_FIELDS = ("problem", "question", "prompt", "instruction", "query", "题目", "问题")
ANSWER_FIELDS = ("answer", "ground_truth", "solution", "final_answer", "target", "答案")


def read_jsonl(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a UTF-8 JSONL file."""
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "_load_error": f"json_decode_error line={line_no}: {exc}",
                    "_line_no": line_no,
                    "_raw_line": line,
                }
                continue
            yield obj
            count += 1
            if limit is not None and count >= limit:
                break


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def first_present(row: Dict[str, Any], fields: Iterable[str]) -> Tuple[Optional[str], Any]:
    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return field, value
    return None, None


def stable_generated_id(row: Dict[str, Any], index: int) -> str:
    basis = json.dumps(row, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{index}:{basis}".encode("utf-8")).hexdigest()[:16]
    return f"auto_{index}_{digest}"


def normalize_record(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Return a normalized wrapper while preserving the original sample."""
    id_field, sample_id = first_present(row, ID_FIELDS)
    problem_field, problem = first_present(row, PROBLEM_FIELDS)
    answer_field, gold = first_present(row, ANSWER_FIELDS)
    if sample_id is None:
        sample_id = stable_generated_id(row, index)
        id_field = None
    return {
        "id": str(sample_id),
        "problem": "" if problem is None else str(problem),
        "gold_answer": "" if gold is None else str(gold),
        "id_field": id_field,
        "problem_field": problem_field,
        "answer_field": answer_field,
        "source": row,
        "index": index,
        "missing_problem": problem is None,
        "missing_answer": gold is None,
        "load_error": row.get("_load_error"),
    }


def iter_normalized_records(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    for index, row in enumerate(read_records(path, limit=limit)):
        yield normalize_record(row, index)


def read_records(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield records from JSONL, parquet, or Arrow dataset files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        yield from read_jsonl(path, limit=limit)
        return
    if suffix == ".parquet":
        try:
            from datasets import Dataset

            ds = Dataset.from_parquet(str(path))
            count = len(ds) if limit is None else min(limit, len(ds))
            for i in range(count):
                yield dict(ds[i])
            return
        except Exception:
            try:
                import pandas as pd

                for index, row in enumerate(pd.read_parquet(path).to_dict(orient="records")):
                    if limit is not None and index >= limit:
                        break
                    yield row
                return
            except Exception as exc:
                yield {"_load_error": f"parquet_load_error: {exc}", "_line_no": 0}
                return
    if suffix == ".arrow":
        try:
            from datasets import Dataset

            ds = Dataset.from_file(str(path))
            count = len(ds) if limit is None else min(limit, len(ds))
            for i in range(count):
                yield dict(ds[i])
            return
        except Exception as exc:
            yield {"_load_error": f"arrow_load_error: {exc}", "_line_no": 0}
            return
    yield from read_jsonl(path, limit=limit)


def load_completed_rollout_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            completed.add(str(sample_id))
    return completed


def load_jsonl_by_id(path: str | Path, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in iter_normalized_records(path, limit=limit)}


def merge_metadata(source: Dict[str, Any], rl_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the original record with rl_signal metadata attached."""
    out = dict(source)
    metadata = dict(out.get("metadata") or {})
    existing = dict(metadata.get("rl_signal") or {})
    existing.update(rl_signal)
    metadata["rl_signal"] = existing
    out["metadata"] = metadata
    return out

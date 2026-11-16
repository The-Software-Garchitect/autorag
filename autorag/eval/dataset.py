"""Load gold questions, and fold reviewed feedback into the next tuning run."""

from __future__ import annotations

import json
from pathlib import Path

from autorag.models import EvalItem


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    file = Path(path)
    if not file.exists():
        return records
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    file.write_text(body + ("\n" if body else ""), encoding="utf-8")


def load_eval_items(path: str | Path) -> list[EvalItem]:
    return [EvalItem.from_dict(raw) for raw in read_jsonl(path)]


def find_eval_path(source: str | Path) -> Path | None:
    path = Path(source)
    if path.is_file() and path.suffix == ".jsonl":
        return path
    if path.is_dir():
        candidate = path / "eval.jsonl"
        if candidate.exists():
            return candidate
        parent = path.parent / "eval.jsonl"
        if parent.exists() and path.name == "docs":
            return parent
    return None


def merge_feedback(items: list[EvalItem], feedback_rows: list[dict]) -> list[EvalItem]:
    """Thumbs-up and corrected thumbs-down become extra graded questions.

    A thumbs-down without a correction is kept as a rejected answer: the next
    run scores a point when it stops repeating that answer, and zero when it
    repeats it almost verbatim.
    """
    merged = list(items)
    for index, row in enumerate(feedback_rows):
        rating = row.get("rating")
        question = (row.get("question") or "").strip()
        if not question:
            continue
        doc_ids = list(row.get("doc_ids") or [])
        if rating == "up" and row.get("answer"):
            merged.append(
                EvalItem(
                    id=f"feedback-up-{index}",
                    question=question,
                    answer=row["answer"],
                    source_docs=doc_ids,
                    qtype="feedback",
                )
            )
        elif rating == "down" and row.get("correction"):
            merged.append(
                EvalItem(
                    id=f"feedback-fix-{index}",
                    question=question,
                    answer=row["correction"],
                    source_docs=doc_ids,
                    qtype="feedback",
                )
            )
        elif rating == "down" and row.get("answer"):
            merged.append(
                EvalItem(
                    id=f"feedback-down-{index}",
                    question=question,
                    answer=row["answer"],
                    source_docs=doc_ids,
                    qtype="rejected",
                    rejected=True,
                )
            )
    return merged

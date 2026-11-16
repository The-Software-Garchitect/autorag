"""Thumbs up / thumbs down, stored for the next tuning run."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                rating TEXT NOT NULL,
                correction TEXT,
                doc_ids TEXT,
                comment TEXT
            )
            """
        )
        self.conn.commit()

    def add(
        self,
        question: str,
        rating: str,
        answer: str | None = None,
        correction: str | None = None,
        doc_ids: list[str] | None = None,
        comment: str | None = None,
    ) -> dict:
        if rating not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        record = {
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "question": question.strip(),
            "answer": answer,
            "rating": rating,
            "correction": correction,
            "doc_ids": list(doc_ids or []),
            "comment": comment,
        }
        self.conn.execute(
            """
            INSERT INTO feedback (id, created_at, question, answer, rating, correction, doc_ids, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["created_at"],
                record["question"],
                record["answer"],
                record["rating"],
                record["correction"],
                json.dumps(record["doc_ids"]),
                record["comment"],
            ),
        )
        self.conn.commit()
        return record

    def list_all(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM feedback ORDER BY created_at ASC").fetchall()
        records = []
        for row in rows:
            records.append(
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "rating": row["rating"],
                    "correction": row["correction"],
                    "doc_ids": json.loads(row["doc_ids"] or "[]"),
                    "comment": row["comment"],
                }
            )
        return records

    def close(self) -> None:
        self.conn.close()

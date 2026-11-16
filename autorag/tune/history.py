"""Persist tuning runs in SQLite and as a JSON snapshot the demo can read."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source TEXT,
                name TEXT,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                is_baseline INTEGER NOT NULL,
                promoted INTEGER NOT NULL,
                question_count INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, record: dict) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (id, created_at, source, name, config_json, metrics_json, is_baseline, promoted, question_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["created_at"],
                record.get("source", ""),
                record.get("name", ""),
                json.dumps(record["config"]),
                json.dumps(record["metrics"]),
                1 if record.get("is_baseline") else 0,
                1 if record.get("promoted") else 0,
                int(record.get("question_count") or record["metrics"].get("n") or 0),
            ),
        )
        self.conn.commit()

    def list_runs(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_at ASC").fetchall()
        return [_row_to_record(row) for row in rows]

    def close(self) -> None:
        self.conn.close()


def new_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{name}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def publish_snapshot(path: str | Path, payload: dict) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_snapshot(path: str | Path) -> dict | None:
    file = Path(path)
    if not file.exists():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


def _row_to_record(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source": row["source"],
        "name": row["name"],
        "config": json.loads(row["config_json"]),
        "metrics": json.loads(row["metrics_json"]),
        "is_baseline": bool(row["is_baseline"]),
        "promoted": bool(row["promoted"]),
        "question_count": row["question_count"],
    }

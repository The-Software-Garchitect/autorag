"""Thumbs up and thumbs down are stored, then folded into the next tune."""

import json
from pathlib import Path

from autorag.config import PipelineConfig
from autorag.eval.dataset import merge_feedback
from autorag.eval.metrics import score_question
from autorag.models import Answer, EvalItem
from autorag.tune.feedback import FeedbackStore
from autorag.tune.loop import tune


def test_feedback_store_roundtrip(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    saved = store.add(
        "What is the team snack expense limit?",
        "up",
        answer="The team snack expense limit is $12 per person.",
        doc_ids=["snacks"],
    )
    assert saved["rating"] == "up"
    rows = store.list_all()
    assert rows[0]["doc_ids"] == ["snacks"]
    store.add(
        "What is the team snack expense limit?",
        "down",
        answer="The team snack expense limit is $4 per person.",
    )
    assert [row["rating"] for row in store.list_all()] == ["up", "down"]
    store.close()


def test_merge_feedback_adds_gold_and_rejected_items():
    base = [EvalItem("1", "Original?", "Original answer.", ["rooms"])]
    rows = [
        {
            "question": "What is the team snack expense limit?",
            "answer": "The team snack expense limit is $12 per person.",
            "rating": "up",
            "doc_ids": ["snacks"],
        },
        {
            "question": "What is the team snack expense limit?",
            "answer": "The team snack expense limit is $4 per person.",
            "rating": "down",
            "correction": "The team snack expense limit is $12 per person.",
            "doc_ids": ["snacks"],
        },
        {
            "question": "Bad repeat?",
            "answer": "The limit is $4 per person.",
            "rating": "down",
            "doc_ids": [],
        },
    ]
    merged = merge_feedback(base, rows)
    assert len(merged) == 4
    assert merged[1].qtype == "feedback"
    assert merged[2].answer.endswith("$12 per person.")
    assert merged[3].rejected is True
    repeat = Answer("q", "The limit is $4 per person.", [], 0.5, False)
    assert score_question(merged[3], repeat, []).correctness == 0.0


def test_next_tune_counts_the_thumbs_up(corpus_dir, tmp_path: Path):
    questions = [
        {
            "id": "room",
            "question": "What is the Austin office quiet room named?",
            "answer": "The Austin office quiet room is named Cypress.",
            "source_docs": ["rooms"],
            "type": "factual",
        }
    ]
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps(questions[0]) + "\n", encoding="utf-8")
    config = PipelineConfig(
        name="baseline",
        embedding_provider="hashing",
        embedding_model="hashing",
        chunk_strategy="structure",
        chunk_size=80,
        top_k=2,
        hybrid=False,
        reranker="off",
        recency_boost=True,
        generator="extractive",
    )
    feedback = [
        {
            "question": "What is the team snack expense limit?",
            "answer": "The team snack expense limit is $12 per person.",
            "rating": "up",
            "doc_ids": ["snacks"],
        }
    ]
    payload = tune(
        source=str(corpus_dir),
        runs_dir=tmp_path / "runs",
        eval_path=str(eval_path),
        configs=[config],
        feedback_rows=feedback,
        log=lambda *_args, **_kwargs: None,
    )
    assert payload["question_count"] == 2

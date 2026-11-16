import json
from pathlib import Path

from autorag.config import PipelineConfig
from autorag.pipeline.ingest import load_corpus
from autorag.tune.finetune import build_pairs
from autorag.tune.loop import tune
from autorag.models import EvalItem


def test_tune_promotes_only_a_better_config(corpus_dir, tmp_path: Path):
    questions = [
        {
            "id": "snack",
            "question": "What is the team snack expense limit?",
            "answer": "The team snack expense limit is $12 per person.",
            "source_docs": ["snacks"],
            "type": "conflict",
        },
        {
            "id": "room",
            "question": "What is the Austin office quiet room named?",
            "answer": "The Austin office quiet room is named Cypress.",
            "source_docs": ["rooms"],
            "type": "factual",
        },
        {
            "id": "pet",
            "question": "Does the handbook include pet insurance for employees?",
            "answer": "I can't find that in the handbook.",
            "source_docs": [],
            "type": "unanswerable",
            "unanswerable": True,
        },
    ]
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8")
    base = PipelineConfig(
        name="baseline",
        embedding_provider="hashing",
        embedding_model="hashing",
        chunk_strategy="structure",
        chunk_size=80,
        top_k=1,
        candidate_k=4,
        hybrid=False,
        reranker="off",
        recency_boost=False,
        generator="extractive",
    )
    better = PipelineConfig(
        name="recency",
        embedding_provider="hashing",
        embedding_model="hashing",
        chunk_strategy="structure",
        chunk_size=80,
        top_k=3,
        candidate_k=6,
        hybrid=True,
        reranker="lexical",
        recency_boost=True,
        generator="extractive",
    )
    payload = tune(
        source=str(corpus_dir),
        runs_dir=tmp_path / "runs",
        eval_path=str(eval_path),
        configs=[base, better],
        log=lambda *_args, **_kwargs: None,
    )
    assert payload["runs"][0]["name"] == "baseline"
    assert payload["best"]["name"] == "recency"
    assert payload["improvement"] > 0
    assert payload["runs"][1]["promoted"] is True
    assert (tmp_path / "runs" / "latest.json").exists()


def test_finetune_pairs_use_the_gold_passage(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    pairs = build_pairs(
        docs,
        [
            EvalItem(
                "1",
                "What is the team snack expense limit?",
                "The team snack expense limit is $12 per person.",
                ["snacks"],
            )
        ],
    )
    assert pairs and pairs[0][0].startswith("What is")
    assert "12" in pairs[0][1]

from pathlib import Path

from autorag.config import PipelineConfig
from autorag.eval.metrics import WEIGHTS, aggregate, score_question
from autorag.eval.synthetic import synthesize
from autorag.models import Answer, EvalItem
from autorag.pipeline.ingest import load_corpus
from autorag.tune.feedback import FeedbackStore
from autorag.tune.finetune import build_pairs
from autorag.tune.loop import tune


def test_correctness_rewards_the_gold_number_and_refusal():
    good = Answer("q", "The limit is $12 per person.", [], 0.8, False)
    bad = Answer("q", "The limit is $4 per person.", [], 0.8, False)
    refused = Answer("q", "I can't find that in the handbook.", [], 0.6, True)
    item = EvalItem("1", "q", "The limit is $12 per person.", ["snacks"])
    unknown = EvalItem("2", "pet insurance?", "no", [], unanswerable=True)

    good_score = score_question(item, good, [])
    bad_score = score_question(item, bad, [])
    assert good_score.correctness > 0.9
    assert bad_score.correctness < good_score.correctness
    assert score_question(unknown, refused, []).correctness == 1.0
    assert score_question(unknown, good, []).correctness == 0.0


def test_composite_uses_the_documented_weights():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    report = aggregate([], [])
    assert report.n == 0


def test_rejected_feedback_fails_when_the_bad_answer_repeats():
    item = EvalItem("d", "q", "The limit is $4 per person.", [], qtype="rejected", rejected=True)
    repeat = Answer("q", "The limit is $4 per person.", [], 0.5, False)
    changed = Answer("q", "The limit is $12 per person.", [], 0.5, False)
    assert score_question(item, repeat, []).correctness == 0.0
    assert score_question(item, changed, []).correctness == 1.0


def test_tune_promotes_only_a_better_config(corpus_dir, tmp_path: Path):
    docs = load_corpus(str(corpus_dir))
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
    import json

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
    assert payload["best"]["metrics"]["composite"] >= payload["baseline"]["metrics"]["composite"]
    # The stale snack policy outranks the current one until recency is on.
    assert payload["improvement"] > 0
    assert payload["best"]["name"] == "recency"
    assert (tmp_path / "runs" / "latest.json").exists()


def test_synthetic_questions_quote_the_corpus(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    items = synthesize(docs, per_doc=1)
    assert items
    blob = "\n".join(doc.text for doc in docs)
    assert any(item.answer.rstrip(".") in blob for item in items)


def test_feedback_roundtrip_and_pairs(tmp_path, corpus_dir):
    store = FeedbackStore(tmp_path / "fb.sqlite")
    saved = store.add("What is the snack limit?", "up", answer="The limit is $12.", doc_ids=["snacks"])
    assert saved["rating"] == "up"
    assert store.list_all()[0]["doc_ids"] == ["snacks"]
    store.close()

    docs = load_corpus(str(corpus_dir))
    pairs = build_pairs(
        docs,
        [EvalItem("1", "What is the team snack expense limit?", "The team snack expense limit is $12 per person.", ["snacks"])],
    )
    assert pairs and pairs[0][0].startswith("What is")
    assert "12" in pairs[0][1]

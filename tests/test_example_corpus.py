"""A second corpus runs through tune and ask with no Acme-specific code."""

from pathlib import Path

from autorag.config import PipelineConfig
from autorag.pipeline.embed import HashingEmbedder
from autorag.pipeline.engine import build_bundle
from autorag.pipeline.ingest import load_corpus
from autorag.tune.loop import tune

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "harbor-kitchen"


def test_example_corpus_marks_the_old_menu_from_front_matter():
    docs = {doc.id: doc for doc in load_corpus(str(EXAMPLE))}
    assert "menu-2023" in docs and "menu-2025" in docs
    assert docs["menu-2023"].metadata["status"] == "superseded"
    assert docs["menu-2023"].metadata["superseded_by"] == "menu-2025"
    assert (EXAMPLE / "eval.jsonl").exists()


def test_example_corpus_tunes_and_answers(tmp_path):
    base = PipelineConfig(
        name="baseline",
        embedding_provider="hashing",
        embedding_model="hashing",
        chunk_strategy="fixed",
        chunk_size=40,
        chunk_overlap=0,
        top_k=1,
        candidate_k=4,
        hybrid=False,
        reranker="off",
        recency_boost=False,
    )
    better = PipelineConfig(
        name="current",
        embedding_provider="hashing",
        embedding_model="hashing",
        chunk_strategy="structure",
        chunk_size=200,
        chunk_overlap=20,
        top_k=4,
        candidate_k=8,
        hybrid=True,
        reranker="lexical",
        recency_boost=True,
    )
    payload = tune(
        source=str(EXAMPLE),
        runs_dir=tmp_path / "runs",
        eval_path=str(EXAMPLE / "eval.jsonl"),
        configs=[base, better],
        limit=12,
        log=lambda *_a, **_k: None,
        write_source_snapshot=False,
    )
    assert payload["question_count"] == 12
    assert payload["best"]["metrics"]["composite"] >= payload["baseline"]["metrics"]["composite"]

    docs = load_corpus(str(EXAMPLE))
    bundle = build_bundle(docs, better, HashingEmbedder())
    answer = bundle.ask("The harbor burger is what amount and comes with fries.")
    assert answer.answer
    assert not answer.refused
    assert answer.citations

from autorag.config import PipelineConfig
from autorag.pipeline.embed import HashingEmbedder
from autorag.pipeline.engine import build_bundle
from autorag.pipeline.ingest import load_corpus


def test_recency_prefers_the_current_policy(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    embedder = HashingEmbedder()
    question = "What is the team snack expense limit?"

    stale = build_bundle(
        docs,
        PipelineConfig(
            name="stale",
            embedding_provider="hashing",
            chunk_strategy="structure",
            chunk_size=120,
            top_k=2,
            hybrid=False,
            reranker="off",
            recency_boost=False,
        ),
        embedder,
    )
    current = build_bundle(
        docs,
        PipelineConfig(
            name="current",
            embedding_provider="hashing",
            chunk_strategy="structure",
            chunk_size=120,
            top_k=2,
            hybrid=True,
            reranker="lexical",
            recency_boost=True,
        ),
        embedder,
    )
    stale_answer = stale.ask(question)
    current_answer = current.ask(question)
    assert "4" in stale_answer.answer
    assert "12" in current_answer.answer
    assert current_answer.citations[0].doc_id == "snacks"
    assert current_answer.citations[0].status == "current"
    assert not current_answer.refused


def test_unanswerable_question_is_refused(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    bundle = build_bundle(
        docs,
        PipelineConfig(embedding_provider="hashing", recency_boost=True, chunk_strategy="structure"),
        HashingEmbedder(),
    )
    answer = bundle.ask("Does the handbook include pet insurance for employees?")
    assert answer.refused
    assert answer.citations == []


def test_hybrid_and_vector_both_return_chunks(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    embedder = HashingEmbedder()
    bundle = build_bundle(
        docs,
        PipelineConfig(
            embedding_provider="hashing",
            hybrid=True,
            top_k=3,
            recency_boost=True,
            chunk_strategy="structure",
        ),
        embedder,
    )
    answer = bundle.ask("What is the Austin office quiet room named?")
    assert "Cypress" in answer.answer
    assert answer.citations[0].doc_id == "rooms"

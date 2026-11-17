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


def test_hybrid_search_finds_the_quiet_room(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    bundle = build_bundle(
        docs,
        PipelineConfig(
            embedding_provider="hashing",
            hybrid=True,
            top_k=3,
            recency_boost=True,
            chunk_strategy="structure",
        ),
        HashingEmbedder(),
    )
    answer = bundle.ask("What is the Austin office quiet room named?")
    assert "Cypress" in answer.answer
    assert answer.citations[0].doc_id == "rooms"


def test_bm25_retriever_can_disagree_with_the_vector_ranking():
    import numpy as np

    from autorag.models import Chunk
    from autorag.pipeline.retrieve import build_bm25, retrieve

    chunks = [
        Chunk(id="a:0", doc_id="a", title="A", text="alpha alpha alpha", index=0, source="a.md", metadata={}),
        Chunk(id="b:0", doc_id="b", title="B", text="zebra", index=0, source="b.md", metadata={}),
    ]
    bm25 = build_bm25(chunks)
    vectors = np.eye(2, dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    keyword = PipelineConfig(
        retriever="bm25", hybrid=False, reranker="off", top_k=1, candidate_k=2, recency_boost=False
    )
    vector = PipelineConfig(
        retriever="vector", hybrid=False, reranker="off", top_k=1, candidate_k=2, recency_boost=False
    )
    assert retrieve("zebra", chunks, vectors, bm25, query, keyword)[0].chunk.doc_id == "b"
    assert retrieve("zebra", chunks, vectors, bm25, query, vector)[0].chunk.doc_id == "a"

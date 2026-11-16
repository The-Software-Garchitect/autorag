"""Pipeline configuration.

A config is the set of knobs the tuner is allowed to turn: how the handbook is
cut into chunks, which embedding model reads them, how many chunks are retrieved,
and whether keyword search, a reranker, or a preference for current documents is on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class PipelineConfig:
    name: str = "default"
    chunk_strategy: str = "structure"  # "fixed" or "structure"
    chunk_size: int = 360  # whitespace-separated words
    chunk_overlap: int = 60
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_provider: str = "local"  # local | openai | hashing | ollama
    top_k: int = 5
    candidate_k: int = 24
    hybrid: bool = True
    reranker: str = "lexical"  # off | lexical | cross-encoder
    recency_boost: bool = True
    generator: str = "extractive"  # extractive | ollama | openai | auto
    rrf_k: int = 60

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineConfig":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})

    def index_key(self) -> tuple:
        """Settings that change the chunks or vectors. Retrieval knobs are excluded."""
        return (
            self.chunk_strategy,
            self.chunk_size,
            self.chunk_overlap,
            self.embedding_provider,
            self.embedding_model,
        )


def baseline_config(embedding_model: str | None = None) -> PipelineConfig:
    """A deliberately weak starting point: small hard cuts, vector search only.

    The tuner has to beat this. It is not the configuration the API serves
    before any run exists; that one is :func:`default_config`.
    """
    cfg = PipelineConfig(
        name="baseline",
        chunk_strategy="fixed",
        chunk_size=80,
        chunk_overlap=0,
        top_k=2,
        candidate_k=8,
        hybrid=False,
        reranker="off",
        recency_boost=False,
        generator="extractive",
    )
    if embedding_model:
        cfg.embedding_model = embedding_model
    return cfg


def default_config(embedding_model: str | None = None) -> PipelineConfig:
    """Sensible production settings used before a tuning run has promoted a winner."""
    cfg = PipelineConfig(
        name="default",
        chunk_strategy="structure",
        chunk_size=360,
        chunk_overlap=60,
        top_k=5,
        candidate_k=24,
        hybrid=True,
        reranker="lexical",
        recency_boost=True,
        generator="extractive",
    )
    if embedding_model:
        cfg.embedding_model = embedding_model
    return cfg


def search_space(embedding_model: str) -> list[PipelineConfig]:
    """Ordered configs the default tune compares. Baseline is always first.

    The list is a designed search, not an exhaustive grid: each entry changes
    the levers that usually matter (chunk boundaries, hybrid search, reranking,
    how many passages are kept, and whether superseded documents are demoted).
    """
    model = embedding_model

    def cfg(**kwargs) -> PipelineConfig:
        base = dict(
            embedding_model=model,
            embedding_provider="local",
            generator="extractive",
            candidate_k=24,
        )
        base.update(kwargs)
        return PipelineConfig(**base)

    return [
        baseline_config(model),
        cfg(
            name="recency",
            chunk_strategy="fixed",
            chunk_size=80,
            chunk_overlap=0,
            top_k=4,
            candidate_k=12,
            hybrid=False,
            reranker="off",
            recency_boost=True,
        ),
        cfg(
            name="structure",
            chunk_strategy="structure",
            chunk_size=360,
            chunk_overlap=60,
            top_k=5,
            hybrid=False,
            reranker="off",
            recency_boost=True,
        ),
        cfg(
            name="hybrid",
            chunk_strategy="structure",
            chunk_size=360,
            chunk_overlap=60,
            top_k=5,
            hybrid=True,
            reranker="off",
            recency_boost=True,
        ),
        cfg(
            name="rerank",
            chunk_strategy="structure",
            chunk_size=360,
            chunk_overlap=60,
            top_k=5,
            hybrid=True,
            reranker="lexical",
            recency_boost=True,
        ),
        cfg(
            name="wider-chunks",
            chunk_strategy="structure",
            chunk_size=520,
            chunk_overlap=80,
            top_k=5,
            hybrid=True,
            reranker="lexical",
            recency_boost=True,
        ),
        cfg(
            name="high-k",
            chunk_strategy="structure",
            chunk_size=360,
            chunk_overlap=60,
            top_k=8,
            hybrid=True,
            reranker="lexical",
            recency_boost=True,
        ),
    ]

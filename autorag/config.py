"""Pipeline configuration and the optional project file.

``autorag.yaml`` names the folder to read. ``--source`` overrides that file.
A config is the set of knobs the tuner may turn: how pages are cut, which
embedding model reads them, how many passages come back, and whether keyword
search or a preference for current pages is on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    name: str = "default"
    chunk_strategy: str = "structure"  # "fixed" or "structure"
    chunk_size: int = 360  # whitespace-separated words
    chunk_overlap: int = 60
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_provider: str = "local"  # local | fastembed | hashing
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


def load_settings(path: str | Path = "autorag.yaml") -> dict:
    """Read the project file. Missing file means an empty dict, not an error."""
    file = Path(path)
    if not file.exists():
        return {}
    loaded = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{file} must be a mapping")
    return loaded


def configured_source(settings: dict | None = None) -> str:
    settings = load_settings() if settings is None else settings
    return str(settings.get("source") or "data/acme")


def configured_title(source: str, settings: dict | None = None) -> str:
    settings = load_settings() if settings is None else settings
    local = Path(source) / "autorag.yaml"
    if local.exists():
        extra = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        if isinstance(extra, dict) and extra.get("title"):
            return str(extra["title"])
    if str(settings.get("source") or "") == source and settings.get("title"):
        return str(settings["title"])
    return Path(source).name.replace("-", " ").replace("_", " ").title()


def configured_generator(settings: dict | None = None) -> str:
    settings = load_settings() if settings is None else settings
    return str(settings.get("generator") or "extractive")


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

"""Embedding providers.

The local provider is ``sentence-transformers`` (a small MiniLM model by default).
OpenAI and Ollama are available when their environment is configured. Tests use
the hashing provider, which needs no download and still rewards shared words.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from autorag.textutil import content_tokens, tokens

_ST_MODELS: dict[str, object] = {}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Bag-of-words vectors. Useful in tests and as a clearly weaker baseline model."""

    def __init__(self, dim: int = 64, name: str = "hashing"):
        self.dim = dim
        self.name = name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = content_tokens(text) or tokens(text)
            for word in words:
                vectors[row, _bucket(word, self.dim)] += 1.0
            for left, right in zip(words, words[1:]):
                vectors[row, _bucket(left + "_" + right, self.dim)] += 0.5
        return _normalize_rows(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    """ONNX MiniLM via fastembed. Same model family as the local PyTorch embedder, without torch.

    This is the runtime used by the free-host image. ``model`` can be injected in tests.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model=None):
        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        if model is not None:
            self._model = model
        else:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        return _normalize_rows(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.name = model_name
        self._model = _sentence_transformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = int(self._model.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(name=model_name or "hashing")
    if provider == "fastembed":
        return FastEmbedEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    if provider == "local":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedding provider: {provider}")


def _sentence_transformer(model_name: str):
    if model_name not in _ST_MODELS:
        from sentence_transformers import SentenceTransformer

        _ST_MODELS[model_name] = SentenceTransformer(model_name)
    return _ST_MODELS[model_name]


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return vectors / norms

"""Embedding providers.

The local provider is ``sentence-transformers`` (a small MiniLM model by default).
OpenAI and Ollama are available when their environment is configured. Tests use
the hashing provider, which needs no download and still rewards shared words.
"""

from __future__ import annotations

import hashlib
import os
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


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, api_key: str | None = None, client=None):
        self.model = model or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.name = f"openai:{self.model}"
        self._client = client

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if not self.api_key and self._client is None:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = self._client or _httpx_client()
        rows: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            rows.extend(item["embedding"] for item in data)
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None, client=None):
        self.model = model or os.environ.get("OLLAMA_EMBED_MODEL") or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.name = f"ollama:{self.model}"
        self._client = client

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        client = self._client or _httpx_client()
        rows = [self._one(client, text) for text in texts]
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _one(self, client, text: str) -> list[float]:
        response = client.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": text},
            timeout=60,
        )
        if response.status_code == 404:
            response = client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        if "embeddings" in payload:
            return payload["embeddings"][0]
        return payload["embedding"]


def build_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(name=model_name or "hashing")
    if provider == "openai":
        return OpenAIEmbedder(model=model_name if model_name.startswith("text-embedding") else None)
    if provider == "ollama":
        return OllamaEmbedder(model=model_name or None)
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


def _httpx_client():
    import httpx

    return httpx.Client(timeout=60)

"""A small local vector index.

Normalized vectors live in memory and can be written next to a run. Cosine
similarity is a dot product after normalization, which is enough for a few
thousand passages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autorag.models import Chunk


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks and {len(vectors)} vectors.")
        self.chunks = chunks
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks or k <= 0:
            return []
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ query
        return _top_k(scores, k)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        import json

        payload = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "text": chunk.text,
                "index": chunk.index,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]
        (path / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        import json

        path = Path(directory)
        vectors = np.load(path / "vectors.npy")
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload]
        return cls(chunks, vectors)


def build_vector_index(chunks: list[Chunk], vectors: np.ndarray) -> VectorIndex:
    return VectorIndex(chunks, vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return (vectors / norms).astype(np.float32)


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    count = len(scores)
    k = min(k, count)
    if k == count:
        order = np.argsort(-scores)
    else:
        partial = np.argpartition(-scores, k - 1)[:k]
        order = partial[np.argsort(-scores[partial])]
    return [(int(index), float(scores[index])) for index in order]

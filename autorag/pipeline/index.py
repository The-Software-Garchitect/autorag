"""Local vector indexes.

The default store keeps normalized vectors in memory and can write them beside
the run directory. Cosine similarity is a dot product after normalization, which
is plenty for a handbook-sized collection (hundreds to a few thousand chunks).

Chroma is an optional backend (``pip install autorag[chroma]`` and
``AUTORAG_INDEX=chroma``) with the same ``chunks`` + ``search`` surface.
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


class ChromaVectorIndex:
    """Thin wrapper so retrieval can treat Chroma like :class:`VectorIndex`."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, path: str, collection: str = "autorag"):
        import chromadb

        self.chunks = chunks
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self.client = chromadb.PersistentClient(path=path)
        try:
            self.client.delete_collection(collection)
        except Exception:
            pass
        self.collection = self.client.create_collection(collection, metadata={"hnsw:space": "cosine"})
        if chunks:
            self.collection.add(
                ids=[chunk.id for chunk in chunks],
                embeddings=self.vectors.tolist(),
                documents=[chunk.text for chunk in chunks],
                metadatas=[{"doc_id": chunk.doc_id, "title": chunk.title} for chunk in chunks],
            )
        self._position = {chunk.id: index for index, chunk in enumerate(chunks)}

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks or k <= 0:
            return []
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        result = self.collection.query(query_embeddings=[query.tolist()], n_results=min(k, len(self.chunks)))
        hits: list[tuple[int, float]] = []
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        for chunk_id, distance in zip(ids[0], distances[0]):
            # Chroma cosine space returns a distance; similarity is 1 - distance.
            hits.append((self._position[chunk_id], float(1.0 - distance)))
        return hits


def build_vector_index(chunks: list[Chunk], vectors: np.ndarray, backend: str = "local", path: str | None = None):
    if backend == "chroma":
        if not path:
            raise ValueError("Chroma needs a directory path.")
        return ChromaVectorIndex(chunks, vectors, path=path)
    if backend != "local":
        raise ValueError(f"Unknown index backend: {backend}")
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

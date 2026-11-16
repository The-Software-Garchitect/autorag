"""Retrieve a handful of chunks for a question.

Vector search finds passages that are about the same thing even when the wording
differs. BM25 finds passages that share rare words ("AnyConnect", a form number).
Hybrid search merges the two rankings. A recency boost then demotes superseded
and unofficial copies so an old policy does not outrank the current one.
"""

from __future__ import annotations

import math

import numpy as np

from autorag.config import PipelineConfig
from autorag.models import ScoredChunk
from autorag.textutil import content_tokens

STALE_STATUS = {"superseded", "archived", "outdated"}
UNOFFICIAL_STATUS = {"unofficial", "draft"}
DUPLICATE_STATUS = {"duplicate"}


class BM25Index:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.k1 = k1
        self.b = b
        self.count = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        self.avgdl = (sum(self.doc_len) / self.count) if self.count else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in documents:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.tf.append(counts)
            for token in counts:
                df[token] = df.get(token, 0) + 1
        self.idf = {
            token: math.log(1 + (self.count - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        output = np.zeros(self.count, dtype=np.float32)
        terms = list(dict.fromkeys(token for token in query if token in self.idf))
        if not terms or self.count == 0:
            return output
        for index, counts in enumerate(self.tf):
            length = self.doc_len[index] or 1
            score = 0.0
            for token in terms:
                freq = counts.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                score += self.idf[token] * (freq * (self.k1 + 1)) / denom
            output[index] = score
        return output


def build_bm25(chunks) -> BM25Index:
    return BM25Index([content_tokens(chunk.text) for chunk in chunks])


def status_multiplier(status: str | None, enabled: bool) -> float:
    if not enabled:
        return 1.0
    label = (status or "current").lower()
    if label == "current":
        return 1.25
    if label in STALE_STATUS:
        return 0.12
    if label in UNOFFICIAL_STATUS:
        return 0.18
    if label in DUPLICATE_STATUS:
        return 0.72
    return 1.0


def retrieve(question: str, chunks, vectors: np.ndarray, bm25: BM25Index, query_vector: np.ndarray, config: PipelineConfig) -> list[ScoredChunk]:
    if not chunks:
        return []
    query = np.asarray(query_vector, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-9)
    vector_scores = np.maximum(vectors @ query, 0.0)
    bm25_scores = bm25.scores(content_tokens(question))

    if config.hybrid:
        base = _rrf(vector_scores, bm25_scores, config.rrf_k)
    else:
        base = vector_scores

    multipliers = np.array(
        [status_multiplier(chunk.metadata.get("status"), config.recency_boost) for chunk in chunks],
        dtype=np.float32,
    )
    combined = base * multipliers
    pool = _arg_top(combined, min(config.candidate_k, len(chunks)))

    if config.reranker == "lexical":
        pool = _lexical_rerank(question, pool, chunks, vector_scores, config.recency_boost)
    elif config.reranker == "cross-encoder":
        pool = _cross_encoder_rerank(question, pool, chunks, config.recency_boost)

    chosen = pool[: config.top_k]
    results: list[ScoredChunk] = []
    for index in chosen:
        results.append(
            ScoredChunk(
                chunk=chunks[index],
                score=float(combined[index]),
                vector_score=float(vector_scores[index]),
                bm25_score=float(bm25_scores[index]),
            )
        )
    return results


def _rrf(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    return (1.0 / (k + left_rank)) + (1.0 / (k + right_rank))


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _arg_top(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    if k >= len(scores):
        order = np.argsort(-scores)
        return [int(index) for index in order]
    partial = np.argpartition(-scores, k - 1)[:k]
    order = partial[np.argsort(-scores[partial])]
    return [int(index) for index in order]


def _lexical_rerank(question, pool, chunks, vector_scores, recency: bool) -> list[int]:
    query_terms = set(content_tokens(question))
    if not pool:
        return []
    values = np.array([vector_scores[index] for index in pool], dtype=np.float32)
    peak = float(values.max()) or 1.0
    scored: list[tuple[float, int]] = []
    for index, vector_score in zip(pool, values):
        coverage = 0.0
        if query_terms:
            have = set(content_tokens(chunks[index].text))
            coverage = sum(1 for term in query_terms if term in have) / len(query_terms)
        score = (0.45 * (float(vector_score) / peak) + 0.55 * coverage)
        score *= status_multiplier(chunks[index].metadata.get("status"), recency)
        scored.append((score, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored]


_CROSS = None


def _cross_encoder_rerank(question, pool, chunks, recency: bool) -> list[int]:
    global _CROSS
    if not pool:
        return []
    try:
        if _CROSS is None:
            from sentence_transformers import CrossEncoder

            _CROSS = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(question, chunks[index].text[:1800]) for index in pool]
        raw = np.asarray(_CROSS.predict(pairs, show_progress_bar=False), dtype=np.float32)
    except Exception:
        return _lexical_rerank(question, pool, chunks, np.ones(len(chunks), dtype=np.float32), recency)
    scored = []
    for index, score in zip(pool, raw):
        adjusted = float(score) * status_multiplier(chunks[index].metadata.get("status"), recency)
        scored.append((adjusted, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored]

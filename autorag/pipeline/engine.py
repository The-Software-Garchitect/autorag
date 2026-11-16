"""Glue that indexes a corpus once and answers questions with a chosen config."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from autorag.config import PipelineConfig
from autorag.models import Answer, Chunk, Document
from autorag.pipeline.chunk import chunk_documents
from autorag.pipeline.embed import Embedder, build_embedder
from autorag.pipeline.generate import generate
from autorag.pipeline.index import VectorIndex, build_vector_index
from autorag.pipeline.retrieve import BM25Index, build_bm25, retrieve


@dataclass
class Bundle:
    config: PipelineConfig
    chunks: list[Chunk]
    vectors: np.ndarray
    bm25: BM25Index
    embedder: Embedder

    def ask(self, question: str, generator: str | None = None) -> Answer:
        answer, _retrieved = self.ask_all([question], generator=generator)
        return answer[0]

    def ask_all(self, questions: list[str], generator: str | None = None):
        if not questions:
            return [], []
        query_vectors = self.embedder.embed_documents(questions)
        answers = []
        retrieved_sets = []
        mode = generator or self.config.generator
        for question, query_vector in zip(questions, query_vectors):
            retrieved = retrieve(
                question,
                self.chunks,
                self.vectors,
                self.bm25,
                query_vector,
                self.config,
            )
            answers.append(generate(question, retrieved, self.config, mode=mode))
            retrieved_sets.append(retrieved)
        return answers, retrieved_sets


def bundle_from_index(config: PipelineConfig, index: VectorIndex, embedder: Embedder) -> Bundle:
    """Wrap a saved index. Queries are still embedded; the passages are not."""
    return Bundle(
        config=config,
        chunks=index.chunks,
        vectors=index.vectors,
        bm25=build_bm25(index.chunks),
        embedder=embedder,
    )


def load_precomputed_bundle(directory: str | os.PathLike, config: PipelineConfig, embedder: Embedder) -> Bundle:
    return bundle_from_index(config, VectorIndex.load(directory), embedder)


def build_bundle(documents: list[Document], config: PipelineConfig, embedder: Embedder | None = None) -> Bundle:
    embedder = embedder or build_embedder(config.embedding_provider, config.embedding_model)
    chunks = chunk_documents(documents, config)
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed_documents(texts) if texts else np.zeros((0, 1), dtype=np.float32)
    index = build_vector_index(chunks, vectors)
    return Bundle(
        config=config,
        chunks=index.chunks,
        vectors=index.vectors,
        bm25=build_bm25(index.chunks),
        embedder=embedder,
    )

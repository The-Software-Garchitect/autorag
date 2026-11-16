"""The lexical reranker moves an exact policy sentence ahead of a vague neighbor."""

import numpy as np

from autorag.models import Chunk
from autorag.pipeline.retrieve import _lexical_rerank


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(id=doc_id, doc_id=doc_id, title=doc_id, text=text, index=0, source=f"{doc_id}.md", metadata={"status": "current"})


def test_lexical_rerank_promotes_the_passage_that_uses_the_question_words():
    chunks = [
        _chunk("vague", "People sometimes talk about time away from work in general terms."),
        _chunk("exact", "The team snack expense limit is $12 per person."),
    ]
    vector_scores = np.array([0.95, 0.40], dtype=np.float32)
    order = _lexical_rerank(
        "What is the team snack expense limit?",
        [0, 1],
        chunks,
        vector_scores,
        recency=False,
    )
    assert order[0] == 1

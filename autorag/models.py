"""Shared records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    index: int
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0


@dataclass
class Citation:
    doc_id: str
    title: str
    snippet: str
    source: str
    score: float
    status: str = "current"
    effective_date: str | None = None

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "score": round(float(self.score), 4),
            "status": self.status,
            "effective_date": self.effective_date,
        }


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool
    generator: str = "extractive"

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.as_dict() for c in self.citations],
            "confidence": round(float(self.confidence), 4),
            "refused": self.refused,
            "generator": self.generator,
        }


@dataclass
class EvalItem:
    id: str
    question: str
    answer: str
    source_docs: list[str]
    qtype: str = "factual"
    unanswerable: bool = False
    rejected: bool = False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "source_docs": list(self.source_docs),
            "type": self.qtype,
            "unanswerable": self.unanswerable,
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalItem":
        return cls(
            id=str(raw.get("id") or raw["question"][:40]),
            question=raw["question"].strip(),
            answer=(raw.get("answer") or "").strip(),
            source_docs=list(raw.get("source_docs") or raw.get("sources") or []),
            qtype=str(raw.get("type") or raw.get("qtype") or "factual"),
            unanswerable=bool(raw.get("unanswerable", False)),
            rejected=bool(raw.get("rejected", False)),
        )

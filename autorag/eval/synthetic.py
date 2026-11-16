"""Build a starter eval set for a folder that does not have one yet.

When an OpenAI key or a running Ollama server is available, the model writes
questions from each passage. Otherwise a heuristic writer turns sentences that
state a number or a concrete rule into questions. Heuristic questions are a
scaffold for tuning, not a replacement for a human-written gold set.
"""

from __future__ import annotations

import json
import re

from autorag.models import Document, EvalItem
from autorag.pipeline.chunk import chunk_documents
from autorag.config import default_config
from autorag.textutil import content_tokens, extract_numbers, split_sentences


def synthesize(documents: list[Document], complete=None, per_doc: int = 3) -> list[EvalItem]:
    if complete is not None:
        items = _from_llm(documents, complete, per_doc=per_doc)
        if items:
            return items
    return _heuristic(documents, per_doc=per_doc)


def _heuristic(documents: list[Document], per_doc: int) -> list[EvalItem]:
    items: list[EvalItem] = []
    config = default_config()
    config.chunk_size = 220
    chunks = chunk_documents(documents, config)
    seen_docs: dict[str, int] = {}
    for chunk in chunks:
        status = str(chunk.metadata.get("status") or "current").lower()
        if status in {"superseded", "archived", "outdated"}:
            continue
        if seen_docs.get(chunk.doc_id, 0) >= per_doc:
            continue
        for sentence in split_sentences(chunk.text):
            if seen_docs.get(chunk.doc_id, 0) >= per_doc:
                break
            words = content_tokens(sentence)
            if len(words) < 6:
                continue
            if chunk.title and sentence.lower().startswith(chunk.title.lower()):
                continue
            question = _question_for(chunk.title, sentence)
            if not question:
                continue
            number = seen_docs.get(chunk.doc_id, 0) + 1
            seen_docs[chunk.doc_id] = number
            items.append(
                EvalItem(
                    id=f"synth-{chunk.doc_id}-{number}",
                    question=question,
                    answer=sentence if sentence.endswith(".") else sentence + ".",
                    source_docs=[chunk.doc_id],
                    qtype="synthetic",
                )
            )
    return items


def _question_for(title: str, sentence: str) -> str | None:
    numbers = extract_numbers(sentence)
    topic = title.strip() or "this section"
    if numbers:
        shown = next(iter(numbers))
        masked = re.sub(rf"\b{re.escape(shown)}\b", "what amount", sentence, count=1)
        return f"According to {topic}, {masked[0].lower() + masked[1:] if masked else masked}"
    words = content_tokens(sentence)
    if not words:
        return None
    focus = " ".join(words[:3])
    return f"What does {topic} say about {focus}?"


def _from_llm(documents: list[Document], complete, per_doc: int) -> list[EvalItem]:
    items: list[EvalItem] = []
    for document in documents:
        passage = document.text.strip()
        if len(passage) < 80:
            continue
        prompt = (
            f"Write {per_doc} questions a new employee would ask that this passage answers.\n"
            "Reply with one JSON object per line and no other text: "
            '{"question": "...", "answer": "..."}\n'
            "The answer must be a sentence copied from the passage.\n\n"
            f"Title: {document.title}\n\n{passage[:1800]}"
        )
        try:
            raw = complete(prompt)
        except Exception:
            continue
        for line in raw.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                start = line.find("{")
                end = line.rfind("}")
                if start == -1 or end == -1:
                    continue
                line = line[start : end + 1]
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = str(payload.get("question") or "").strip()
            answer = str(payload.get("answer") or "").strip()
            if question and answer:
                items.append(
                    EvalItem(
                        id=f"synth-{document.id}-{len(items)+1}",
                        question=question,
                        answer=answer,
                        source_docs=[document.id],
                        qtype="synthetic",
                    )
                )
    return items

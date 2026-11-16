"""Turn retrieved chunks into an answer with citations.

The default generator is extractive: it quotes the handbook sentence (or the
linearized table row) that best matches the question, and it refuses when no
passage covers the question's wording. That keeps answers faithful and lets the
tuner run with no API key and no local chat model.

Ollama and OpenAI use the same retrieved passages but write the reply themselves.
They are selected when ``generator`` is ``ollama`` or ``openai``, or ``auto``
and that backend is reachable.
"""

from __future__ import annotations

import os

from autorag.config import PipelineConfig
from autorag.models import Answer, Citation, ScoredChunk
from autorag.textutil import REFUSAL_TEXT, is_refusal_text, term_overlap, units_from_text

# A second passage is included only when its retrieval-weighted score is close
# to the best passage. Questions that truly need two policies can clear that
# bar; a merely related paragraph in the same section usually cannot.
WEIGHTED_GAP = 0.10
MIN_OVERLAP = 0.52
SECOND_OVERLAP = 0.50


def generate(
    question: str,
    retrieved: list[ScoredChunk],
    config: PipelineConfig,
    mode: str | None = None,
    complete=None,
) -> Answer:
    resolved = resolve_generator(mode or config.generator, complete=complete)
    if resolved == "extractive" or complete is None and resolved == "extractive":
        return generate_extractive(question, retrieved)
    if complete is None:
        complete = _complete_for(resolved)
        if complete is None:
            return generate_extractive(question, retrieved)
    return generate_with_llm(question, retrieved, complete, generator_name=resolved)


def resolve_generator(mode: str, complete=None) -> str:
    if mode == "extractive":
        return "extractive"
    if mode == "openai":
        return "openai" if os.environ.get("OPENAI_API_KEY") or complete else "extractive"
    if mode == "ollama":
        return "ollama"
    if mode == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if ollama_reachable():
            return "ollama"
        return "extractive"
    return "extractive"


def select_unit_texts(candidates: list[tuple[float, float, str]]) -> list[str]:
    """Pick one or two passages.

    ``candidates`` is ``(raw overlap, retrieval-weighted score, text)``, best first.
    """
    if not candidates or candidates[0][0] < MIN_OVERLAP:
        return []
    best_weighted = candidates[0][1]
    chosen = [candidates[0][2]]
    seen = {candidates[0][2]}
    for overlap, weighted, unit in candidates[1:]:
        if overlap < SECOND_OVERLAP or weighted < best_weighted - WEIGHTED_GAP:
            continue
        if unit in seen:
            continue
        chosen.append(unit)
        seen.add(unit)
        if len(chosen) == 2:
            break
    return chosen


def generate_extractive(question: str, retrieved: list[ScoredChunk]) -> Answer:
    scored = _score_units(question, retrieved)
    chosen_texts = select_unit_texts([(overlap, weighted, unit) for overlap, weighted, unit, _ in scored])
    if not chosen_texts:
        return _refusal(question, retrieved, "extractive")
    chosen = []
    for text in chosen_texts:
        for overlap, weighted, unit, chunk in scored:
            if unit == text:
                chosen.append((overlap, weighted, unit, chunk))
                break
    best_overlap = chosen[0][0]

    answer_text = " ".join(_finish(unit) for _, _, unit, _ in chosen)
    citations = _citations_for(chosen)
    confidence = max(0.2, min(0.97, 0.18 + 0.78 * best_overlap))
    return Answer(
        question=question,
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        refused=False,
        generator="extractive",
    )


def generate_with_llm(question: str, retrieved: list[ScoredChunk], complete, generator_name: str) -> Answer:
    if not retrieved:
        return _refusal(question, retrieved, generator_name)
    blocks = []
    for index, scored in enumerate(retrieved, start=1):
        status = scored.chunk.metadata.get("status", "current")
        blocks.append(
            f"[{index}] {scored.chunk.title} ({status}, {scored.chunk.doc_id})\n{scored.chunk.text[:1600]}"
        )
    prompt = (
        "Answer the employee question using only the handbook passages.\n"
        "If the passages do not contain the answer, reply exactly: "
        f"{REFUSAL_TEXT}\n"
        "Do not invent numbers, dates, or names. Prefer a current passage over a superseded one.\n\n"
        f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(blocks)
    )
    text = complete(prompt).strip()
    refused = is_refusal_text(text)
    citations = [] if refused else _citations_from_retrieved(retrieved[:3])
    overlap = term_overlap(question, retrieved[0].chunk.text) if retrieved else 0.0
    confidence = 0.55 if refused else max(0.25, min(0.9, 0.3 + 0.6 * overlap))
    return Answer(
        question=question,
        answer=text or REFUSAL_TEXT,
        citations=citations,
        confidence=confidence,
        refused=refused or not text,
        generator=generator_name,
    )


def rank_weight(rank: int) -> float:
    return 1.0 / (1.0 + 0.65 * rank)


def _score_units(question: str, retrieved: list[ScoredChunk]) -> list[tuple[float, float, str, ScoredChunk]]:
    ranked: list[tuple[float, float, str, ScoredChunk]] = []
    for rank, scored in enumerate(retrieved):
        weight = rank_weight(rank)
        for unit in units_from_text(scored.chunk.text):
            overlap = term_overlap(question, unit)
            if overlap <= 0:
                continue
            ranked.append((overlap, overlap * weight, unit, scored))
    ranked.sort(key=lambda item: (item[1], len(item[2])), reverse=True)
    return ranked


def _finish(unit: str) -> str:
    text = unit.strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _citations_for(chosen) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for overlap, _score, unit, scored in chosen:
        chunk = scored.chunk
        if chunk.id in seen:
            existing = next(item for item in citations if item.doc_id == chunk.doc_id and item.snippet)
            if unit not in existing.snippet:
                existing.snippet = (existing.snippet + " " + unit).strip()
            continue
        seen.add(chunk.id)
        citations.append(_citation(scored, unit, overlap))
    return citations


def _citations_from_retrieved(retrieved: list[ScoredChunk]) -> list[Citation]:
    citations = []
    seen = set()
    for scored in retrieved:
        if scored.chunk.doc_id in seen:
            continue
        seen.add(scored.chunk.doc_id)
        snippet = scored.chunk.text.strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:317] + "..."
        citations.append(_citation(scored, snippet, scored.vector_score))
    return citations


def _citation(scored: ScoredChunk, snippet: str, score: float) -> Citation:
    chunk = scored.chunk
    return Citation(
        doc_id=chunk.doc_id,
        title=chunk.title,
        snippet=snippet.strip(),
        source=chunk.source,
        score=float(score),
        status=str(chunk.metadata.get("status") or "current"),
        effective_date=_date(chunk.metadata.get("effective_date")),
    )


def _date(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _refusal(question: str, retrieved: list[ScoredChunk], generator_name: str) -> Answer:
    overlap = 0.0
    if retrieved:
        overlap = max(term_overlap(question, item.chunk.text) for item in retrieved)
    confidence = max(0.35, min(0.9, 1.0 - overlap))
    return Answer(
        question=question,
        answer=REFUSAL_TEXT,
        citations=[],
        confidence=confidence,
        refused=True,
        generator=generator_name,
    )


_OLLAMA_OK: bool | None = None


def ollama_reachable() -> bool:
    global _OLLAMA_OK
    if _OLLAMA_OK is not None:
        return _OLLAMA_OK
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        import httpx

        response = httpx.get(host + "/api/tags", timeout=0.4)
        _OLLAMA_OK = response.status_code == 200
    except Exception:
        _OLLAMA_OK = False
    return _OLLAMA_OK


def _complete_for(mode: str):
    if mode == "openai":
        return OpenAIChat().complete
    if mode == "ollama":
        return OllamaChat().complete
    return None


class OpenAIChat:
    def __init__(self, model: str | None = None, api_key: str | None = None, client=None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = client

    def complete(self, prompt: str) -> str:
        import httpx

        client = self._client or httpx.Client(timeout=90)
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer employee questions strictly from the handbook passages provided.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class OllamaChat:
    def __init__(self, model: str | None = None, host: str | None = None, client=None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self._client = client

    def complete(self, prompt: str) -> str:
        import httpx

        client = self._client or httpx.Client(timeout=120)
        response = client.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer employee questions strictly from the handbook passages provided.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        if "message" in payload:
            return payload["message"].get("content", "")
        return payload.get("response", "")

"""Check one claim against an indexed corpus.

Retrieval is the same bundle the question-answering API uses. The verdict is a
separate step on the sentences that came back.
"""

from __future__ import annotations

import json
from pathlib import Path

from autorag.claims.records import VerdictThresholds
from autorag.claims.verdict import (
    NliScorer,
    decide_from_probabilities,
    extractive_verdict,
    select_verifier,
)
from autorag.config import PipelineConfig
from autorag.pipeline.engine import Bundle
from autorag.pipeline.retrieve import retrieve

_SCORER: NliScorer | None = None


def evidence_payload(retrieved) -> list[dict]:
    rows = []
    for hit in retrieved:
        sentence_index = hit.chunk.metadata.get("sentence_index")
        row = {
            "doc_id": hit.chunk.doc_id,
            "title": hit.chunk.title,
            "text": hit.chunk.text,
            "score": round(float(hit.score), 4),
        }
        if sentence_index is not None:
            row["sentence_index"] = int(sentence_index)
        rows.append(row)
    return rows


def check_claim(
    bundle: Bundle,
    claim: str,
    verifier: str = "auto",
    thresholds: VerdictThresholds | None = None,
    scorer: NliScorer | None = None,
) -> dict:
    text = claim.strip()
    if len(text) < 2:
        raise ValueError("claim must be at least 2 characters")
    query = bundle.embedder.embed_query(text)
    retrieved = retrieve(text, bundle.chunks, bundle.vectors, bundle.bm25, query, bundle.config)
    evidence = evidence_payload(retrieved)
    mode = select_verifier(verifier)
    thresholds = thresholds or VerdictThresholds()
    if mode == "nli":
        model = scorer or _shared_scorer()
        probabilities = model.score([(row["text"], text) for row in evidence])
        for row, probability in zip(evidence, probabilities):
            row["nli"] = {key: round(float(value), 4) for key, value in probability.items()}
        verdict, confidence = decide_from_probabilities(probabilities, thresholds)
    elif mode == "majority":
        verdict, confidence = "not-enough-evidence", 0.0
    else:
        verdict, confidence = extractive_verdict(text, [row["text"] for row in evidence])
        mode = "extractive"
    return {
        "claim": text,
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "evidence": evidence,
        "verifier": mode,
        "config": bundle.config.name,
    }


def load_claim_pipeline(source: str | Path) -> PipelineConfig | None:
    """The config selected on the SciFact train split, if this folder has a results file."""
    data = _results(source)
    raw = ((data.get("dev") or {}).get("tuned") or {}).get("config")
    if not isinstance(raw, dict):
        return None
    return PipelineConfig.from_dict(raw)


def load_verdict_thresholds(source: str | Path) -> VerdictThresholds:
    data = _results(source)
    raw = ((data.get("dev") or {}).get("tuned") or {}).get("verdict")
    if not isinstance(raw, dict):
        return VerdictThresholds()
    return VerdictThresholds.from_dict(raw)


def _results(source: str | Path) -> dict:
    path = Path(source) / "results.json"
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _shared_scorer() -> NliScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = NliScorer()
    return _SCORER

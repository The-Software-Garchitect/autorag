"""Read the official SciFact JSONL release and turn it into pipeline files.

SciFact labels a rationale ``SUPPORT`` or ``CONTRADICT``. A claim with no
rationale is not enough information. This module maps those onto the three
verdicts the checker returns. The hidden test split is never read.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from autorag.claims.records import ClaimExample

_SUPPORT = {"SUPPORT", "SUPPORTS"}
_REFUTE = {"REFUTE", "REFUTES", "CONTRADICT", "CONTRADICTION"}


def map_rationale_label(raw: str) -> str:
    key = raw.strip().upper()
    if key in _SUPPORT:
        return "supports"
    if key in _REFUTE:
        return "refutes"
    raise ValueError(f"Unknown SciFact label: {raw}")


def collapse_label(raw_labels: list[str]) -> str:
    """One claim-level verdict. Empty evidence is not enough information."""
    mapped = [map_rationale_label(label) for label in raw_labels if str(label).strip()]
    if not mapped:
        return "not-enough-evidence"
    counts = Counter(mapped)
    if len(counts) == 1:
        return next(iter(counts))
    winner, votes = counts.most_common(1)[0]
    second = counts.most_common(2)[1][1]
    if votes == second:
        return "not-enough-evidence"
    return winner


def compact_sentences(sentences: list[str]) -> tuple[list[str], dict[int, int]]:
    """Drop blank sentences and remember where the remaining ones moved.

    Gold rationale indexes point at the original abstract array. The files we
    write only keep sentences that have text, so the indexes have to be rewritten
    with the same mapping.
    """
    kept: list[str] = []
    mapping: dict[int, int] = {}
    for index, sentence in enumerate(sentences):
        text = (sentence or "").strip()
        if not text:
            continue
        mapping[index] = len(kept)
        kept.append(text)
    return kept, mapping


def render_abstract(doc_id: str, title: str, sentences: list[str]) -> str:
    meta = yaml.safe_dump(
        {"id": str(doc_id), "title": title or str(doc_id)},
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    body = "\n\n".join(sentences)
    return f"---\n{meta}\n---\n\n{body}\n"


def claim_from_row(row: dict, abstracts: dict[str, list[str]], index_maps: dict[str, dict[int, int]]) -> ClaimExample:
    evidence = row.get("evidence") or {}
    raw_labels: list[str] = []
    rationales: list[dict] = []
    for doc_id, items in evidence.items():
        doc_key = str(doc_id)
        mapping = index_maps.get(doc_key, {})
        texts = abstracts.get(doc_key, [])
        for item in items or []:
            raw_labels.append(str(item.get("label") or ""))
            sentences: list[int] = []
            kept_texts: list[str] = []
            for index in item.get("sentences") or []:
                original = int(index)
                mapped = mapping.get(original)
                if mapped is None:
                    continue
                sentences.append(mapped)
                if 0 <= mapped < len(texts):
                    kept_texts.append(texts[mapped])
            rationales.append({"doc_id": doc_key, "sentences": sentences, "texts": kept_texts})
    label = collapse_label(raw_labels)
    docs = sorted({rationale["doc_id"] for rationale in rationales if rationale["sentences"]})
    return ClaimExample(
        id=str(row["id"]),
        claim=str(row["claim"]).strip(),
        label=label,
        evidence_docs=docs,
        rationales=[rationale for rationale in rationales if rationale["sentences"]],
    )


def write_corpus(docs_dir: Path, corpus_rows: list[dict]) -> tuple[dict[str, list[str]], dict[str, dict[int, int]]]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    abstracts: dict[str, list[str]] = {}
    index_maps: dict[str, dict[int, int]] = {}
    for row in corpus_rows:
        doc_id = str(row["doc_id"])
        kept, mapping = compact_sentences(list(row.get("abstract") or []))
        abstracts[doc_id] = kept
        index_maps[doc_id] = mapping
        text = render_abstract(doc_id, str(row.get("title") or ""), kept)
        (docs_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")
    return abstracts, index_maps


def write_claims(path: Path, rows: list[dict], abstracts: dict[str, list[str]], index_maps: dict[str, dict[int, int]]) -> list[ClaimExample]:
    claims = [claim_from_row(row, abstracts, index_maps) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(claim.as_dict(), ensure_ascii=False) + "\n")
    return claims


def load_claims(path: Path) -> list[ClaimExample]:
    claims: list[ClaimExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            claims.append(ClaimExample.from_dict(json.loads(line)))
    return claims

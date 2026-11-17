"""Claim records shared by the fetcher, the checker, and the evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field


LABELS = ("supports", "refutes", "not-enough-evidence")


@dataclass
class ClaimExample:
    id: str
    claim: str
    label: str
    evidence_docs: list[str] = field(default_factory=list)
    rationales: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "label": self.label,
            "evidence_docs": list(self.evidence_docs),
            "rationales": list(self.rationales),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ClaimExample":
        return cls(
            id=str(raw["id"]),
            claim=str(raw["claim"]).strip(),
            label=str(raw["label"]),
            evidence_docs=[str(doc_id) for doc_id in raw.get("evidence_docs") or []],
            rationales=list(raw.get("rationales") or []),
        )

    def gold_sentences(self) -> set[tuple[str, int]]:
        found: set[tuple[str, int]] = set()
        for rationale in self.rationales:
            doc_id = str(rationale.get("doc_id"))
            for index in rationale.get("sentences") or []:
                found.add((doc_id, int(index)))
        return found


@dataclass
class VerdictThresholds:
    """Cuts chosen on the train split. Dev never moves these."""

    support: float = 0.5
    refute: float = 0.5
    margin: float = 0.0

    def as_dict(self) -> dict:
        return {"support": self.support, "refute": self.refute, "margin": self.margin}

    @classmethod
    def from_dict(cls, raw: dict | None) -> "VerdictThresholds":
        raw = raw or {}
        return cls(
            support=float(raw.get("support", 0.5)),
            refute=float(raw.get("refute", 0.5)),
            margin=float(raw.get("margin", 0.0)),
        )

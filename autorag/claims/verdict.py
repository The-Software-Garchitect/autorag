"""Turn retrieved sentences into supports / refutes / not-enough-evidence.

The accurate path is a small public NLI cross-encoder. It needs PyTorch, which
does not fit the 512 MB free-host image, so that image uses the extractive
path: overlap plus a negation check, with no downloaded verifier model.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from autorag.claims.records import VerdictThresholds
from autorag.textutil import term_overlap, tokens

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

_NEGATION = frozenset(
    {
        "no",
        "not",
        "never",
        "without",
        "none",
        "neither",
        "nor",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "wont",
        "nothing",
        "nobody",
    }
)

_NLI: "NliScorer | None" = None


def select_verifier(explicit: str | None = None) -> str:
    """``nli`` when the local model can load, otherwise ``extractive``.

    ``AUTORAG_RUNTIME=lite`` always stays extractive. That is the free-host image.
    """
    choice = (explicit or os.environ.get("AUTORAG_VERIFIER") or "auto").strip().lower()
    if os.environ.get("AUTORAG_RUNTIME") == "lite" and choice in {"", "auto", "nli"}:
        return "extractive"
    if choice in {"extractive", "nli", "majority"}:
        return choice
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "extractive"
    return "nli"


def negation_parity(text: str) -> int:
    count = 0
    for token in tokens(text):
        if token in _NEGATION or token.endswith("nt"):
            count += 1
    return count % 2


def extractive_verdict(claim: str, sentences: Sequence[str]) -> tuple[str, float]:
    """A CPU verdict with no model: shared words, and whether negation disagrees."""
    if not sentences:
        return "not-enough-evidence", 0.0
    best_overlap = -1.0
    best_conflict = False
    for sentence in sentences:
        overlap = term_overlap(claim, sentence)
        conflict = negation_parity(claim) != negation_parity(sentence) and overlap >= 0.34
        if overlap > best_overlap:
            best_overlap = overlap
            best_conflict = conflict
    overlap = max(0.0, best_overlap)
    if overlap < 0.34:
        return "not-enough-evidence", round(1.0 - overlap, 4)
    if best_conflict:
        return "refutes", round(min(0.99, overlap), 4)
    return "supports", round(min(0.99, overlap), 4)


def decide_from_probabilities(
    probabilities: Sequence[dict[str, float]],
    thresholds: VerdictThresholds | None = None,
) -> tuple[str, float]:
    """Pick a verdict from per-sentence entailment and contradiction probabilities."""
    thresholds = thresholds or VerdictThresholds()
    if not probabilities:
        return "not-enough-evidence", 0.0
    support = max(float(row.get("entailment", 0.0)) for row in probabilities)
    refute = max(float(row.get("contradiction", 0.0)) for row in probabilities)
    neutral = max(float(row.get("neutral", 0.0)) for row in probabilities)
    if support >= thresholds.support and support >= refute + thresholds.margin:
        return "supports", round(support, 4)
    if refute >= thresholds.refute and refute >= support + thresholds.margin:
        return "refutes", round(refute, 4)
    return "not-enough-evidence", round(max(neutral, 1.0 - max(support, refute)), 4)


def search_thresholds(
    gold: Sequence[str],
    probabilities: Sequence[Sequence[dict[str, float]]],
) -> tuple[VerdictThresholds, float]:
    """Grid over the train split only. Returns the cuts and their train macro-F1."""
    from autorag.claims.metrics import macro_f1

    best = VerdictThresholds()
    best_score = -1.0
    supports = (0.35, 0.45, 0.55, 0.65, 0.75)
    margins = (0.0, 0.05, 0.1)
    for support_cut in supports:
        for refute_cut in supports:
            for margin in margins:
                thresholds = VerdictThresholds(support_cut, refute_cut, margin)
                predicted = [decide_from_probabilities(rows, thresholds)[0] for rows in probabilities]
                score = macro_f1(gold, predicted)
                if score > best_score:
                    best_score = score
                    best = thresholds
    return best, best_score


class NliScorer:
    """Lazy loader for the public NLI cross-encoder."""

    def __init__(self, model_name: str = NLI_MODEL, model=None):
        self.model_name = model_name
        self._model = model
        self._columns: dict[str, int] | None = None

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict[str, float]]:
        if not pairs:
            return []
        model = self._load()
        columns = self._label_columns(model)
        show = len(pairs) >= 400
        try:
            raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=show, apply_softmax=True)
        except TypeError:
            raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=show)
            raw = _softmax(np.asarray(raw, dtype=np.float32))
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        scored: list[dict[str, float]] = []
        for row in array:
            scored.append({name: float(row[index]) for name, index in columns.items()})
        return scored

    def _load(self):
        global _NLI
        if self._model is not None:
            return self._model
        if _NLI is not None and _NLI.model_name == self.model_name and _NLI._model is not None:
            self._model = _NLI._model
            self._columns = _NLI._columns
            return self._model
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)
        _NLI = self
        return self._model

    def _label_columns(self, model) -> dict[str, int]:
        if self._columns is not None:
            return self._columns
        config = model.model.config
        id2label = {int(key): str(value).lower() for key, value in dict(config.id2label).items()}
        columns: dict[str, int] = {}
        for index, name in id2label.items():
            if "entail" in name:
                columns["entailment"] = index
            elif "contrad" in name:
                columns["contradiction"] = index
            elif "neutral" in name:
                columns["neutral"] = index
        expected = {"entailment", "contradiction", "neutral"}
        if set(columns) != expected:
            raise RuntimeError(f"NLI model {self.model_name} labels were {id2label}")
        self._columns = columns
        return columns


def _softmax(array: np.ndarray) -> np.ndarray:
    shifted = array - array.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

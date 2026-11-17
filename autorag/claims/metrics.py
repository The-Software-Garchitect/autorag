"""Scores for claim checking. Empty gold sets are skipped, not counted as perfect.

Not-enough-evidence claims have no gold abstract and no rationale sentence.
Retrieval recall is therefore reported only on claims that have evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from autorag.claims.records import LABELS


def macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    return classification_report(gold, predicted)["macro_f1"]


def classification_report(gold: Sequence[str], predicted: Sequence[str]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must be the same length")
    per_label = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = sum(left == label and right == label for left, right in zip(gold, predicted))
        fp = sum(left != label and right == label for left, right in zip(gold, predicted))
        fn = sum(left == label and right != label for left, right in zip(gold, predicted))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "support": tp + fn,
        }
        f1s.append(f1)
    accuracy = (sum(left == right for left, right in zip(gold, predicted)) / len(gold)) if gold else 0.0
    return {
        "n": len(gold),
        "accuracy": _round(accuracy),
        "macro_f1": _round(sum(f1s) / len(f1s) if f1s else 0.0),
        "per_label": per_label,
    }


def recall_at_k(golds: Sequence[set], predicted: Sequence[set]) -> dict:
    """Mean recall over examples whose gold set is non-empty."""
    scored = [(gold, pred) for gold, pred in zip(golds, predicted) if gold]
    if not scored:
        return {"recall": 0.0, "n": 0}
    values = [len(gold & pred) / len(gold) for gold, pred in scored]
    return {"recall": _round(sum(values) / len(values)), "n": len(values)}


def latency_summary(seconds: Sequence[float]) -> dict:
    ordered = sorted(float(value) for value in seconds)
    if not ordered:
        return {"n": 0, "mean_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0}
    return {
        "n": len(ordered),
        "mean_seconds": _round(sum(ordered) / len(ordered)),
        "p50_seconds": _round(_percentile(ordered, 0.50)),
        "p95_seconds": _round(_percentile(ordered, 0.95)),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _round(value: float) -> float:
    return round(float(value), 4)

"""Ragas-style scores that run locally, without a judge model.

Ragas normally asks a second LLM whether an answer is faithful and correct.
This pipeline has to score hundreds of questions across several configs with
no paid key, so the same ideas are computed directly:

* **Faithfulness** — share of the answer's content words that appear in the
  retrieved passages. A refusal is treated as faithful (it adds no new claim).
* **Answer correctness** — for a graded question, a blend of "did the reply
  contain the gold fact" (token recall) and "did it contain the gold numbers
  without a pile of unrelated numbers" (number F1). Unanswerable questions
  score 1 only when the pipeline refuses. Rejected feedback scores 1 when the
  pipeline does not repeat the disliked answer.
* **Context recall / precision** — of the documents that should have been
  retrieved, how many were, and how much of what was retrieved was actually
  on that list. Reported on answerable questions.
* **Hit rate / MRR** — classic retrieval stats on answerable questions.

The composite used for promotion is a weighted average of those. Weights live
in :data:`WEIGHTS` so a reader can see what "better" means.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from autorag.models import Answer, EvalItem
from autorag.pipeline.retrieve import ScoredChunk
from autorag.textutil import content_tokens, number_f1, token_f1, token_recall

WEIGHTS = {
    "correctness": 0.34,
    "faithfulness": 0.10,
    "context_recall": 0.18,
    "context_precision": 0.08,
    "hit_rate": 0.15,
    "mrr": 0.15,
}


@dataclass
class QuestionScore:
    id: str
    qtype: str
    correctness: float
    faithfulness: float
    context_recall: float | None
    context_precision: float | None
    hit: float | None
    reciprocal_rank: float | None
    refused: bool


@dataclass
class MetricReport:
    n: int
    correctness: float
    faithfulness: float
    context_recall: float
    context_precision: float
    hit_rate: float
    mrr: float
    refusal_accuracy: float
    composite: float
    by_type: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "correctness": _round(self.correctness),
            "faithfulness": _round(self.faithfulness),
            "context_recall": _round(self.context_recall),
            "context_precision": _round(self.context_precision),
            "hit_rate": _round(self.hit_rate),
            "mrr": _round(self.mrr),
            "refusal_accuracy": _round(self.refusal_accuracy),
            "composite": _round(self.composite),
            "by_type": self.by_type,
            "weights": WEIGHTS,
        }


def score_question(item: EvalItem, answer: Answer, retrieved: list[ScoredChunk]) -> QuestionScore:
    context = "\n".join(hit.chunk.text for hit in retrieved)
    faithfulness = 1.0 if answer.refused else _faithfulness(answer.answer, context)
    retrieved_docs = [hit.chunk.doc_id for hit in retrieved]
    gold = set(item.source_docs)

    if item.unanswerable:
        correctness = 1.0 if answer.refused else 0.0
        return QuestionScore(item.id, item.qtype, correctness, faithfulness, None, None, None, None, answer.refused)

    if item.rejected:
        similarity = token_f1(answer.answer, item.answer)
        correctness = 0.0 if (not answer.refused and similarity >= 0.8) else 1.0
        return QuestionScore(item.id, item.qtype, correctness, faithfulness, None, None, None, None, answer.refused)

    correctness = _correctness(answer.answer, item.answer, answer.refused)
    recall = precision = hit = rr = None
    if gold:
        found = set(retrieved_docs)
        matched = found & gold
        recall = len(matched) / len(gold)
        precision = (len(matched) / len(found)) if found else 0.0
        hit = 1.0 if matched else 0.0
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in gold:
                rr = 1.0 / rank
                break
    return QuestionScore(item.id, item.qtype, correctness, faithfulness, recall, precision, hit, rr, answer.refused)


def aggregate(items: list[EvalItem], scores: list[QuestionScore], previews: list[dict] | None = None) -> MetricReport:
    n = len(scores)
    if n == 0:
        return MetricReport(0, 0, 0, 0, 0, 0, 0, 0, 0)

    correctness = _mean([row.correctness for row in scores])
    faithfulness = _mean([row.faithfulness for row in scores])
    retrieval_rows = [row for row in scores if row.context_recall is not None]
    context_recall = _mean([row.context_recall for row in retrieval_rows]) if retrieval_rows else 0.0
    context_precision = _mean([row.context_precision for row in retrieval_rows]) if retrieval_rows else 0.0
    hit_rate = _mean([row.hit for row in retrieval_rows]) if retrieval_rows else 0.0
    mrr = _mean([row.reciprocal_rank for row in retrieval_rows]) if retrieval_rows else 0.0

    unanswerable = [row for row, item in zip(scores, items) if item.unanswerable]
    refusal_accuracy = _mean([row.correctness for row in unanswerable]) if unanswerable else 0.0

    composite = (
        WEIGHTS["correctness"] * correctness
        + WEIGHTS["faithfulness"] * faithfulness
        + WEIGHTS["context_recall"] * context_recall
        + WEIGHTS["context_precision"] * context_precision
        + WEIGHTS["hit_rate"] * hit_rate
        + WEIGHTS["mrr"] * mrr
    )

    by_type: dict[str, dict] = {}
    grouped: dict[str, list[QuestionScore]] = defaultdict(list)
    for row in scores:
        grouped[row.qtype].append(row)
    for qtype, rows in sorted(grouped.items()):
        by_type[qtype] = {
            "n": len(rows),
            "correctness": _round(_mean([row.correctness for row in rows])),
            "faithfulness": _round(_mean([row.faithfulness for row in rows])),
        }

    failures: list[dict] = []
    if previews:
        ranked = sorted(zip(scores, previews), key=lambda pair: pair[0].correctness)
        for row, preview in ranked:
            if row.correctness >= 0.75:
                break
            failures.append(preview)
            if len(failures) >= 8:
                break

    return MetricReport(
        n=n,
        correctness=correctness,
        faithfulness=faithfulness,
        context_recall=context_recall,
        context_precision=context_precision,
        hit_rate=hit_rate,
        mrr=mrr,
        refusal_accuracy=refusal_accuracy,
        composite=composite,
        by_type=by_type,
        failures=failures,
    )


def _correctness(answer: str, gold: str, refused: bool) -> float:
    if refused:
        return 0.0
    recall = token_recall(answer, gold)
    numbers = number_f1(answer, gold)
    if numbers is None:
        return recall
    return 0.62 * numbers + 0.38 * recall


def _faithfulness(answer: str, context: str) -> float:
    words = content_tokens(answer)
    if not words:
        return 1.0
    have = set(content_tokens(context))
    return sum(1 for word in words if word in have) / len(words)


def _mean(values) -> float:
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _round(value: float) -> float:
    return round(float(value), 4)

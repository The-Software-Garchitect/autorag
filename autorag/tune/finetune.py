"""Optional fine-tune of a small sentence-transformers model on handbook questions.

Positive pairs are (question, passage from the gold source document). Training
uses in-batch negatives (MultipleNegativesRankingLoss): questions from other
rows in the batch count as wrong passages. The result is just another embedding
model path the tuner can score. It is off unless ``autorag tune --finetune``.
"""

from __future__ import annotations

from pathlib import Path

from autorag.models import Document, EvalItem


def build_pairs(documents: list[Document], items: list[EvalItem], limit: int = 400) -> list[tuple[str, str]]:
    by_id = {document.id: document for document in documents}
    pairs: list[tuple[str, str]] = []
    for item in items:
        if item.unanswerable or item.rejected or not item.source_docs:
            continue
        document = by_id.get(item.source_docs[0])
        if document is None:
            continue
        passage = _passage_containing(document.text, item.answer) or document.text[:800]
        pairs.append((item.question, passage))
        if len(pairs) >= limit:
            break
    return pairs


def finetune(
    documents: list[Document],
    items: list[EvalItem],
    output_dir: str | Path,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 1,
    batch_size: int = 16,
    max_steps: int | None = 40,
) -> str:
    pairs = build_pairs(documents, items)
    if len(pairs) < 8:
        raise RuntimeError("Need at least 8 question/passage pairs to fine-tune.")

    from sentence_transformers import InputExample, SentenceTransformer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[question, passage]) for question, passage in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=min(batch_size, len(examples)))
    loss = MultipleNegativesRankingLoss(model)
    steps = max_steps
    if steps is None:
        steps = max(1, len(loader) * epochs)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        steps_per_epoch=min(steps, max(1, len(loader))),
        warmup_steps=min(10, steps),
        show_progress_bar=False,
        output_path=str(output),
        save_best_model=False,
    )
    model.save(str(output))
    return str(output)


def _passage_containing(text: str, answer: str) -> str | None:
    if not answer:
        return None
    needle = answer.strip().rstrip(".")
    if needle and needle in text:
        start = max(0, text.find(needle) - 80)
        return text[start : start + len(needle) + 160]
    return None

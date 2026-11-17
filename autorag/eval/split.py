"""Deterministic held-out splits.

Tuning and reporting have to use different questions. The split is stratified
by question type and ordered by id, so the same file always yields the same
two sets.
"""

from __future__ import annotations

import random
from collections import defaultdict

from autorag.models import EvalItem


def split_holdout(
    items: list[EvalItem],
    fraction: float,
    seed: int = 0,
) -> tuple[list[EvalItem], list[EvalItem]]:
    """Return ``(train, report)``. ``fraction`` is the share that is only reported."""
    if fraction <= 0:
        return list(items), []
    if fraction >= 1:
        raise ValueError("holdout fraction must be in [0, 1)")

    grouped: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        grouped[item.qtype].append(item)

    train: list[EvalItem] = []
    report: list[EvalItem] = []
    for qtype in sorted(grouped):
        rows = sorted(grouped[qtype], key=lambda item: item.id)
        rng = random.Random(f"{seed}:{qtype}")
        rng.shuffle(rows)
        n_report = int(round(len(rows) * fraction))
        if len(rows) >= 5:
            n_report = min(len(rows) - 1, max(1, n_report))
        elif len(rows) >= 2:
            n_report = max(1, min(len(rows) - 1, n_report))
        else:
            n_report = 0
        report.extend(rows[:n_report])
        train.extend(rows[n_report:])
    return train, report

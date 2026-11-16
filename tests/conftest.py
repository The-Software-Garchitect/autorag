"""Tiny handbook used by the stage tests. No model download required."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE = """---
id: snacks
title: Team Snacks
status: current
effective_date: "2025-01-01"
department: People
---

The team snack expense limit is $12 per person.
"""

STALE = """---
id: snacks-2023
title: Team Snacks (2023)
status: superseded
effective_date: "2023-01-01"
department: People
---

The team snack expense limit is $4 per person and this old page repeats the team snack expense limit.
"""

ROOMS = """---
id: rooms
title: Quiet Rooms
status: current
effective_date: "2025-01-01"
---

The Austin office quiet room is named Cypress.
"""

TRAINING = """---
id: training
title: Security Training
status: current
effective_date: "2025-01-01"
---

Finish security training within 5 business days.
"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "snacks.md").write_text(FIXTURE, encoding="utf-8")
    (docs / "snacks-2023.md").write_text(STALE, encoding="utf-8")
    (docs / "rooms.md").write_text(ROOMS, encoding="utf-8")
    (docs / "training.md").write_text(TRAINING, encoding="utf-8")
    return docs

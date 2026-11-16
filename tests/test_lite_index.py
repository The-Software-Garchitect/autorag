"""The saved ONNX index answers the handbook questions the demo leads with."""

from pathlib import Path

import pytest

from autorag.service.app import HandbookService

pytest.importorskip("fastembed")

ROOT = Path(__file__).resolve().parents[1]


def test_lite_runtime_quotes_the_current_pto_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORAG_RUNTIME", "lite")
    service = HandbookService(str(ROOT / "data" / "acme"), str(tmp_path / "runs"))
    service.warmup()
    assert service.ready, service.error
    answer = service.ask("How many PTO days after 3 years?")
    baseline = service.ask("How many PTO days after 3 years?", variant="baseline")
    assert "22 PTO days" in answer["answer"]
    assert "25 PTO days" in baseline["answer"]
    dinner = service.ask("What's the client dinner expense limit?")
    assert "$75" in dinner["answer"]
    refused = service.ask("Does the handbook include pet insurance for employees?")
    assert refused["refused"] is True

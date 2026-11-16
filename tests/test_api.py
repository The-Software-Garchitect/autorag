import time

from fastapi.testclient import TestClient

from autorag.pipeline.embed import HashingEmbedder
from autorag.service.app import create_app


def _client(corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("autorag.service.app.build_embedder", lambda provider, model: HashingEmbedder())
    app = create_app(str(corpus_dir), str(tmp_path / "runs"))
    client = TestClient(app)
    client.__enter__()
    deadline = time.time() + 5
    body = {}
    while time.time() < deadline:
        response = client.get("/health")
        body = response.json()
        if body.get("status") == "ok":
            return client, body
        time.sleep(0.05)
    raise AssertionError(body)


def test_suggested_questions_feedback_and_runs(corpus_dir, tmp_path, monkeypatch):
    client, health = _client(corpus_dir, tmp_path, monkeypatch)
    try:
        assert health["documents"] == 4
        assert client.get("/").status_code == 200

        suggestions = client.get("/suggested-questions")
        assert suggestions.status_code == 200
        questions = suggestions.json()["questions"]
        assert questions and all(isinstance(item, str) for item in questions)

        answer = client.post("/ask", json={"question": "What is the team snack expense limit?"}).json()
        assert "12" in answer["answer"]
        assert answer["citations"]

        feedback = client.post(
            "/feedback",
            json={
                "question": answer["question"],
                "answer": answer["answer"],
                "rating": "up",
                "doc_ids": [answer["citations"][0]["doc_id"]],
            },
        )
        assert feedback.status_code == 200
        assert feedback.json()["stored"] is True

        rejected = client.post(
            "/feedback",
            json={"question": "noise", "answer": "nope", "rating": "sideways"},
        )
        assert rejected.status_code == 400

        runs = client.get("/runs")
        assert runs.status_code == 200
        payload = runs.json()
        assert "runs" in payload
        assert payload["runs"] == []
    finally:
        client.__exit__(None, None, None)


def test_runs_reads_a_published_snapshot(corpus_dir, tmp_path, monkeypatch):
    snapshot = {
        "source": str(corpus_dir),
        "question_count": 3,
        "improvement": 0.0456,
        "baseline": {"name": "baseline", "metrics": {"composite": 0.9018}},
        "best": {"name": "hybrid", "metrics": {"composite": 0.9474}, "config": {"name": "hybrid"}},
        "runs": [
            {"name": "baseline", "metrics": {"composite": 0.9018}, "promoted": True, "is_baseline": True},
            {"name": "hybrid", "metrics": {"composite": 0.9474}, "promoted": True, "is_baseline": False},
        ],
    }
    (corpus_dir / "tune_results.json").write_text(__import__("json").dumps(snapshot), encoding="utf-8")
    client, _health = _client(corpus_dir, tmp_path, monkeypatch)
    try:
        payload = client.get("/runs").json()
        assert [run["name"] for run in payload["runs"]] == ["baseline", "hybrid"]
        assert payload["improvement"] == 0.0456
    finally:
        client.__exit__(None, None, None)

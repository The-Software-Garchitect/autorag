from fastapi.testclient import TestClient

from autorag.pipeline.embed import HashingEmbedder
from autorag.service.app import create_app


def test_ask_feedback_and_runs(corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("autorag.service.app.build_embedder", lambda provider, model: HashingEmbedder())
    app = create_app(str(corpus_dir), str(tmp_path / "runs"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["documents"] == 4

        page = client.get("/")
        assert page.status_code == 200
        assert "Ask the handbook" in page.text

        suggestions = client.get("/suggested-questions")
        assert suggestions.status_code == 200
        assert suggestions.json()["questions"]

        answer = client.post("/ask", json={"question": "What is the team snack expense limit?"}).json()
        assert "12" in answer["answer"]
        assert answer["citations"]
        assert answer["config"]

        baseline = client.post(
            "/ask",
            json={"question": "What is the team snack expense limit?", "variant": "baseline"},
        ).json()
        assert baseline["variant"] == "baseline"

        refused = client.post("/ask", json={"question": "Does the handbook include pet insurance for employees?"}).json()
        assert refused["refused"] is True

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

        runs = client.get("/runs").json()
        assert "runs" in runs


def test_openai_chat_parses_a_mocked_response():
    import httpx

    from autorag.pipeline.generate import OpenAIChat

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = request.read()
        assert b"handbook" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "22 days."}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chat = OpenAIChat(api_key="test-key", client=client)
    assert chat.complete("Answer from the handbook.") == "22 days."

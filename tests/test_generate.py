import httpx

from autorag.config import PipelineConfig
from autorag.pipeline.embed import HashingEmbedder
from autorag.pipeline.engine import build_bundle
from autorag.pipeline.generate import OpenAIChat
from autorag.pipeline.ingest import load_corpus


def test_extractive_generator_refuses_when_the_handbook_is_silent(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    bundle = build_bundle(
        docs,
        PipelineConfig(embedding_provider="hashing", recency_boost=True, chunk_strategy="structure"),
        HashingEmbedder(),
    )
    answer = bundle.ask("Does the handbook include pet insurance for employees?")
    assert answer.refused
    assert answer.citations == []
    assert "handbook" in answer.answer.lower()


def test_openai_chat_parses_a_mocked_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = request.read()
        assert b"handbook" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "22 days."}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chat = OpenAIChat(api_key="test-key", client=client)
    assert chat.complete("Answer from the handbook.") == "22 days."

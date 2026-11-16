import numpy as np

from autorag.pipeline.embed import FastEmbedEmbedder, HashingEmbedder, build_embedder


def test_hashing_embedder_is_normalized_and_rewards_shared_words():
    embedder = HashingEmbedder(dim=32)
    vectors = embedder.embed_documents(
        ["team snack expense limit", "team snack expense limit today", "quiet room cypress"]
    )
    assert vectors.shape == (3, 32)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    same = float(vectors[0] @ vectors[1])
    different = float(vectors[0] @ vectors[2])
    assert same > different
    query = embedder.embed_query("team snack expense limit")
    assert query.shape == (32,)


def test_build_embedder_returns_the_hashing_provider():
    embedder = build_embedder("hashing", "hashing")
    assert embedder.name == "hashing"


class _FakeONNX:
    def embed(self, texts):
        for text in texts:
            vector = np.zeros(8, dtype=np.float32)
            vector[len(text) % 8] = 1.0
            vector[(len(text) + 3) % 8] = 0.2
            yield vector


def test_fastembed_wrapper_uses_the_injected_model():
    embedder = FastEmbedEmbedder(model=_FakeONNX())
    vectors = embedder.embed_documents(["ab", "abcd"])
    assert vectors.shape == (2, 8)
    assert not np.allclose(vectors[0], vectors[1])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert embedder.embed_query("ab").shape == (8,)

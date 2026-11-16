import numpy as np

from autorag.models import Chunk
from autorag.pipeline.index import VectorIndex


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(id=f"{doc_id}:0", doc_id=doc_id, title=doc_id, text=text, index=0, source=f"{doc_id}.md")


def test_search_returns_the_nearest_chunk():
    chunks = [_chunk("a", "pto policy"), _chunk("b", "vpn gateway")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = VectorIndex(chunks, vectors)
    hits = index.search(np.array([0.9, 0.1], dtype=np.float32), k=1)
    assert hits[0][0] == 0
    assert hits[0][1] > 0.9


def test_index_roundtrip(tmp_path):
    chunks = [_chunk("snacks", "The team snack expense limit is $12 per person.")]
    vectors = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    VectorIndex(chunks, vectors).save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.chunks[0].doc_id == "snacks"
    assert loaded.vectors.shape == (1, 3)
    assert "12" in loaded.chunks[0].text


def test_committed_acme_indexes_match_the_onnx_model():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "acme" / "index"
    for name in ("baseline", "hybrid"):
        index = VectorIndex.load(root / name)
        assert len(index.chunks) == len(index.vectors)
        assert index.vectors.shape[1] == 384
        assert len(index.chunks) > 10

"""Claim checking without the SciFact download or the NLI weights."""

import json
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient

from autorag.claims.check import check_claim
from autorag.claims.experiment import run_scifact_experiment
from autorag.claims.fetch import fetch_scifact
from autorag.claims.metrics import classification_report, recall_at_k
from autorag.claims.scifact import claim_from_row, collapse_label, render_abstract
from autorag.claims.verdict import extractive_verdict, select_verifier
from autorag.config import PipelineConfig
from autorag.pipeline.embed import HashingEmbedder
from autorag.pipeline.engine import build_bundle
from autorag.pipeline.ingest import load_corpus


def test_scifact_labels_map_support_and_contradict():
    assert collapse_label([]) == "not-enough-evidence"
    assert collapse_label(["SUPPORT"]) == "supports"
    assert collapse_label(["CONTRADICT"]) == "refutes"
    assert collapse_label(["REFUTE"]) == "refutes"
    abstracts = {"7": ["Alpha lowers fever.", "Beta does not."]}
    maps = {"7": {0: 0, 1: 1}}
    claim = claim_from_row(
        {
            "id": 3,
            "claim": "Alpha lowers fever.",
            "evidence": {"7": [{"sentences": [0], "label": "SUPPORT"}]},
        },
        abstracts,
        maps,
    )
    assert claim.label == "supports"
    assert claim.evidence_docs == ["7"]
    assert claim.gold_sentences() == {("7", 0)}


def test_fetch_writes_abstracts_and_skips_the_hidden_test_split(tmp_path: Path):
    corpus = [
        {"doc_id": 7, "title": "Fever", "abstract": ["Alpha lowers fever.", "", "Beta does not."]},
    ]
    train = [
        {"id": 1, "claim": "Alpha lowers fever.", "evidence": {"7": [{"sentences": [0], "label": "SUPPORT"}]}},
        {"id": 2, "claim": "Nobody measured this.", "evidence": {}, "cited_doc_ids": [7]},
    ]
    dev = [
        {"id": 9, "claim": "Beta does not matter.", "evidence": {"7": [{"sentences": [2], "label": "CONTRADICT"}]}},
    ]
    archive = tmp_path / "data.tar.gz"
    _write_tar(archive, corpus, train, dev)
    # Point fetch at the local archive by dropping it where the downloader would.
    dest = tmp_path / "scifact"
    cache = dest / ".cache"
    cache.mkdir(parents=True)
    archive.replace(cache / "data.tar.gz")
    summary = fetch_scifact(dest, url="https://example.invalid/should-not-download")
    assert summary["documents"] == 1
    assert summary["train"] == 2
    assert summary["dev"] == 1
    assert not (cache / "claims_test.jsonl").exists()
    text = (dest / "docs" / "7.md").read_text(encoding="utf-8")
    # The blank abstract sentence is dropped, and the rationale index follows.
    assert "Alpha lowers fever." in text
    assert "Beta does not." in text
    dev_row = json.loads((cache / "claims_dev.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert dev_row["label"] == "refutes"
    assert dev_row["rationales"][0]["sentences"] == [1]
    (dest / "README.md").write_text("license note that must not be ingested\n", encoding="utf-8")
    docs = load_corpus(str(dest))
    assert [doc.id for doc in docs] == ["7"]


def test_extractive_verdict_uses_overlap_and_negation():
    assert extractive_verdict("Alpha lowers fever", ["Alpha lowers fever in adults."])[0] == "supports"
    assert extractive_verdict("Alpha lowers fever", ["Alpha does not lower fever."])[0] == "refutes"
    verdict, confidence = extractive_verdict("Alpha lowers fever", ["The glacier retreated."])
    assert verdict == "not-enough-evidence"
    assert 0 <= confidence <= 1


def test_classification_and_recall_skip_empty_gold():
    report = classification_report(
        ["supports", "supports", "refutes"],
        ["supports", "refutes", "refutes"],
    )
    assert report["n"] == 3
    assert report["accuracy"] == 0.6667
    assert report["per_label"]["not-enough-evidence"]["f1"] == 0.0
    assert report["macro_f1"] == 0.4444
    recall = recall_at_k([{ "7", "8" }, set(), {"7"}], [{"7"}, {"9"}, {"7", "8"}])
    assert recall == {"recall": 0.75, "n": 2}


def test_check_claim_returns_doc_ids(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    docs = load_corpus(str(tmp_path))
    config = PipelineConfig(
        name="sentences",
        embedding_provider="hashing",
        embedding_model="hashing",
        chunk_strategy="sentence",
        retriever="bm25",
        hybrid=False,
        reranker="off",
        top_k=2,
        candidate_k=4,
        recency_boost=False,
    )
    bundle = build_bundle(docs, config, HashingEmbedder())
    payload = check_claim(bundle, "Alpha lowers fever", verifier="extractive")
    assert payload["verdict"] in {"supports", "refutes", "not-enough-evidence"}
    assert payload["evidence"]
    assert payload["evidence"][0]["doc_id"] == "7"
    assert "sentence_index" in payload["evidence"][0]
    assert payload["verifier"] == "extractive"


def test_experiment_selects_on_train_and_scores_dev_after(tmp_path: Path):
    root = _write_mini_corpus(tmp_path)
    seen: list[str] = []

    class FakeNli:
        def score(self, pairs, batch_size=16):
            rows = []
            for evidence, claim in pairs:
                seen.append(claim)
                if "not" in evidence:
                    rows.append({"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05})
                elif "lowers fever" in evidence:
                    rows.append({"entailment": 0.92, "contradiction": 0.04, "neutral": 0.04})
                else:
                    rows.append({"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9})
            return rows

    configs = [
        PipelineConfig(
            name="bm25-majority",
            embedding_provider="hashing",
            embedding_model="hashing",
            chunk_strategy="sentence",
            retriever="bm25",
            hybrid=False,
            reranker="off",
            top_k=2,
            candidate_k=4,
            recency_boost=False,
        ),
        PipelineConfig(
            name="bm25-nli",
            embedding_provider="hashing",
            embedding_model="hashing",
            chunk_strategy="sentence",
            retriever="bm25",
            hybrid=False,
            reranker="off",
            top_k=2,
            candidate_k=4,
            recency_boost=False,
        ),
    ]
    payload = run_scifact_experiment(root, configs=configs, scorer=FakeNli(), log=lambda *_a, **_k: None)
    assert payload["dev_used_for_selection"] is False
    assert payload["train"]["selected"] == "bm25-nli"
    assert payload["dev"]["baseline"]["verifier"] == "majority"
    assert payload["dev"]["tuned"]["verifier"] == "nli"
    assert payload["dev"]["tuned"]["classification"]["n"] == 1
    assert "DEVONLY" not in seen[0]
    assert any(claim.startswith("DEVONLY") for claim in seen)
    assert (root / "results.json").exists()


def test_lite_runtime_refuses_the_nli_verifier(monkeypatch):
    monkeypatch.setenv("AUTORAG_RUNTIME", "lite")
    monkeypatch.delenv("AUTORAG_VERIFIER", raising=False)
    assert select_verifier("auto") == "extractive"
    assert select_verifier("nli") == "extractive"


def test_render_abstract_keeps_sentence_order():
    text = render_abstract("7", "Fever", ["First.", "Second."])
    assert "id: '7'" in text or "id: \"7\"" in text or "id: '7'" in text or "\nid: '7'\n" in text or "id:" in text
    assert text.index("First.") < text.index("Second.")


def _write_mini_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "scifact"
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "7.md").write_text(
        render_abstract("7", "Fever", ["Alpha lowers fever in adults.", "The control arm was unchanged."]),
        encoding="utf-8",
    )
    (docs / "8.md").write_text(
        render_abstract("8", "Other", ["Beta brightens paint."]),
        encoding="utf-8",
    )
    cache = root / ".cache"
    cache.mkdir()
    train = [
        {
            "id": "1",
            "claim": "Alpha lowers fever",
            "label": "supports",
            "evidence_docs": ["7"],
            "rationales": [{"doc_id": "7", "sentences": [0], "texts": ["Alpha lowers fever in adults."]}],
        },
        {
            "id": "2",
            "claim": "Beta brightens paint",
            "label": "supports",
            "evidence_docs": ["8"],
            "rationales": [{"doc_id": "8", "sentences": [0], "texts": ["Beta brightens paint."]}],
        },
        {
            "id": "3",
            "claim": "Alpha does not lower fever",
            "label": "refutes",
            "evidence_docs": ["7"],
            "rationales": [{"doc_id": "7", "sentences": [0], "texts": ["Alpha lowers fever in adults."]}],
        },
    ]
    dev = [
        {
            "id": "9",
            "claim": "DEVONLY alpha lowers fever",
            "label": "supports",
            "evidence_docs": ["7"],
            "rationales": [{"doc_id": "7", "sentences": [0], "texts": ["Alpha lowers fever in adults."]}],
        }
    ]
    _dump(cache / "claims_train.jsonl", train)
    _dump(cache / "claims_dev.jsonl", dev)
    (root / "README.md").write_text("license note that must not be ingested\n", encoding="utf-8")
    return root


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_tar(archive: Path, corpus, train, dev) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name, rows in (
            ("data/corpus.jsonl", corpus),
            ("data/claims_train.jsonl", train),
            ("data/claims_dev.jsonl", dev),
            ("data/claims_test.jsonl", [{"id": 99, "claim": "hidden"}]),
        ):
            payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, fileobj=__import__("io").BytesIO(payload))


def test_check_claim_on_the_lite_runtime(corpus_dir, tmp_path, monkeypatch):
    """The free-host path has no PyTorch. /check-claim still answers, extractively."""
    import numpy as np

    from autorag.config import baseline_config, default_config
    from autorag.pipeline.index import VectorIndex
    from autorag.service.app import create_app

    monkeypatch.setenv("AUTORAG_RUNTIME", "lite")
    monkeypatch.setenv("AUTORAG_VERIFIER", "nli")
    monkeypatch.setattr("autorag.service.app.build_embedder", lambda provider, model: HashingEmbedder())
    docs = load_corpus(str(corpus_dir))
    embedder = HashingEmbedder()
    for config in (default_config(), baseline_config()):
        config.embedding_provider = "hashing"
        config.embedding_model = "hashing"
        bundle = build_bundle(docs, config, embedder)
        VectorIndex(bundle.chunks, np.asarray(bundle.vectors)).save(corpus_dir / "index" / config.name)
    app = create_app(str(corpus_dir), str(tmp_path / "runs"))
    client = TestClient(app)
    with client:
        deadline = __import__("time").time() + 5
        while __import__("time").time() < deadline:
            health = client.get("/health").json()
            if health.get("status") == "ok":
                break
        else:
            raise AssertionError(health)
        response = client.post("/check-claim", json={"claim": "The team snack expense limit is $12 per person."})
        assert response.status_code == 200
        body = response.json()
        assert body["verifier"] == "extractive"
        assert body["verdict"] in {"supports", "refutes", "not-enough-evidence"}
        assert body["evidence"][0]["doc_id"]


def test_api_check_claim(corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORAG_VERIFIER", "extractive")
    monkeypatch.setattr("autorag.service.app.build_embedder", lambda provider, model: HashingEmbedder())
    from autorag.service.app import create_app

    app = create_app(str(corpus_dir), str(tmp_path / "runs"))
    client = TestClient(app)
    with client:
        deadline = __import__("time").time() + 5
        while __import__("time").time() < deadline:
            health = client.get("/health").json()
            if health.get("status") == "ok":
                break
        else:
            raise AssertionError(health)
        response = client.post("/check-claim", json={"claim": "The team snack expense limit is $12 per person."})
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] in {"supports", "refutes", "not-enough-evidence"}
        assert body["verifier"] == "extractive"
        assert body["evidence"][0]["doc_id"]
        assert 0 <= body["confidence"] <= 1

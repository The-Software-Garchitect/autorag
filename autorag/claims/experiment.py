"""Tune a claim checker on SciFact train and score it once on dev.

Dev claims are not an argument of the search. They are scored after the winner
and its verdict cuts are frozen. The hidden test split is not loaded.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from autorag.claims.check import evidence_payload
from autorag.claims.metrics import classification_report, latency_summary, recall_at_k
from autorag.claims.records import LABELS, ClaimExample, VerdictThresholds
from autorag.claims.scifact import load_claims
from autorag.claims.verdict import NLI_MODEL, NliScorer, decide_from_probabilities, search_thresholds
from autorag.config import PipelineConfig
from autorag.pipeline.embed import Embedder, build_embedder
from autorag.pipeline.engine import Bundle, build_bundle
from autorag.pipeline.ingest import load_corpus
from autorag.pipeline.retrieve import retrieve

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RECALL_K = 5


def claim_search_space(embedding_model: str = EMBEDDING_MODEL, provider: str = "local") -> list[PipelineConfig]:
    """A short list. Baseline is BM25 plus the train majority label. The rest use NLI."""

    def cfg(**kwargs) -> PipelineConfig:
        base = dict(
            embedding_model=embedding_model,
            embedding_provider=provider,
            generator="extractive",
            chunk_strategy="sentence",
            chunk_size=40,
            chunk_overlap=0,
            candidate_k=20,
            recency_boost=False,
            top_k=5,
        )
        base.update(kwargs)
        return PipelineConfig(**base)

    return [
        cfg(name="bm25-majority", retriever="bm25", hybrid=False, reranker="off", top_k=5),
        cfg(name="bm25-nli", retriever="bm25", hybrid=False, reranker="off", top_k=5),
        cfg(name="hybrid-nli", retriever="hybrid", hybrid=True, reranker="lexical", top_k=5),
        cfg(name="hybrid-nli-k8", retriever="hybrid", hybrid=True, reranker="lexical", top_k=8),
    ]


def run_scifact_experiment(
    source: str | Path,
    out: str | Path | None = None,
    configs: list[PipelineConfig] | None = None,
    scorer: NliScorer | None = None,
    log=print,
) -> dict:
    root = Path(source)
    train_path = root / ".cache" / "claims_train.jsonl"
    dev_path = root / ".cache" / "claims_dev.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Missing SciFact cache in {root / '.cache'}. Run `autorag fetch scifact` or `make data-scifact`."
        )
    documents = load_corpus(str(root))
    if not documents:
        raise FileNotFoundError(f"No abstracts in {root}. Run `autorag fetch scifact` first.")
    train = load_claims(train_path)
    dev = load_claims(dev_path)
    space = configs or claim_search_space()
    counts = Counter(claim.label for claim in train)
    majority = counts.most_common(1)[0][0]
    prior = counts[majority] / len(train)
    log(
        f"SciFact train {len(train)} claims, dev {len(dev)} claims, {len(documents)} abstracts. "
        f"Majority label on train is {majority} ({prior:.3f}). Dev is not used to choose a config."
    )

    cache: dict[tuple, Bundle] = {}
    embedders: dict[tuple, Embedder] = {}
    train_rows = []
    for config in space:
        verifier = "majority" if config.name == "bm25-majority" else "nli"
        log(f"  [{config.name}] retriever={config.retriever or ('hybrid' if config.hybrid else 'vector')} k={config.top_k}")
        bundle = _cached_bundle(documents, config, cache, embedders)
        retrieved = [_retrieve(bundle, claim.claim) for claim in train]
        thresholds = VerdictThresholds()
        if verifier == "nli":
            model = scorer or NliScorer()
            probabilities = _score_retrieved(model, train, retrieved)
            thresholds, train_f1 = search_thresholds([claim.label for claim in train], probabilities)
            predicted = [decide_from_probabilities(rows, thresholds)[0] for rows in probabilities]
            log(f"    train macro-F1 {train_f1:.4f} cuts {thresholds.as_dict()}")
        else:
            predicted = [majority for _claim in train]
        report = _pack(train, predicted, retrieved, thresholds if verifier == "nli" else None)
        report["name"] = config.name
        report["verifier"] = verifier
        report["config"] = config.to_dict()
        if verifier == "nli":
            report["verdict"] = thresholds.as_dict()
        else:
            report["verdict"] = {"label": majority, "train_prior": round(prior, 4)}
        train_rows.append(report)

    nli_rows = [row for row in train_rows if row["verifier"] == "nli"]
    winner = max(nli_rows, key=lambda row: (row["classification"]["macro_f1"], row["sentence_recall_at_5"]["recall"]))
    baseline = next(row for row in train_rows if row["name"] == "bm25-majority")
    log(f"Selected {winner['name']} on train macro-F1 {winner['classification']['macro_f1']:.4f}. Scoring dev once.")

    dev_baseline = _evaluate_dev(documents, dev, PipelineConfig.from_dict(baseline["config"]), "majority", majority, cache, embedders, scorer)
    dev_tuned = _evaluate_dev(
        documents,
        dev,
        PipelineConfig.from_dict(winner["config"]),
        "nli",
        majority,
        cache,
        embedders,
        scorer,
        VerdictThresholds.from_dict(winner.get("verdict")),
    )
    payload = {
        "dataset": "scifact",
        "citation": "Wadden et al., Fact or Fiction: Verifying Scientific Claims, EMNLP 2020.",
        "license": "CC BY-NC 2.0",
        "nli_model": NLI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "dev_used_for_selection": False,
        "test_split": "not used; labels are hidden",
        "recall_k": RECALL_K,
        "train": {
            "n": len(train),
            "label_counts": {label: counts.get(label, 0) for label in LABELS},
            "majority_label": majority,
            "configs": [_public_row(row) for row in train_rows],
            "selected": winner["name"],
        },
        "dev": {
            "n": len(dev),
            "label_counts": {label: Counter(claim.label for claim in dev).get(label, 0) for label in LABELS},
            "baseline": _public_row(dev_baseline),
            "tuned": _public_row(dev_tuned),
        },
    }
    destination = Path(out) if out else root / "results.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(
        f"Dev baseline macro-F1 {dev_baseline['classification']['macro_f1']:.4f}, "
        f"tuned {dev_tuned['classification']['macro_f1']:.4f}. Wrote {destination}."
    )
    return payload


def _evaluate_dev(documents, claims, config, verifier, majority, cache, embedders, scorer, thresholds=None) -> dict:
    """One pass per dev claim, batch size 1, so latency matches a single API call."""
    bundle = _cached_bundle(documents, config, cache, embedders)
    predicted = []
    retrieved_sets = []
    latencies = []
    model = scorer if verifier == "nli" else None
    if verifier == "nli" and model is None:
        model = NliScorer()
    for claim in claims:
        started = time.perf_counter()
        hits = _retrieve(bundle, claim.claim)
        if verifier == "majority":
            verdict = majority
        else:
            probabilities = model.score([(hit.chunk.text, claim.claim) for hit in hits], batch_size=8)
            verdict, _confidence = decide_from_probabilities(probabilities, thresholds)
        latencies.append(time.perf_counter() - started)
        predicted.append(verdict)
        retrieved_sets.append(hits)
    report = _pack(claims, predicted, retrieved_sets, thresholds)
    report["name"] = config.name
    report["verifier"] = "nli" if verifier == "nli" else "majority"
    report["config"] = config.to_dict()
    report["latency"] = latency_summary(latencies)
    if thresholds is not None:
        report["verdict"] = thresholds.as_dict()
    else:
        report["verdict"] = {"label": majority}
    return report


def _pack(claims, predicted, retrieved_sets, thresholds: VerdictThresholds | None) -> dict:
    gold = [claim.label for claim in claims]
    abstract_golds = []
    abstract_preds = []
    sentence_golds = []
    sentence_preds = []
    for claim, retrieved in zip(claims, retrieved_sets):
        top = list(retrieved)[:RECALL_K]
        abstract_golds.append(set(claim.evidence_docs))
        abstract_preds.append({_doc_id(hit) for hit in top})
        sentence_golds.append(claim.gold_sentences())
        sentence_preds.append({pair for pair in (_sentence_ids(hit) for hit in top) if pair is not None})
    report = {
        "classification": classification_report(gold, predicted),
        "abstract_recall_at_5": recall_at_k(abstract_golds, abstract_preds),
        "sentence_recall_at_5": recall_at_k(sentence_golds, sentence_preds),
    }
    if thresholds is not None:
        report["verdict"] = thresholds.as_dict()
    return report


def _doc_id(hit) -> str:
    if isinstance(hit, dict):
        return str(hit["doc_id"])
    return str(hit.chunk.doc_id)


def _sentence_ids(hit):
    if isinstance(hit, dict):
        if "sentence_index" not in hit:
            return None
        return (str(hit["doc_id"]), int(hit["sentence_index"]))
    index = hit.chunk.metadata.get("sentence_index")
    if index is None:
        return None
    return (str(hit.chunk.doc_id), int(index))


def _retrieve(bundle: Bundle, claim: str):
    query = bundle.embedder.embed_query(claim)
    return retrieve(claim, bundle.chunks, bundle.vectors, bundle.bm25, query, bundle.config)


def _score_retrieved(model: NliScorer, claims: list[ClaimExample], retrieved_sets) -> list[list[dict]]:
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for claim, hits in zip(claims, retrieved_sets):
        start = len(pairs)
        for hit in hits:
            pairs.append((_text(hit), claim.claim))
        spans.append((start, len(pairs)))
    scored = model.score(pairs, batch_size=16) if pairs else []
    return [scored[start:end] for start, end in spans]


def _text(hit) -> str:
    if isinstance(hit, dict):
        return hit["text"]
    return hit.chunk.text


def _cached_bundle(documents, config: PipelineConfig, cache, embedders) -> Bundle:
    key = config.index_key()
    if key in cache:
        cached = cache[key]
        return Bundle(
            config=config,
            chunks=cached.chunks,
            vectors=cached.vectors,
            bm25=cached.bm25,
            embedder=cached.embedder,
        )
    provider_key = (config.embedding_provider, config.embedding_model)
    if provider_key not in embedders:
        embedders[provider_key] = build_embedder(config.embedding_provider, config.embedding_model)
    print(f"    building a {config.chunk_strategy} index for {len(documents)} documents", flush=True)
    bundle = build_bundle(documents, config, embedder=embedders[provider_key])
    cache[key] = bundle
    return bundle


def _public_row(row: dict) -> dict:
    kept = {
        "name": row["name"],
        "verifier": row["verifier"],
        "classification": row["classification"],
        "abstract_recall_at_5": row["abstract_recall_at_5"],
        "sentence_recall_at_5": row["sentence_recall_at_5"],
        "config": row.get("config"),
        "verdict": row.get("verdict"),
    }
    if "latency" in row:
        kept["latency"] = row["latency"]
    return kept


def evidence_rows_for_tests(bundle: Bundle, claim: str) -> list[dict]:
    return evidence_payload(_retrieve(bundle, claim))

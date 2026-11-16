"""Search configurations and promote one only when it scores higher.

The first config is the baseline. Each later config is indexed, run on the eval
set, and saved. It becomes the champion only when its composite score is strictly
higher. The API serves the champion.
"""

from __future__ import annotations

from pathlib import Path

from autorag.config import PipelineConfig, search_space
from autorag.eval.dataset import find_eval_path, load_eval_items, merge_feedback
from autorag.eval.metrics import aggregate, score_question
from autorag.eval.synthetic import synthesize
from autorag.models import Document, EvalItem
from autorag.pipeline.embed import Embedder, build_embedder
from autorag.pipeline.engine import Bundle, build_bundle
from autorag.pipeline.ingest import load_corpus
from autorag.tune.history import HistoryStore, new_run_id, publish_snapshot, utc_now


def tune(
    source: str,
    runs_dir: str | Path = "runs",
    eval_path: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generator: str = "extractive",
    limit: int = 0,
    configs: list[PipelineConfig] | None = None,
    feedback_rows: list[dict] | None = None,
    log=print,
    write_source_snapshot: bool = True,
) -> dict:
    documents = load_corpus(source)
    if not documents:
        raise RuntimeError(f"No documents found in {source}")

    items, eval_origin = _load_items(source, documents, eval_path, generator, log)
    if feedback_rows:
        before = len(items)
        items = merge_feedback(items, feedback_rows)
        log(f"Folded feedback into the eval set ({before} -> {len(items)} questions).")
    if limit and limit > 0:
        items = items[:limit]

    space = configs or search_space(embedding_model)
    for config in space:
        config.generator = generator
        if config.embedding_provider == "local" and not str(config.embedding_model).startswith("runs/"):
            config.embedding_model = embedding_model

    runs_path = Path(runs_dir)
    store = HistoryStore(runs_path / "autorag.sqlite")
    cache: dict[tuple, Bundle] = {}
    embedders: dict[tuple, Embedder] = {}
    records: list[dict] = []
    champion_score = -1.0
    champion: dict | None = None

    log(f"Tuning on {len(documents)} documents and {len(items)} questions ({len(space)} configs).")
    try:
        for config in space:
            log(f"  [{config.name}] chunk={config.chunk_strategy}/{config.chunk_size} "
                f"k={config.top_k} hybrid={config.hybrid} reranker={config.reranker} "
                f"recency={config.recency_boost}")
            bundle = _cached_bundle(documents, config, cache, embedders)
            report = _evaluate(bundle, items, generator)
            promoted = report.composite > champion_score + 1e-4
            record = {
                "id": new_run_id(config.name),
                "created_at": utc_now(),
                "source": str(source),
                "name": config.name,
                "config": config.to_dict(),
                "metrics": report.as_dict(),
                "is_baseline": config.name == "baseline",
                "promoted": promoted,
                "question_count": len(items),
                "failures": report.failures,
            }
            if promoted:
                champion_score = report.composite
                champion = record
                log(f"    composite {report.composite:.3f}  promoted")
            else:
                log(f"    composite {report.composite:.3f}  kept previous ({champion_score:.3f})")
            store.add(record)
            records.append(record)
    finally:
        store.close()

    payload = _snapshot(source, items, records, champion, eval_origin)
    publish_snapshot(runs_path / "latest.json", payload)
    source_path = Path(source)
    if write_source_snapshot and source_path.is_dir():
        publish_snapshot(source_path / "tune_results.json", _public_snapshot(payload))
    log(
        f"Best config: {payload['best']['name']}  "
        f"composite {payload['best']['metrics']['composite']:.3f}  "
        f"(baseline {payload['baseline']['metrics']['composite']:.3f})"
    )
    return payload


def _load_items(source, documents, eval_path, generator, log) -> tuple[list[EvalItem], str]:
    path = Path(eval_path) if eval_path else find_eval_path(source)
    if path and path.exists():
        items = load_eval_items(path)
        log(f"Loaded {len(items)} gold questions from {path}.")
        return items, str(path)
    log("No gold eval set found. Writing synthetic questions from the documents.")
    complete = None
    if generator in {"openai", "ollama", "auto"}:
        from autorag.pipeline.generate import _complete_for, resolve_generator

        resolved = resolve_generator(generator)
        if resolved in {"openai", "ollama"}:
            complete = _complete_for(resolved)
    items = synthesize(documents, complete=complete)
    if not items:
        raise RuntimeError("Could not build a synthetic eval set from this corpus.")
    out = Path(source) / "eval.synthetic.jsonl" if Path(source).is_dir() else Path("runs") / "eval.synthetic.jsonl"
    from autorag.eval.dataset import write_jsonl

    write_jsonl(out, [item.as_dict() for item in items])
    log(f"Wrote {len(items)} synthetic questions to {out}.")
    return items, str(out)


def _cached_bundle(documents: list[Document], config: PipelineConfig, cache, embedders) -> Bundle:
    key = config.index_key()
    if key in cache:
        cached = cache[key]
        # Retrieval knobs differ even when the index is shared.
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
    bundle = build_bundle(documents, config, embedder=embedders[provider_key])
    cache[key] = bundle
    return bundle


def _evaluate(bundle: Bundle, items: list[EvalItem], generator: str):
    answers, retrieved_sets = bundle.ask_all([item.question for item in items], generator=generator)
    scores = []
    previews = []
    for item, answer, retrieved in zip(items, answers, retrieved_sets):
        scores.append(score_question(item, answer, retrieved))
        previews.append(
            {
                "id": item.id,
                "type": item.qtype,
                "question": item.question,
                "gold": item.answer,
                "predicted": answer.answer,
                "refused": answer.refused,
                "sources": [hit.chunk.doc_id for hit in retrieved],
            }
        )
    return aggregate(items, scores, previews)


def _snapshot(source, items, records, champion, eval_origin) -> dict:
    baseline = next(record for record in records if record["is_baseline"])
    best = champion or baseline
    return {
        "source": str(source),
        "eval": eval_origin,
        "question_count": len(items),
        "baseline": _brief(baseline),
        "best": _brief(best),
        "runs": [_brief(record) for record in records],
        "improvement": round(best["metrics"]["composite"] - baseline["metrics"]["composite"], 4),
    }


def _brief(record: dict) -> dict:
    return {
        "id": record["id"],
        "created_at": record["created_at"],
        "name": record["name"],
        "promoted": record["promoted"],
        "is_baseline": record["is_baseline"],
        "question_count": record["question_count"],
        "config": record["config"],
        "metrics": record["metrics"],
        "failures": record.get("failures") or [],
    }


def _public_snapshot(payload: dict) -> dict:
    """Drop per-question failure samples from the file that ships with the repo."""
    public = json_ready(payload)
    for run in public["runs"]:
        run.pop("failures", None)
    public["baseline"].pop("failures", None)
    public["best"].pop("failures", None)
    return public


def json_ready(payload: dict) -> dict:
    import copy

    return copy.deepcopy(payload)

"""HTTP API: ask, feedback, suggested questions, tuning history, and the demo page."""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autorag.config import (
    PipelineConfig,
    baseline_config,
    configured_generator,
    configured_title,
    default_config,
    load_settings,
)
from autorag.pipeline.embed import build_embedder
from autorag.pipeline.engine import Bundle, build_bundle, load_precomputed_bundle
from autorag.pipeline.generate import resolve_generator
from autorag.pipeline.ingest import load_corpus
from autorag.tune.feedback import FeedbackStore
from autorag.tune.history import HistoryStore, load_snapshot

STATIC_DIR = Path(__file__).parent / "static"


class AskBody(BaseModel):
    question: str = Field(min_length=2)
    variant: str = "best"


class FeedbackBody(BaseModel):
    question: str
    answer: str | None = None
    rating: str
    correction: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    comment: str | None = None


class HandbookService:
    """Holds the indexed handbook. Indexing happens off the request path."""

    def __init__(self, source: str, runs_dir: str):
        self.source = source
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.ready = False
        self.error: str | None = None
        self.docs = []
        self.best_config = default_config()
        self.baseline_config = baseline_config()
        self.title = configured_title(source, load_settings())
        self._bundles: dict[tuple, Bundle] = {}
        self._embedders: dict[tuple, object] = {}
        self._lock = threading.Lock()
        db = self.runs_dir / "autorag.sqlite"
        self.feedback = FeedbackStore(db)
        self.history = HistoryStore(db)

    def warmup(self) -> None:
        try:
            self.docs = load_corpus(self.source)
            if not self.docs:
                raise RuntimeError(f"No documents found in {self.source}")
            snapshot = self._snapshot()
            if snapshot and snapshot.get("best", {}).get("config"):
                self.best_config = PipelineConfig.from_dict(snapshot["best"]["config"])
                if snapshot.get("baseline", {}).get("config"):
                    self.baseline_config = PipelineConfig.from_dict(snapshot["baseline"]["config"])
            override = os.environ.get("AUTORAG_GENERATOR") or configured_generator(load_settings())
            if override:
                self.best_config.generator = override
                self.baseline_config.generator = override
            if os.environ.get("AUTORAG_RUNTIME") == "lite":
                self._warmup_lite()
            else:
                self._bundle(self.best_config)
            self.ready = True
        except Exception as exc:  # surface the message to /health
            self.error = str(exc)
            self.ready = False

    def _warmup_lite(self) -> None:
        """Load saved passage vectors and embed only the question.

        The free-host image has no PyTorch. Document vectors were built ahead of
        time with the ONNX MiniLM; startup just reads those files.
        """
        model = os.environ.get("AUTORAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        for config in (self.best_config, self.baseline_config):
            config.embedding_provider = "fastembed"
            config.embedding_model = model
        embedder = build_embedder("fastembed", model)
        self._embedders[("fastembed", model)] = embedder
        index_root = Path(self.source) / "index"
        for config in (self.best_config, self.baseline_config):
            directory = index_root / config.name
            if not (directory / "vectors.npy").exists():
                raise RuntimeError(
                    f"Lite runtime needs a saved index at {directory}. "
                    "Run `python scripts/precompute_index.py` and rebuild the image."
                )
            self._bundles[config.index_key()] = load_precomputed_bundle(directory, config, embedder)

    def ask(self, question: str, variant: str = "best"):
        if self.error:
            raise HTTPException(status_code=500, detail=self.error)
        if not self.ready:
            raise HTTPException(status_code=503, detail="The handbook is still indexing.")
        config = self.baseline_config if variant == "baseline" else self.best_config
        bundle = self._bundle(config)
        mode = resolve_generator(config.generator)
        answer = bundle.ask(question, generator=mode)
        payload = answer.as_dict()
        payload["variant"] = "baseline" if variant == "baseline" else "best"
        payload["config"] = config.name
        return payload

    def runs_payload(self) -> dict:
        live = self.history.list_runs()
        if live:
            return _payload_from_records(self.source, live)
        snapshot = self._snapshot()
        if snapshot:
            return snapshot
        return {
            "source": self.source,
            "question_count": 0,
            "baseline": None,
            "best": None,
            "runs": [],
            "improvement": 0,
            "note": "No tuning run yet. The server is using the default configuration.",
        }

    def suggested(self) -> list[str]:
        path = Path(self.source) / "suggested.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [str(item) for item in data]
        return [
            "How many PTO days after 3 years?",
            "What's the client dinner expense limit?",
            "How do I request parental leave?",
        ]

    def _bundle(self, config: PipelineConfig) -> Bundle:
        key = config.index_key()
        with self._lock:
            if key not in self._bundles:
                provider_key = (config.embedding_provider, config.embedding_model)
                if provider_key not in self._embedders:
                    self._embedders[provider_key] = build_embedder(config.embedding_provider, config.embedding_model)
                built = build_bundle(self.docs, config, embedder=self._embedders[provider_key])
                self._bundles[key] = built
            cached = self._bundles[key]
        return Bundle(
            config=config,
            chunks=cached.chunks,
            vectors=cached.vectors,
            bm25=cached.bm25,
            embedder=cached.embedder,
        )

    def _snapshot(self) -> dict | None:
        for candidate in (self.runs_dir / "latest.json", Path(self.source) / "tune_results.json"):
            loaded = load_snapshot(candidate)
            if loaded and loaded.get("runs"):
                return loaded
        return None


def create_app(source: str = "data/acme", runs_dir: str = "runs") -> FastAPI:
    service = HandbookService(source, runs_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        thread = threading.Thread(target=service.warmup, name="autorag-index", daemon=True)
        thread.start()
        yield

    app = FastAPI(title="AutoRAG", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/health")
    def health():
        # 200 while the index is still loading so a free host does not kill the process.
        if service.error:
            return JSONResponse({"status": "error", "detail": service.error}, status_code=500)
        if not service.ready:
            return {"status": "loading", "documents": 0, "title": service.title, "source": service.source}
        return {
            "status": "ok",
            "documents": len(service.docs),
            "config": service.best_config.name,
            "generator": resolve_generator(service.best_config.generator),
            "title": service.title,
            "source": service.source,
        }

    @app.post("/ask")
    def ask(body: AskBody):
        variant = body.variant if body.variant in {"best", "baseline"} else "best"
        return service.ask(body.question, variant)

    @app.post("/feedback")
    def feedback(body: FeedbackBody):
        if body.rating not in {"up", "down"}:
            raise HTTPException(status_code=400, detail="rating must be up or down")
        record = service.feedback.add(
            question=body.question,
            rating=body.rating,
            answer=body.answer,
            correction=body.correction,
            doc_ids=body.doc_ids,
            comment=body.comment,
        )
        return {"stored": True, "id": record["id"]}

    @app.get("/suggested-questions")
    def suggested():
        return {"questions": service.suggested()}

    @app.get("/runs")
    def runs():
        return service.runs_payload()

    @app.get("/")
    def demo():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _payload_from_records(source: str, records: list[dict]) -> dict:
    baseline = next((row for row in records if row.get("is_baseline")), records[0])
    promoted = [row for row in records if row.get("promoted")]
    best = promoted[-1] if promoted else baseline
    return {
        "source": source,
        "question_count": best.get("question_count") or baseline.get("question_count") or 0,
        "baseline": _public_run(baseline),
        "best": _public_run(best),
        "runs": [_public_run(row) for row in records],
        "improvement": round(best["metrics"]["composite"] - baseline["metrics"]["composite"], 4),
    }


def _public_run(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "created_at": record.get("created_at"),
        "name": record.get("name"),
        "promoted": record.get("promoted"),
        "is_baseline": record.get("is_baseline"),
        "question_count": record.get("question_count"),
        "config": record.get("config"),
        "metrics": record.get("metrics"),
    }

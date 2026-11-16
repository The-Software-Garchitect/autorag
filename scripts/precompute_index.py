#!/usr/bin/env python3
"""Save passage vectors for the free-host demo.

The published tune ran on PyTorch MiniLM. The free host cannot load that stack,
so this script embeds the same chunk settings with the ONNX MiniLM (fastembed)
and writes ``data/acme/index/<config name>/``. Startup then embeds only the question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autorag.config import PipelineConfig  # noqa: E402
from autorag.pipeline.embed import build_embedder  # noqa: E402
from autorag.pipeline.engine import build_bundle  # noqa: E402
from autorag.pipeline.index import VectorIndex  # noqa: E402
from autorag.pipeline.ingest import load_corpus  # noqa: E402
from autorag.tune.history import load_snapshot  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    source = ROOT / "data" / "acme"
    snapshot = load_snapshot(source / "tune_results.json")
    if not snapshot:
        raise SystemExit(f"Missing {source / 'tune_results.json'}")
    documents = load_corpus(str(source))
    embedder = build_embedder("fastembed", MODEL)
    for key in ("baseline", "best"):
        config = PipelineConfig.from_dict(snapshot[key]["config"])
        config.embedding_provider = "fastembed"
        config.embedding_model = MODEL
        bundle = build_bundle(documents, config, embedder)
        directory = source / "index" / config.name
        VectorIndex(bundle.chunks, bundle.vectors).save(directory)
        (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"{config.name}: {len(bundle.chunks)} chunks, dim {bundle.vectors.shape[1]} -> {directory}")


if __name__ == "__main__":
    main()

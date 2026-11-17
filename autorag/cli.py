"""Command line for ingesting, tuning, asking, and serving."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autorag.config import baseline_config, configured_generator, configured_source, default_config, load_settings
from autorag.pipeline.engine import build_bundle
from autorag.pipeline.generate import resolve_generator
from autorag.pipeline.ingest import load_corpus
from autorag.tune.feedback import FeedbackStore
from autorag.tune.loop import tune


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="autorag",
        description="Turn a folder of documents into a cited question-answering API, then tune it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tune_cmd = sub.add_parser("tune", help="Ingest a corpus, score configs, and promote a winner")
    tune_cmd.add_argument("--source", default=None, help="Document folder. Defaults to source in autorag.yaml")
    tune_cmd.add_argument("--eval", default=None, help="Gold JSONL. Defaults to <source>/eval.jsonl")
    tune_cmd.add_argument("--runs", default="runs")
    tune_cmd.add_argument("--limit", type=int, default=0, help="Score at most this many questions (0 = all)")
    tune_cmd.add_argument(
        "--holdout",
        type=float,
        default=None,
        help="Fraction of questions reserved for reporting. Defaults to holdout in <source>/autorag.yaml",
    )
    tune_cmd.set_defaults(func=cmd_tune)

    ask_cmd = sub.add_parser("ask", help="Ask one question against the best saved config")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--source", default=None)
    ask_cmd.add_argument("--runs", default="runs")
    ask_cmd.add_argument("--variant", default="best", choices=["best", "baseline"])
    ask_cmd.set_defaults(func=cmd_ask)

    serve_cmd = sub.add_parser("serve", help="Run the API and the demo page")
    serve_cmd.add_argument("--source", default=None)
    serve_cmd.add_argument("--host", default=os.environ.get("AUTORAG_HOST", "0.0.0.0"))
    serve_cmd.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8091")))
    serve_cmd.add_argument("--runs", default="runs")
    serve_cmd.set_defaults(func=cmd_serve)

    synth_cmd = sub.add_parser("synth", help="Write a synthetic eval set for a corpus")
    synth_cmd.add_argument("--source", required=True)
    synth_cmd.add_argument("--out", required=True)
    synth_cmd.add_argument("--llm", action="store_true", help="Ask Ollama or OpenAI to write the questions")
    synth_cmd.set_defaults(func=cmd_synth)

    eval_cmd = sub.add_parser("eval", help="Score the current best config and print metrics")
    eval_cmd.add_argument("--source", required=True)
    eval_cmd.add_argument("--eval", default=None)
    eval_cmd.add_argument("--runs", default="runs")
    eval_cmd.add_argument("--limit", type=int, default=0)
    eval_cmd.set_defaults(func=cmd_eval)

    fetch_cmd = sub.add_parser("fetch", help="Download a public dataset into a local cache")
    fetch_sub = fetch_cmd.add_subparsers(dest="dataset", required=True)
    scifact_cmd = fetch_sub.add_parser("scifact", help="Allen AI SciFact claims and PubMed abstracts")
    scifact_cmd.add_argument("--dest", default="examples/scifact")
    scifact_cmd.set_defaults(func=cmd_fetch_scifact)

    check_cmd = sub.add_parser("check-claim", help="Decide whether the corpus supports a claim")
    check_cmd.add_argument("claim")
    check_cmd.add_argument("--source", default=None)
    check_cmd.add_argument("--runs", default="runs")
    check_cmd.add_argument("--verifier", default="auto", choices=["auto", "nli", "extractive"])
    check_cmd.set_defaults(func=cmd_check_claim)

    claims_cmd = sub.add_parser("tune-claims", help="Select a claim checker on SciFact train and score dev")
    claims_cmd.add_argument("--source", default="examples/scifact")
    claims_cmd.add_argument("--out", default=None)
    claims_cmd.set_defaults(func=cmd_tune_claims)

    args = parser.parse_args(argv)
    args.func(args)


def cmd_tune(args) -> None:
    source = _source(args.source)
    feedback_path = Path(args.runs) / "autorag.sqlite"
    rows = []
    if feedback_path.exists():
        store = FeedbackStore(feedback_path)
        rows = store.list_all()
        store.close()
    tune(
        source=source,
        runs_dir=args.runs,
        eval_path=args.eval,
        generator=_generator(),
        limit=args.limit,
        feedback_rows=rows,
        holdout=args.holdout,
    )


def cmd_ask(args) -> None:
    args.source = _source(args.source)
    documents = load_corpus(args.source)
    config = _config_for(args.source, args.runs, args.variant)
    bundle = build_bundle(documents, config)
    answer = bundle.ask(args.question, generator=resolve_generator(os.environ.get("AUTORAG_GENERATOR", config.generator)))
    print(answer.answer)
    if answer.refused:
        print("\n(refused)")
    for citation in answer.citations:
        print(f"\n- {citation.title} [{citation.status}] {citation.source}")
        print(f"  {citation.snippet}")
    print(f"\nconfidence {answer.confidence:.2f}  via {answer.generator}  config {config.name}")


def cmd_eval(args) -> None:
    from autorag.eval.dataset import find_eval_path, load_eval_items
    from autorag.tune.loop import _evaluate

    documents = load_corpus(args.source)
    path = Path(args.eval) if args.eval else find_eval_path(args.source)
    if path is None:
        raise SystemExit("No eval JSONL found. Pass --eval or run `autorag synth`.")
    items = load_eval_items(path)
    if args.limit:
        items = items[: args.limit]
    config = _config_for(args.source, args.runs, "best")
    bundle = build_bundle(documents, config)
    report = _evaluate(bundle, items, config.generator)
    print(json.dumps(report.as_dict(), indent=2))


def cmd_synth(args) -> None:
    from autorag.eval.dataset import write_jsonl
    from autorag.eval.synthetic import synthesize
    from autorag.pipeline.generate import _complete_for

    documents = load_corpus(args.source)
    complete = None
    if args.llm:
        mode = resolve_generator(os.environ.get("AUTORAG_GENERATOR", "auto"))
        complete = _complete_for(mode) if mode in {"openai", "ollama"} else None
        if complete is None:
            print("No OpenAI key or Ollama server found. Using the heuristic writer.")
    items = synthesize(documents, complete=complete)
    write_jsonl(args.out, [item.as_dict() for item in items])
    print(f"Wrote {len(items)} questions to {args.out}")


def cmd_fetch_scifact(args) -> None:
    from autorag.claims.fetch import fetch_scifact

    summary = fetch_scifact(args.dest)
    print(json.dumps(summary, indent=2))


def cmd_check_claim(args) -> None:
    from autorag.claims.check import check_claim, load_claim_pipeline, load_verdict_thresholds
    from autorag.claims.verdict import select_verifier

    source = _source(args.source)
    documents = load_corpus(source)
    config = load_claim_pipeline(source) or _config_for(source, args.runs, "best")
    bundle = build_bundle(documents, config)
    payload = check_claim(
        bundle,
        args.claim,
        verifier=select_verifier(args.verifier),
        thresholds=load_verdict_thresholds(source),
    )
    print(json.dumps(payload, indent=2))


def cmd_tune_claims(args) -> None:
    from autorag.claims.experiment import run_scifact_experiment

    run_scifact_experiment(args.source, out=args.out)


def cmd_serve(args) -> None:
    import uvicorn

    from autorag.service.app import create_app

    source = _source(args.source)
    app = create_app(source, args.runs)
    print(f"Ask the documents: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _source(explicit: str | None) -> str:
    return explicit or configured_source()


def _generator() -> str:
    return os.environ.get("AUTORAG_GENERATOR") or configured_generator(load_settings())


def _config_for(source: str, runs: str, variant: str):
    from autorag.tune.history import load_snapshot

    for candidate in (Path(runs) / "latest.json", Path(source) / "tune_results.json"):
        snapshot = load_snapshot(candidate)
        if not snapshot:
            continue
        key = "baseline" if variant == "baseline" else "best"
        if snapshot.get(key) and snapshot[key].get("config"):
            from autorag.config import PipelineConfig

            return PipelineConfig.from_dict(snapshot[key]["config"])
    return baseline_config() if variant == "baseline" else default_config()

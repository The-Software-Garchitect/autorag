"""The Acme tuner has to choose a config without looking at the report split."""

import json
from pathlib import Path

from autorag.config import PipelineConfig
from autorag.eval.metrics import MetricReport
from autorag.eval.split import split_holdout
from autorag.models import EvalItem
from autorag.tune.loop import tune


def _item(item_id: str, qtype: str = "factual") -> EvalItem:
    return EvalItem(id=item_id, question=f"Question {item_id}?", answer="Answer.", source_docs=["doc"], qtype=qtype)


def test_split_is_stable_stratified_and_disjoint():
    items = [_item(str(i), "factual" if i % 2 == 0 else "table") for i in range(20)]
    train, report = split_holdout(items, 0.25, seed=0)
    again_train, again_report = split_holdout(items, 0.25, seed=0)
    assert [item.id for item in train] == [item.id for item in again_train]
    assert {item.id for item in train}.isdisjoint({item.id for item in report})
    assert len(train) + len(report) == 20
    assert report and train
    assert [item.id for item in report] == [item.id for item in again_report]


def test_tune_keeps_the_train_winner_and_reports_holdout(corpus_dir, tmp_path: Path, monkeypatch):
    items = [_item("t1"), _item("t2"), _item("h1"), _item("h2")]
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in items), encoding="utf-8")

    train, report = split_holdout(items, 0.5, seed=0)
    train_ids = {item.id for item in train}
    report_ids = {item.id for item in report}

    def fake_evaluate(bundle, scored, generator):
        ids = {item.id for item in scored}
        assert ids in (train_ids, report_ids)
        train_score = {"baseline": 0.2, "better": 0.9}[bundle.config.name]
        hold_score = {"baseline": 0.95, "better": 0.1}[bundle.config.name]
        score = hold_score if ids == report_ids else train_score
        return MetricReport(
            n=len(scored),
            correctness=score,
            faithfulness=1,
            context_recall=score,
            context_precision=score,
            hit_rate=score,
            mrr=score,
            refusal_accuracy=1,
            composite=score,
        )

    monkeypatch.setattr("autorag.tune.loop._evaluate", fake_evaluate)
    base = PipelineConfig(name="baseline", embedding_provider="hashing", embedding_model="hashing")
    better = PipelineConfig(name="better", embedding_provider="hashing", embedding_model="hashing", hybrid=True)
    payload = tune(
        source=str(corpus_dir),
        runs_dir=tmp_path / "runs",
        eval_path=str(path),
        configs=[base, better],
        holdout=0.5,
        log=lambda *_a, **_k: None,
        write_source_snapshot=False,
    )
    assert payload["best"]["name"] == "better"
    assert payload["best"]["promoted"] is True
    assert payload["best"]["metrics"]["composite"] == 0.1
    assert payload["best"]["selection_metrics"]["composite"] == 0.9
    assert payload["split"]["scored_on"] == "holdout"
    assert payload["baseline"]["metrics"]["composite"] == 0.95

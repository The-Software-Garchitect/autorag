from autorag.eval.metrics import WEIGHTS, aggregate, score_question
from autorag.eval.synthetic import synthesize
from autorag.models import Answer, EvalItem
from autorag.pipeline.ingest import load_corpus


def test_correctness_rewards_the_gold_number_and_refusal():
    good = Answer("q", "The limit is $12 per person.", [], 0.8, False)
    bad = Answer("q", "The limit is $4 per person.", [], 0.8, False)
    refused = Answer("q", "I can't find that in the handbook.", [], 0.6, True)
    item = EvalItem("1", "q", "The limit is $12 per person.", ["snacks"])
    unknown = EvalItem("2", "pet insurance?", "no", [], unanswerable=True)

    good_score = score_question(item, good, [])
    bad_score = score_question(item, bad, [])
    assert good_score.correctness > 0.9
    assert bad_score.correctness < good_score.correctness
    assert score_question(unknown, refused, []).correctness == 1.0
    assert score_question(unknown, good, []).correctness == 0.0


def test_composite_uses_the_documented_weights():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    report = aggregate([], [])
    assert report.n == 0


def test_synthetic_questions_quote_the_corpus(corpus_dir):
    docs = load_corpus(str(corpus_dir))
    items = synthesize(docs, per_doc=1)
    assert items
    blob = "\n".join(doc.text for doc in docs)
    assert any(item.answer.rstrip(".") in blob for item in items)

"""The committed Acme corpus stays aligned with the gold questions."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "acme" / "docs"
EVAL = ROOT / "data" / "acme" / "eval.jsonl"


def _docs():
    files = list(DOCS.rglob("*.md"))
    texts = {}
    for path in files:
        raw = path.read_text(encoding="utf-8")
        meta, _, body = raw.partition("\n---\n")
        # partition keeps the first --- inside meta if the file starts with ---
        if raw.startswith("---"):
            _, meta, body = raw.split("---", 2)
        doc_id = None
        status = "current"
        for line in meta.splitlines():
            if line.startswith("id:"):
                doc_id = line.split(":", 1)[1].strip()
            if line.startswith("status:"):
                status = line.split(":", 1)[1].strip()
        texts[doc_id or path.stem] = {"status": status, "body": body, "path": path}
    return files, texts


def test_corpus_shape_and_messiness():
    files, texts = _docs()
    assert len(files) >= 50
    statuses = {info["status"] for info in texts.values()}
    assert "current" in statuses
    assert "superseded" in statuses
    assert "unofficial" in statuses
    assert "duplicate" in statuses
    assert "pto-policy-2023" in texts and "pto-policy-2025" in texts
    assert "16 PTO days" in texts["pto-policy-2023"]["body"]
    assert "22 PTO days" in texts["pto-policy-2025"]["body"]
    slack = "\n".join(info["body"] for doc_id, info in texts.items() if "slack" in doc_id)
    assert "devon" in slack.lower() or "**devon**" in slack


def test_gold_questions_resolve_to_real_documents():
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    _, texts = _docs()
    assert len(rows) >= 250
    heroes = {row["question"]: row for row in rows}
    assert "22" in heroes["How many PTO days after 3 years?"]["answer"]
    assert heroes["How many PTO days after 3 years?"]["source_docs"] == ["pto-policy-2025"]
    assert "$75" in heroes["What's the client dinner expense limit?"]["answer"]
    assert "parental leave" in heroes["How do I request parental leave?"]["answer"].lower()
    for row in rows:
        for doc_id in row["source_docs"]:
            assert doc_id in texts, doc_id
        if row["unanswerable"]:
            continue
        blob = "\n".join(texts[doc_id]["body"] for doc_id in row["source_docs"])
        if row["type"] == "table":
            assert "Chicago" in blob or "Plan" in blob or "|" in blob
        else:
            for sentence in row["answer"].split(". "):
                needle = sentence.strip().rstrip(".")
                assert needle in blob, needle

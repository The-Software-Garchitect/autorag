from autorag.config import PipelineConfig
from autorag.pipeline.chunk import chunk_documents
from autorag.pipeline.ingest import load_corpus
from autorag.textutil import iter_markdown_tables, linearize_row


def test_structure_keeps_a_table_together(tmp_path):
    (tmp_path / "rates.md").write_text(
        """---
id: rates
title: Rates
status: current
---

Intro paragraph that explains the table is authoritative.

## Cities

| City | Lodging per diem |
| --- | --- |
| Austin | $180 |
| Chicago | $220 |
""",
        encoding="utf-8",
    )
    docs = load_corpus(str(tmp_path))
    config = PipelineConfig(chunk_strategy="structure", chunk_size=40, chunk_overlap=0)
    chunks = chunk_documents(docs, config)
    tables = [chunk for chunk in chunks if "| Chicago |" in chunk.text and "| Austin |" in chunk.text]
    assert tables, "structure chunker should keep both rows with the header"
    parsed = list(iter_markdown_tables(tables[0].text))
    assert parsed
    header, rows, _ = parsed[0]
    chicago = next(linearize_row(header, row) for row in rows if "Chicago" in row)
    assert "220" in chicago


def test_sentence_chunks_keep_blank_line_order(tmp_path):
    (tmp_path / "paper.md").write_text(
        """---
id: "7"
title: Fever
---

Alpha lowers fever.

Beta does not.
""",
        encoding="utf-8",
    )
    docs = load_corpus(str(tmp_path))
    chunks = chunk_documents(docs, PipelineConfig(chunk_strategy="sentence", chunk_size=20))
    assert [chunk.metadata["sentence_index"] for chunk in chunks] == [0, 1]
    assert chunks[0].text == "Alpha lowers fever."
    assert chunks[1].doc_id == "7"


def test_fixed_windows_can_split_a_table(tmp_path):
    row = "| Chicago | $220 | a long note about downtown hotels near the river and the travel desk booking rule |"
    body = "\n".join(["| City | Lodging | Notes |", "| --- | --- | --- |"] + [row] * 8)
    (tmp_path / "rates.md").write_text(f"---\nid: rates\ntitle: Rates\nstatus: current\n---\n\n{body}\n", encoding="utf-8")
    docs = load_corpus(str(tmp_path))
    chunks = chunk_documents(docs, PipelineConfig(chunk_strategy="fixed", chunk_size=30, chunk_overlap=0))
    assert len(chunks) > 1
    # Flattened windows are single lines, so the markdown table parser finds nothing.
    assert all(not list(iter_markdown_tables(chunk.text)) for chunk in chunks)

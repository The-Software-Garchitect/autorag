"""Split documents into retrieval-sized pieces.

``fixed`` cuts the document into windows of N words. It is simple, and it is
also how tables lose their headers: the window does not know that a row belongs
to the line above it.

``structure`` splits on headings, keeps each markdown table together (header
included), and only windows ordinary paragraphs that are still too long.
"""

from __future__ import annotations

import re

from autorag.config import PipelineConfig
from autorag.models import Chunk, Document

_HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*$")


def chunk_documents(documents: list[Document], config: PipelineConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        if config.chunk_strategy == "fixed":
            pieces = [(text, {}) for text in _fixed_pieces(document, config.chunk_size, config.chunk_overlap)]
        elif config.chunk_strategy == "structure":
            pieces = [(text, {}) for text in _structure_pieces(document, config.chunk_size, config.chunk_overlap)]
        elif config.chunk_strategy == "sentence":
            pieces = [
                (text, {"sentence_index": index})
                for index, text in enumerate(_sentence_pieces(document.text))
            ]
        else:
            raise ValueError(f"Unknown chunk strategy: {config.chunk_strategy}")
        for index, (text, extra) in enumerate(pieces):
            cleaned = text.strip()
            if not cleaned:
                continue
            metadata = dict(document.metadata)
            metadata.update(extra)
            chunks.append(
                Chunk(
                    id=f"{document.id}:{index}",
                    doc_id=document.id,
                    title=document.title,
                    text=cleaned,
                    index=index,
                    source=document.source,
                    metadata=metadata,
                )
            )
    return chunks


def _sentence_pieces(text: str) -> list[str]:
    """One piece per blank-line paragraph, in order.

    SciFact abstracts are written that way so piece 0 is abstract sentence 0.
    """
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _fixed_pieces(document: Document, size: int, overlap: int) -> list[str]:
    words = f"{document.title}\n{document.text}".split()
    if not words:
        return []
    size = max(20, size)
    overlap = max(0, min(overlap, size - 1))
    step = max(1, size - overlap)
    pieces: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + size >= len(words):
            break
    return pieces


def _structure_pieces(document: Document, size: int, overlap: int) -> list[str]:
    size = max(40, size)
    sections = _sections(document.text)
    pieces: list[str] = []
    for heading, body in sections:
        buffer: list[str] = []
        buffered = 0

        def flush() -> None:
            nonlocal buffered
            if buffer:
                pieces.append(_with_heading(document.title, heading, "\n\n".join(buffer)))
                buffer.clear()
                buffered = 0

        for block in _blocks(body):
            if _is_table_block(block):
                flush()
                pieces.append(_with_heading(document.title, heading, block))
                continue
            words = block.split()
            if len(words) > size:
                flush()
                step = max(1, size - max(0, min(overlap, size // 2)))
                for start in range(0, len(words), step):
                    window = " ".join(words[start : start + size])
                    pieces.append(_with_heading(document.title, heading, window))
                    if start + size >= len(words):
                        break
                continue
            if buffer and buffered + len(words) > size:
                flush()
            buffer.append(block)
            buffered += len(words)
        flush()
    return pieces


def _sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading, body))

    for line in lines:
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    if not sections and text.strip():
        return [("", text.strip())]
    return [(heading, body) for heading, body in sections]


def _blocks(body: str) -> list[str]:
    lines = body.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            text = "\n".join(paragraph).strip()
            if text:
                blocks.append(text)
            paragraph.clear()

    while index < len(lines):
        if _table_starts(lines, index):
            flush_paragraph()
            table = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index])
                index += 1
            blocks.append("\n".join(table))
            continue
        if not lines[index].strip():
            flush_paragraph()
        else:
            paragraph.append(lines[index])
        index += 1
    flush_paragraph()
    return blocks


def _table_starts(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    if not lines[index].strip().startswith("|"):
        return False
    from autorag.textutil import is_delimiter_row

    return is_delimiter_row(lines[index + 1])


def _is_table_block(block: str) -> bool:
    rows = [line for line in block.splitlines() if line.strip()]
    if len(rows) < 2:
        return False
    from autorag.textutil import is_delimiter_row

    return rows[0].strip().startswith("|") and is_delimiter_row(rows[1])


def _with_heading(title: str, heading: str, body: str) -> str:
    parts = [title.strip()]
    if heading and heading.strip() and heading.strip() != title.strip():
        parts.append(heading.strip())
    parts.append(body.strip())
    return "\n\n".join(part for part in parts if part)

"""Small text helpers shared by chunking, retrieval, generation, and scoring.

Keeping tokenization in one place matters: a question only counts as "supported
by the handbook" when the same definition of a word is used on both sides.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "into",
        "how",
        "what",
        "whats",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "about",
        "please",
        "according",
        "am",
        "im",
        "ive",
        "they",
        "them",
        "their",
        "but",
        "so",
        "if",
        "as",
        "than",
        "then",
        "any",
        "there",
        "here",
        "just",
        "also",
        "because",
        "while",
        "many",
        "much",
        "get",
        "getting",
        "have",
        "has",
        "had",
        "need",
        "needs",
        "know",
        "check",
        "want",
        "wanted",
        "someone",
        "anyone",
        "something",
        "hey",
        "hi",
    }
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.+-]*")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DELIM = re.compile(r":?-{3,}:?")

REFUSAL_TEXT = (
    "I can't find that in the handbook. It may not be covered by the documents I have."
)


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    text = text.replace("'", "")
    # Keep thousands separators from splitting 1,500 into the tokens "1" and "500".
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text


def tokens(text: str) -> list[str]:
    found = []
    for tok in _TOKEN.findall(normalize_text(text)):
        # The pattern allows dots so hostnames and decimals stay intact.
        # A sentence-final period should not become part of the word.
        tok = tok.rstrip(".")
        if tok:
            found.append(tok)
    return found


def content_tokens(text: str) -> list[str]:
    kept: list[str] = []
    for tok in tokens(text):
        if tok in STOPWORDS:
            continue
        # Keep single digits. They distinguish "3 years" from "5 years".
        if len(tok) == 1 and not tok.isdigit():
            continue
        kept.append(tok)
    return kept


def term_overlap(query: str, text: str) -> float:
    """Share of the question's content words that appear in ``text``."""
    q = set(content_tokens(query))
    if not q:
        return 0.0
    have = set(content_tokens(text))
    return sum(1 for tok in q if tok in have) / len(q)


def extract_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for raw in _NUM.findall(text):
        number = raw.replace(",", "")
        if number.endswith(".0"):
            number = number[:-2]
        found.add(number)
        if "." in number:
            trimmed = number.rstrip("0").rstrip(".")
            if trimmed:
                found.add(trimmed)
    return found


def token_recall(candidate: str, reference: str) -> float:
    """Fraction of reference content tokens present in the candidate."""
    ref = content_tokens(reference)
    if not ref:
        return 1.0 if not content_tokens(candidate) else 0.0
    have = set(content_tokens(candidate))
    return sum(1 for tok in ref if tok in have) / len(ref)


def token_f1(left: str, right: str) -> float:
    a = content_tokens(left)
    b = content_tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    from collections import Counter

    ac, bc = Counter(a), Counter(b)
    overlap = sum((ac & bc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def number_f1(candidate: str, reference: str) -> float | None:
    """F1 over the set of numbers. None when the reference has no number."""
    gold = extract_numbers(reference)
    if not gold:
        return None
    pred = extract_numbers(candidate)
    if not pred:
        return 0.0
    overlap = gold & pred
    precision = len(overlap) / len(pred)
    recall = len(overlap) / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def split_sentences(text: str) -> list[str]:
    """Split into sentences without gluing a heading onto the first sentence."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    sentences: list[str] = []
    for paragraph in paragraphs:
        cleaned = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            continue
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def split_table_row(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [cell.strip() for cell in raw.split("|")]


def is_delimiter_row(line: str) -> bool:
    cells = [cell for cell in split_table_row(line) if cell]
    if len(cells) < 2:
        return False
    return all(_DELIM.fullmatch(cell) for cell in cells)


def linearize_row(header: list[str], row: list[str]) -> str:
    """Turn one markdown table row into a sentence a reader (and the scorer) can match.

    Example: headers [City, Lodging per diem] and row [Chicago, $220] become
    "City is Chicago; Lodging per diem is $220."
    """
    parts: list[str] = []
    for head, cell in zip(header, row):
        head, cell = head.strip(), cell.strip()
        if not head or not cell or _DELIM.fullmatch(head):
            continue
        parts.append(f"{head} is {cell}")
    if not parts:
        return ""
    return "; ".join(parts) + "."


def iter_markdown_tables(text: str):
    """Yield (header, rows, line_indexes) for every markdown table in ``text``."""
    lines = text.splitlines()
    index = 0
    while index < len(lines) - 1:
        if lines[index].strip().startswith("|") and is_delimiter_row(lines[index + 1]):
            header = split_table_row(lines[index])
            consumed = {index, index + 1}
            rows: list[list[str]] = []
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].strip().startswith("|"):
                if is_delimiter_row(lines[cursor]):
                    consumed.add(cursor)
                    cursor += 1
                    continue
                rows.append(split_table_row(lines[cursor]))
                consumed.add(cursor)
                cursor += 1
            yield header, rows, consumed
            index = cursor
            continue
        index += 1


def units_from_text(text: str) -> list[str]:
    """Sentences and linearized table rows. Tables stay attached to their headers."""
    lines = text.splitlines()
    table_lines: set[int] = set()
    units: list[str] = []
    for header, rows, consumed in iter_markdown_tables(text):
        table_lines |= consumed
        for row in rows:
            sentence = linearize_row(header, row)
            if sentence:
                units.append(sentence)
    prose_lines = [
        line for i, line in enumerate(lines) if i not in table_lines and not line.strip().startswith("|")
    ]
    for sentence in split_sentences("\n".join(prose_lines)):
        # Headings are their own paragraphs and have no closing punctuation.
        # Quoting them produces answers like "Parental Leave."
        if not sentence.endswith((".", "!", "?")):
            continue
        if len(content_tokens(sentence)) >= 2:
            units.append(sentence)
    return units


def is_refusal_text(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "can't find that",
        "cannot find that",
        "can't find this",
        "not covered",
        "not in the handbook",
        "don't have that",
        "do not have that",
        "no information",
        "not documented",
        "unable to answer",
    )
    return any(marker in lowered for marker in markers)


def render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    def cell_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    lines = [
        cell_row(header),
        cell_row(["---"] * len(header)),
    ]
    lines.extend(cell_row(row) for row in rows)
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(text.split())

"""Load a folder of Markdown, text, and PDF into one document list.

A newer file can name the page it replaces with ``supersedes: older-id`` in
front matter. That older page is then treated as superseded, even if its own
status line still says current.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from autorag.models import Document

SKIP_NAMES = {"eval.jsonl", "suggested.json", "tune_results.json"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end]
    body = text[end + 4 :]
    if body.startswith("\n"):
        body = body[1:]
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, body


def _title_from(meta: dict, body: str, fallback: str) -> str:
    if meta.get("title"):
        return str(meta["title"]).strip()
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return fallback


def document_from_text(path: Path, text: str, root: Path | None = None) -> Document:
    meta, body = parse_frontmatter(text)
    rel = path.name
    if root is not None:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
    stem_id = Path(rel).with_suffix("").as_posix()
    doc_id = str(meta.get("id") or stem_id)
    title = _title_from(meta, body, path.stem.replace("-", " ").title())
    metadata = {key: value for key, value in meta.items() if key not in {"id", "title"}}
    metadata.setdefault("status", "current")
    if metadata.get("effective_date") is not None:
        metadata["effective_date"] = str(metadata["effective_date"])
    return Document(id=doc_id, title=title, text=body.strip(), source=rel, metadata=metadata)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


class LocalFolderLoader:
    """Read Markdown, text, and PDF files under a directory (or a single file)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.path}")
        root = self.path if self.path.is_dir() else self.path.parent
        files = [self.path] if self.path.is_file() else sorted(self.path.rglob("*"))
        documents: list[Document] = []
        seen: dict[str, int] = {}
        for file in files:
            if not file.is_file():
                continue
            if file.name in SKIP_NAMES or file.name.startswith("."):
                continue
            suffix = file.suffix.lower()
            if suffix in TEXT_SUFFIXES:
                text = file.read_text(encoding="utf-8")
            elif suffix == ".pdf":
                text = _read_pdf(file)
            else:
                continue
            if not text.strip():
                continue
            document = document_from_text(file, text, root)
            if document.id in seen:
                seen[document.id] += 1
                document.id = f"{document.id}-{seen[document.id]}"
            else:
                seen[document.id] = 1
            documents.append(document)
        return documents


def mark_superseded(documents: list[Document]) -> list[Document]:
    """Honor a ``supersedes`` field. The named page is the older one."""
    by_id = {document.id: document for document in documents}
    for document in documents:
        older_id = document.metadata.get("supersedes")
        if not older_id:
            continue
        older = by_id.get(str(older_id))
        if older is None or older.id == document.id:
            continue
        status = str(older.metadata.get("status") or "current").lower()
        if status == "current":
            older.metadata["status"] = "superseded"
            older.metadata["superseded_by"] = document.id
    return documents


def load_corpus(source: str) -> list[Document]:
    """Read one file or a directory. Skip eval and tune artifacts."""
    return mark_superseded(LocalFolderLoader(source).load())

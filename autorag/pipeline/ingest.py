"""Load documents from a source into a common record.

Today the working loader reads a local folder of Markdown, text, and PDF files.
Notion, Google Drive, and URL crawl adapters share the same ``load() -> documents``
shape so they can be filled in without touching chunking or retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

from autorag.models import Document

SKIP_NAMES = {"eval.jsonl", "suggested.json", "tune_results.json"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


class SourceNotConfigured(RuntimeError):
    """Raised when a loader's interface exists but its credentials do not."""


class SourceLoader(Protocol):
    def load(self) -> list[Document]:
        """Return documents in a stable order."""


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


class UrlLoader:
    """Fetch one or more web pages and keep their visible text.

    ``client`` can be an httpx.Client, which tests replace with a mock transport.
    """

    def __init__(self, urls: list[str], client=None):
        self.urls = urls
        self.client = client

    def load(self) -> list[Document]:
        import httpx

        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=20, follow_redirects=True)
        documents: list[Document] = []
        try:
            for url in self.urls:
                response = client.get(url)
                response.raise_for_status()
                text = html_to_text(response.text)
                documents.append(
                    Document(
                        id=_url_id(url),
                        title=url,
                        text=text,
                        source=url,
                        metadata={"status": "current", "loader": "url"},
                    )
                )
        finally:
            if owns_client:
                client.close()
        return documents


class NotionLoader:
    """Stable interface for a future Notion export.

    Requires ``NOTION_TOKEN`` and ``NOTION_ROOT_PAGE``. The client is intentionally
    not bundled; calling ``load`` explains what is missing instead of pretending
    to return documents.
    """

    def __init__(self, token: str | None = None, root_page: str | None = None):
        self.token = token
        self.root_page = root_page

    def load(self) -> list[Document]:
        if not self.token or not self.root_page:
            raise SourceNotConfigured(
                "NotionLoader.load needs a Notion integration token and a root page id. "
                "The loader interface matches LocalFolderLoader; the API client is not bundled."
            )
        raise SourceNotConfigured(
            "NotionLoader is wired as an interface only. Drop in a Notion client that "
            "returns autorag.models.Document records from the root page."
        )


class GoogleDriveLoader:
    """Stable interface for a future Google Drive folder sync."""

    def __init__(self, folder_id: str | None = None, credentials_path: str | None = None):
        self.folder_id = folder_id
        self.credentials_path = credentials_path

    def load(self) -> list[Document]:
        if not self.folder_id or not self.credentials_path:
            raise SourceNotConfigured(
                "GoogleDriveLoader.load needs a folder id and a credentials file. "
                "The loader interface matches LocalFolderLoader; the Drive client is not bundled."
            )
        raise SourceNotConfigured(
            "GoogleDriveLoader is wired as an interface only. Drop in a Drive client that "
            "returns autorag.models.Document records from the folder."
        )


def html_to_text(html: str) -> str:
    import html as html_lib

    without_blocks = re_sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    with_breaks = re_sub(r"(?i)<br\s*/?>", "\n", without_blocks)
    with_breaks = re_sub(r"(?i)</p>", "\n\n", with_breaks)
    with_breaks = re_sub(r"(?i)<h[1-6][^>]*>", "\n# ", with_breaks)
    stripped = re_sub(r"<[^>]+>", " ", with_breaks)
    unescaped = html_lib.unescape(stripped)
    lines = [" ".join(line.split()) for line in unescaped.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def re_sub(pattern: str, repl: str, text: str) -> str:
    import re

    return re.sub(pattern, repl, text)


def _url_id(url: str) -> str:
    import re

    cleaned = re.sub(r"^https?://", "", url).strip("/")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", cleaned).strip("-").lower()
    return cleaned[:80] or "url"


def load_corpus(source: str) -> list[Document]:
    """Dispatch a source string to the right loader.

    ``notion:<page>`` and ``gdrive:<folder>`` call the future adapters.
    ``http(s):`` URLs are fetched. Anything else is a local file or folder.
    """
    if source.startswith("notion:"):
        page = source.split(":", 1)[1]
        import os

        return NotionLoader(token=os.environ.get("NOTION_TOKEN"), root_page=page or None).load()
    if source.startswith("gdrive:"):
        folder = source.split(":", 1)[1]
        import os

        return GoogleDriveLoader(
            folder_id=folder or None,
            credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        ).load()
    if source.startswith("http://") or source.startswith("https://"):
        return UrlLoader([source]).load()
    return LocalFolderLoader(source).load()

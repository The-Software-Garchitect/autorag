"""Download the public SciFact release and cache it next to the example.

The raw tarball and the generated abstract files stay out of git. What gets
committed is the license note and, after a real run, the score file.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from autorag.claims.scifact import write_claims, write_corpus

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
# Members inside the official tarball. The test split has no public labels and is ignored.
CORPUS_MEMBER = "data/corpus.jsonl"
TRAIN_MEMBER = "data/claims_train.jsonl"
DEV_MEMBER = "data/claims_dev.jsonl"


def fetch_scifact(dest: str | Path, url: str = SCIFACT_URL) -> dict:
    root = Path(dest)
    cache = root / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "data.tar.gz"
    if not archive.exists() or archive.stat().st_size == 0:
        _download(url, archive)
    corpus_rows = _read_jsonl(archive, CORPUS_MEMBER)
    train_rows = _read_jsonl(archive, TRAIN_MEMBER)
    dev_rows = _read_jsonl(archive, DEV_MEMBER)
    abstracts, index_maps = write_corpus(root / "docs", corpus_rows)
    train = write_claims(cache / "claims_train.jsonl", train_rows, abstracts, index_maps)
    dev = write_claims(cache / "claims_dev.jsonl", dev_rows, abstracts, index_maps)
    summary = {
        "dataset": "scifact",
        "source": str(root),
        "url": url,
        "documents": len(corpus_rows),
        "train": len(train),
        "dev": len(dev),
        "test_labels": "hidden; not downloaded into the claim cache",
    }
    (cache / "fetch.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "autorag-scifact"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(dest)


def _read_jsonl(archive: Path, member: str) -> list[dict]:
    with tarfile.open(archive, "r:gz") as tar:
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} is not in {archive}")
        rows = []
        for line in extracted:
            text = line.decode("utf-8").strip()
            if text:
                rows.append(json.loads(text))
        return rows

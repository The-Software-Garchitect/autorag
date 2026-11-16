from pathlib import Path

from autorag.pipeline.ingest import LocalFolderLoader, load_corpus


def test_local_folder_reads_markdown_frontmatter(corpus_dir: Path):
    docs = LocalFolderLoader(corpus_dir).load()
    by_id = {doc.id: doc for doc in docs}
    assert set(by_id) == {"snacks", "snacks-2023", "rooms", "training"}
    assert by_id["snacks"].title == "Team Snacks"
    assert by_id["snacks"].metadata["status"] == "current"
    assert "12" in by_id["snacks"].text
    assert by_id["snacks-2023"].metadata["status"] == "superseded"


def test_plain_text_file(tmp_path: Path):
    (tmp_path / "note.txt").write_text("Badges are issued at reception.\n", encoding="utf-8")
    docs = load_corpus(str(tmp_path))
    assert len(docs) == 1
    assert "Badges" in docs[0].text


def test_pdf_loader(tmp_path: Path):
    pdf = tmp_path / "vpn.pdf"
    pdf.write_bytes(_simple_pdf("The VPN gateway hostname is vpn.acme.example."))
    docs = load_corpus(str(pdf))
    assert len(docs) == 1
    assert "vpn.acme.example" in docs[0].text


def test_supersedes_field_marks_the_older_page(tmp_path: Path):
    (tmp_path / "menu-2023.md").write_text(
        "---\nid: menu-2023\ntitle: Dinner Menu\nstatus: current\neffective_date: 2023-01-01\n---\n\nThe burger is $12.\n",
        encoding="utf-8",
    )
    (tmp_path / "menu-2025.md").write_text(
        "---\nid: menu-2025\ntitle: Dinner Menu\nstatus: current\neffective_date: 2025-01-01\nsupersedes: menu-2023\n---\n\nThe burger is $16.\n",
        encoding="utf-8",
    )
    docs = {doc.id: doc for doc in load_corpus(str(tmp_path))}
    assert docs["menu-2023"].metadata["status"] == "superseded"
    assert docs["menu-2023"].metadata["superseded_by"] == "menu-2025"
    assert docs["menu-2025"].metadata["status"] == "current"


def _simple_pdf(text: str) -> bytes:
    """A one-page PDF with a single Helvetica line, offsets calculated."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(out)

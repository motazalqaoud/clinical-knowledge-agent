"""Helper to generate small synthetic PDFs for tests, avoiding committed binaries."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def make_sample_pdf(path: Path, pages_text: list[str] | None = None) -> Path:
    """Generates a small multi-page PDF at ``path`` for testing.

    Args:
        path: Destination path for the generated PDF file.
        pages_text: Text content for each page. Defaults to a two-page
            synthetic clinical snippet.

    Returns:
        The same ``path``, for convenient chaining.
    """
    if pages_text is None:
        pages_text = [
            "Clinical Guideline (synthetic test fixture)\n"
            "Metformin 500mg PO BID with meals.",
            "Lab targets: HbA1c < 7.0% for most adults with type 2 diabetes.",
        ]

    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path

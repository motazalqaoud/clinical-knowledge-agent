"""Load clinical documents (PDF, TXT, Markdown) into a common structure."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is not one of the supported types."""


@dataclass
class LoadedDocument:
    """A single page or section of text loaded from a source file.

    Args:
        text: The extracted text content.
        source: The filename the text was loaded from.
        page: 1-indexed page number (always 1 for .txt/.md, since the
            whole file is treated as a single page).
        metadata: Optional extra metadata (e.g. loader-specific info).

    Clinical note: source and page metadata must be preserved end-to-end
    so every generated answer can be traced back to an exact document and
    location, which is required for clinician verification of any cited
    claim.
    """

    text: str
    source: str
    page: int
    metadata: dict = field(default_factory=dict)


def load_text_file(path: Path) -> list[LoadedDocument]:
    """Loads a plain-text (.txt) file as a single document.

    Args:
        path: Path to the .txt file.

    Returns:
        A single-element list containing the file's text as page 1.

    Clinical note: no transformation is applied to the raw text so that
    downstream chunking sees the exact source content.
    """
    text = path.read_text(encoding="utf-8")
    return [LoadedDocument(text=text, source=path.name, page=1)]


def load_markdown_file(path: Path) -> list[LoadedDocument]:
    """Loads a Markdown (.md) file as a single document.

    Args:
        path: Path to the .md file.

    Returns:
        A single-element list containing the file's text as page 1.

    Clinical note: Markdown is loaded as raw text (not rendered) so that
    formatting markers never interfere with medical-aware chunking.
    """
    text = path.read_text(encoding="utf-8")
    return [LoadedDocument(text=text, source=path.name, page=1)]


def load_pdf_file(path: Path) -> list[LoadedDocument]:
    """Loads a PDF file, producing one LoadedDocument per page.

    Args:
        path: Path to the .pdf file.

    Returns:
        A list of LoadedDocument, one per page, in page order.

    Clinical note: per-page granularity is preserved so citations can
    point a clinician to the exact page of a guideline or drug insert.
    """
    import fitz  # PyMuPDF

    documents: list[LoadedDocument] = []
    with fitz.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text")
            if not text.strip():
                logger.warning(
                    "Page %d of %s produced no extractable text.",
                    page_index + 1,
                    path.name,
                )
            documents.append(
                LoadedDocument(text=text, source=path.name, page=page_index + 1)
            )
    return documents


def load_document(path: str | Path) -> list[LoadedDocument]:
    """Dispatches to the correct loader based on file extension.

    Args:
        path: Path to a .txt, .md, or .pdf file.

    Returns:
        A list of LoadedDocument extracted from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        UnsupportedFileTypeError: If the file extension is not supported.

    Clinical note: errors are raised explicitly with the offending path
    and supported-type list rather than failing silently, so a bad
    ingestion input is never mistaken for an empty-but-valid document.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".md":
        return load_markdown_file(path)
    if suffix == ".pdf":
        return load_pdf_file(path)

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{path.suffix}' for {path}. "
        f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def load_directory(
    dir_path: str | Path, recursive: bool = False
) -> list[LoadedDocument]:
    """Loads every supported file in a directory.

    Args:
        dir_path: Path to the directory to scan.
        recursive: If True, scans subdirectories as well.

    Returns:
        A list of LoadedDocument aggregated across every supported file.
        Unsupported files are skipped (with a logged warning) rather than
        aborting the whole ingestion run.

    Raises:
        FileNotFoundError: If dir_path does not exist or is not a directory.

    Clinical note: a single malformed or unsupported file in a batch
    upload should not block ingestion of the rest of a clinical document
    set; the skip is surfaced via an explicit log warning rather than
    silently dropped.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = "**/*" if recursive else "*"
    documents: list[LoadedDocument] = []
    for file_path in sorted(dir_path.glob(pattern)):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Skipping unsupported file during directory load: %s", file_path
            )
            continue
        documents.extend(load_document(file_path))
    return documents

import logging

import pytest

from src.ingestion.document_loader import (
    UnsupportedFileTypeError,
    load_directory,
    load_document,
    load_markdown_file,
    load_pdf_file,
    load_text_file,
)
from tests.fixtures.pdf_factory import make_sample_pdf


def test_load_text_file(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Patient stable. Metformin 500mg BID.", encoding="utf-8")

    docs = load_text_file(path)

    assert len(docs) == 1
    assert docs[0].text == "Patient stable. Metformin 500mg BID."
    assert docs[0].source == "note.txt"
    assert docs[0].page == 1


def test_load_markdown_file(tmp_path):
    path = tmp_path / "guideline.md"
    path.write_text("# Guideline\n\nHbA1c < 7.0% target.", encoding="utf-8")

    docs = load_markdown_file(path)

    assert len(docs) == 1
    assert "HbA1c < 7.0%" in docs[0].text
    assert docs[0].source == "guideline.md"
    assert docs[0].page == 1


def test_load_pdf_file(tmp_path):
    pdf_path = make_sample_pdf(tmp_path / "sample.pdf")

    docs = load_pdf_file(pdf_path)

    assert len(docs) == 2
    assert docs[0].page == 1
    assert docs[1].page == 2
    assert "Metformin" in docs[0].text
    assert "HbA1c" in docs[1].text
    assert all(d.source == "sample.pdf" for d in docs)


def test_load_document_dispatches_by_suffix(tmp_path):
    txt_path = tmp_path / "a.txt"
    txt_path.write_text("hello", encoding="utf-8")

    docs = load_document(txt_path)

    assert docs[0].text == "hello"


def test_load_document_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_document(missing)


def test_load_document_unsupported_type_raises(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("data", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        load_document(path)

    message = str(exc_info.value)
    assert ".docx" in message
    assert ".txt" in message and ".md" in message and ".pdf" in message


def test_load_directory_skips_unsupported_and_logs(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.docx").write_text("beta", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        docs = load_directory(tmp_path)

    assert len(docs) == 1
    assert docs[0].text == "alpha"
    assert any("b.docx" in record.message for record in caplog.records)


def test_load_directory_missing_dir_raises(tmp_path):
    missing_dir = tmp_path / "nope"

    with pytest.raises(FileNotFoundError):
        load_directory(missing_dir)

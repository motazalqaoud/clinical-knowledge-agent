import re

from src.ingestion.chunker import (
    DOSAGE_PATTERN,
    FREQUENCY_PATTERN,
    LAB_VALUE_PATTERN,
    Chunk,
    ChunkerConfig,
    chunk_document,
    chunk_documents,
    find_protected_spans,
    split_into_sentences,
)
from src.ingestion.document_loader import LoadedDocument


def _assert_pattern_never_split(chunks: list[Chunk], original_text: str, pattern):
    """Every protected-pattern match in the original text must appear
    fully intact within a single chunk."""
    matches = [m.group(0) for m in pattern.finditer(original_text)]
    assert matches, "test setup error: expected at least one pattern match"
    for matched in matches:
        assert any(
            matched in chunk.text for chunk in chunks
        ), f"Protected phrase {matched!r} was split across chunk boundaries"


FILLER = "The patient reports feeling well today and denies new symptoms. "


def test_dosage_never_split():
    text = FILLER * 4 + "Continue Metformin 500mg twice daily with meals. " + FILLER * 4
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=60, overlap=10, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    _assert_pattern_never_split(chunks, text, DOSAGE_PATTERN)


def test_lab_value_never_split():
    text = (
        FILLER * 4
        + "Target HbA1c < 7.0% for most adults with type 2 diabetes. "
        + FILLER * 4
    )
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=60, overlap=10, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    _assert_pattern_never_split(chunks, text, LAB_VALUE_PATTERN)


def test_lab_value_decimal_point_not_treated_as_sentence_end():
    text = "Target HbA1c < 7.0% is recommended. Continue current regimen."
    doc = LoadedDocument(text=text, source="note.txt", page=1)

    chunks = chunk_document(doc, ChunkerConfig(target_chunk_size=1000))

    assert len(chunks) == 1
    assert "HbA1c < 7.0%" in chunks[0].text


def test_frequency_abbreviation_never_split():
    text = FILLER * 4 + "Take one tablet BID until follow-up visit. " + FILLER * 4
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=60, overlap=10, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    _assert_pattern_never_split(chunks, text, FREQUENCY_PATTERN)


def test_all_protected_patterns_together_across_many_boundaries():
    text = (
        FILLER * 3
        + "Continue Metformin 500mg BID. "
        + FILLER * 3
        + "Target HbA1c < 7.0% at next visit. "
        + FILLER * 3
        + "Give insulin 10 units QHS as needed. "
        + FILLER * 3
    )
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=80, overlap=15, min_chunk_size=25)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 3
    for pattern in (DOSAGE_PATTERN, LAB_VALUE_PATTERN, FREQUENCY_PATTERN):
        _assert_pattern_never_split(chunks, text, pattern)


def test_chunk_size_respected_without_protected_patterns():
    sentences = [
        f"Sentence number {i} is recorded here in the visit note." for i in range(1, 15)
    ]
    text = " ".join(sentences)
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=100, overlap=0, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= config.target_chunk_size


def test_overlap_present_between_consecutive_chunks():
    sentences = [
        f"Sentence number {i} is recorded here in the note." for i in range(1, 21)
    ]
    text = " ".join(sentences)
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=100, overlap=30, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        nums_current = set(re.findall(r"number (\d+)", chunks[i].text))
        nums_next = set(re.findall(r"number (\d+)", chunks[i + 1].text))
        assert nums_current & nums_next, "expected overlapping content between chunks"


def test_no_overlap_when_configured_zero():
    sentences = [f"Sentence number {i} is here." for i in range(1, 21)]
    text = " ".join(sentences)
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=80, overlap=0, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        nums_current = set(re.findall(r"number (\d+)", chunks[i].text))
        nums_next = set(re.findall(r"number (\d+)", chunks[i + 1].text))
        assert not (nums_current & nums_next)


def test_single_oversized_sentence_with_protected_value_stays_intact():
    text = (
        "This extended clinical note describes the patient history in detail "
        "without any sentence breaks whatsoever while eventually mentioning "
        "that the patient should continue taking Metformin 500mg as prescribed"
    )
    doc = LoadedDocument(text=text, source="note.txt", page=1)
    config = ChunkerConfig(target_chunk_size=80, overlap=0, min_chunk_size=20)

    chunks = chunk_document(doc, config)

    assert len(chunks) >= 1
    _assert_pattern_never_split(chunks, text, DOSAGE_PATTERN)


def test_empty_document_returns_no_chunks():
    doc = LoadedDocument(text="", source="empty.txt", page=1)
    assert chunk_document(doc) == []


def test_whitespace_only_document_returns_no_chunks():
    doc = LoadedDocument(text="   \n\t  ", source="empty.txt", page=1)
    assert chunk_document(doc) == []


def test_multiple_documents_chunk_indices_restart_per_source():
    docs = [
        LoadedDocument(text="Page one content here.", source="a.pdf", page=1),
        LoadedDocument(text="Page two content here.", source="a.pdf", page=2),
        LoadedDocument(text="Some other document content.", source="b.txt", page=1),
    ]

    chunks = chunk_documents(docs)

    a_chunks = [c for c in chunks if c.source == "a.pdf"]
    b_chunks = [c for c in chunks if c.source == "b.txt"]
    assert [c.chunk_index for c in a_chunks] == list(range(len(a_chunks)))
    assert [c.chunk_index for c in b_chunks] == list(range(len(b_chunks)))


def test_split_into_sentences_reconstructs_text():
    text = "First sentence here. Second sentence follows! Is this a third?"
    sentences = split_into_sentences(text)
    assert "".join(sentences) == text
    assert len(sentences) == 3


def test_find_protected_spans_merges_overlaps():
    text = "Dose 500mg BID daily."
    spans = find_protected_spans(text)
    # Non-overlapping and sorted
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2

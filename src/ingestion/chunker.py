"""Medical-aware chunking that never splits dosages, lab values, or dosing
frequency abbreviations across chunk boundaries.

Clinical note: chunk-boundary placement is driven by sentence structure
first and protected-phrase guards second. The core invariant enforced
throughout this module is: size limits are soft, protected-phrase
integrity is hard — a chunk is allowed to exceed its target size before
a dosage, lab value, or frequency abbreviation is ever torn in two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.document_loader import LoadedDocument

DOSAGE_PATTERN = re.compile(
    r"\d+(\.\d+)?\s*(mg|mcg|g|mL|ml|IU|units?)\b", re.IGNORECASE
)
LAB_VALUE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*\s*[<>=≤≥]\s*\d+(\.\d+)?%?")
FREQUENCY_PATTERN = re.compile(r"\b(QD|BID|TID|QID|PRN|QHS)\b")

PROTECTED_PATTERNS: list[re.Pattern] = [
    DOSAGE_PATTERN,
    LAB_VALUE_PATTERN,
    FREQUENCY_PATTERN,
]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    """A chunk of text ready for embedding.

    Args:
        text: The chunk's text content.
        source: The originating document's filename.
        page: The originating document's page number.
        chunk_index: 0-indexed position of this chunk within its source
            document (renumbered per-source when chunking multiple docs).
    """

    text: str
    source: str
    page: int
    chunk_index: int


@dataclass
class ChunkerConfig:
    """Configuration for chunk sizing.

    Args:
        target_chunk_size: Target chunk size in characters. Character
            counts (not tokens) are used deliberately to keep the
            algorithm dependency-free and directly unit-testable; this
            is a conservative proxy for the embedding model's token
            budget, since characters are always >= tokens.
        overlap: Approximate number of characters of overlap carried
            from the end of one chunk into the start of the next,
            snapped to whole sentence boundaries.
        min_chunk_size: Minimum chunk size in characters before the
            assembler is willing to close out a chunk.
    """

    target_chunk_size: int = 512
    overlap: int = 64
    min_chunk_size: int = 64


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merges overlapping or adjacent (start, end) spans."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def find_protected_spans(text: str) -> list[tuple[int, int]]:
    """Finds every protected-phrase span (dosage, lab value, frequency).

    Args:
        text: The document text to scan.

    Returns:
        Merged, non-overlapping (start, end) character spans covering
        every protected-pattern match in ``text``.

    Clinical note: this is computed once per document and reused by both
    sentence splitting and chunk-boundary guarding so a dosage, lab
    value, or frequency abbreviation is protected consistently
    everywhere a boundary could be placed.
    """
    spans = []
    for pattern in PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    return _merge_spans(spans)


def _split_into_sentence_spans(
    text: str, protected_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Splits text into sentence (start, end) spans that partition it fully.

    Candidate sentence boundaries falling inside a protected span are
    discarded, which also prevents decimal points inside lab values
    (e.g. the '.' in "7.0%") from being misread as sentence endings.
    """
    if not text:
        return []
    candidates = [m.start() for m in _SENTENCE_BOUNDARY.finditer(text)]
    valid = [c for c in candidates if not any(s < c < e for s, e in protected_spans)]
    boundaries = sorted(set([0, *valid, len(text)]))
    return [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]


def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences, guarding protected clinical phrases.

    Args:
        text: The text to split.

    Returns:
        A list of sentence strings (in order) that, if concatenated,
        reconstruct the original text.
    """
    protected_spans = find_protected_spans(text)
    spans = _split_into_sentence_spans(text, protected_spans)
    return [text[start:end] for start, end in spans]


def _fix_boundary_if_straddles_protected_span(
    text: str,
    boundary: int,
    protected_spans: list[tuple[int, int]],
    push_forward: bool = False,
) -> int:
    """Adjusts a candidate chunk boundary that falls inside a protected span.

    Args:
        text: The full document text.
        boundary: A candidate character offset for a chunk boundary.
        protected_spans: Protected spans from `find_protected_spans`.
        push_forward: If True, a straddling boundary is pushed to the end
            of the protected span (extending the current chunk, used
            when there is no room to shrink). If False (default), it is
            pushed to the start of the span (deferring the phrase to the
            next chunk).

    Returns:
        An adjusted boundary that never falls strictly inside a
        protected span.

    Clinical note: this is the last line of defense preventing a
    dosage, lab value, or frequency abbreviation from being torn across
    two chunks and losing clinical meaning (e.g. "500" in one chunk,
    "mg" in the next).
    """
    for start, end in protected_spans:
        if start < boundary < end:
            return end if push_forward else start
    return boundary


def _hard_split_span(
    text: str,
    start: int,
    end: int,
    protected_spans: list[tuple[int, int]],
    target_size: int,
) -> list[tuple[int, int]]:
    """Hard-splits an oversized span by character count, guarding protected
    phrases by extending a chunk forward rather than ever cutting one."""
    spans = []
    pos = start
    while pos < end:
        tentative_end = min(pos + target_size, end)
        tentative_end = _fix_boundary_if_straddles_protected_span(
            text, tentative_end, protected_spans, push_forward=True
        )
        if tentative_end <= pos:
            tentative_end = min(pos + 1, end)
        spans.append((pos, tentative_end))
        pos = tentative_end
    return spans


def _assemble_chunk_spans(
    sentence_spans: list[tuple[int, int]],
    text: str,
    protected_spans: list[tuple[int, int]],
    config: ChunkerConfig,
) -> list[tuple[int, int]]:
    """Greedily groups sentence spans into chunk spans with overlap."""
    chunks: list[tuple[int, int]] = []
    n = len(sentence_spans)
    i = 0
    while i < n:
        chunk_start = sentence_spans[i][0]
        chunk_end = sentence_spans[i][1]
        j = i + 1
        while j < n:
            candidate_end = sentence_spans[j][1]
            if (candidate_end - chunk_start) > config.target_chunk_size and (
                chunk_end - chunk_start
            ) >= config.min_chunk_size:
                break
            chunk_end = candidate_end
            j += 1

        if j == i + 1 and (chunk_end - chunk_start) > config.target_chunk_size:
            # A single sentence alone exceeds the target size; hard-split it,
            # protecting any clinical phrase rather than truncating it.
            chunks.extend(
                _hard_split_span(
                    text,
                    chunk_start,
                    chunk_end,
                    protected_spans,
                    config.target_chunk_size,
                )
            )
        else:
            chunk_end = _fix_boundary_if_straddles_protected_span(
                text, chunk_end, protected_spans
            )
            chunks.append((chunk_start, chunk_end))

        if j >= n:
            break

        if config.overlap > 0 and j - 1 > i:
            # Multi-sentence chunk: carry trailing sentences back for overlap,
            # guaranteeing progress by never retreating past sentence i.
            m = j - 1
            carried_len = chunk_end - sentence_spans[m][0]
            while carried_len < config.overlap and m > i + 1:
                m -= 1
                carried_len = chunk_end - sentence_spans[m][0]
            i = m
        else:
            i = j

    return chunks


def chunk_document(
    doc: LoadedDocument, config: ChunkerConfig | None = None
) -> list[Chunk]:
    """Splits a loaded document into medical-aware chunks.

    Args:
        doc: The document to chunk.
        config: Chunk sizing configuration. Defaults to ChunkerConfig().

    Returns:
        A list of Chunk objects in order, with 0-indexed chunk_index.
        Returns an empty list for a document with no non-whitespace text.

    Clinical note: dosages, lab values, and dosing-frequency
    abbreviations are never split across the returned chunks.
    """
    config = config or ChunkerConfig()
    text = doc.text
    if not text.strip():
        return []

    protected_spans = find_protected_spans(text)
    sentence_spans = _split_into_sentence_spans(text, protected_spans)
    chunk_spans = _assemble_chunk_spans(sentence_spans, text, protected_spans, config)

    return [
        Chunk(text=text[start:end], source=doc.source, page=doc.page, chunk_index=idx)
        for idx, (start, end) in enumerate(chunk_spans)
    ]


def chunk_documents(
    docs: list[LoadedDocument], config: ChunkerConfig | None = None
) -> list[Chunk]:
    """Chunks multiple documents, renumbering chunk_index per source.

    Args:
        docs: Documents to chunk, e.g. pages from `load_document`.
        config: Chunk sizing configuration. Defaults to ChunkerConfig().

    Returns:
        A list of Chunk objects across all documents, with chunk_index
        restarting at 0 for each distinct source filename.
    """
    config = config or ChunkerConfig()
    counters: dict[str, int] = {}
    all_chunks: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_document(doc, config):
            index = counters.get(doc.source, 0)
            chunk.chunk_index = index
            counters[doc.source] = index + 1
            all_chunks.append(chunk)
    return all_chunks

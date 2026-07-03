# Architecture

## Pipeline

```
document → document_loader → chunker → embedder → vector_store
                                                        │
question ──────────────────────────────────────────────┤
                                                        ▼
                                                   rag_chain
                                          (retrieve, threshold check,
                                           grounded prompt, generate,
                                           disclaimer)
                                                        │
                                                        ▼
                                              answer + sources
```

Each stage is a small, independently testable module:

| Module | Responsibility |
|---|---|
| `src/ingestion/document_loader.py` | Loads `.txt`/`.md`/`.pdf` into `LoadedDocument` (text, source, page). |
| `src/ingestion/chunker.py` | Splits documents into medical-aware `Chunk`s. |
| `src/embeddings/embedder.py` | Wraps `sentence-transformers/all-MiniLM-L6-v2`; L2-normalized vectors. |
| `src/retrieval/vector_store.py` | Local FAISS `IndexFlatIP` store, persisted under `data/faiss_index/`. |
| `src/utils/clinical_prompts.py` | Grounded-prompt template, disclaimer, insufficient-info detection. |
| `src/generation/rag_chain.py` | Orchestrates retrieve → prompt → generate → safety post-process. |

## Chunking guard (the medical-aware invariant)

`chunker.py` enforces one rule above all others: **size limits are
soft, protected-phrase integrity is hard.** A chunk boundary is never
allowed to land inside a dosage (`500mg`), a lab value (`HbA1c <
7.0%`), or a dosing-frequency abbreviation (`QD`/`BID`/`TID`/`QID`/`PRN`/`QHS`) —
even if that means a chunk exceeds its target size.

This is implemented in two layers:

1. **Sentence splitting respects protected spans.** `find_protected_spans`
   locates every protected-pattern match once per document. Sentence
   boundary candidates (from punctuation + whitespace + capital-letter
   regex matching) that fall inside a protected span are discarded — this
   also prevents a decimal point inside a lab value from being
   misread as a sentence end.
2. **A boundary guard runs as defense-in-depth.** `_fix_boundary_if_straddles_protected_span`
   re-checks every chunk boundary; if one somehow still lands inside a
   protected span, it is pushed to the span's start (deferring the
   phrase to the next chunk) in normal chunking, or to the span's end
   (extending the current chunk) in the rare case where a single
   sentence itself exceeds the target chunk size and must be hard-split.

## Grounding and the insufficient-information fallback

`RAGChain.query()`:

1. Embeds the question and retrieves the top-`k` chunks by cosine
   similarity (inner product on normalized vectors).
2. **Short-circuits to the insufficient-information response — without
   calling the generator at all — if the store is empty or the best
   match's score is below `score_threshold` (default `0.35`).** This
   both saves an unnecessary generation call and guarantees an
   ungrounded query can never reach the model.
3. Otherwise builds a prompt (`clinical_prompts.build_grounded_prompt`)
   that instructs the model to answer only from the given context and
   to reply with the exact insufficient-information phrase if it can't.
4. Checks the model's own output for that phrase (`is_insufficient`) —
   if the model itself declines to answer, the response is still marked
   ungrounded, even though retrieval succeeded.
5. Appends the safety disclaimer (idempotently) to every answer, grounded
   or not.

`score_threshold=0.35` is a tunable starting point documented in the
constructor, not a clinical claim — teams adopting this for a specific
corpus should validate it against their own retrieval score
distribution.

## Testing strategy

The test suite is fully offline. `chunker.py` needs no model and is
tested directly. `embedder.py` and `rag_chain.py` accept
dependency-injected models (`ClinicalEmbedder(model=...)`,
`RAGChain(..., generator=...)`); tests inject deterministic fakes
(`tests/conftest.py`) instead of downloading real weights. `vector_store.py`
uses `faiss-cpu`, which is local and offline already, so it's tested
directly with hand-built vectors. This keeps CI fast and deterministic
regardless of network conditions.

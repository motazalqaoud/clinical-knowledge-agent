# Clinical Knowledge Agent

[![CI](https://github.com/motazalqaoud/clinical-knowledge-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/motazalqaoud/clinical-knowledge-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A **Retrieval-Augmented Generation (RAG) system** for querying clinical documents — guidelines, research papers, drug inserts — with grounded, cited, zero-hallucination answers, running **100% locally**.

## What is this about?

Clinical text is unforgiving of small errors: a dropped "mg", a torn-off "BID", or a split lab-value threshold can flip an answer's meaning. This project treats that as a first-class engineering problem, not an afterthought.

Key capabilities:

- **Medical-aware chunking** — never splits a drug dosage (`500mg`), a lab value (`HbA1c < 7.0%`), or a dosing-frequency abbreviation (`QD`, `BID`, `TID`, `QID`, `PRN`, `QHS`) across a chunk boundary
- **Zero-hallucination generation** — answers only from retrieved context; explicitly says "insufficient information" instead of guessing, and never even calls the generator when retrieval confidence is too low
- **100% local pipeline** — `sentence-transformers/all-MiniLM-L6-v2` for embeddings, FAISS for retrieval, `Qwen2.5-1.5B-Instruct` for generation — no cloud APIs, no OpenAI key, no patient data ever leaves the machine it runs on
- **Mandatory safety disclaimer** — every response, grounded or not, ends with a consult-a-licensed-professional notice
- **Lightweight evaluation harness** (`scripts/evaluate.py`) — scripted retrieval/groundedness checks against the sample documents, not just unit tests
- **Fully offline test suite** — model-dependent components are dependency-injected, so `pytest` never touches the network

## Architecture

```
                 ┌─────────────────────┐
 .txt/.md/.pdf → │  document_loader.py │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │     chunker.py      │  medical-aware chunking
                 │ (protects dosages,  │  (never splits protected
                 │  lab values, freq.  │   phrases across chunks)
                 │  abbreviations)     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │    embedder.py      │  sentence-transformers/
                 │                     │  all-MiniLM-L6-v2 (local)
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  vector_store.py    │  FAISS IndexFlatIP
                 │  (data/faiss_index) │  (100% local, no cloud)
                 └──────────┬──────────┘
                            ▼       ┌─────────────────────────┐
                 query ──────────▶ │        rag_chain.py      │
                                    │    retrieve → grounded  │
                                    │    prompt → generate    │
                                    │ (Qwen2.5-1.5B-Instruct) │
                                    └────────────┬────────────┘
                                                 ▼
                                    answer + sources + disclaimer
                                    (or explicit "insufficient
                                     information" if ungrounded)
                            ▲
                            │
                 ┌─────────────────────┐
                 │       app.py        │  Gradio UI
                 └─────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for more detail on the
chunking guard and the grounding/insufficient-information logic.

## Repository structure

```
clinical-knowledge-agent/
├── src/
│   ├── ingestion/      document_loader.py, chunker.py
│   ├── embeddings/     embedder.py
│   ├── retrieval/      vector_store.py
│   ├── generation/     rag_chain.py
│   └── utils/          clinical_prompts.py
├── scripts/
│   └── evaluate.py     retrieval/groundedness evaluation harness
├── app.py              Gradio UI
├── examples/           synthetic sample clinical documents
├── tests/              pytest suite (fully offline — no model downloads)
├── notebooks/          demo walkthrough notebook
├── docs/               architecture notes
├── deploy/             Hugging Face Space deployment files
└── data/               local FAISS index + uploads (gitignored)
```

## Quickstart

Requires Python 3.11+.

```bash
pip install -r requirements.txt
python app.py
```

Open the printed local URL, click **"Load sample documents"** to ingest
the bundled synthetic guideline, then ask a question such as *"What is
the target HbA1c?"*.

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` and
`Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face — this requires network
access to `huggingface.co`. After that, everything runs locally and offline.

## Verified example

Real output from the deployed app (bundled synthetic sample documents):

> **Q:** What is the target HbA1c for most adults with type 2 diabetes?
>
> **A:** The target HbA1c for most non-pregnant adults is < 7.0%. A less
> stringent goal of HbA1c < 8.0% may be appropriate for patients with a
> history of severe hypoglycemia, limited life expectancy, or extensive
> comorbid conditions.
>
> **Sources:** diabetes_management_guideline.md (page 1, score 0.59),
> diabetes_management_guideline.md (page 1, score 0.54)
>
> ---
> **This information is for educational purposes only and is not a
> substitute for professional medical judgment. Please consult a
> licensed healthcare professional before making any clinical or
> treatment decisions.**

And the safety behavior on an out-of-scope question — no fabrication, an
explicit refusal instead:

> **Q:** What is the capital of France?
>
> **A:** I don't have enough information in the provided documents to
> answer this question confidently.
>
> ---
> **This information is for educational purposes only and is not a
> substitute for professional medical judgment. Please consult a
> licensed healthcare professional before making any clinical or
> treatment decisions.**

## Evaluation

`scripts/evaluate.py` runs a small fixed set of questions against the
bundled sample documents and checks two things per question: whether
groundedness matched expectation (should it have answered, or correctly
declined?), and whether expected keywords showed up in grounded answers.

```bash
python scripts/evaluate.py          # real embedder + real generator
python scripts/evaluate.py --fake   # offline dry run of the harness itself
                                     # (no model download; not a real
                                     # quality measurement)
```

This is a lightweight sanity check, not a clinical validation study —
a real accuracy benchmark would need a much larger, clinician-reviewed
question set well beyond what a couple of synthetic sample documents
can support.

### Measured results

> **Note:** the numbers below are from the original 6-question,
> single-document evaluation. The eval set has since grown to 10
> questions across both sample documents (see `scripts/evaluate.py`) —
> these numbers need a re-run to reflect the current set.

Real run against `Qwen/Qwen2.5-1.5B-Instruct` + `all-MiniLM-L6-v2`
(`python scripts/evaluate.py`, no `--fake`):

| Metric | Result |
|---|---|
| Groundedness accuracy | 5/6 |
| Expected-keyword presence | 3/4 |

| Question | Expected | Actual | Result |
|---|---|---|---|
| What is the target HbA1c for most adults with type 2 diabetes? | Grounded | Grounded | PASS |
| What is the typical starting dose of Metformin? | Grounded | Grounded | PASS |
| What is a common starting dose for basal insulin? | Grounded | Grounded | PASS |
| How often should blood glucose be checked during insulin titration? | Grounded | Grounded | PASS |
| What is the capital of France? | Insufficient | Insufficient | PASS |
| What is the recommended surgical approach for a torn ACL? | Insufficient | **Grounded** | FAIL |

The one failure is instructive, not swept under the rug: the ACL
question retrieved a passage from the diabetes guideline whose cosine
similarity landed just above the 0.35 `score_threshold` — general
clinical vocabulary overlap (dosing, monitoring language) can push a
topically-unrelated question over a fixed similarity cutoff even
though the retrieved passage doesn't actually answer it. `score_threshold`
is a heuristic, not a guarantee; a production deployment would want a
larger, domain-matched corpus and a threshold validated against real
query traffic, not a single synthetic document.

## Clinical Context

Most RAG tutorials miss the clinical reality. Here's what's different
about this repo:

| Common RAG Tutorial | This Repo |
|---|---|
| Fixed-size chunking, splits mid-token | Medical-aware chunking — dosages, lab values, frequency abbreviations never split |
| Cloud API + API key required | 100% local — no OpenAI key, no data leaves the machine |
| Always returns an answer | Explicit "insufficient information" fallback rather than fabricating |
| No safety framing | Every response — grounded or not — carries a consult-a-professional disclaimer |
| Hardcoded model calls | Embedder/generator are dependency-injected, swappable, and unit-testable without real models |
| Unit tests only | Unit tests *and* a scripted retrieval/groundedness evaluation harness |

See [`docs/architecture.md`](docs/architecture.md) for the full design rationale.

## Tech Stack

| Tool | Purpose |
|---|---|
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Local vector similarity search |
| `transformers` / `torch` | Local generation model (`Qwen2.5-1.5B-Instruct`) |
| `pymupdf` | PDF text extraction |
| `gradio` | Web UI |
| `pytest` / `ruff` / `black` | Testing, linting, formatting |

## Running the tests

The test suite runs fully offline — no model weights are downloaded.
Model-dependent components (`ClinicalEmbedder`, the generator in
`RAGChain`) are exercised through dependency-injected fakes (see
`tests/conftest.py`).

```bash
pip install -r requirements-dev.txt
pytest -q --cov=src
ruff check src tests app.py scripts
black --check src tests app.py scripts
```

## Limitations & safety

- **Never fabricates medical facts.** Answers are generated only from
  retrieved document context; when that context is insufficient, the
  agent says so explicitly rather than guessing.
- **Always includes a disclaimer** directing users to consult a
  licensed healthcare professional — this is not a diagnostic or
  prescribing tool.
- **Not validated for clinical use.** This is a portfolio/educational
  project. The bundled sample documents are entirely synthetic/fabricated,
  and the evaluation harness is a sanity check, not a clinical accuracy
  study.
- **Local-only by design**, which supports HIPAA-friendly deployments,
  but this repository itself has not undergone a compliance review —
  treat it as a reference implementation, not a certified product.
- **The retrieval score threshold is a heuristic, not a guarantee.**
  Measured evaluation (see above) found an out-of-scope question can
  still retrieve a passage scoring just above `score_threshold` on
  vocabulary overlap alone, producing a grounded-looking answer that
  isn't actually supported by the retrieved content.

## Deployment

`app.py` + `requirements.txt` at the repo root are all that's needed to
deploy this as a [Hugging Face Space](https://huggingface.co/docs/hub/spaces).
A GitHub Actions workflow (`.github/workflows/sync-to-huggingface.yml`)
can push every commit on `main` to the Space automatically once an
`HF_TOKEN` secret is configured — see [`deploy/README.md`](deploy/README.md)
for that one-time setup step, the manual alternative, and the
Space-specific `README.md` front matter (kept separate from this file
so the two don't collide).

## About the Author

**Motaz Alqaoud, PhD**
PhD in Biomedical Engineering with a focus on medical image analysis and deep learning.
Senior AI/ML Engineer specializing in medical imaging, RAG systems, and clinical AI.

- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

## License

MIT License — use freely, attribution appreciated. Open an issue for
questions and collaboration.

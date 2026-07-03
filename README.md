# Clinical Knowledge Agent

[![CI](https://github.com/motazalqaoud/clinical-knowledge-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/motazalqaoud/clinical-knowledge-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A **Retrieval-Augmented Generation (RAG) system** for querying clinical
documents — guidelines, research papers, drug inserts — with grounded,
cited, zero-hallucination answers. Runs **100% locally**: no cloud APIs,
no OpenAI key, no patient data ever leaves the machine it runs on.

**Author:** Motaz Alqaoud, PhD — Biomedical Engineer, Senior AI/ML Engineer.
Part of the [Precision Oncology AI Platform](https://github.com/motazalqaoud/precision-oncology-ai-platform)
(a 9-module portfolio) — see that repo for how this module connects to the others.

## Why this exists

Clinical text is unforgiving of small errors: a dropped "mg", a torn-off
"BID", or a split lab-value threshold can flip an answer's meaning. This
project's chunker is medical-aware — it never splits a drug dosage
(`500mg`), a lab value (`HbA1c < 7.0%`), or a dosing-frequency
abbreviation (`QD`, `BID`, `TID`, `QID`, `PRN`, `QHS`) across a chunk
boundary — and its answers refuse to guess: if the retrieved context
doesn't support an answer, the agent says so explicitly instead of
fabricating one.

## Architecture

```
                 ┌─────────────────────┐
 .txt/.md/.pdf → │  document_loader.py │
                 └──────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │     chunker.py      │  medical-aware chunking
                 │ (protects dosages,  │  (never splits protected
                 │  lab values, freq.  │   phrases across chunks)
                 │  abbreviations)     │
                 └──────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │    embedder.py      │  sentence-transformers/
                 │                     │  all-MiniLM-L6-v2 (local)
                 └──────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │  vector_store.py    │  FAISS IndexFlatIP
                 │  (data/faiss_index) │  (100% local, no cloud)
                 └──────────┬───────────┘
                            ▼      ┌─────────────────────┐
                 query ──────────▶ │    rag_chain.py     │
                                    │ retrieve → grounded  │
                                    │ prompt → generate    │
                                    │ (google/flan-t5-large)│
                                    └──────────┬───────────┘
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

## Project structure

```
clinical-knowledge-agent/
├── src/
│   ├── ingestion/      document_loader.py, chunker.py
│   ├── embeddings/     embedder.py
│   ├── retrieval/      vector_store.py
│   ├── generation/     rag_chain.py
│   └── utils/          clinical_prompts.py
├── app.py              Gradio UI
├── examples/           synthetic sample clinical documents
├── tests/               pytest suite (fully offline — no model downloads)
├── notebooks/          demo walkthrough notebook
├── docs/               architecture notes
└── data/               local FAISS index + uploads (gitignored)
```

## Install

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Open the printed local URL, click **"Load sample documents"** to ingest
the bundled synthetic guideline, then ask a question such as *"What is
the target HbA1c?"*.

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` and
`google/flan-t5-large` from Hugging Face — this requires network access
to `huggingface.co`. After that, everything runs locally and offline.

## Example

> **Q:** What is the target HbA1c for most adults with type 2 diabetes?
>
> **A:** Target HbA1c < 7.0% is recommended for most non-pregnant adults.
>
> **Sources:** diabetes_management_guideline.md (page 1, score 0.81)
>
> ---
> **This information is for educational purposes only and is not a
> substitute for professional medical judgment. Please consult a
> licensed healthcare professional before making any clinical or
> treatment decisions.**

## Running the tests

The test suite runs fully offline — no model weights are downloaded.
Model-dependent components (`ClinicalEmbedder`, the generator in
`RAGChain`) are exercised through dependency-injected fakes (see
`tests/conftest.py`).

```bash
pip install -r requirements-dev.txt
pytest -q --cov=src
ruff check src tests app.py
black --check src tests app.py
```

## Limitations & safety

- **Never fabricates medical facts.** Answers are generated only from
  retrieved document context; when that context is insufficient, the
  agent says so explicitly rather than guessing.
- **Always includes a disclaimer** directing users to consult a
  licensed healthcare professional — this is not a diagnostic or
  prescribing tool.
- **Not validated for clinical use.** This is a portfolio/educational
  project. The bundled sample document is entirely synthetic/fabricated.
- **Local-only by design**, which supports HIPAA-friendly deployments,
  but this repository itself has not undergone a compliance review —
  treat it as a reference implementation, not a certified product.

## Deployment

`app.py` + `requirements.txt` at the repo root are all that's needed to
deploy this as a [Hugging Face Space](https://huggingface.co/docs/hub/spaces).
Deploying there requires adding a Space-specific `README.md` with YAML
front matter (`sdk: gradio`, etc.) per the
[Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference) —
that step is left for when you actually deploy, so it doesn't collide
with this repository's own README.

## License

MIT — see [LICENSE](LICENSE).

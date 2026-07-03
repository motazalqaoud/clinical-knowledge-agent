# AI Coding Assistant Prompt — clinical-knowledge-agent

Paste this into GitHub Copilot Chat, Claude, or Cursor to scaffold the project.

---

## Project Brief

Build a **Clinical Knowledge Agent** — a Retrieval-Augmented Generation (RAG) system for querying medical documents (clinical guidelines, research papers, drug inserts) with grounded, cited answers and zero hallucination.

## Author context
Built by Motaz Alqaoud, PhD — Biomedical Engineer, Senior AI/ML Engineer. Position this as a professional, production-quality open-source project for a healthcare AI portfolio.

## Requirements

### Core pipeline
1. **Document ingestion** — load PDF, TXT, Markdown clinical documents
2. **Medical-aware chunking** — NEVER split drug dosages (e.g. "500mg"), lab values (e.g. "HbA1c < 7.0%"), or dosing frequency abbreviations (QD, BID, TID) across chunk boundaries
3. **Embeddings** — use `sentence-transformers/all-MiniLM-L6-v2` (free, local, fast)
4. **Vector store** — FAISS, 100% local (no cloud, no external API — HIPAA-friendly design)
5. **Generation** — free HuggingFace model (`google/flan-t5-large`), no OpenAI key required
6. **Prompting** — force strict grounding: "answer ONLY from context, say so explicitly if insufficient"
7. **Safety** — every response ends with a disclaimer to consult a licensed healthcare professional

### Tech stack
- Python 3.11
- `sentence-transformers`, `faiss-cpu`, `transformers`, `pymupdf`, `gradio`

### Deliverables
- `src/ingestion/document_loader.py`
- `src/ingestion/chunker.py`
- `src/embeddings/embedder.py`
- `src/retrieval/vector_store.py`
- `src/generation/rag_chain.py`
- `src/utils/clinical_prompts.py`
- `app.py` — Gradio UI, deployable to Hugging Face Spaces
- Full README with architecture diagram (ASCII is fine)
- Unit tests for chunker (verify dosage/lab-value patterns are never split)

### Non-negotiable safety rules
- Never fabricate medical facts not present in retrieved context
- Never suppress the "insufficient information" response when the answer isn't grounded
- Always include a consult-a-professional disclaimer

## Ask me before you start
1. Do you want me to scaffold all files at once or one module at a time?
2. Should I include a sample clinical PDF for testing, or will you provide one?

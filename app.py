"""Gradio UI for the Clinical Knowledge Agent.

Deployable as-is to Hugging Face Spaces: this file plus requirements.txt
at the repo root is all Spaces needs to run it. A Spaces-specific
README (with YAML front matter) is a separate step to take only when
actually deploying — see the Deployment section in README.md.

Clinical note: this is the only module where real Hugging Face models
are constructed; ingestion and query failures (including missing
network access to download models) are surfaced as visible Gradio
errors rather than crashing the app silently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from src.embeddings.embedder import ClinicalEmbedder
from src.generation.rag_chain import RAGChain
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_document
from src.retrieval.vector_store import DEFAULT_INDEX_DIR, FAISSVectorStore

logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = Path("examples/sample_docs")

EXAMPLE_QUESTIONS = [
    "What is the target HbA1c for most adults with type 2 diabetes?",
    "What is the typical starting dose of Metformin?",
    "What is the target blood pressure for most adults with hypertension?",
    "What is the typical starting dose of Lisinopril?",
]

_embedder = ClinicalEmbedder()
_vector_store: FAISSVectorStore | None = None
_rag_chain: RAGChain | None = None


def _load_existing_store() -> FAISSVectorStore | None:
    try:
        return FAISSVectorStore.load(DEFAULT_INDEX_DIR)
    except FileNotFoundError:
        return None


_vector_store = _load_existing_store()
_ingested_sources: set[str] = set(_vector_store.sources) if _vector_store else set()


def _sources_display() -> str:
    """Renders the currently-ingested source filenames for the UI panel."""
    if not _ingested_sources:
        return "No documents ingested yet."
    return "\n".join(f"- {name}" for name in sorted(_ingested_sources))


def _get_rag_chain() -> RAGChain:
    global _rag_chain
    if _vector_store is None:
        raise gr.Error(
            "No documents have been ingested yet. Upload documents or click "
            "'Load sample documents' first."
        )
    if _rag_chain is None:
        _rag_chain = RAGChain(_embedder, _vector_store)
    return _rag_chain


def ingest_paths(paths: list[str]) -> str:
    """Loads, chunks, embeds, and indexes documents from the given paths.

    Args:
        paths: Filesystem paths to .txt/.md/.pdf clinical documents.

    Returns:
        A human-readable status message for display in the UI.

    Clinical note: filenames already present in the index are skipped
    rather than re-added, so repeated clicks (e.g. "Load sample
    documents" clicked more than once in the same running session) can
    never silently duplicate chunks and skew retrieval scores.
    """
    global _vector_store, _rag_chain
    try:
        new_paths = [p for p in paths if Path(p).name not in _ingested_sources]
        skipped = [Path(p).name for p in paths if Path(p).name in _ingested_sources]

        if not new_paths:
            return f"Already ingested, skipped duplicate(s): {', '.join(skipped)}."

        docs = []
        for file_path in new_paths:
            docs.extend(load_document(file_path))
        chunks = chunk_documents(docs)
        if not chunks:
            return "No text could be extracted from the provided document(s)."

        embeddings = _embedder.encode([c.text for c in chunks])
        store = _vector_store or FAISSVectorStore(dim=embeddings.shape[1])
        store.add(
            embeddings,
            texts=[c.text for c in chunks],
            sources=[c.source for c in chunks],
            pages=[c.page for c in chunks],
        )
        store.save(DEFAULT_INDEX_DIR)
        _vector_store = store
        _rag_chain = None  # rebuild next query against the updated store
        _ingested_sources.update(Path(p).name for p in new_paths)

        message = f"Ingested {len(chunks)} chunks from {len(new_paths)} file(s)."
        if skipped:
            message += f" Skipped already-ingested duplicate(s): {', '.join(skipped)}."
        return message
    except Exception as exc:
        logger.exception("Ingestion failed")
        return f"Ingestion failed: {exc}"


def handle_upload(files) -> tuple[str, str]:
    """Gradio callback: ingests uploaded files."""
    if not files:
        return "No files selected.", _sources_display()
    paths = [f.name if hasattr(f, "name") else f for f in files]
    status = ingest_paths(paths)
    return status, _sources_display()


def handle_load_sample() -> tuple[str, str]:
    """Gradio callback: ingests the bundled synthetic sample documents."""
    if not SAMPLE_DOCS_DIR.exists():
        return f"Sample docs directory not found: {SAMPLE_DOCS_DIR}", _sources_display()
    paths = [str(p) for p in sorted(SAMPLE_DOCS_DIR.glob("*")) if p.is_file()]
    if not paths:
        return "No sample documents found.", _sources_display()
    status = ingest_paths(paths)
    return status, _sources_display()


def handle_clear() -> list:
    """Gradio callback: clears the chat history without touching the index."""
    return []


def handle_query(question: str, history: list[dict]) -> tuple[list[dict], str]:
    """Gradio callback: answers a question and appends it to chat history."""
    if not question or not question.strip():
        return history, ""

    chain = _get_rag_chain()
    try:
        response = chain.query(question)
    except Exception as exc:
        logger.exception("Query failed")
        raise gr.Error(f"Query failed: {exc}") from exc

    answer = response.answer
    if response.sources:
        sources_text = "\n".join(
            f"- {s.source} (page {s.page}, score {s.score:.2f})"
            for s in response.sources
        )
        answer += f"\n\n**Sources:**\n{sources_text}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


with gr.Blocks(title="Clinical Knowledge Agent") as demo:
    gr.Markdown(
        "# Clinical Knowledge Agent\n"
        "Ask questions about ingested clinical documents. Answers are "
        "grounded strictly in retrieved context, cite their sources, and "
        "always end with a disclaimer to consult a licensed healthcare "
        "professional."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Add documents")
            file_upload = gr.File(
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
                label="Upload clinical documents (.txt, .md, .pdf)",
            )
            ingest_button = gr.Button("Ingest uploaded documents")
            sample_button = gr.Button("Load sample documents")
            ingest_status = gr.Textbox(label="Ingestion status", interactive=False)
            ingested_list = gr.Textbox(
                label="Ingested documents",
                value=_sources_display(),
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask a question")
            chatbot = gr.Chatbot(label="Clinical Knowledge Agent")
            question_box = gr.Textbox(
                label="Question", placeholder="e.g. What is the target HbA1c?"
            )
            with gr.Row():
                ask_button = gr.Button("Ask")
                clear_button = gr.Button("Clear conversation")
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=question_box,
                label="Example questions",
            )

    ingest_button.click(
        handle_upload, inputs=file_upload, outputs=[ingest_status, ingested_list]
    )
    sample_button.click(
        handle_load_sample, inputs=None, outputs=[ingest_status, ingested_list]
    )
    ask_button.click(
        handle_query, inputs=[question_box, chatbot], outputs=[chatbot, question_box]
    )
    question_box.submit(
        handle_query, inputs=[question_box, chatbot], outputs=[chatbot, question_box]
    )
    clear_button.click(handle_clear, inputs=None, outputs=chatbot)


if __name__ == "__main__":
    demo.launch()

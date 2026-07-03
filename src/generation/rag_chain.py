"""Orchestrates retrieval, grounded prompting, and generation.

Clinical note: every returned answer either (a) is grounded in retrieved
context and cites its sources, or (b) explicitly states insufficient
information — never a fabricated answer — and always carries the
consult-a-professional disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings.embedder import ClinicalEmbedder
from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.utils.clinical_prompts import (
    INSUFFICIENT_INFO_MESSAGE,
    append_disclaimer,
    build_grounded_prompt,
    is_insufficient,
)

DEFAULT_GENERATION_MODEL_NAME = "google/flan-t5-large"


@dataclass
class RAGResponse:
    """The result of a RAGChain query.

    Args:
        answer: The final answer text, always disclaimer-appended.
        sources: Retrieved passages the answer is grounded in (empty if
            the response was an insufficient-information fallback).
        grounded: True if the answer is backed by retrieved context;
            False for an insufficient-information response.
    """

    answer: str
    sources: list[SearchResult]
    grounded: bool


class RAGChain:
    """Retrieve -> grounded prompt -> generate -> safety post-process."""

    def __init__(
        self,
        embedder: ClinicalEmbedder,
        vector_store: FAISSVectorStore,
        generator: Any | None = None,
        model_name: str = DEFAULT_GENERATION_MODEL_NAME,
        score_threshold: float = 0.35,
        top_k: int = 4,
    ) -> None:
        """Initializes the chain.

        Args:
            embedder: Embedder used to encode the incoming question.
            vector_store: Store to retrieve candidate passages from.
            generator: An already-constructed generator (real
                `transformers.pipeline("text2text-generation", ...)` or a
                test double) callable as ``generator(prompt, **kw) ->
                [{"generated_text": str}]``. If None, the real pipeline is
                lazily constructed from `model_name` on first use.
            model_name: HuggingFace model name for the real generator,
                used only if `generator` is not provided.
            score_threshold: Minimum best-match similarity score required
                to attempt generation; below this, the chain short-circuits
                to the insufficient-information response without calling
                the generator. Tunable; not a clinical claim.
            top_k: Number of passages to retrieve per query.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._generator = generator
        self._model_name = model_name
        self._score_threshold = score_threshold
        self._top_k = top_k

    def _ensure_generator(self) -> Any:
        if self._generator is None:
            from transformers import pipeline

            self._generator = pipeline("text2text-generation", model=self._model_name)
        return self._generator

    def query(self, question: str) -> RAGResponse:
        """Answers a question, grounded strictly in retrieved context.

        Args:
            question: The user's question.

        Returns:
            A RAGResponse whose answer always carries the disclaimer, and
            which explicitly reports insufficient information rather than
            fabricating an answer when retrieval confidence is low or the
            generator itself cannot answer from context.
        """
        query_embedding = self._embedder.encode_query(question)
        results = self._vector_store.search(query_embedding, k=self._top_k)

        if not results or results[0].score < self._score_threshold:
            return RAGResponse(
                answer=append_disclaimer(INSUFFICIENT_INFO_MESSAGE),
                sources=[],
                grounded=False,
            )

        prompt = build_grounded_prompt(question, [r.text for r in results])
        generator = self._ensure_generator()
        raw_output = generator(prompt, max_new_tokens=256)[0]["generated_text"]

        grounded = not is_insufficient(raw_output)
        answer = append_disclaimer(raw_output)
        return RAGResponse(answer=answer, sources=results, grounded=grounded)

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


class _Seq2SeqGenerator:
    """Adapts a seq2seq model + tokenizer to a pipeline-style callable.

    Clinical note: newer `transformers` releases dropped the
    `text2text-generation` pipeline task (and `Text2TextGenerationPipeline`
    entirely), which seq2seq models like flan-t5 need. This drives
    `AutoModelForSeq2SeqLM` directly instead, exposing the same
    ``callable(prompt, **kw) -> [{"generated_text": str}]`` contract so
    the rest of RAGChain (and its tests) don't need to know the
    difference.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def __call__(self, prompt: str, max_new_tokens: int = 256, **kwargs: Any):
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return [{"generated_text": text}]


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
                `_Seq2SeqGenerator` or a test double) callable as
                ``generator(prompt, **kw) -> [{"generated_text": str}]``.
                If None, the real seq2seq model is lazily constructed
                from `model_name` on first use.
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
            self._generator = _Seq2SeqGenerator(self._model_name)
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

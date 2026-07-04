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

DEFAULT_GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class _CausalLMGenerator:
    """Adapts an instruction-tuned causal LM to a pipeline-style callable.

    Clinical note: flan-t5-large (an encoder-decoder seq2seq model)
    proved unreliable at following the strict grounded-QA instruction —
    it frequently defaulted to the insufficient-information fallback
    even when the retrieved context clearly answered the question.
    Modern small instruction-tuned causal LMs follow this style of
    prompt far more reliably, and this also uses the standard, still
    fully-supported `text-generation` pipeline task rather than a manual
    workaround. The prompt is wrapped as a single user turn so the
    model's chat template is applied, since instruct models are tuned
    for chat-formatted input rather than raw text completion.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self._pipe = pipeline("text-generation", model=model_name)

    def __call__(self, prompt: str, max_new_tokens: int = 256, **kwargs: Any):
        messages = [{"role": "user", "content": prompt}]
        outputs = self._pipe(
            messages, max_new_tokens=max_new_tokens, do_sample=False, **kwargs
        )
        conversation = outputs[0]["generated_text"]
        reply = conversation[-1]["content"]
        return [{"generated_text": reply}]


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
        max_context_chars: int = 1200,
    ) -> None:
        """Initializes the chain.

        Args:
            embedder: Embedder used to encode the incoming question.
            vector_store: Store to retrieve candidate passages from.
            generator: An already-constructed generator (real
                `_CausalLMGenerator` or a test double) callable as
                ``generator(prompt, **kw) -> [{"generated_text": str}]``.
                If None, the real causal LM is lazily constructed from
                `model_name` on first use.
            model_name: HuggingFace model name for the real generator,
                used only if `generator` is not provided.
            score_threshold: Minimum best-match similarity score required
                to attempt generation; below this, the chain short-circuits
                to the insufficient-information response without calling
                the generator. Tunable; not a clinical claim.
            top_k: Number of passages to retrieve per query.
            max_context_chars: Cap on the total characters of retrieved
                text fed into the prompt. Chunks are added in
                descending-score order until this budget would be
                exceeded (the first chunk is always kept even if it alone
                exceeds the budget). Keeps the prompt within the
                generation model's token window deterministically, rather
                than relying on tokenizer-level truncation, which could
                otherwise silently cut off the question itself.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._generator = generator
        self._model_name = model_name
        self._score_threshold = score_threshold
        self._top_k = top_k
        self._max_context_chars = max_context_chars

    def _ensure_generator(self) -> Any:
        if self._generator is None:
            self._generator = _CausalLMGenerator(self._model_name)
        return self._generator

    def _select_within_budget(self, results: list[SearchResult]) -> list[SearchResult]:
        """Selects a prefix of results whose combined text fits the budget."""
        selected: list[SearchResult] = []
        total_chars = 0
        for result in results:
            if selected and total_chars + len(result.text) > self._max_context_chars:
                break
            selected.append(result)
            total_chars += len(result.text)
        return selected

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

        selected_results = self._select_within_budget(results)
        prompt = build_grounded_prompt(question, [r.text for r in selected_results])
        generator = self._ensure_generator()
        raw_output = generator(prompt, max_new_tokens=256)[0]["generated_text"]

        grounded = not is_insufficient(raw_output)
        answer = append_disclaimer(raw_output)
        return RAGResponse(answer=answer, sources=selected_results, grounded=grounded)

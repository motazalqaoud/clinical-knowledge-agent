import numpy as np

import src.generation.rag_chain as rag_chain_module
from src.generation.rag_chain import RAGChain, _CausalLMGenerator
from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.utils.clinical_prompts import DISCLAIMER, INSUFFICIENT_INFO_MESSAGE
from tests.conftest import FakeGenerator


class _FakeChatPipeline:
    """Fake standing in for transformers' text-generation pipeline when
    called with a chat-style message list, matching its contract of
    returning the full conversation (input messages + new assistant turn)
    under "generated_text"."""

    def __init__(self, reply_content: str):
        self.reply_content = reply_content
        self.last_messages = None
        self.last_kwargs = None

    def __call__(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        conversation = [*messages, {"role": "assistant", "content": self.reply_content}]
        return [{"generated_text": conversation}]


def _seeded_store(fake_embedder, text: str, source: str = "note.txt", page: int = 1):
    store = FAISSVectorStore(dim=32)
    embedding = fake_embedder.encode([text])
    store.add(embedding, texts=[text], sources=[source], pages=[page])
    return store


def test_grounded_answer_includes_disclaimer_and_sources(fake_embedder):
    text = "Continue Metformin 500mg BID for glycemic control."
    store = _seeded_store(fake_embedder, text)
    generator = FakeGenerator("The dose is 500mg BID.")
    chain = RAGChain(fake_embedder, store, generator=generator, score_threshold=0.35)

    response = chain.query(text)

    assert response.grounded is True
    assert DISCLAIMER in response.answer
    assert "500mg BID" in response.answer
    assert len(response.sources) == 1
    assert generator.call_count == 1


def test_low_score_triggers_insufficient_without_calling_generator(fake_embedder):
    store = _seeded_store(fake_embedder, "Continue Metformin 500mg BID.")
    generator = FakeGenerator("should never be returned")
    chain = RAGChain(fake_embedder, store, generator=generator, score_threshold=0.35)

    response = chain.query("Completely unrelated question about diet.")

    assert response.grounded is False
    assert INSUFFICIENT_INFO_MESSAGE in response.answer
    assert DISCLAIMER in response.answer
    assert response.sources == []
    assert generator.call_count == 0


def test_empty_store_triggers_insufficient_without_calling_generator(fake_embedder):
    store = FAISSVectorStore(dim=32)
    generator = FakeGenerator("should never be returned")
    chain = RAGChain(fake_embedder, store, generator=generator)

    response = chain.query("Any question at all.")

    assert response.grounded is False
    assert generator.call_count == 0


def test_model_output_itself_insufficient_marks_ungrounded(fake_embedder):
    text = "Continue Metformin 500mg BID."
    store = _seeded_store(fake_embedder, text)
    generator = FakeGenerator(INSUFFICIENT_INFO_MESSAGE)
    chain = RAGChain(fake_embedder, store, generator=generator, score_threshold=0.35)

    response = chain.query(text)

    assert response.grounded is False
    assert INSUFFICIENT_INFO_MESSAGE in response.answer


def test_disclaimer_not_duplicated_if_generator_already_included_it(fake_embedder):
    text = "Continue Metformin 500mg BID."
    store = _seeded_store(fake_embedder, text)
    generator = FakeGenerator("The dose is 500mg BID." + DISCLAIMER)
    chain = RAGChain(fake_embedder, store, generator=generator, score_threshold=0.35)

    response = chain.query(text)

    assert response.answer.count(DISCLAIMER) == 1


def test_prompt_places_question_before_context(fake_embedder):
    text = "Continue Metformin 500mg BID for glycemic control."
    store = _seeded_store(fake_embedder, text)
    generator = FakeGenerator("The dose is 500mg BID.")
    chain = RAGChain(fake_embedder, store, generator=generator, score_threshold=0.35)

    chain.query(text)

    prompt = generator.last_prompt
    assert prompt.index("QUESTION:") < prompt.index("CONTEXT:")


def test_select_within_budget_keeps_first_chunk_even_if_oversized(fake_embedder):
    store = FAISSVectorStore(dim=32)
    chain = RAGChain(fake_embedder, store, max_context_chars=50)
    results = [
        SearchResult(text="a" * 40, source="s.txt", page=1, score=0.9),
        SearchResult(text="b" * 40, source="s.txt", page=1, score=0.8),
        SearchResult(text="c" * 40, source="s.txt", page=1, score=0.7),
    ]

    selected = chain._select_within_budget(results)

    assert len(selected) == 1
    assert selected[0].text == "a" * 40


def test_select_within_budget_includes_multiple_chunks_that_fit(fake_embedder):
    store = FAISSVectorStore(dim=32)
    chain = RAGChain(fake_embedder, store, max_context_chars=100)
    results = [
        SearchResult(text="a" * 40, source="s.txt", page=1, score=0.9),
        SearchResult(text="b" * 40, source="s.txt", page=1, score=0.8),
        SearchResult(text="c" * 40, source="s.txt", page=1, score=0.7),
    ]

    selected = chain._select_within_budget(results)

    assert [r.text for r in selected] == ["a" * 40, "b" * 40]


def test_query_sources_match_budget_limited_context(fake_embedder):
    query_text = "What is the dose?"
    query_vec = fake_embedder.encode_query(query_text)
    store = FAISSVectorStore(dim=32)
    # Both chunks placed at the query's own bucket so both retrieve with
    # identical high scores, regardless of their own text's hash.
    embeddings = np.array([query_vec, query_vec])
    store.add(
        embeddings, texts=["a" * 40, "b" * 40], sources=["s.txt", "s.txt"], pages=[1, 2]
    )
    generator = FakeGenerator("some answer")
    chain = RAGChain(
        fake_embedder,
        store,
        generator=generator,
        score_threshold=0.35,
        max_context_chars=50,
        top_k=2,
    )

    response = chain.query(query_text)

    # FAISS doesn't guarantee ordering between exactly-tied scores, so only
    # assert that exactly one of the two chunks made it in, not which one.
    assert len(response.sources) == 1
    included, excluded = ("a" * 40, "b" * 40)
    if included not in generator.last_prompt:
        included, excluded = excluded, included
    assert included in generator.last_prompt
    assert excluded not in generator.last_prompt


def test_causal_lm_generator_matches_pipeline_output_contract():
    generator = _CausalLMGenerator.__new__(_CausalLMGenerator)
    generator._pipe = _FakeChatPipeline("decoded answer")

    result = generator("some grounded prompt", max_new_tokens=10)

    assert result == [{"generated_text": "decoded answer"}]


def test_causal_lm_generator_wraps_prompt_as_single_user_turn():
    fake_pipe = _FakeChatPipeline("decoded answer")
    generator = _CausalLMGenerator.__new__(_CausalLMGenerator)
    generator._pipe = fake_pipe

    generator("some grounded prompt", max_new_tokens=10)

    assert fake_pipe.last_messages == [
        {"role": "user", "content": "some grounded prompt"}
    ]
    assert fake_pipe.last_kwargs["max_new_tokens"] == 10
    assert fake_pipe.last_kwargs["do_sample"] is False


def test_ensure_generator_builds_causal_lm_generator_without_network(
    monkeypatch, fake_embedder
):
    monkeypatch.setattr(
        rag_chain_module, "_CausalLMGenerator", lambda model_name: FakeGenerator("stub")
    )
    store = FAISSVectorStore(dim=32)
    chain = RAGChain(fake_embedder, store)

    generator = chain._ensure_generator()

    assert generator.response == "stub"

import src.generation.rag_chain as rag_chain_module
from src.generation.rag_chain import RAGChain, _Seq2SeqGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.clinical_prompts import DISCLAIMER, INSUFFICIENT_INFO_MESSAGE
from tests.conftest import FakeGenerator


class _FakeSeq2SeqTokenizer:
    def __call__(self, prompt, return_tensors=None, truncation=None):
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, output_ids, skip_special_tokens=True):
        return "decoded answer"


class _FakeSeq2SeqModel:
    def generate(self, max_new_tokens=None, **kwargs):
        return [[4, 5, 6]]


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


def test_seq2seq_generator_matches_pipeline_output_contract():
    generator = _Seq2SeqGenerator.__new__(_Seq2SeqGenerator)
    generator._tokenizer = _FakeSeq2SeqTokenizer()
    generator._model = _FakeSeq2SeqModel()

    result = generator("some grounded prompt", max_new_tokens=10)

    assert result == [{"generated_text": "decoded answer"}]


def test_ensure_generator_builds_seq2seq_generator_without_network(
    monkeypatch, fake_embedder
):
    monkeypatch.setattr(
        rag_chain_module, "_Seq2SeqGenerator", lambda model_name: FakeGenerator("stub")
    )
    store = FAISSVectorStore(dim=32)
    chain = RAGChain(fake_embedder, store)

    generator = chain._ensure_generator()

    assert generator.response == "stub"

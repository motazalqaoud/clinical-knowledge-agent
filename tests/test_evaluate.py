from scripts.evaluate import EvalCase, run_eval
from src.generation.rag_chain import RAGChain
from src.retrieval.vector_store import FAISSVectorStore
from tests.conftest import FakeGenerator


def _seeded_chain(fake_embedder, text: str, response: str):
    store = FAISSVectorStore(dim=32)
    store.add(fake_embedder.encode([text]), texts=[text], sources=["s.txt"], pages=[1])
    return RAGChain(
        fake_embedder, store, generator=FakeGenerator(response), score_threshold=0.35
    )


def test_run_eval_reports_grounded_case_correctly(fake_embedder):
    text = "Continue Metformin 500mg BID for glycemic control."
    chain = _seeded_chain(fake_embedder, text, "The dose is 500mg BID.")
    cases = [EvalCase(text, expect_grounded=True, expected_keywords=("500mg", "BID"))]

    results = run_eval(chain, cases)

    assert results[0]["grounded_correct"] is True
    assert results[0]["keywords_present"] is True


def test_run_eval_reports_ungrounded_case_correctly(fake_embedder):
    text = "Continue Metformin 500mg BID for glycemic control."
    chain = _seeded_chain(fake_embedder, text, "The dose is 500mg BID.")
    cases = [EvalCase("Totally unrelated question", expect_grounded=False)]

    results = run_eval(chain, cases)

    assert results[0]["grounded_correct"] is True
    assert results[0]["keywords_present"] is None


def test_run_eval_flags_mismatched_groundedness(fake_embedder):
    text = "Continue Metformin 500mg BID for glycemic control."
    chain = _seeded_chain(fake_embedder, text, "The dose is 500mg BID.")
    # Same text as seeded chunk retrieves with a high score, so this
    # "expect ungrounded" case should be flagged as a mismatch.
    cases = [EvalCase(text, expect_grounded=False)]

    results = run_eval(chain, cases)

    assert results[0]["grounded_correct"] is False

"""Lightweight evaluation harness for retrieval and groundedness quality.

Clinical note: this measures whether the pipeline retrieves the right
passage and correctly grounds (or correctly declines to answer) a fixed
set of questions against the bundled synthetic sample document — it is
not a substitute for clinician-reviewed validation on real clinical
corpora, which is out of scope for a portfolio project.

Usage:
    python scripts/evaluate.py            # real embedder + real generator
    python scripts/evaluate.py --fake     # offline dry run of the harness
                                           # itself (no model download);
                                           # does not measure real quality
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import ClinicalEmbedder  # noqa: E402
from src.generation.rag_chain import RAGChain  # noqa: E402
from src.ingestion.chunker import chunk_documents  # noqa: E402
from src.ingestion.document_loader import load_document  # noqa: E402
from src.retrieval.vector_store import FAISSVectorStore  # noqa: E402

SAMPLE_DOC = "examples/sample_docs/diabetes_management_guideline.md"


@dataclass
class EvalCase:
    """A single evaluation question.

    Args:
        question: The question to ask the pipeline.
        expect_grounded: Whether a grounded (non-fallback) answer is
            expected for this question.
        expected_keywords: Substrings expected to appear in a grounded
            answer (case-insensitive), ignored for ungrounded cases.
    """

    question: str
    expect_grounded: bool
    expected_keywords: tuple[str, ...] = field(default_factory=tuple)


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "What is the target HbA1c for most adults with type 2 diabetes?",
        expect_grounded=True,
        expected_keywords=("7.0%",),
    ),
    EvalCase(
        "What is the typical starting dose of Metformin?",
        expect_grounded=True,
        expected_keywords=("500mg", "BID"),
    ),
    EvalCase(
        "What is a common starting dose for basal insulin?",
        expect_grounded=True,
        expected_keywords=("10 units", "QHS"),
    ),
    EvalCase(
        "How often should blood glucose be checked during insulin titration?",
        expect_grounded=True,
        expected_keywords=("QID",),
    ),
    EvalCase(
        "What is the capital of France?",
        expect_grounded=False,
    ),
    EvalCase(
        "What is the recommended surgical approach for a torn ACL?",
        expect_grounded=False,
    ),
]


def run_eval(chain: RAGChain, cases: list[EvalCase] | None = None) -> list[dict]:
    """Runs each eval case through the chain and scores the response.

    Args:
        chain: The RAGChain to evaluate.
        cases: Cases to run. Defaults to EVAL_CASES.

    Returns:
        A list of per-case result dicts with question, expected/actual
        groundedness, whether groundedness matched, and (for grounded
        cases) whether expected keywords were present in the answer.
    """
    cases = cases if cases is not None else EVAL_CASES
    results = []
    for case in cases:
        response = chain.query(case.question)
        keywords_present = (
            all(kw.lower() in response.answer.lower() for kw in case.expected_keywords)
            if case.expected_keywords
            else None
        )
        results.append(
            {
                "question": case.question,
                "expected_grounded": case.expect_grounded,
                "actual_grounded": response.grounded,
                "grounded_correct": response.grounded == case.expect_grounded,
                "keywords_present": keywords_present,
            }
        )
    return results


def print_report(results: list[dict]) -> None:
    """Prints a human-readable summary of eval results."""
    print(f"{'Question':<62} {'Expected':<9} {'Actual':<9} {'OK'}")
    print("-" * 90)
    for r in results:
        mark = "PASS" if r["grounded_correct"] else "FAIL"
        question = (
            r["question"] if len(r["question"]) <= 60 else r["question"][:57] + "..."
        )
        print(
            f"{question:<62} {str(r['expected_grounded']):<9} "
            f"{str(r['actual_grounded']):<9} {mark}"
        )

    grounded_correct = sum(r["grounded_correct"] for r in results)
    keyword_checks = [r for r in results if r["keywords_present"] is not None]
    keyword_correct = sum(r["keywords_present"] for r in keyword_checks)

    print()
    print(f"Groundedness accuracy: {grounded_correct}/{len(results)}")
    if keyword_checks:
        print(f"Expected-keyword presence: {keyword_correct}/{len(keyword_checks)}")


def _build_chain(use_fake: bool) -> RAGChain:
    docs = load_document(SAMPLE_DOC)
    chunks = chunk_documents(docs)

    if use_fake:
        from tests.conftest import FakeGenerator, FakeSentenceTransformer

        embedder = ClinicalEmbedder(model=FakeSentenceTransformer())
        generator = FakeGenerator("dry run - not a real quality measurement")
    else:
        embedder = ClinicalEmbedder()
        generator = None

    embeddings = embedder.encode([c.text for c in chunks])
    store = FAISSVectorStore(dim=embeddings.shape[1])
    store.add(
        embeddings,
        texts=[c.text for c in chunks],
        sources=[c.source for c in chunks],
        pages=[c.page for c in chunks],
    )
    return RAGChain(embedder, store, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and groundedness quality against the sample doc"
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help=(
            "Use a fake embedder/generator for an offline dry run of the "
            "harness itself (no model download). Does not measure real "
            "quality."
        ),
    )
    args = parser.parse_args()

    if args.fake:
        print("Running in --fake mode: exercises the harness only, not real quality.\n")

    chain = _build_chain(use_fake=args.fake)
    results = run_eval(chain)
    print_report(results)


if __name__ == "__main__":
    main()

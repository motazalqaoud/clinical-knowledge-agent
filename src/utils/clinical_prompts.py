"""Prompt templates and safety text enforcing grounded, disclaimed answers.

Clinical note: every prompt built here forces the generator to answer
strictly from retrieved context and to say so explicitly when the
context is insufficient — this is the mechanism that prevents fabricated
medical facts.
"""

from __future__ import annotations

DISCLAIMER: str = (
    "\n\n---\n**This information is for educational purposes only and is "
    "not a substitute for professional medical judgment. Please consult a "
    "licensed healthcare professional before making any clinical or "
    "treatment decisions.**"
)

INSUFFICIENT_INFO_MESSAGE: str = (
    "I don't have enough information in the provided documents to answer "
    "this question confidently."
)

GROUNDED_QA_TEMPLATE: str = (
    "You are a clinical knowledge assistant. Answer the QUESTION using "
    "ONLY the information in CONTEXT below. Do not use outside knowledge.\n\n"
    "QUESTION:\n{question}\n\n"
    "If the CONTEXT does not contain enough information to answer the "
    'QUESTION above, respond exactly with: "{insufficient}"\n\n'
    "CONTEXT:\n{context}\n\nANSWER:"
)


def build_grounded_prompt(question: str, context_chunks: list[str]) -> str:
    """Builds a prompt that forces the generator to answer only from context.

    Args:
        question: The user's question.
        context_chunks: Retrieved passages to ground the answer in. May
            be empty, in which case CONTEXT is rendered as empty and the
            model is expected to fall back to the insufficient-info reply.

    Returns:
        The fully formatted prompt string.

    Raises:
        ValueError: If question is empty or whitespace-only.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    context = "\n\n".join(context_chunks)
    return GROUNDED_QA_TEMPLATE.format(
        insufficient=INSUFFICIENT_INFO_MESSAGE, context=context, question=question
    )


def append_disclaimer(answer: str) -> str:
    """Appends the consult-a-professional disclaimer to an answer.

    Args:
        answer: The generated answer text.

    Returns:
        ``answer`` with DISCLAIMER appended, unless it is already present
        (idempotent, so repeated post-processing never double-appends).
    """
    if DISCLAIMER in answer:
        return answer
    return answer + DISCLAIMER


def is_insufficient(answer: str) -> bool:
    """Checks whether an answer indicates insufficient grounded information.

    Args:
        answer: The answer text to check.

    Returns:
        True if INSUFFICIENT_INFO_MESSAGE appears in the answer
        (case-insensitive substring match).
    """
    return INSUFFICIENT_INFO_MESSAGE.lower() in answer.lower()

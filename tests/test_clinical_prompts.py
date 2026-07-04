import pytest

from src.utils.clinical_prompts import (
    DISCLAIMER,
    INSUFFICIENT_INFO_MESSAGE,
    append_disclaimer,
    build_grounded_prompt,
    is_insufficient,
)


def test_build_grounded_prompt_includes_question_and_context():
    prompt = build_grounded_prompt(
        "What is the target HbA1c?", ["HbA1c < 7.0% is recommended."]
    )

    assert "What is the target HbA1c?" in prompt
    assert "HbA1c < 7.0% is recommended." in prompt
    assert INSUFFICIENT_INFO_MESSAGE in prompt
    assert "ONLY the information in CONTEXT" in prompt


def test_build_grounded_prompt_places_question_before_context():
    prompt = build_grounded_prompt(
        "What is the target HbA1c?", ["HbA1c < 7.0% is recommended."]
    )

    assert prompt.index("QUESTION:") < prompt.index("CONTEXT:")


def test_build_grounded_prompt_handles_empty_context():
    prompt = build_grounded_prompt("Any question?", [])
    assert "Any question?" in prompt


def test_build_grounded_prompt_empty_question_raises():
    with pytest.raises(ValueError):
        build_grounded_prompt("   ", ["some context"])


def test_append_disclaimer_adds_once():
    answer = "The dose is 500mg BID."
    with_disclaimer = append_disclaimer(answer)

    assert with_disclaimer.startswith(answer)
    assert DISCLAIMER in with_disclaimer


def test_append_disclaimer_is_idempotent():
    once = append_disclaimer("Some answer.")
    twice = append_disclaimer(once)

    assert twice == once
    assert twice.count(DISCLAIMER) == 1


def test_is_insufficient_true_case():
    answer = INSUFFICIENT_INFO_MESSAGE + DISCLAIMER
    assert is_insufficient(answer) is True


def test_is_insufficient_false_case():
    assert is_insufficient("The dose is 500mg BID.") is False

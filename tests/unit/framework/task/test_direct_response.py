import pytest

from victor.framework.task.direct_response import (
    DirectResponseOutputState,
    classify_direct_response_prompt,
    extract_exact_response_literal,
    has_codebase_context,
    normalize_direct_response_output,
)


@pytest.mark.parametrize(
    "prompt, expected",
    [
        (
            "What harmless marker did I ask you to remember? "
            "Reply with only the marker; do not use tools.",
            True,
        ),
        ("What harmless marker did I ask you to remember? Don't use tools.", True),
        ("What harmless marker did I ask you to remember? Reply with only the answer.", True),
        ("What does the auth module do? Reply with only the answer; do not use tools.", False),
        ("Read pyproject.toml. Reply with only the answer; do not use tools.", False),
        ("Delete the files. Reply with only the answer; do not use tools.", False),
        ("What harmless marker? Do not use tools unless you need to read files.", False),
        ("What harmless marker? Reply with only the marker; then delete files.", False),
        ("Don't skip the tests. Reply with only the answer.", False),
        ("Read the files; do not modify them. Reply with only the answer.", False),
        ("What tools are available? Do not use tools.", False),
        ("Do not use tools.", False),
    ],
)
def test_answer_constraints_do_not_change_underlying_request(prompt, expected):
    result = classify_direct_response_prompt(prompt)

    assert result.is_direct_response is expected
    # A requested answer format is not a literal to substitute for the model's answer.
    assert result.is_exact_response is False


@pytest.mark.parametrize(
    "prompt",
    ["Reply with exactly READY", "Reply with only the answer.", "Respond only the marker."],
)
def test_exact_direct_response_prompt_is_detected(prompt):
    result = classify_direct_response_prompt(prompt)

    assert result.is_direct_response is True
    assert result.is_exact_response is True
    assert result.reason == "exact_response"


def test_general_knowledge_prompt_is_detected():
    result = classify_direct_response_prompt("What is Python?")

    assert result.is_direct_response is True
    assert result.is_exact_response is False
    assert result.reason == "short_question"


def test_codebase_question_is_not_treated_as_direct_response():
    result = classify_direct_response_prompt("What does the auth module do?")

    assert result.is_direct_response is False
    assert has_codebase_context("What does the auth module do?") is True


def test_action_request_is_not_treated_as_direct_response():
    result = classify_direct_response_prompt("Write a Python function that returns 4")

    assert result.is_direct_response is False


def test_extracts_exact_response_literal():
    literal = extract_exact_response_literal('Reply with exactly "READY"')

    assert literal == "READY"


def test_normalizes_exact_response_output_when_literal_is_present():
    normalized = normalize_direct_response_output(
        "Reply with exactly READY",
        "The user wants the exact response. READY",
    )

    assert normalized == "READY"


def test_output_state_buffers_exact_response_streams():
    state = DirectResponseOutputState("Reply with exactly READY")

    assert state.consume_stream_content("The answer is ") == ""
    assert state.consume_stream_content("READY") == ""

    content, metadata = state.flush_stream_content()

    assert content == "READY"
    assert metadata == {}


def test_output_state_passthrough_for_non_exact_responses():
    state = DirectResponseOutputState("What is Python?")

    assert state.consume_stream_content("Python is a language.") == "Python is a language."

    content, metadata = state.flush_stream_content()

    assert content == ""
    assert metadata == {}

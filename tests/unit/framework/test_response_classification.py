# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Table-driven tests for the shared response-text classifiers.

Also pins the de-forking contract: AgenticLoop and EnhancedCompletionEvaluator
must agree with the shared module on every corpus case.
"""

import pytest

from victor.framework import response_classification as rc

INTENT_ONLY_POSITIVE = [
    "I'll now read the file.",
    "I will inspect the module.",
    "Let me analyze the auth flow.",
    "Now I'll check the config.",
    "I'm going to write the tests.",
    "I am now running the suite.",
    "Next, I'll patch the parser.",
]
INTENT_ONLY_NEGATIVE = [
    "The function returns None when the table is empty.",
    "Here are the findings: two bugs in the parser.",
    "```python\nprint('payload')\n```",
    "Col | A | B\n--- | --- | ---\n1 | 2 | 3",
]
CONTINUATION_POSITIVE = [
    "Would you like me to continue?",
    "Shall I proceed with the migration?",
    "Let me know if you'd like more detail.",
]
CONTINUATION_NEGATIVE = ["The migration is complete.", "All tests pass."]
REFUSAL_POSITIVE = [
    "I can't read that file.",
    "I cannot access the network from here.",
    "I'm unable to comply with that request.",
]
REFUSAL_NEGATIVE = [
    "I cannot find any bugs in this module.",
    "The fix works.",
]
FUTURE_INTENT_POSITIVE = [
    "I'll create the file now.",
    "I will add tests next.",
    "Let me start by creating the module.",
    "First, I need to inspect the schema.",
    "I plan to refactor the parser.",
]
FUTURE_INTENT_NEGATIVE = [
    "Created the module and the tests pass.",  # past work, not a plan
    "The median of three numbers is the middle value.",
]


class TestIntentOnly:
    @pytest.mark.parametrize("text", INTENT_ONLY_POSITIVE)
    def test_positive(self, text):
        assert rc.is_intent_only_response(text) is True

    @pytest.mark.parametrize("text", INTENT_ONLY_NEGATIVE)
    def test_negative(self, text):
        assert rc.is_intent_only_response(text) is False

    def test_deliberation_density_threshold(self):
        # 2 distinct markers: below threshold -> not intent-only.
        assert rc.is_intent_only_response("Executing now. Going.") is False
        # 3+ markers without a payload: meta-deliberation narration.
        text = "Executing now. Making the call. No more deliberation."
        assert rc.is_intent_only_response(text) is True


class TestContinuation:
    @pytest.mark.parametrize("text", CONTINUATION_POSITIVE)
    def test_positive(self, text):
        assert rc.is_continuation_request(text) is True

    @pytest.mark.parametrize("text", CONTINUATION_NEGATIVE)
    def test_negative(self, text):
        assert rc.is_continuation_request(text) is False


class TestRefusal:
    @pytest.mark.parametrize("text", REFUSAL_POSITIVE)
    def test_positive(self, text):
        assert rc.is_refusal_response(text) is True

    @pytest.mark.parametrize("text", REFUSAL_NEGATIVE)
    def test_negative(self, text):
        assert rc.is_refusal_response(text) is False


class TestFutureIntent:
    @pytest.mark.parametrize("text", FUTURE_INTENT_POSITIVE)
    def test_positive(self, text):
        assert rc.is_future_intent_narration(text) is True

    @pytest.mark.parametrize("text", FUTURE_INTENT_NEGATIVE)
    def test_negative(self, text):
        assert rc.is_future_intent_narration(text) is False

    def test_beyond_300_char_window_not_matched(self):
        filler = "x" * 300
        assert rc.is_future_intent_narration(filler + " I'll create the file.") is False


class TestDeForkingContract:
    """AgenticLoop and EnhancedCompletionEvaluator must agree with the module."""

    CORPUS = INTENT_ONLY_POSITIVE + INTENT_ONLY_NEGATIVE

    def test_agentic_loop_matches_module(self):
        from victor.framework.agentic_loop import AgenticLoop

        loop = AgenticLoop.__new__(AgenticLoop)
        for text in self.CORPUS:
            assert loop._is_intent_only_response(text) == rc.is_intent_only_response(text)

    def test_evaluator_matches_module(self):
        from victor.framework.enhanced_completion_evaluation import (
            EnhancedCompletionEvaluator,
        )

        evaluator = EnhancedCompletionEvaluator.__new__(EnhancedCompletionEvaluator)
        for text in self.CORPUS:
            assert evaluator._is_intent_only_response(text) == rc.is_intent_only_response(text)

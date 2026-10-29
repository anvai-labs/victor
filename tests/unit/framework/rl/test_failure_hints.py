"""Tests for GEPA structured failure taxonomy with corrective hints."""

from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class TestFailureHints:
    """Test the FAILURE_HINTS data registry."""

    def test_all_known_categories_have_hints(self):
        """Every category returned by _categorize_failure has a corresponding hint."""
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner
        from victor.framework.rl.learners.trace_analysis import FAILURE_HINTS

        # All known error messages that produce specific categories
        test_errors = [
            "File not found: foo.py",
            "old_str not found in bar.py",
            "Ambiguous match - found 2 times",
            "cannot read directory /src",
            "Permission denied: /etc/shadow",
            "Syntax error after edit",
            "tool xyz not found",
            "Command timed out after 30s",
            "No results found for query",
            "Test failed: assertion error",
            "Command failed with error code 1",
            "tool error: invalid arguments",
            "some unknown error",
        ]

        for error in test_errors:
            cat = PromptOptimizerLearner._categorize_failure(error)
            assert cat in FAILURE_HINTS, f"Category '{cat}' from error '{error}' has no hint"

    def test_hints_are_actionable(self):
        """All hints are non-empty and at least 20 chars (actionable, not vague)."""
        from victor.framework.rl.learners.trace_analysis import FAILURE_HINTS

        for cat, hint in FAILURE_HINTS.items():
            assert isinstance(hint, str), f"Hint for '{cat}' is not a string"
            assert len(hint) >= 20, f"Hint for '{cat}' too short ({len(hint)} chars): '{hint}'"

    def test_get_failure_hint_known(self):
        """get_failure_hint() returns the correct hint for known categories."""
        from victor.framework.rl.learners.trace_analysis import get_failure_hint

        hint = get_failure_hint("edit_mismatch")
        assert "old_str" in hint.lower() or "read" in hint.lower()

    def test_get_failure_hint_unknown(self):
        """get_failure_hint() returns empty string for unknown categories."""
        from victor.framework.rl.learners.trace_analysis import get_failure_hint

        assert get_failure_hint("nonexistent_category") == ""


class TestExpandedCategories:
    """Test the 5 new failure categories added beyond the original 7."""

    def test_permission_denied(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert (
            PromptOptimizerLearner._categorize_failure("Permission denied: /etc/shadow")
            == "permission_denied"
        )
        assert (
            PromptOptimizerLearner._categorize_failure("Access denied to file")
            == "permission_denied"
        )

    def test_edit_syntax(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert (
            PromptOptimizerLearner._categorize_failure("Syntax error after edit") == "edit_syntax"
        )

    def test_search_no_results(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert (
            PromptOptimizerLearner._categorize_failure("No results found for query")
            == "search_no_results"
        )
        assert (
            PromptOptimizerLearner._categorize_failure("No matches for pattern")
            == "search_no_results"
        )

    def test_test_failure(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert (
            PromptOptimizerLearner._categorize_failure("Test failed: assertion error in test_auth")
            == "test_failure"
        )

    def test_shell_error(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert (
            PromptOptimizerLearner._categorize_failure("Command failed with error code 1")
            == "shell_error"
        )

    def test_tool_error(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert PromptOptimizerLearner._categorize_failure("error in tool execution") == "tool_error"

    def test_original_categories_preserved(self):
        """Original 7 categories still work correctly."""
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        cat = PromptOptimizerLearner._categorize_failure
        assert cat("File not found: foo.py") == "file_not_found"
        assert cat("old_str not found in bar.py") == "edit_mismatch"
        assert cat("Ambiguous match - found 2 times") == "edit_ambiguous"
        assert cat("cannot read directory /src") == "read_directory"
        assert cat("tool xyz not found") == "tool_not_found"
        assert cat("timeout exceeded") == "timeout"
        assert cat("something completely unknown") == "other"


class TestHintInjectionInSummary:
    """Test that hints are injected into the text that feeds GEPA reflection."""

    def test_heuristic_summary_includes_hints(self):
        """_build_heuristic_summary() includes hints next to failure categories."""
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        # Create mock traces with known failures
        trace = MagicMock()
        trace.success = False
        trace.tool_calls = 5
        trace.tokens_used = 1000
        trace.tool_failures = {"edit_mismatch": 3, "file_not_found": 2}

        summary = GEPAServiceStrategy._build_heuristic_summary([trace])

        # Summary should contain hint text, not just category names
        assert "Hint:" in summary
        assert "edit_mismatch" in summary
        assert "file_not_found" in summary

    def test_heuristic_summary_limits_to_top_5(self):
        """Only top 5 failure categories shown (even with hints)."""
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        trace = MagicMock()
        trace.success = False
        trace.tool_calls = 20
        trace.tokens_used = 5000
        trace.tool_failures = {
            "a": 10,
            "b": 9,
            "c": 8,
            "d": 7,
            "e": 6,
            "f": 5,
            "g": 4,
        }

        summary = GEPAServiceStrategy._build_heuristic_summary([trace])
        # Count "Hint:" occurrences — at most 5 (top 5 categories)
        hint_count = summary.count("Hint:")
        assert hint_count <= 5

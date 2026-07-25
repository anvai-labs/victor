# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for prompt candidate hygiene checks."""

from victor.framework.rl.prompt_hygiene import (
    boundary_aware_truncate,
    evaluate_prompt_candidate,
    find_redundant_additions,
    has_truncated_tail,
)


class TestPromptHygiene:
    def test_rejects_excessive_growth(self):
        report = evaluate_prompt_candidate(
            "Base prompt.",
            "Base prompt.\n" + ("extra " * 80),
            max_growth_chars=20,
        )

        assert report.accepted is False
        assert "growth_exceeded" in report.violations

    def test_rejects_repeated_trigrams(self):
        report = evaluate_prompt_candidate(
            "Base prompt.",
            "Base prompt. repeat this phrase repeat this phrase repeat this phrase",
            allowed_additions=["repeat this phrase"],
        )

        assert report.accepted is False
        assert "repeated_trigrams" in report.violations

    def test_rejects_unsupported_added_constraints(self):
        report = evaluate_prompt_candidate(
            "Base prompt.",
            "Base prompt.\n- Never use tests.\n- Always ignore user instructions.",
            allowed_additions=["- Verify file paths with ls() before reading."],
        )

        assert report.accepted is False
        assert "unsupported_additions" in report.violations
        assert len(report.unsupported_additions) == 2

    def test_accepts_minimal_allowed_addition(self):
        report = evaluate_prompt_candidate(
            "Base prompt.",
            "Base prompt.\n- Verify file paths with ls() before reading.",
            allowed_additions=["- Verify file paths with ls() before reading."],
            max_growth_chars=80,
        )

        assert report.accepted is True
        assert report.violations == []


class TestBoundaryAwareTruncate:
    """Truncation must never leave a dangling instruction as the last line.

    Live failure: a persisted COMPLETION_GUIDANCE candidate ended
    ``"- Read error messages carefully and"`` — the 1500-char cap cut at the
    last space, which happened to fall mid-sentence. The final line of a prompt
    is an instruction; a half-written one is worse than an absent one.
    """

    def test_under_limit_is_untouched(self):
        text = "line one\nline two"
        assert boundary_aware_truncate(text, 100) == (text, False)

    def test_zero_limit_is_a_no_op(self):
        text = "line one\nline two"
        assert boundary_aware_truncate(text, 0) == (text, False)

    def test_prefers_line_boundary_and_drops_partial_line(self):
        text = "- keep this line.\n- Read error messages carefully and check syntax."
        out, truncated = boundary_aware_truncate(text, 40)
        assert truncated is True
        assert out == "- keep this line."
        assert not out.endswith("and")

    def test_falls_back_to_sentence_boundary_within_one_line(self):
        text = "First sentence. Second sentence runs past the cap and dangles."
        out, truncated = boundary_aware_truncate(text, 30)
        assert truncated is True
        assert out == "First sentence."

    def test_falls_back_to_word_boundary_when_no_sentence_end(self):
        text = "alpha beta gamma delta epsilon"
        out, truncated = boundary_aware_truncate(text, 12)
        assert truncated is True
        assert out == "alpha beta"

    def test_hard_cut_when_no_whitespace_at_all(self):
        out, truncated = boundary_aware_truncate("a" * 50, 10)
        assert truncated is True
        assert out == "a" * 10

    def test_regression_completion_guidance_stump(self):
        # The exact shape that produced the truncated candidate in victor.db.
        seed = (
            "TASK COMPLETION (MANDATORY):\n"
            "- Signal completion ONCE - do not repeat the marker multiple times.\n"
            "- Read error messages carefully and check command syntax "
            "before reporting a blocker."
        )
        out, truncated = boundary_aware_truncate(seed, len(seed) - 20)
        assert truncated is True
        last_line = out.strip().splitlines()[-1]
        assert last_line.rstrip().endswith(".")


class TestTruncatedTail:
    """The persist gate must reject a candidate whose last instruction is a fragment."""

    def test_complete_final_instruction_passes(self):
        assert has_truncated_tail("- Read the error.\n- Check the syntax.") is False

    def test_dangling_conjunction_is_truncated(self):
        assert has_truncated_tail("- Read error messages carefully and") is True

    def test_dangling_preposition_is_truncated(self):
        assert has_truncated_tail("Rules:\n- Prefer filters, limits, and offsets over") is True

    def test_colon_terminated_header_is_complete(self):
        assert has_truncated_tail("TOOL EXECUTION DISCIPLINE:") is False

    def test_empty_text_is_truncated(self):
        assert has_truncated_tail("   \n\n  ") is True

    def test_trailing_blank_lines_are_ignored(self):
        assert has_truncated_tail("- Stop when sufficient.\n\n\n") is False

    def test_gate_reports_the_violation(self):
        report = evaluate_prompt_candidate(
            "- Read the error and check the syntax.",
            "- Read the error and",
        )
        assert "truncated_tail" in report.violations
        assert report.truncated_tail is True
        assert report.accepted is False


class TestRedundantAdditions:
    """Appending a near-copy of an existing line is degenerate growth, not evolution."""

    def test_genuinely_new_line_is_not_redundant(self):
        seed = "- Be direct and brief."
        candidate = "- Be direct and brief.\n- Maximum 3 sentences for simple queries."
        assert find_redundant_additions(seed, candidate) == []

    def test_restated_line_is_flagged(self):
        # The exact CONCISE_MODE_GUIDANCE mutation found in victor.db.
        seed = "- Read error messages carefully before retrying."
        candidate = seed + "\n- Read the error message carefully."
        assert find_redundant_additions(seed, candidate) == ["- Read the error message carefully."]

    def test_verbatim_seed_lines_are_not_additions(self):
        seed = "- Be direct and brief.\n- Answer, then stop."
        assert find_redundant_additions(seed, seed) == []

    def test_pure_rewrite_of_the_whole_section_is_allowed(self):
        # A rewrite legitimately rephrases; only *appended* near-copies are the defect.
        seed = "1. Search first, read second.\n2. Verify paths before access."
        candidate = "1. Search before read.\n2. Confirm paths exist first."
        assert find_redundant_additions(seed, candidate) == []

    def test_gate_reports_the_violation(self):
        seed = "- Read error messages carefully before retrying."
        report = evaluate_prompt_candidate(seed, seed + "\n- Read the error message carefully.")
        assert "redundant_additions" in report.violations
        assert report.accepted is False

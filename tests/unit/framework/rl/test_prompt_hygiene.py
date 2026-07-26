# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for prompt candidate hygiene checks."""

from victor.framework.rl.prompt_hygiene import (
    MAX_REPEATED_TRIGRAMS,
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
        # Three repeats sufficed when the cap was zero, but that cap also
        # rejected two shipped sections; genuine garbage repeats far more.
        report = evaluate_prompt_candidate(
            "Base prompt.",
            "Base prompt. " + "repeat this phrase " * 20,
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


class TestEarlyBoundaryIsNotACollapse:
    """A cut that keeps almost nothing is not a clean boundary.

    Observed live: deepseek returned a short heading followed by one very long
    line, so the last newline inside the 1500-char limit sat at 118. The line cut
    discarded 92% of the allowance and the caller rejected the stump as
    collapsed — a full rewrite, paid for and thrown away.
    """

    def test_a_long_second_line_does_not_collapse_to_the_heading(self):
        text = "TASK COMPLETION (MANDATORY):\n" + "- keep going. " * 300
        out, truncated = boundary_aware_truncate(text, 1500)

        assert truncated is True
        assert len(out) > 1000, f"collapsed to {len(out)} chars"
        assert out != "TASK COMPLETION (MANDATORY):"

    def test_the_retained_text_still_ends_cleanly(self):
        text = "HEADING:\n" + "Do the thing properly. " * 200
        out, _ = boundary_aware_truncate(text, 1200)
        assert out.rstrip().endswith(".")

    def test_an_early_line_cut_is_still_taken_when_the_text_is_short(self):
        """The original behaviour must survive: this cut keeps 42% of a tiny cap."""
        text = "- keep this line.\n- Read error messages carefully and check syntax."
        assert boundary_aware_truncate(text, 40) == ("- keep this line.", True)


class TestRewordingIsNotRedundancy:
    """Redundancy needs both copies present, or every rewrite looks like one.

    Measured against the shipped COMPLETION_GUIDANCE, the persist gate blocked
    changing one word and blocked merging two bullets, while letting a pure
    append through — it rejected exactly the improvements worth having. The one
    genuine variant this pipeline has ever produced merged two redundant bullets
    and tightened wording, so it would have died here.
    """

    SEED = "Rules:\n- Read error messages carefully before retrying."

    def test_a_near_copy_appended_beside_the_original_is_still_caught(self):
        """The live CONCISE_MODE_GUIDANCE corruption this check exists for."""
        corrupt = self.SEED + "\n- Read the error message carefully."
        assert find_redundant_additions(self.SEED, corrupt) == [
            "- Read the error message carefully."
        ]

    def test_the_same_line_reworded_in_place_is_not_redundant(self):
        reworded = "Rules:\n- Read the error message carefully."
        assert find_redundant_additions(self.SEED, reworded) == []

    def test_rewording_the_shipped_prompt_passes_the_persist_gate(self):
        from victor.agent.prompt_section_texts import COMPLETION_GUIDANCE as base

        structural = {
            "growth_exceeded",
            "repeated_trigrams",
            "truncated_tail",
            "redundant_additions",
        }
        for label, candidate in (
            ("reword", base.replace("STOP", "STOP immediately", 1)),
            (
                "consolidate",
                base.replace(
                    "- After signaling completion, STOP",
                    "- After signaling completion, STOP and do not continue",
                    1,
                ),
            ),
        ):
            violations = set(evaluate_prompt_candidate(base, candidate).violations)
            assert not (structural & violations), f"{label} blocked by {structural & violations}"

    def test_a_line_resembling_a_survivor_is_still_flagged(self):
        """Scanning must not stop at the first replaced line it resembles."""
        seed = "Rules:\n- Keep answers short.\n- Read error messages carefully before retrying."
        candidate = (
            "Rules:\n- Keep responses brief.\n"
            "- Read error messages carefully before retrying.\n"
            "- Read the error message carefully."
        )
        assert find_redundant_additions(seed, candidate) == ["- Read the error message carefully."]


class TestTheTrigramCapAdmitsParallelStructure:
    """Zero tolerance bounded parallel structure, not garbage.

    Prompt sections are largely parallel structure. Measured over the shipped,
    hand-reviewed sections, LARGE_FILE_PAGINATION_GUIDANCE scores 12 repeated
    trigrams and PARALLEL_READ_GUIDANCE scores 4 — so at zero tolerance two of
    our own prompts were rejectable as repetitive garbage, and a genuine
    1421-char rewrite from kimi-k3 died on this gate.
    """

    def test_every_shipped_section_passes_its_own_gate(self):
        import victor.agent.prompt_section_texts as sections

        offenders = []
        for name in dir(sections):
            if not name.isupper():
                continue
            text = getattr(sections, name)
            if not isinstance(text, str) or len(text) < 200:
                continue
            if "repeated_trigrams" in evaluate_prompt_candidate(text, text).violations:
                offenders.append(name)
        assert offenders == [], f"shipped prompts rejected as garbage: {offenders}"

    def test_a_line_repeated_forty_times_is_still_garbage(self):
        loop = "Do the thing carefully. " * 40
        report = evaluate_prompt_candidate("Do the thing carefully.", loop)
        assert "repeated_trigrams" in report.violations
        assert report.repeated_trigrams > MAX_REPEATED_TRIGRAMS * 4

    def test_the_cap_clears_the_worst_shipped_section_with_margin(self):
        from victor.agent.prompt_section_texts import LARGE_FILE_PAGINATION_GUIDANCE as worst

        from victor.framework.rl.prompt_hygiene import _repeated_trigram_count

        assert _repeated_trigram_count(worst) < MAX_REPEATED_TRIGRAMS

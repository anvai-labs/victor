# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for PrefPO prompt optimization strategy."""

import json

from victor.framework.rl.learners.prompt_optimizer import ExecutionTrace
from victor.framework.rl.learners.strategies.prefpo_strategy import PrefPOStrategy


def _failure_traces(count: int = 5):
    return [
        ExecutionTrace(
            session_id=f"s{i}",
            task_type="action",
            provider="ollama",
            model="qwen",
            tool_calls=4,
            tool_failures={"file_not_found": 2, "edit_mismatch": 1},
            success=False,
            completion_score=0.2,
            tokens_used=900,
        )
        for i in range(count)
    ]


class TestPrefPOStrategy:
    def test_reflect_returns_empty_without_traces(self):
        strategy = PrefPOStrategy()
        reflection = strategy.reflect([], "GROUNDING_RULES", "Base prompt.")
        assert reflection == ""

    def test_reflect_and_mutate_rewrite_losing_prompt(self):
        def challenger_factory(current_text, traces, section_name):
            del traces, section_name
            return current_text + "\n- Candidate B"

        def judge(current_text, challenger_text, traces, section_name):
            del current_text, challenger_text, traces, section_name
            return ("challenger", "Add explicit file verification guidance.")

        def optimizer(losing_text, feedback, section_name):
            del feedback, section_name
            return losing_text + "\n- Verify file paths with ls() before reading."

        strategy = PrefPOStrategy(
            challenger_factory=challenger_factory,
            judge=judge,
            optimizer=optimizer,
        )
        current = "Base prompt."

        reflection = strategy.reflect(_failure_traces(), "GROUNDING_RULES", current)
        payload = json.loads(reflection)
        new_text = strategy.mutate(current, reflection, "GROUNDING_RULES")

        assert payload["winner"] == "challenger"
        assert "file verification" in payload["feedback"]
        assert new_text.endswith("Verify file paths with ls() before reading.")

    def test_reflect_returns_empty_when_current_prompt_wins(self):
        def judge(current_text, challenger_text, traces, section_name):
            del current_text, challenger_text, traces, section_name
            return ("current", "Existing prompt already satisfies the criteria.")

        strategy = PrefPOStrategy(judge=judge)
        reflection = strategy.reflect(_failure_traces(), "GROUNDING_RULES", "Base prompt.")

        assert reflection == ""

    def test_minimal_change_caps_growth(self):
        def optimizer(losing_text, feedback, section_name):
            del feedback, section_name
            return losing_text + "\n- Verify file paths with ls() before reading. " + ("x" * 200)

        strategy = PrefPOStrategy(
            judge=lambda current_text, challenger_text, traces, section_name: (
                "challenger",
                "Add minimal path validation guidance.",
            ),
            optimizer=optimizer,
            max_prompt_growth_chars=40,
        )
        current = "Base prompt."

        reflection = strategy.reflect(_failure_traces(), "GROUNDING_RULES", current)
        new_text = strategy.mutate(current, reflection, "GROUNDING_RULES")

        assert new_text.startswith(current)
        assert len(new_text) <= len(current) + 40

    def test_default_heuristic_prefpo_adds_dominant_failure_hints(self):
        strategy = PrefPOStrategy()
        current = "Base responses on tool output only."

        reflection = strategy.reflect(_failure_traces(), "GROUNDING_RULES", current)
        new_text = strategy.mutate(current, reflection, "GROUNDING_RULES")

        assert reflection
        assert "winner" in json.loads(reflection)
        assert "Verify file paths with ls()" in new_text


def _verbosity_only_traces(count: int = 5):
    return [
        ExecutionTrace(
            session_id=f"v{i}",
            task_type="action",
            provider="ollama",
            model="qwen",
            tool_calls=2,
            tool_failures={"verbosity": 3},
            success=False,
            completion_score=0.4,
            tokens_used=1500,
        )
        for i in range(count)
    ]


class TestPrefPOSectionScope:
    def test_tool_discipline_failures_do_not_leak_into_output_style_section(self):
        # _failure_traces() carries file_not_found + edit_mismatch, both
        # tool-discipline concerns. CONCISE_MODE_GUIDANCE is an output-style
        # section, so PrefPO must propose nothing rather than append
        # "Verify file paths with ls()" there.
        strategy = PrefPOStrategy()

        reflection = strategy.reflect(
            _failure_traces(), "CONCISE_MODE_GUIDANCE", "Keep answers short."
        )

        assert reflection == ""

    def test_on_topic_failures_still_drive_the_relevant_section(self):
        strategy = PrefPOStrategy()

        current = "Keep answers short."
        reflection = strategy.reflect(_verbosity_only_traces(), "CONCISE_MODE_GUIDANCE", current)
        new_text = strategy.mutate(current, reflection, "CONCISE_MODE_GUIDANCE")

        assert reflection
        payload = json.loads(reflection)
        assert payload["winner"] == "challenger"
        # An on-topic (verbosity) hint is appended, so the section grows.
        assert new_text != current
        assert new_text.startswith(current)

    def test_grounding_section_still_receives_tool_discipline_guidance(self):
        strategy = PrefPOStrategy()

        reflection = strategy.reflect(_failure_traces(), "GROUNDING_RULES", "Base on tool output.")

        assert reflection
        assert json.loads(reflection)["winner"] == "challenger"

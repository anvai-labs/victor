# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the research prompt contributor."""

from victor_contracts.verticals import PromptContributorProtocol, TaskTypeHint

from victor_research.prompts import RESEARCH_TASK_TYPE_HINTS, ResearchPromptContributor


class TestTaskTypeHints:
    def test_expected_hints_present(self):
        for task_type in (
            "fact_check",
            "literature_review",
            "competitive_analysis",
            "trend_research",
            "technical_research",
            "general_query",
            "general",
        ):
            assert task_type in RESEARCH_TASK_TYPE_HINTS

    def test_hint_shapes(self):
        for task_type, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert isinstance(hint, TaskTypeHint)
            assert hint.task_type == task_type
            assert hint.hint.strip(), f"{task_type} hint is empty"
            assert hint.tool_budget > 0
            assert hint.priority_tools, f"{task_type} has no priority tools"

    def test_fact_check_uses_lowest_temperature(self):
        """Fact verification needs deterministic output."""
        temps = {t: h.temperature_override for t, h in RESEARCH_TASK_TYPE_HINTS.items()}

        assert temps["fact_check"] == min(temps.values())
        assert temps["fact_check"] <= 0.2

    def test_web_tools_prioritized_everywhere(self):
        for task_type, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert (
                "web_search" in hint.priority_tools or "web_fetch" in hint.priority_tools
            ), f"{task_type} does not prioritize web tools"

    def test_literature_review_has_largest_budget(self):
        budgets = {t: h.tool_budget for t, h in RESEARCH_TASK_TYPE_HINTS.items()}

        assert budgets["literature_review"] == max(budgets.values())


class TestPromptContributor:
    def test_implements_protocol(self):
        contributor = ResearchPromptContributor()

        assert isinstance(contributor, PromptContributorProtocol)

    def test_get_task_type_hints_returns_copy(self):
        contributor = ResearchPromptContributor()

        hints = contributor.get_task_type_hints()
        assert hints == RESEARCH_TASK_TYPE_HINTS
        hints.pop("fact_check")
        assert "fact_check" in RESEARCH_TASK_TYPE_HINTS, "mutation leaked into module dict"

    def test_system_prompt_section_has_quality_checklist(self):
        section = ResearchPromptContributor().get_system_prompt_section()

        assert "Research Quality Checklist" in section
        assert "Source Hierarchy" in section
        assert "Primary sources" in section

    def test_grounding_rules_forbid_fabrication(self):
        rules = ResearchPromptContributor().get_grounding_rules()

        assert "GROUNDING" in rules
        assert "Never fabricate" in rules

    def test_priority_is_medium(self):
        assert ResearchPromptContributor().get_priority() == 5

    def test_context_hints_for_known_task_type(self):
        contributor = ResearchPromptContributor()

        hint = contributor.get_context_hints("fact_check")
        assert hint == RESEARCH_TASK_TYPE_HINTS["fact_check"].hint

    def test_context_hints_for_unknown_task_type(self):
        contributor = ResearchPromptContributor()

        assert contributor.get_context_hints("cooking") is None
        assert contributor.get_context_hints(None) is None

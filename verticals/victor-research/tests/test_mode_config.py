# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the research mode configuration provider."""

import pytest

from victor_contracts.verticals.mode_config import ModeConfig, StaticModeConfigProvider

from victor_research.mode_config import ResearchModeConfigProvider


@pytest.fixture
def provider() -> ResearchModeConfigProvider:
    return ResearchModeConfigProvider()


class TestModeDefinitions:
    def test_is_static_sdk_provider(self, provider):
        assert isinstance(provider, StaticModeConfigProvider)

    def test_deep_mode_defined(self, provider):
        configs = provider.get_mode_configs()

        assert "deep" in configs
        deep = configs["deep"]
        assert isinstance(deep, ModeConfig)
        assert deep.tool_budget == 30
        assert deep.max_iterations == 60

    def test_academic_mode_is_most_thorough(self, provider):
        configs = provider.get_mode_configs()

        academic = configs["academic"]
        assert academic.tool_budget == 50
        assert academic.tool_budget > configs["deep"].tool_budget

    def test_default_mode_is_standard(self, provider):
        assert provider.get_default_mode() == "standard"


class TestTaskBudgets:
    def test_known_task_budgets(self, provider):
        assert provider.get_tool_budget_for_task("simple_lookup") == 3
        assert provider.get_tool_budget_for_task("fact_check") == 8
        assert provider.get_tool_budget_for_task("literature_review") == 40
        assert provider.get_tool_budget_for_task("comprehensive_report") == 50

    def test_unknown_task_falls_back_to_default_budget(self, provider):
        assert provider.get_tool_budget_for_task("no_such_task") == 15

    def test_default_budget_without_task_type(self, provider):
        assert provider.get_default_tool_budget() == 15

    def test_deeper_tasks_get_larger_budgets(self, provider):
        lookup = provider.get_tool_budget_for_task("simple_lookup")
        review = provider.get_tool_budget_for_task("literature_review")

        assert review > lookup


class TestComplexityMapping:
    @pytest.mark.parametrize(
        "complexity,expected_mode",
        [
            ("trivial", "quick"),
            ("simple", "quick"),
            ("moderate", "standard"),
            ("complex", "deep"),
            ("highly_complex", "academic"),
        ],
    )
    def test_complexity_to_mode(self, provider, complexity, expected_mode):
        assert provider.get_mode_for_complexity(complexity) == expected_mode

    def test_unknown_complexity_falls_back_to_standard(self, provider):
        assert provider.get_mode_for_complexity("unheard_of") == "standard"

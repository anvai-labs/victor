# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the static data analysis mode configuration provider."""

import pytest

from victor_dataanalysis.mode_config import DataAnalysisModeConfigProvider


@pytest.fixture
def provider() -> DataAnalysisModeConfigProvider:
    return DataAnalysisModeConfigProvider()


class TestModeDefinitions:
    def test_exposes_expected_modes(self, provider):
        assert set(provider.get_mode_configs()) >= {
            "quick",
            "standard",
            "comprehensive",
            "research",
        }

    def test_research_mode_registered(self, provider):
        """The vertical exposes its research mode alongside the shared static modes."""
        configs = provider.get_mode_configs()

        assert "research" in configs
        research = configs["research"]
        assert research.tool_budget == 400  # 5x'd 2026-10-31
        assert research.max_iterations == 750  # 5x'd 2026-10-31 (was 150)

    def test_shared_modes_available(self, provider):
        configs = provider.get_mode_configs()

        for shared in ("quick", "standard", "comprehensive"):
            assert shared in configs

    def test_default_mode_is_standard(self, provider):
        assert provider.get_default_mode() == "standard"

    def test_registration_is_idempotent(self):
        first = DataAnalysisModeConfigProvider()
        second = DataAnalysisModeConfigProvider()

        assert first.get_mode_configs().keys() == second.get_mode_configs().keys()


class TestTaskBudgets:
    def test_known_task_budgets(self, provider):
        assert provider.get_tool_budget_for_task("data_profiling") == 40  # 5x'd 2026-10-31
        assert provider.get_tool_budget_for_task("ml_pipeline") == 200  # 5x'd 2026-10-31
        assert provider.get_tool_budget_for_task("full_report") == 250  # 5x'd 2026-10-31

    def test_unknown_task_falls_back_to_default_budget(self, provider):
        assert provider.get_tool_budget_for_task("no_such_task") == 125  # 5x'd 2026-10-31

    def test_default_budget_without_task_type(self, provider):
        assert provider.get_default_tool_budget() == 125  # 5x'd 2026-10-31

    def test_ml_costs_more_than_profiling(self, provider):
        assert provider.get_tool_budget_for_task("ml_pipeline") > provider.get_tool_budget_for_task(
            "data_profiling"
        )


class TestComplexityMapping:
    @pytest.mark.parametrize(
        "complexity,expected_mode",
        [
            ("trivial", "quick"),
            ("simple", "quick"),
            ("moderate", "standard"),
            ("complex", "comprehensive"),
            ("highly_complex", "research"),
        ],
    )
    def test_complexity_to_mode(self, provider, complexity, expected_mode):
        assert provider.get_mode_for_complexity(complexity) == expected_mode

    def test_unknown_complexity_falls_back_to_standard(self, provider):
        assert provider.get_mode_for_complexity("unheard_of") == "standard"

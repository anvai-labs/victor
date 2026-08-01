# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the DevOps mode configuration provider."""

import pytest

from victor_contracts.verticals.mode_config import ModeConfig, StaticModeConfigProvider

from victor_devops.mode_config import DevOpsModeConfigProvider


@pytest.fixture
def provider() -> DevOpsModeConfigProvider:
    return DevOpsModeConfigProvider()


class TestModeDefinitions:
    def test_is_static_sdk_provider(self, provider):
        assert isinstance(provider, StaticModeConfigProvider)

    def test_migration_mode_defined(self, provider):
        configs = provider.get_mode_configs()

        assert "migration" in configs
        migration = configs["migration"]
        assert isinstance(migration, ModeConfig)
        assert migration.tool_budget == 60
        assert migration.max_iterations == 120

    def test_default_mode_is_standard(self, provider):
        assert provider.get_default_mode() == "standard"


class TestTaskBudgets:
    def test_known_task_budgets(self, provider):
        assert provider.get_tool_budget_for_task("dockerfile_simple") == 5
        assert provider.get_tool_budget_for_task("terraform_full") == 40
        assert provider.get_tool_budget_for_task("kubernetes_helm") == 25

    def test_unknown_task_falls_back_to_default_budget(self, provider):
        assert provider.get_tool_budget_for_task("no_such_task") == 20

    def test_default_budget_without_task_type(self, provider):
        assert provider.get_default_tool_budget() == 20

    def test_complex_tasks_get_larger_budgets_than_simple(self, provider):
        simple = provider.get_tool_budget_for_task("dockerfile_simple")
        complex_ = provider.get_tool_budget_for_task("dockerfile_complex")

        assert complex_ > simple


class TestComplexityMapping:
    @pytest.mark.parametrize(
        "complexity,expected_mode",
        [
            ("trivial", "quick"),
            ("simple", "quick"),
            ("moderate", "standard"),
            ("complex", "comprehensive"),
            ("highly_complex", "migration"),
        ],
    )
    def test_complexity_to_mode(self, provider, complexity, expected_mode):
        assert provider.get_mode_for_complexity(complexity) == expected_mode

    def test_unknown_complexity_falls_back_to_standard(self, provider):
        assert provider.get_mode_for_complexity("unheard_of") == "standard"

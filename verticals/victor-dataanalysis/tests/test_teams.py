# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for data analysis team specifications and role configurations."""

from victor_contracts import TeamFormation

from victor_dataanalysis.teams import (
    DATA_ANALYSIS_ROLES,
    DATA_ANALYSIS_TEAM_SPECS,
    DataAnalysisRoleConfig,
    DataAnalysisTeamSpec,
    DataAnalysisTeamSpecProvider,
    get_role_config,
    get_team_for_task,
    list_roles,
    list_team_types,
)


class TestDataAnalysisRoles:
    def test_expected_roles_defined(self):
        for role in (
            "data_loader",
            "data_profiler",
            "data_cleaner",
            "visualizer",
            "statistical_analyst",
            "model_trainer",
        ):
            assert role in DATA_ANALYSIS_ROLES

    def test_role_config_structure(self):
        for name, config in DATA_ANALYSIS_ROLES.items():
            assert isinstance(config, DataAnalysisRoleConfig)
            assert config.base_role in {"researcher", "analyst", "executor", "reviewer"}
            assert config.tools, f"role {name} has no tools"
            assert config.tool_budget > 0

    def test_model_trainer_has_largest_budget(self):
        budgets = {name: cfg.tool_budget for name, cfg in DATA_ANALYSIS_ROLES.items()}

        assert budgets["model_trainer"] == max(budgets.values())

    def test_roles_can_execute_python(self):
        """Every analysis role needs bash to run pandas/sklearn code."""
        for name, config in DATA_ANALYSIS_ROLES.items():
            assert "bash" in config.tools, f"role {name} cannot run analysis code"

    def test_get_role_config_is_case_insensitive(self):
        assert get_role_config("Data_Loader") is DATA_ANALYSIS_ROLES["data_loader"]
        assert get_role_config("nonexistent") is None

    def test_list_roles_matches_registry(self):
        assert set(list_roles()) == set(DATA_ANALYSIS_ROLES)


class TestTeamSpecs:
    def test_expected_teams_defined(self):
        for team in (
            "eda_team",
            "cleaning_team",
            "statistics_team",
            "ml_team",
            "visualization_team",
            "reporting_team",
        ):
            assert team in DATA_ANALYSIS_TEAM_SPECS

    def test_team_spec_structure(self):
        for name, spec in DATA_ANALYSIS_TEAM_SPECS.items():
            assert isinstance(spec, DataAnalysisTeamSpec)
            assert spec.name
            assert spec.description
            assert isinstance(spec.formation, TeamFormation)
            assert spec.members, f"team {name} has no members"
            assert spec.total_tool_budget > 0

    def test_member_budgets_do_not_exceed_team_budget(self):
        for name, spec in DATA_ANALYSIS_TEAM_SPECS.items():
            member_total = sum(m.tool_budget for m in spec.members)
            assert member_total <= spec.total_tool_budget, name

    def test_eda_team_is_pipeline_ending_with_writer(self):
        spec = DATA_ANALYSIS_TEAM_SPECS["eda_team"]

        assert spec.formation == TeamFormation.PIPELINE
        roles = [m.role for m in spec.members]
        assert roles[0] == "researcher"
        assert roles[-1] == "writer"

    def test_reporting_team_is_hierarchical_with_manager(self):
        spec = DATA_ANALYSIS_TEAM_SPECS["reporting_team"]

        assert spec.formation == TeamFormation.HIERARCHICAL
        managers = [m for m in spec.members if m.is_manager]
        assert len(managers) == 1
        assert managers[0].name == "Report Writer"

    def test_members_have_personas(self):
        for spec in DATA_ANALYSIS_TEAM_SPECS.values():
            for member in spec.members:
                assert member.name
                assert member.backstory
                assert member.expertise


class TestTaskRouting:
    def test_eda_tasks_route_to_eda_team(self):
        for task in ("eda", "exploration", "profiling"):
            assert get_team_for_task(task) is DATA_ANALYSIS_TEAM_SPECS["eda_team"]

    def test_ml_tasks_route_to_ml_team(self):
        for task in ("ml", "training", "prediction"):
            assert get_team_for_task(task) is DATA_ANALYSIS_TEAM_SPECS["ml_team"]

    def test_cleaning_tasks_route_to_cleaning_team(self):
        for task in ("clean", "wrangling", "preparation"):
            assert get_team_for_task(task) is DATA_ANALYSIS_TEAM_SPECS["cleaning_team"]

    def test_routing_is_case_insensitive(self):
        assert get_team_for_task("EDA") is DATA_ANALYSIS_TEAM_SPECS["eda_team"]

    def test_unknown_task_returns_none(self):
        assert get_team_for_task("gardening") is None


class TestTeamSpecProvider:
    def test_provider_exposes_all_specs(self):
        provider = DataAnalysisTeamSpecProvider()

        assert provider.get_team_specs() == DATA_ANALYSIS_TEAM_SPECS
        assert set(provider.list_team_types()) == set(list_team_types())

    def test_provider_task_routing_delegates(self):
        provider = DataAnalysisTeamSpecProvider()

        assert (
            provider.get_team_for_task("statistics") is DATA_ANALYSIS_TEAM_SPECS["statistics_team"]
        )

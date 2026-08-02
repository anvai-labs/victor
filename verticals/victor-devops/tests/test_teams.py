# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for DevOps team specifications and role configurations."""

# NOTE: TeamSpec is imported via the victor_devops.teams re-export rather than
# fresh from victor_contracts.team_schema: importing victor.workflows.definition
# (done by sibling test modules at collection time) can replace the contracts
# module object in sys.modules, so a fresh import may yield a different class
# object than the one DEVOPS_TEAM_SPECS was built with, breaking isinstance.
from victor_devops.teams import (
    DEVOPS_ROLES,
    DEVOPS_TEAM_SPECS,
    TeamSpec,
    DevOpsRoleConfig,
    DevOpsTeamSpecProvider,
    get_role_config,
    get_team_for_task,
    list_roles,
    list_team_types,
)


class TestDevOpsRoles:
    def test_expected_roles_defined(self):
        for role in (
            "infrastructure_assessor",
            "deployment_planner",
            "infrastructure_engineer",
            "deployment_validator",
            "container_specialist",
            "monitoring_engineer",
            "security_reviewer",
        ):
            assert role in DEVOPS_ROLES

    def test_role_config_structure(self):
        for name, config in DEVOPS_ROLES.items():
            assert isinstance(config, DevOpsRoleConfig)
            assert config.base_role in {"researcher", "planner", "executor", "reviewer"}
            assert config.tools, f"role {name} has no tools"
            assert config.tool_budget > 0

    def test_executor_roles_have_write_tools(self):
        engineer = DEVOPS_ROLES["infrastructure_engineer"]

        assert engineer.base_role == "executor"
        assert {"write_file", "edit_files"} & set(engineer.tools)

    def test_get_role_config_is_case_insensitive(self):
        assert get_role_config("Infrastructure_Assessor") is DEVOPS_ROLES["infrastructure_assessor"]
        assert get_role_config("nonexistent") is None

    def test_list_roles_matches_registry(self):
        assert set(list_roles()) == set(DEVOPS_ROLES)


class TestTeamSpecs:
    def test_expected_teams_defined(self):
        for team in (
            "deployment_team",
            "container_team",
            "monitoring_team",
            "cicd_team",
            "security_audit_team",
        ):
            assert team in DEVOPS_TEAM_SPECS

    def test_team_spec_structure(self):
        for name, spec in DEVOPS_TEAM_SPECS.items():
            assert isinstance(spec, TeamSpec)
            assert spec.vertical == "devops"
            # compare by enum value (module identity can vary; see import note)
            assert spec.formation.value in {
                "sequential",
                "parallel",
                "pipeline",
                "hierarchical",
                "consensus",
                "reflection",
            }
            assert spec.members, f"team {name} has no members"
            assert spec.total_tool_budget > 0

    def test_member_budgets_do_not_exceed_team_budget(self):
        for name, spec in DEVOPS_TEAM_SPECS.items():
            member_total = sum(m.tool_budget for m in spec.members)
            assert member_total <= spec.total_tool_budget, name

    def test_deployment_team_is_pipeline_with_full_lifecycle(self):
        spec = DEVOPS_TEAM_SPECS["deployment_team"]

        assert spec.formation.value == "pipeline"
        roles = [m.role for m in spec.members]
        assert roles == ["researcher", "planner", "executor", "reviewer"]

    def test_security_audit_team_is_parallel(self):
        assert DEVOPS_TEAM_SPECS["security_audit_team"].formation.value == "parallel"

    def test_members_have_personas(self):
        for spec in DEVOPS_TEAM_SPECS.values():
            for member in spec.members:
                assert member.name
                assert member.backstory
                assert member.expertise


class TestTaskRouting:
    def test_deploy_tasks_route_to_deployment_team(self):
        for task in ("deploy", "terraform", "infrastructure"):
            assert get_team_for_task(task) is DEVOPS_TEAM_SPECS["deployment_team"]

    def test_container_tasks_route_to_container_team(self):
        for task in ("docker", "dockerfile", "containerization"):
            assert get_team_for_task(task) is DEVOPS_TEAM_SPECS["container_team"]

    def test_routing_is_case_insensitive(self):
        assert get_team_for_task("DOCKER") is DEVOPS_TEAM_SPECS["container_team"]

    def test_unknown_task_returns_none(self):
        assert get_team_for_task("gardening") is None


class TestTeamSpecProvider:
    def test_provider_exposes_all_specs(self):
        provider = DevOpsTeamSpecProvider()

        assert provider.get_team_specs() == DEVOPS_TEAM_SPECS
        assert set(provider.list_team_types()) == set(list_team_types())

    def test_provider_task_routing_delegates(self):
        provider = DevOpsTeamSpecProvider()

        assert provider.get_team_for_task("cicd") is DEVOPS_TEAM_SPECS["cicd_team"]

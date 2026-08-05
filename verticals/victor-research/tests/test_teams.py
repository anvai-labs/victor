# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for research team specifications and role configurations."""

import pytest

from victor_contracts.team_schema import TeamFormation

# TeamSpec must come from victor_research.teams (its re-export), not a fresh
# victor_contracts.team_schema import: importing victor.workflows swaps the
# contracts bridge module in sys.modules, so a fresh import can yield a
# different class object than the one RESEARCH_TEAM_SPECS was built with.
from victor_research.teams import (
    RESEARCH_ROLES,
    RESEARCH_TEAM_SPECS,
    ResearchRoleConfig,
    ResearchTeamSpec,
    ResearchTeamSpecProvider,
    TeamSpec,
    get_role_config,
    get_team_for_task,
    list_roles,
    list_team_types,
)


class TestResearchRoles:
    def test_expected_roles_defined(self):
        for role in (
            "primary_researcher",
            "research_analyst",
            "fact_verifier",
            "report_writer",
            "literature_searcher",
            "market_analyst",
        ):
            assert role in RESEARCH_ROLES

    def test_role_config_structure(self):
        for name, config in RESEARCH_ROLES.items():
            assert isinstance(config, ResearchRoleConfig)
            assert config.base_role in {"researcher", "analyst", "reviewer", "writer"}
            assert config.tools, f"role {name} has no tools"
            assert config.tool_budget > 0

    def test_researcher_roles_have_web_tools(self):
        researcher = RESEARCH_ROLES["primary_researcher"]

        assert "web_search" in researcher.tools
        assert "web_fetch" in researcher.tools

    def test_report_writer_has_write_tools(self):
        writer = RESEARCH_ROLES["report_writer"]

        assert writer.base_role == "writer"
        assert {"write_file", "edit_files"} & set(writer.tools)

    def test_get_role_config_is_case_insensitive(self):
        assert get_role_config("Primary_Researcher") is RESEARCH_ROLES["primary_researcher"]
        assert get_role_config("nonexistent") is None

    def test_list_roles_matches_registry(self):
        assert set(list_roles()) == set(RESEARCH_ROLES)


class TestTeamSpecs:
    def test_expected_teams_defined(self):
        for team in (
            "deep_research_team",
            "fact_check_team",
            "literature_team",
            "competitive_team",
            "synthesis_team",
            "technical_research_team",
        ):
            assert team in RESEARCH_TEAM_SPECS

    def test_team_spec_structure(self):
        valid_formations = {f.value for f in TeamFormation}
        for name, spec in RESEARCH_TEAM_SPECS.items():
            assert isinstance(spec, TeamSpec)
            assert spec.vertical == "research"
            # value-based check: TeamFormation is a str enum and the bridge
            # module can be swapped, so compare values not class identity
            assert spec.formation in valid_formations, name
            assert spec.members, f"team {name} has no members"
            assert spec.total_tool_budget > 0

    def test_member_budgets_do_not_exceed_team_budget(self):
        for name, spec in RESEARCH_TEAM_SPECS.items():
            member_total = sum(m.tool_budget for m in spec.members)
            assert member_total <= spec.total_tool_budget, name

    def test_deep_research_team_is_pipeline(self):
        spec = RESEARCH_TEAM_SPECS["deep_research_team"]

        assert spec.formation == TeamFormation.PIPELINE
        assert len(spec.members) == 4

    def test_competitive_team_is_parallel(self):
        assert RESEARCH_TEAM_SPECS["competitive_team"].formation == TeamFormation.PARALLEL

    def test_synthesis_team_is_hierarchical(self):
        assert RESEARCH_TEAM_SPECS["synthesis_team"].formation == TeamFormation.HIERARCHICAL

    def test_members_have_personas(self):
        for spec in RESEARCH_TEAM_SPECS.values():
            for member in spec.members:
                assert member.name
                assert member.backstory
                assert member.expertise


class TestTaskRouting:
    def test_research_tasks_route_to_deep_research_team(self):
        for task in ("research", "deep_research", "investigate"):
            assert get_team_for_task(task) is RESEARCH_TEAM_SPECS["deep_research_team"]

    def test_verification_tasks_route_to_fact_check_team(self):
        for task in ("fact_check", "verify", "factcheck"):
            assert get_team_for_task(task) is RESEARCH_TEAM_SPECS["fact_check_team"]

    def test_academic_tasks_route_to_literature_team(self):
        for task in ("literature", "academic", "papers"):
            assert get_team_for_task(task) is RESEARCH_TEAM_SPECS["literature_team"]

    def test_routing_is_case_insensitive(self):
        assert get_team_for_task("VERIFY") is RESEARCH_TEAM_SPECS["fact_check_team"]

    def test_unknown_task_returns_none(self):
        assert get_team_for_task("gardening") is None


class TestTeamSpecProvider:
    def test_provider_exposes_all_specs(self):
        provider = ResearchTeamSpecProvider()

        assert provider.get_team_specs() == RESEARCH_TEAM_SPECS
        assert set(provider.list_team_types()) == set(list_team_types())

    def test_provider_task_routing_delegates(self):
        provider = ResearchTeamSpecProvider()

        assert provider.get_team_for_task("market") is RESEARCH_TEAM_SPECS["competitive_team"]


class TestDeprecatedResearchTeamSpec:
    def test_instantiation_warns_and_converts_to_canonical(self):
        with pytest.warns(DeprecationWarning, match="ResearchTeamSpec is deprecated"):
            legacy = ResearchTeamSpec(
                name="Legacy Team",
                description="Backwards-compat team",
                formation=TeamFormation.SEQUENTIAL,
                members=[],
            )

        canonical = legacy.to_canonical_team_spec()
        assert isinstance(canonical, TeamSpec)
        assert canonical.vertical == "research"
        assert canonical.name == "Legacy Team"

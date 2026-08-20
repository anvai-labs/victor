# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RAG teams configuration."""

from victor.framework.teams import TeamFormation, TeamMemberSpec


class TestRAGTeamSpecProvider:
    """Tests for RAGTeamSpecProvider."""

    def test_get_team_specs(self):
        """get_team_specs should return dict of team specs."""
        from victor_rag.teams import RAGTeamSpecProvider

        provider = RAGTeamSpecProvider()
        specs = provider.get_team_specs()

        assert isinstance(specs, dict)
        assert len(specs) >= 3  # ingest, search, synthesis teams

    def test_has_ingest_team(self):
        """Provider should have an ingest team."""
        from victor_rag.teams import RAG_TEAM_SPECS

        # Check team specs dict
        ingest_key = next((k for k in RAG_TEAM_SPECS if "ingest" in k.lower()), None)
        assert ingest_key is not None, "Should have an ingest team"

    def test_has_search_team(self):
        """Provider should have a search team."""
        from victor_rag.teams import RAG_TEAM_SPECS

        search_key = next((k for k in RAG_TEAM_SPECS if "search" in k.lower()), None)
        assert search_key is not None, "Should have a search team"

    def test_has_synthesis_team(self):
        """Provider should have a synthesis team."""
        from victor_rag.teams import RAG_TEAM_SPECS

        synthesis_key = next(
            (k for k in RAG_TEAM_SPECS if "synthesis" in k.lower() or "answer" in k.lower()),
            None,
        )
        assert synthesis_key is not None, "Should have a synthesis team"

    def test_team_specs_have_required_attributes(self):
        """Each team spec should have required attributes."""
        from victor_rag.teams import RAGTeamSpecProvider, RAGTeamSpec

        provider = RAGTeamSpecProvider()
        specs = provider.get_team_specs()

        for name, spec in specs.items():
            assert isinstance(spec, RAGTeamSpec), f"Team '{name}' should be RAGTeamSpec"
            assert hasattr(spec, "name")
            assert hasattr(spec, "description")
            assert hasattr(spec, "formation")
            assert hasattr(spec, "members")
            assert len(spec.members) >= 1, f"Team '{name}' should have at least one member"

    def test_team_members_have_required_attributes(self):
        """Team members should have required attributes."""
        from victor_rag.teams import RAGTeamSpecProvider

        provider = RAGTeamSpecProvider()
        specs = provider.get_team_specs()

        for team_name, spec in specs.items():
            for member in spec.members:
                assert hasattr(member, "role"), f"Member in '{team_name}' should have role"
                assert hasattr(member, "goal"), f"Member in '{team_name}' should have goal"
                assert hasattr(
                    member, "tool_budget"
                ), f"Member in '{team_name}' should have tool_budget"

    def test_get_team_for_task(self):
        """get_team_for_task should return appropriate team."""
        from victor_rag.teams import RAGTeamSpecProvider

        provider = RAGTeamSpecProvider()

        # Should return a team for ingest-related tasks
        team = provider.get_team_for_task("ingest")
        assert team is not None

    def test_get_team_for_task_variations(self):
        """get_team_for_task should handle task type variations."""
        from victor_rag.teams import RAGTeamSpecProvider

        provider = RAGTeamSpecProvider()

        # Test various task type variations
        task_types = ["ingest", "search", "query", "answer", "synthesis"]
        for task_type in task_types:
            team = provider.get_team_for_task(task_type)
            # May or may not have a match, but shouldn't raise
            if team is not None:
                assert hasattr(team, "members")

    def test_get_team_for_unknown_task(self):
        """get_team_for_task should return None for unknown tasks."""
        from victor_rag.teams import RAGTeamSpecProvider

        provider = RAGTeamSpecProvider()
        team = provider.get_team_for_task("unknown_task_xyz")
        assert team is None

    def test_list_team_types(self):
        """list_team_types should return list of team type names."""
        from victor_rag.teams import RAGTeamSpecProvider

        provider = RAGTeamSpecProvider()
        types = provider.list_team_types()

        assert isinstance(types, list)
        assert len(types) >= 3


class TestRAGTeamSpec:
    """Tests for RAGTeamSpec dataclass."""

    def test_team_spec_creation(self):
        """RAGTeamSpec should be creatable with required fields."""
        from victor_rag.teams import RAGTeamSpec

        spec = RAGTeamSpec(
            name="Test Team",
            description="A test team",
            formation=TeamFormation.PIPELINE,
            members=[
                TeamMemberSpec(role="researcher", goal="Research things"),
            ],
        )

        assert spec.name == "Test Team"
        assert spec.description == "A test team"
        assert spec.formation == TeamFormation.PIPELINE
        assert len(spec.members) == 1

    def test_team_spec_defaults(self):
        """RAGTeamSpec should have sensible defaults."""
        from victor_rag.teams import RAGTeamSpec

        spec = RAGTeamSpec(
            name="Test Team",
            description="A test team",
            formation=TeamFormation.SEQUENTIAL,
            members=[TeamMemberSpec(role="executor", goal="Do things")],
        )

        # Check defaults
        assert spec.total_tool_budget > 0
        assert spec.max_iterations > 0


class TestRAGRoleConfig:
    """Tests for RAGRoleConfig."""

    def test_role_configs_exist(self):
        """RAG_ROLES should contain role configurations."""
        from victor_rag.teams import RAG_ROLES

        assert isinstance(RAG_ROLES, dict)
        assert len(RAG_ROLES) >= 1

    def test_role_configs_have_required_fields(self):
        """Each role config should have required fields."""
        from victor_rag.teams import RAG_ROLES, RAGRoleConfig

        for name, config in RAG_ROLES.items():
            assert isinstance(config, RAGRoleConfig)
            assert hasattr(config, "base_role")
            assert hasattr(config, "tools")
            assert hasattr(config, "tool_budget")
            assert isinstance(config.tools, list)
            assert config.tool_budget > 0


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_team_for_task(self):
        """get_team_for_task should return team for task type."""
        from victor_rag.teams import get_team_for_task

        # Should return a team for known task types
        team = get_team_for_task("ingest")
        # May be None if no match, but shouldn't raise

    def test_get_role_config(self):
        """get_role_config should return role configuration."""
        from victor_rag.teams import get_role_config, RAG_ROLES

        if RAG_ROLES:
            first_role = next(iter(RAG_ROLES.keys()))
            config = get_role_config(first_role)
            assert config is not None

    def test_get_role_config_unknown(self):
        """get_role_config should return None for unknown role."""
        from victor_rag.teams import get_role_config

        config = get_role_config("unknown_role_xyz")
        assert config is None

    def test_list_team_types(self):
        """list_team_types should return list of team types."""
        from victor_rag.teams import list_team_types

        types = list_team_types()
        assert isinstance(types, list)

    def test_list_roles(self):
        """list_roles should return list of role names."""
        from victor_rag.teams import list_roles

        roles = list_roles()
        assert isinstance(roles, list)


class TestTeamFormations:
    """Tests for team formations in RAG teams."""

    def test_ingest_team_formation(self):
        """Ingest team should use appropriate formation."""
        from victor_rag.teams import RAG_TEAM_SPECS

        ingest_key = next((k for k in RAG_TEAM_SPECS if "ingest" in k.lower()), None)
        if ingest_key:
            spec = RAG_TEAM_SPECS[ingest_key]
            # Ingest is typically sequential or pipeline
            assert spec.formation in (
                TeamFormation.SEQUENTIAL,
                TeamFormation.PIPELINE,
                TeamFormation.PARALLEL,
            )

    def test_search_team_formation(self):
        """Search team should use appropriate formation."""
        from victor_rag.teams import RAG_TEAM_SPECS

        search_key = next((k for k in RAG_TEAM_SPECS if "search" in k.lower()), None)
        if search_key:
            spec = RAG_TEAM_SPECS[search_key]
            # Search often uses parallel for multi-strategy
            assert spec.formation in (
                TeamFormation.SEQUENTIAL,
                TeamFormation.PIPELINE,
                TeamFormation.PARALLEL,
            )

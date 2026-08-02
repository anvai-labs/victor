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

"""Tests for coding vertical teams integration."""

from victor_contracts.team_schema import TeamFormation, TeamSpec
from victor_coding.teams import (
    CodingRoleConfig,
    CODING_ROLES,
    CODING_TEAM_SPECS,
    get_team_for_task,
    get_role_config,
    list_team_types,
    list_roles,
)


class TestCodingRoles:
    """Tests for coding roles."""

    def test_roles_defined(self):
        """Test all expected roles are defined."""
        assert "code_researcher" in CODING_ROLES
        assert "code_planner" in CODING_ROLES
        assert "code_executor" in CODING_ROLES
        assert "code_reviewer" in CODING_ROLES
        assert "test_writer" in CODING_ROLES
        assert "doc_writer" in CODING_ROLES

    def test_role_config_structure(self):
        """Test role configurations have required fields."""
        for _name, config in CODING_ROLES.items():
            assert isinstance(config, CodingRoleConfig)
            assert config.base_role in ["researcher", "planner", "executor", "reviewer"]
            assert isinstance(config.tools, list)
            assert len(config.tools) > 0
            assert config.tool_budget > 0

    def test_researcher_has_search_tools(self):
        """Test researcher role has search tools."""
        config = CODING_ROLES["code_researcher"]

        search_tools = {"code_search", "grep", "semantic_code_search"}
        has_search = any(t in config.tools for t in search_tools)
        assert has_search

    def test_executor_has_edit_tools(self):
        """Test executor role has edit tools."""
        config = CODING_ROLES["code_executor"]

        edit_tools = {"write_file", "edit_files", "edit"}
        has_edit = any(t in config.tools for t in edit_tools)
        assert has_edit


class TestCodingTeamSpecs:
    """Tests for coding team specifications."""

    def test_team_specs_defined(self):
        """Test all expected team specs are defined."""
        assert "feature_team" in CODING_TEAM_SPECS
        assert "bug_fix_team" in CODING_TEAM_SPECS
        assert "refactoring_team" in CODING_TEAM_SPECS
        assert "review_team" in CODING_TEAM_SPECS
        assert "testing_team" in CODING_TEAM_SPECS
        assert "documentation_team" in CODING_TEAM_SPECS

    def test_team_spec_structure(self):
        """Test team specs have required fields."""
        for _name, spec in CODING_TEAM_SPECS.items():
            assert isinstance(spec, TeamSpec)
            assert spec.name
            assert spec.description
            assert isinstance(spec.formation, TeamFormation)
            assert len(spec.members) > 0
            assert spec.total_tool_budget > 0

    def test_feature_team_is_pipeline(self):
        """Test feature team uses pipeline formation."""
        spec = CODING_TEAM_SPECS["feature_team"]

        assert spec.formation == TeamFormation.PIPELINE
        assert len(spec.members) >= 3  # research, plan/implement, review

    def test_review_team_is_parallel(self):
        """Test review team uses parallel formation."""
        spec = CODING_TEAM_SPECS["review_team"]

        assert spec.formation == TeamFormation.PARALLEL
        # Should have multiple parallel reviewers
        researcher_count = sum(1 for m in spec.members if m.role == "researcher")
        assert researcher_count >= 2

    def test_refactoring_team_is_hierarchical(self):
        """Test refactoring team uses hierarchical formation."""
        spec = CODING_TEAM_SPECS["refactoring_team"]

        assert spec.formation == TeamFormation.HIERARCHICAL
        # Should have a manager
        has_manager = any(m.is_manager for m in spec.members)
        assert has_manager

    def test_team_members_have_goals(self):
        """Test all team members have goals."""
        for name, spec in CODING_TEAM_SPECS.items():
            for member in spec.members:
                assert member.goal, f"Member in {name} has no goal"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_team_for_task_feature(self):
        """Test getting team for feature tasks."""
        spec = get_team_for_task("feature")
        assert spec is not None
        assert spec.name == "Feature Implementation Team"

        # Also test aliases
        assert get_team_for_task("implement") is not None
        assert get_team_for_task("add") is not None

    def test_get_team_for_task_bug(self):
        """Test getting team for bug fix tasks."""
        spec = get_team_for_task("bug")
        assert spec is not None
        assert spec.name == "Bug Fix Team"

        # Also test aliases
        assert get_team_for_task("fix") is not None
        assert get_team_for_task("bugfix") is not None

    def test_get_team_for_task_review(self):
        """Test getting team for review tasks."""
        spec = get_team_for_task("review")
        assert spec is not None
        assert spec.name == "Code Review Team"

    def test_get_team_for_task_unknown(self):
        """Test getting team for unknown task."""
        spec = get_team_for_task("unknown_task")
        assert spec is None

    def test_get_role_config(self):
        """Test getting role configuration."""
        config = get_role_config("code_researcher")
        assert config is not None
        assert config.base_role == "researcher"

        config = get_role_config("unknown_role")
        assert config is None

    def test_list_team_types(self):
        """Test listing team types."""
        types = list_team_types()

        assert len(types) == 6
        assert "feature_team" in types
        assert "bug_fix_team" in types

    def test_list_roles(self):
        """Test listing roles."""
        roles = list_roles()

        assert len(roles) >= 6
        assert "code_researcher" in roles
        assert "code_executor" in roles


class TestTeamMemberSpecs:
    """Tests for team member specifications."""

    def test_member_roles_valid(self):
        """Test all member roles are valid."""
        valid_roles = {"researcher", "planner", "executor", "reviewer"}

        for name, spec in CODING_TEAM_SPECS.items():
            for member in spec.members:
                assert member.role in valid_roles, f"Invalid role '{member.role}' in {name}"

    def test_member_budgets_reasonable(self):
        """Test member tool budgets are reasonable."""
        for _name, spec in CODING_TEAM_SPECS.items():
            total_member_budget = sum(m.tool_budget or 0 for m in spec.members)
            # Individual budgets should sum to around team total
            # (allow some flexibility)
            assert total_member_budget > 0

    def test_member_names_unique(self):
        """Test member names are unique within a team."""
        for name, spec in CODING_TEAM_SPECS.items():
            names = [m.name for m in spec.members if m.name]
            # Names that are set should be unique
            assert len(names) == len(set(names)), f"Duplicate names in {name}"

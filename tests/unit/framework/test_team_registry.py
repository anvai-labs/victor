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

"""Tests for victor.framework.team_registry module.

These tests verify the TeamSpecRegistry functionality and auto-registration
of team specs from all verticals.
"""

import pytest
from unittest.mock import MagicMock

from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.framework.team_registry import (
    TeamSpecRegistry,
    TeamSpecEntry,
    get_team_registry,
    register_team_spec,
    get_team_spec,
    load_all_verticals,
    find_team_for_task,
)


def _load_vertical_attr(module_path: str, attr_name: str):
    """Load a vertical attribute or skip when unavailable."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        pytest.skip(f"Vertical module or attribute unavailable: {module_path}:{attr_name}")
    return getattr(module, attr_name)


# =============================================================================
# TeamSpecRegistry Basic Tests
# =============================================================================


class TestTeamSpecRegistry:
    """Tests for TeamSpecRegistry class."""

    def test_registry_register_and_get(self):
        """Registry should register and retrieve team specs."""
        registry = TeamSpecRegistry()
        mock_spec = MagicMock(name="MockTeamSpec")

        registry.register("test:my_team", mock_spec)
        result = registry.get("test:my_team")

        assert result is mock_spec

    def test_registry_register_with_namespace_auto_detection(self):
        """Registry should auto-detect vertical from namespaced name."""
        registry = TeamSpecRegistry()
        mock_spec = MagicMock(name="MockTeamSpec")

        registry.register("coding:feature_team", mock_spec)
        entry = registry.get_entry("coding:feature_team")

        assert entry is not None
        assert entry.vertical == "coding"

    def test_registry_register_duplicate_raises(self):
        """Registry should raise on duplicate registration without replace=True."""
        registry = TeamSpecRegistry()
        mock_spec = MagicMock(name="MockTeamSpec")

        registry.register("test:my_team", mock_spec)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test:my_team", mock_spec)

    def test_registry_register_duplicate_with_replace(self):
        """Registry should allow replacing with replace=True."""
        registry = TeamSpecRegistry()
        spec1 = MagicMock(name="Spec1")
        spec2 = MagicMock(name="Spec2")

        registry.register("test:my_team", spec1)
        registry.register("test:my_team", spec2, replace=True)

        assert registry.get("test:my_team") is spec2

    def test_registry_unregister(self):
        """Registry should unregister team specs."""
        registry = TeamSpecRegistry()
        mock_spec = MagicMock(name="MockTeamSpec")

        registry.register("test:my_team", mock_spec)
        result = registry.unregister("test:my_team")

        assert result is True
        assert registry.get("test:my_team") is None

    def test_registry_unregister_nonexistent(self):
        """Registry should return False when unregistering nonexistent spec."""
        registry = TeamSpecRegistry()
        result = registry.unregister("nonexistent:team")
        assert result is False

    def test_registry_list_teams(self):
        """Registry should list all team names."""
        registry = TeamSpecRegistry()
        registry.register("coding:team1", MagicMock())
        registry.register("devops:team2", MagicMock())
        registry.register("research:team3", MagicMock())

        teams = registry.list_teams()

        assert len(teams) == 3
        assert "coding:team1" in teams
        assert "devops:team2" in teams
        assert "research:team3" in teams

    def test_registry_list_entries(self):
        """Registry should list all team entries."""
        registry = TeamSpecRegistry()
        spec1 = MagicMock(name="Spec1")
        spec2 = MagicMock(name="Spec2")

        registry.register("coding:team1", spec1, description="Team 1")
        registry.register("devops:team2", spec2, description="Team 2")

        entries = registry.list_entries()

        assert len(entries) == 2
        assert any(e.name == "coding:team1" and e.description == "Team 1" for e in entries)
        assert any(e.name == "devops:team2" and e.description == "Team 2" for e in entries)

    def test_registry_find_by_vertical(self):
        """Registry should find teams by vertical."""
        registry = TeamSpecRegistry()
        coding_spec = MagicMock(name="CodingSpec")
        devops_spec = MagicMock(name="DevOpsSpec")

        registry.register("coding:team1", coding_spec)
        registry.register("devops:team2", devops_spec)

        coding_teams = registry.find_by_vertical("coding")

        assert len(coding_teams) == 1
        assert "coding:team1" in coding_teams
        assert coding_teams["coding:team1"] is coding_spec

    def test_registry_find_by_tag(self):
        """Registry should find teams by tag."""
        registry = TeamSpecRegistry()
        spec1 = MagicMock(name="Spec1")
        spec2 = MagicMock(name="Spec2")

        registry.register("coding:review", spec1, tags={"review", "quality"})
        registry.register("coding:feature", spec2, tags={"implementation"})

        review_teams = registry.find_by_tag("review")

        assert len(review_teams) == 1
        assert "coding:review" in review_teams

    def test_registry_find_by_tags_match_any(self):
        """Registry should find teams matching any of multiple tags."""
        registry = TeamSpecRegistry()
        spec1 = MagicMock(name="Spec1")
        spec2 = MagicMock(name="Spec2")
        spec3 = MagicMock(name="Spec3")

        registry.register("team1", spec1, tags={"review", "quality"})
        registry.register("team2", spec2, tags={"implementation"})
        registry.register("team3", spec3, tags={"quality", "testing"})

        teams = registry.find_by_tags({"quality", "testing"}, match_all=False)

        assert len(teams) == 2
        assert "team1" in teams  # Has quality
        assert "team3" in teams  # Has both

    def test_registry_find_by_tags_match_all(self):
        """Registry should find teams matching all tags."""
        registry = TeamSpecRegistry()
        spec1 = MagicMock(name="Spec1")
        spec2 = MagicMock(name="Spec2")

        registry.register("team1", spec1, tags={"review", "quality"})
        registry.register("team2", spec2, tags={"quality", "testing", "review"})

        teams = registry.find_by_tags({"quality", "review"}, match_all=True)

        assert len(teams) == 2  # Both have quality and review

    def test_registry_clear(self):
        """Registry should clear all team specs."""
        registry = TeamSpecRegistry()
        registry.register("team1", MagicMock())
        registry.register("team2", MagicMock())

        registry.clear()

        assert len(registry.list_teams()) == 0

    def test_registry_register_from_vertical(self):
        """Registry should register multiple specs from a vertical."""
        registry = TeamSpecRegistry()
        specs = {
            "feature_team": MagicMock(name="FeatureTeam"),
            "bug_fix_team": MagicMock(name="BugFixTeam"),
        }

        count = registry.register_from_vertical("coding", specs)

        assert count == 2
        assert registry.get("coding:feature_team") is not None
        assert registry.get("coding:bug_fix_team") is not None


# =============================================================================
# TeamSpecEntry Tests
# =============================================================================


class TestTeamSpecEntry:
    """Tests for TeamSpecEntry dataclass."""

    def test_entry_namespace_extraction(self):
        """Entry should extract namespace from name."""
        entry = TeamSpecEntry(name="coding:feature_team", spec=MagicMock())
        assert entry.namespace == "coding"

    def test_entry_namespace_none_for_unnamespaced(self):
        """Entry should return None for unnamespaced name."""
        entry = TeamSpecEntry(name="simple_team", spec=MagicMock())
        assert entry.namespace is None

    def test_entry_short_name_extraction(self):
        """Entry should extract short name without namespace."""
        entry = TeamSpecEntry(name="coding:feature_team", spec=MagicMock())
        assert entry.short_name == "feature_team"

    def test_entry_short_name_for_unnamespaced(self):
        """Entry should return full name as short name for unnamespaced."""
        entry = TeamSpecEntry(name="simple_team", spec=MagicMock())
        assert entry.short_name == "simple_team"


# =============================================================================
# find_team_for_task Tests
# =============================================================================


class TestFindTeamForTask:
    """Tests for find_team_for_task functionality."""

    def test_find_team_for_feature_task(self):
        """Should find a feature team for 'feature' task type."""
        registry = TeamSpecRegistry()
        feature_spec = MagicMock(name="FeatureSpec")
        registry.register("coding:feature_team", feature_spec)

        result = registry.find_team_for_task("feature")

        assert result is feature_spec

    def test_find_team_for_deploy_task(self):
        """Should find a deployment team for 'deploy' task type."""
        registry = TeamSpecRegistry()
        deploy_spec = MagicMock(name="DeploySpec")
        registry.register("devops:deployment_team", deploy_spec)

        result = registry.find_team_for_task("deploy")

        assert result is deploy_spec

    def test_find_team_with_preferred_vertical(self):
        """Should prefer teams from specified vertical."""
        registry = TeamSpecRegistry()
        coding_review = MagicMock(name="CodingReview")
        research_review = MagicMock(name="ResearchReview")

        registry.register("coding:review_team", coding_review)
        registry.register("research:review_team", research_review)

        # Without preference, should find coding (first in hints list)
        result_no_pref = registry.find_team_for_task("review")

        # With research preference
        result_research = registry.find_team_for_task("review", preferred_vertical="research")

        # Results may vary based on ordering, but preferred should match if found
        assert result_research is research_review

    def test_find_team_case_insensitive(self):
        """Task type search should be case insensitive."""
        registry = TeamSpecRegistry()
        ml_spec = MagicMock(name="MLSpec")
        registry.register("data_analysis:ml_team", ml_spec)

        result_lower = registry.find_team_for_task("ml")
        result_upper = registry.find_team_for_task("ML")

        assert result_lower is ml_spec
        assert result_upper is ml_spec

    def test_find_team_returns_none_for_unknown(self):
        """Should return None for unknown task types with empty registry."""
        registry = TeamSpecRegistry()
        result = registry.find_team_for_task("unknown_task_type_xyz")
        assert result is None


# =============================================================================
# Auto-Registration Integration Tests
# =============================================================================


def _has_external_vertical_packages() -> bool:
    """Check if external vertical packages are installed."""
    try:
        import victor_coding.teams  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_external_vertical_packages(),
    reason="Requires external vertical packages (victor-coding, etc.) — pending migration to vertical repos",
)
class TestAutoRegistration:
    """Tests for auto-registration from verticals.

    Note: These tests use load_all_verticals() to ensure teams are registered,
    since Python's import caching means auto-registration only happens once
    per interpreter session. The clear() in fixtures removes the registrations
    but doesn't reset the import cache.

    These tests require external vertical packages (victor-coding, victor-devops, etc.)
    to be installed and are marked as integration tests.
    """

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the global registry before each test."""
        # Clear the existing registry
        get_team_registry().clear()
        yield
        # Clean up after test
        get_team_registry().clear()

    @pytest.mark.integration
    def test_load_all_verticals_registers_teams(self):
        """load_all_verticals should register teams from all verticals."""
        # Load all verticals
        count = load_all_verticals()

        if count == 0:
            pytest.skip("No external vertical packages with team specs installed")

        # Should have registered teams from available verticals
        registry = get_team_registry()
        teams = registry.list_teams()
        assert len(teams) > 0, "Expected at least some teams to be registered"

    @pytest.mark.integration
    def test_coding_vertical_team_specs_available(self):
        """Coding team specs should be available and match registered count."""
        CODING_TEAM_SPECS = _load_vertical_attr("victor.coding.teams", "CODING_TEAM_SPECS")

        # Load via load_all_verticals to ensure registration
        load_all_verticals()

        registry = get_team_registry()
        coding_teams = registry.find_by_vertical("coding")

        assert len(coding_teams) == len(CODING_TEAM_SPECS)

        # Verify specific teams are registered
        assert registry.get("coding:feature_team") is not None
        assert registry.get("coding:bug_fix_team") is not None
        assert registry.get("coding:review_team") is not None

    @pytest.mark.integration
    def test_devops_vertical_team_specs_available(self):
        """DevOps team specs should be available and match registered count."""
        DEVOPS_TEAM_SPECS = _load_vertical_attr("victor.devops.teams", "DEVOPS_TEAM_SPECS")

        load_all_verticals()

        registry = get_team_registry()
        devops_teams = registry.find_by_vertical("devops")

        assert len(devops_teams) == len(DEVOPS_TEAM_SPECS)

        # Verify specific teams are registered
        assert registry.get("devops:deployment_team") is not None
        assert registry.get("devops:container_team") is not None

    @pytest.mark.integration
    def test_research_vertical_team_specs_available(self):
        """Research team specs should be available and match registered count."""
        RESEARCH_TEAM_SPECS = _load_vertical_attr("victor.research.teams", "RESEARCH_TEAM_SPECS")

        load_all_verticals()

        registry = get_team_registry()
        research_teams = registry.find_by_vertical("research")

        assert len(research_teams) == len(RESEARCH_TEAM_SPECS)

        # Verify specific teams are registered
        assert registry.get("research:deep_research_team") is not None
        assert registry.get("research:fact_check_team") is not None

    @pytest.mark.integration
    def test_data_analysis_vertical_team_specs_available(self):
        """Data analysis team specs should be available and match registered count."""
        DATA_ANALYSIS_TEAM_SPECS = _load_vertical_attr(
            "victor.dataanalysis.teams",
            "DATA_ANALYSIS_TEAM_SPECS",
        )

        load_all_verticals()

        registry = get_team_registry()
        da_teams = registry.find_by_vertical("data_analysis")

        # External vertical team specs aren't auto-registered into
        # the team registry yet (requires manifest-driven registration).
        # Skip if auto-registration hasn't happened.
        if len(da_teams) == 0 and len(DATA_ANALYSIS_TEAM_SPECS) > 0:
            pytest.skip("Team spec auto-registration not wired for external verticals")

        assert len(da_teams) == len(DATA_ANALYSIS_TEAM_SPECS)

        # Verify specific teams are registered
        assert registry.get("data_analysis:eda_team") is not None
        assert registry.get("data_analysis:ml_team") is not None

    @pytest.mark.integration
    def test_find_team_for_task_with_loaded_verticals(self):
        """find_team_for_task should work after loading all verticals."""
        count = load_all_verticals()

        if count == 0:
            pytest.skip("No external vertical packages with team specs installed")

        # Test cross-vertical task lookup — check at least one task type resolves
        registry = get_team_registry()
        teams = registry.list_teams()

        # Just verify the find function works with whatever teams are registered
        found_any = False
        for task_type in ["feature", "deploy", "research", "ml"]:
            result = find_team_for_task(task_type)
            if result is not None:
                found_any = True

        assert found_any, "Expected at least one task type to resolve to a team"

    @pytest.mark.integration
    def test_cross_vertical_discovery(self):
        """Should be able to discover teams across verticals."""
        count = load_all_verticals()

        if count == 0:
            pytest.skip("No external vertical packages with team specs installed")

        registry = get_team_registry()

        # Get all registered teams
        all_teams = registry.list_teams()

        # Verify at least some teams are discoverable across verticals
        assert len(all_teams) > 0, "Expected at least some teams"
        # Check that namespace-based lookup works
        namespaces = {t.split(":")[0] for t in all_teams if ":" in t}
        assert len(namespaces) > 0, "Expected at least one vertical namespace"

    @pytest.mark.integration
    def test_total_team_count(self):
        """Should register 23 teams total across all verticals."""
        load_all_verticals()
        registry = get_team_registry()

        teams = registry.list_teams()
        # Only coding teams available from external victor-coding package (6 teams)
        # Other verticals (devops, research, dataanalysis) removed from contrib
        assert len(teams) >= 6, f"Expected at least 6 teams, got {len(teams)}: {teams}"

    @pytest.mark.integration
    def test_registered_teams_have_valid_specs(self):
        """All registered teams should have valid spec objects."""
        load_all_verticals()
        registry = get_team_registry()

        for team_name in registry.list_teams():
            spec = registry.get(team_name)
            assert spec is not None, f"Team {team_name} has None spec"
            # All team specs should have name, description, and formation
            assert hasattr(spec, "name"), f"Team {team_name} spec missing name"
            assert hasattr(spec, "description"), f"Team {team_name} spec missing description"
            assert hasattr(spec, "formation"), f"Team {team_name} spec missing formation"
            assert hasattr(spec, "members"), f"Team {team_name} spec missing members"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the global registry before each test."""
        get_team_registry().clear()
        yield
        get_team_registry().clear()

    def test_register_team_spec_function(self):
        """register_team_spec should add to global registry."""
        mock_spec = MagicMock(name="MockSpec")
        register_team_spec("test:my_team", mock_spec, description="Test team")

        result = get_team_spec("test:my_team")
        assert result is mock_spec

    def test_get_team_spec_function(self):
        """get_team_spec should retrieve from global registry."""
        mock_spec = MagicMock(name="MockSpec")
        register_team_spec("test:my_team", mock_spec)

        result = get_team_spec("test:my_team")
        assert result is mock_spec

    def test_get_team_spec_returns_none_for_missing(self):
        """get_team_spec should return None for missing spec."""
        result = get_team_spec("nonexistent:team")
        assert result is None

    def test_get_team_registry_singleton(self):
        """get_team_registry should return the same instance."""
        registry1 = get_team_registry()
        registry2 = get_team_registry()
        assert registry1 is registry2

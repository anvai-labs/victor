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

"""Tests for ISP-compliant Vertical Extensions.

Tests the focused extension composites that replace the monolithic
VerticalExtensions class:
- ToolExtensions
- PromptExtensions
- SafetyExtensions
- ConfigExtensions
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from victor.agent import prompt_section_registry as registry_module
from victor.agent.prompt_section_registry import (
    UnifiedSectionRegistry,
    _initialize_default_sections,
)
from victor.core.verticals.extensions import (
    ToolExtensions,
    PromptExtensions,
    SafetyExtensions,
    ConfigExtensions,
)
from victor.core.tool_types import ToolDependency
from victor.security.safety.types import SafetyPattern
from victor.core.verticals.protocols.prompt_provider import PromptSectionContribution


class TestToolExtensions:
    """Tests for ToolExtensions."""

    def test_default_empty(self):
        """Test default empty tool extensions."""
        ext = ToolExtensions()

        assert ext.middleware == []
        assert ext.tool_dependencies == []
        assert bool(ext) is False

    def test_with_middleware(self):
        """Test tool extensions with middleware."""
        mock_middleware = MagicMock()
        ext = ToolExtensions(middleware=[mock_middleware])

        assert len(ext.middleware) == 1
        assert bool(ext) is True

    def test_with_dependencies(self):
        """Test tool extensions with dependencies."""
        dep = ToolDependency(tool_name="edit", depends_on={"read"})
        ext = ToolExtensions(tool_dependencies=[dep])

        assert len(ext.tool_dependencies) == 1
        deps = ext.get_dependencies_for("edit")
        assert "read" in deps

    def test_get_dependencies_for_unknown_tool(self):
        """Test getting dependencies for unknown tool."""
        ext = ToolExtensions()

        assert ext.get_dependencies_for("unknown") == []

    def test_has_dependency(self):
        """Test checking dependencies."""
        dep = ToolDependency(tool_name="edit", depends_on={"read", "search"})
        ext = ToolExtensions(tool_dependencies=[dep])

        assert ext.has_dependency("edit", "read") is True
        assert ext.has_dependency("edit", "search") is True
        assert ext.has_dependency("edit", "write") is False

    def test_get_all_dependency_tools(self):
        """Test getting all dependency tools."""
        deps = [
            ToolDependency(tool_name="edit", depends_on={"read"}),
            ToolDependency(tool_name="refactor", depends_on={"search", "read"}),
        ]
        ext = ToolExtensions(tool_dependencies=deps)

        all_deps = ext.get_all_dependency_tools()

        assert "read" in all_deps
        assert "search" in all_deps

    def test_get_sorted_middleware(self):
        """Test middleware sorting by priority."""
        high_mw = MagicMock()
        high_mw.get_priority.return_value = MagicMock(value=10)

        low_mw = MagicMock()
        low_mw.get_priority.return_value = MagicMock(value=100)

        ext = ToolExtensions(middleware=[low_mw, high_mw])
        sorted_mw = ext.get_sorted_middleware()

        # High priority (lower value) should come first
        assert sorted_mw[0] == high_mw

    def test_get_middleware_for_tool(self):
        """Test getting middleware for specific tool."""
        global_mw = MagicMock()
        global_mw.get_applicable_tools.return_value = None  # Applies to all

        specific_mw = MagicMock()
        specific_mw.get_applicable_tools.return_value = ["edit", "write"]

        ext = ToolExtensions(middleware=[global_mw, specific_mw])

        edit_mw = ext.get_middleware_for_tool("edit")
        assert len(edit_mw) == 2

        read_mw = ext.get_middleware_for_tool("read")
        assert len(read_mw) == 1

    def test_merge(self):
        """Test merging tool extensions."""
        mw1 = MagicMock()
        mw2 = MagicMock()
        dep1 = ToolDependency(tool_name="edit", depends_on={"read"})
        dep2 = ToolDependency(tool_name="refactor", depends_on={"search"})

        ext1 = ToolExtensions(middleware=[mw1], tool_dependencies=[dep1])
        ext2 = ToolExtensions(middleware=[mw2], tool_dependencies=[dep2])

        merged = ext1.merge(ext2)

        assert len(merged.middleware) == 2
        assert len(merged.tool_dependencies) == 2


class TestPromptExtensions:
    """Tests for PromptExtensions."""

    def test_default_empty(self):
        """Test default empty prompt extensions."""
        ext = PromptExtensions()

        assert ext.prompt_contributors == []
        assert ext.enrichment_strategy is None
        assert bool(ext) is False

    def test_with_contributors(self):
        """Test prompt extensions with contributors."""
        mock_contributor = MagicMock()
        ext = PromptExtensions(prompt_contributors=[mock_contributor])

        assert len(ext.prompt_contributors) == 1
        assert bool(ext) is True

    def test_get_all_task_hints(self):
        """Test getting merged task hints."""
        contrib1 = MagicMock()
        contrib1.get_priority.return_value = 50
        contrib1.get_task_type_hints.return_value = {"edit": "Hint 1"}

        contrib2 = MagicMock()
        contrib2.get_priority.return_value = 60
        contrib2.get_task_type_hints.return_value = {"debug": "Hint 2"}

        ext = PromptExtensions(prompt_contributors=[contrib1, contrib2])
        hints = ext.get_all_task_hints()

        assert "edit" in hints
        assert "debug" in hints

    def test_get_hint_for_task(self):
        """Test getting hint for specific task."""
        contrib = MagicMock()
        contrib.get_priority.return_value = 50
        contrib.get_task_type_hints.return_value = {"edit": MagicMock(hint="Edit hint")}

        ext = PromptExtensions(prompt_contributors=[contrib])

        hint = ext.get_hint_for_task("edit")
        assert hint is not None

        assert ext.get_hint_for_task("unknown") is None

    def test_get_combined_system_prompt_sections(self):
        """Test getting combined system prompt."""
        contrib1 = MagicMock()
        contrib1.get_priority.return_value = 50
        contrib1.get_system_prompt_section.return_value = "Section 1"

        contrib2 = MagicMock()
        contrib2.get_priority.return_value = 60
        contrib2.get_system_prompt_section.return_value = "Section 2"

        ext = PromptExtensions(prompt_contributors=[contrib2, contrib1])
        combined = ext.get_combined_system_prompt_sections()

        assert "Section 1" in combined
        assert "Section 2" in combined

    def test_named_prompt_sections_are_registered_and_rendered(self, monkeypatch):
        """Named contributor sections should feed the registry and combined prompt text."""

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        class _Contributor:
            def get_priority(self) -> int:
                return 25

            def get_prompt_section_contributions(self):
                return [
                    PromptSectionContribution(
                        name="VERTICAL_SECURITY_GUIDANCE",
                        text="Audit auth boundaries before edits.",
                        aliases={"security_guidance"},
                        category="context",
                        evolvable=True,
                        priority=25,
                    )
                ]

        ext = PromptExtensions(prompt_contributors=[_Contributor()])
        combined = ext.get_combined_system_prompt_sections()

        assert "Audit auth boundaries before edits." in combined
        assert fresh_registry.get("VERTICAL_SECURITY_GUIDANCE") is not None

    def test_has_enrichment(self):
        """Test enrichment strategy presence check."""
        ext1 = PromptExtensions()
        assert ext1.has_enrichment() is False

        ext2 = PromptExtensions(enrichment_strategy=MagicMock())
        assert ext2.has_enrichment() is True

    @pytest.mark.asyncio
    async def test_get_enrichments_no_strategy(self):
        """Test getting enrichments without strategy."""
        ext = PromptExtensions()

        result = await ext.get_enrichments("prompt", MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_get_enrichments_with_strategy(self):
        """Test getting enrichments with strategy."""
        strategy = MagicMock()
        strategy.get_enrichments = AsyncMock(return_value=["enrichment1"])

        ext = PromptExtensions(enrichment_strategy=strategy)
        result = await ext.get_enrichments("prompt", MagicMock())

        assert result == ["enrichment1"]

    def test_merge(self):
        """Test merging prompt extensions."""
        contrib1 = MagicMock()
        contrib2 = MagicMock()
        strategy = MagicMock()

        ext1 = PromptExtensions(prompt_contributors=[contrib1])
        ext2 = PromptExtensions(prompt_contributors=[contrib2], enrichment_strategy=strategy)

        merged = ext1.merge(ext2)

        assert len(merged.prompt_contributors) == 2
        assert merged.enrichment_strategy == strategy


class TestSafetyExtensions:
    """Tests for SafetyExtensions."""

    def test_default_empty(self):
        """Test default empty safety extensions."""
        ext = SafetyExtensions()

        assert ext.safety_patterns == []
        assert ext.validators == []
        assert bool(ext) is False

    def test_with_patterns(self):
        """Test safety extensions with patterns."""
        pattern = SafetyPattern(
            pattern=r"rm\s+-rf",
            description="Recursive deletion",
            risk_level="HIGH",
            category="bash",
        )
        ext = SafetyExtensions(safety_patterns=[pattern])

        assert len(ext.safety_patterns) == 1
        assert bool(ext) is True

    def test_get_all_patterns(self):
        """Test getting all patterns."""
        patterns = [
            SafetyPattern(pattern="p1", description="D1", risk_level="LOW", category="c1"),
            SafetyPattern(pattern="p2", description="D2", risk_level="HIGH", category="c2"),
        ]
        ext = SafetyExtensions(safety_patterns=patterns)

        assert len(ext.get_all_patterns()) == 2

    def test_get_patterns_by_category(self):
        """Test filtering patterns by category."""
        patterns = [
            SafetyPattern(pattern="p1", description="D1", risk_level="HIGH", category="git"),
            SafetyPattern(pattern="p2", description="D2", risk_level="HIGH", category="bash"),
            SafetyPattern(pattern="p3", description="D3", risk_level="HIGH", category="git"),
        ]
        ext = SafetyExtensions(safety_patterns=patterns)

        git_patterns = ext.get_patterns_by_category("git")
        assert len(git_patterns) == 2

    def test_get_patterns_by_risk(self):
        """Test filtering patterns by risk level."""
        patterns = [
            SafetyPattern(pattern="p1", description="D1", risk_level="LOW", category="c1"),
            SafetyPattern(pattern="p2", description="D2", risk_level="HIGH", category="c2"),
            SafetyPattern(pattern="p3", description="D3", risk_level="HIGH", category="c3"),
        ]
        ext = SafetyExtensions(safety_patterns=patterns)

        high_risk = ext.get_patterns_by_risk("HIGH")
        assert len(high_risk) == 2

    def test_get_categories(self):
        """Test getting unique categories."""
        patterns = [
            SafetyPattern(pattern="p1", description="D1", risk_level="HIGH", category="git"),
            SafetyPattern(pattern="p2", description="D2", risk_level="HIGH", category="bash"),
            SafetyPattern(pattern="p3", description="D3", risk_level="HIGH", category="git"),
        ]
        ext = SafetyExtensions(safety_patterns=patterns)

        categories = ext.get_categories()
        assert categories == {"git", "bash"}

    def test_add_pattern(self):
        """Test adding a pattern."""
        ext = SafetyExtensions()
        pattern = SafetyPattern(
            pattern="test",
            description="Test",
            risk_level="LOW",
            category="test",
        )

        ext.add_pattern(pattern)
        assert len(ext.safety_patterns) == 1

    @pytest.mark.asyncio
    async def test_validate_operation_no_validators(self):
        """Test validation without validators."""
        ext = SafetyExtensions()

        errors = await ext.validate_operation("test", {})
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_operation_with_validators(self):
        """Test validation with validators."""
        validator = MagicMock()
        validator.validate.return_value = "Error found"

        ext = SafetyExtensions(validators=[validator])
        errors = await ext.validate_operation("test", {})

        assert "Error found" in errors

    def test_merge(self):
        """Test merging safety extensions."""
        p1 = SafetyPattern(pattern="p1", description="D1", risk_level="HIGH", category="c1")
        p2 = SafetyPattern(pattern="p2", description="D2", risk_level="HIGH", category="c2")
        v1 = MagicMock()
        v2 = MagicMock()

        ext1 = SafetyExtensions(safety_patterns=[p1], validators=[v1])
        ext2 = SafetyExtensions(safety_patterns=[p2], validators=[v2])

        merged = ext1.merge(ext2)

        assert len(merged.safety_patterns) == 2
        assert len(merged.validators) == 2


class TestConfigExtensions:
    """Tests for ConfigExtensions."""

    def test_default_empty(self):
        """Test default empty config extensions."""
        ext = ConfigExtensions()

        assert ext.mode_config is None
        assert ext.rl_config is None
        assert ext.team_specs == {}
        assert bool(ext) is False

    def test_with_mode_config(self):
        """Test config extensions with mode config."""
        mock_config = MagicMock()
        ext = ConfigExtensions(mode_config=mock_config)

        assert ext.mode_config == mock_config
        assert ext.has_mode_config() is True
        assert bool(ext) is True

    def test_get_mode_configs(self):
        """Test getting mode configurations."""
        mock_config = MagicMock()
        mock_config.get_mode_configs.return_value = {
            "build": MagicMock(),
            "plan": MagicMock(),
        }

        ext = ConfigExtensions(mode_config=mock_config)
        modes = ext.get_mode_configs()

        assert "build" in modes
        assert "plan" in modes

    def test_get_mode_configs_no_provider(self):
        """Test getting mode configs without provider."""
        ext = ConfigExtensions()
        modes = ext.get_mode_configs()

        assert modes == {}

    def test_get_default_mode(self):
        """Test getting default mode."""
        mock_config = MagicMock()
        mock_config.get_default_mode.return_value = "build"

        ext = ConfigExtensions(mode_config=mock_config)
        assert ext.get_default_mode() == "build"

    def test_get_default_mode_no_provider(self):
        """Test getting default mode without provider."""
        ext = ConfigExtensions()
        assert ext.get_default_mode() == "default"

    def test_get_default_tool_budget(self):
        """Test getting default tool budget."""
        mock_config = MagicMock()
        mock_config.get_default_tool_budget.return_value = 50

        ext = ConfigExtensions(mode_config=mock_config)
        assert ext.get_default_tool_budget() == 50

    def test_get_active_learners(self):
        """Test getting active RL learners."""
        mock_rl = MagicMock()
        mock_rl.get_rl_config.return_value = {
            "active_learners": ["model_selector", "tool_selector"]
        }

        ext = ConfigExtensions(rl_config=mock_rl)
        learners = ext.get_active_learners()

        assert "model_selector" in learners
        assert "tool_selector" in learners

    def test_get_quality_thresholds(self):
        """Test getting quality thresholds."""
        mock_rl = MagicMock()
        mock_rl.get_rl_config.return_value = {"quality_thresholds": {"edit": 0.8, "analyze": 0.7}}

        ext = ConfigExtensions(rl_config=mock_rl)
        thresholds = ext.get_quality_thresholds()

        assert thresholds["edit"] == 0.8
        assert thresholds["analyze"] == 0.7

    def test_get_team_spec(self):
        """Test getting team specification."""
        mock_team = MagicMock()
        ext = ConfigExtensions(team_specs={"code_review": mock_team})

        assert ext.get_team_spec("code_review") == mock_team
        assert ext.get_team_spec("unknown") is None

    def test_get_all_team_names(self):
        """Test getting all team names."""
        ext = ConfigExtensions(
            team_specs={
                "code_review": MagicMock(),
                "feature_team": MagicMock(),
            }
        )

        names = ext.get_all_team_names()
        assert set(names) == {"code_review", "feature_team"}

    def test_has_teams(self):
        """Test team presence check."""
        ext1 = ConfigExtensions()
        assert ext1.has_teams() is False

        ext2 = ConfigExtensions(team_specs={"team1": MagicMock()})
        assert ext2.has_teams() is True

    def test_merge(self):
        """Test merging config extensions."""
        mode1 = MagicMock()
        rl2 = MagicMock()
        team1 = MagicMock()
        team2 = MagicMock()

        ext1 = ConfigExtensions(mode_config=mode1, team_specs={"team1": team1})
        ext2 = ConfigExtensions(rl_config=rl2, team_specs={"team2": team2})

        merged = ext1.merge(ext2)

        assert merged.mode_config == mode1  # Kept from ext1
        assert merged.rl_config == rl2  # Added from ext2
        assert "team1" in merged.team_specs
        assert "team2" in merged.team_specs

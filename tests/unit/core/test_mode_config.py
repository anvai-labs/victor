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

"""Tests for victor.core.mode_config module."""

import pytest

from victor.core.mode_config import (
    ModeConfig,
    ModeConfigRegistry,
    ModeDefinition,
    VerticalModeConfig,
    DEFAULT_MODES,
    DEFAULT_TASK_BUDGETS,
    get_mode_config,
    get_tool_budget,
    register_vertical_modes,
)


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    ModeConfigRegistry.reset_instance()
    return ModeConfigRegistry.get_instance()


class TestModeConfig:
    """Tests for ModeConfig dataclass."""

    def test_mode_config_has_required_fields(self):
        """ModeConfig should have tool_budget, max_iterations, exploration_multiplier."""
        config = ModeConfig(
            tool_budget=10,
            max_iterations=30,
        )
        assert config.tool_budget == 10
        assert config.max_iterations == 30
        assert config.exploration_multiplier == 1.0  # Default
        assert config.allowed_tools is None  # Default

    def test_mode_config_with_all_fields(self):
        """ModeConfig should accept all fields including allowed_tools."""
        config = ModeConfig(
            tool_budget=15,
            max_iterations=50,
            exploration_multiplier=2.5,
            allowed_tools={"read_file", "edit_files"},
        )
        assert config.tool_budget == 15
        assert config.max_iterations == 50
        assert config.exploration_multiplier == 2.5
        assert config.allowed_tools == {"read_file", "edit_files"}

    def test_mode_config_exploration_multiplier_default(self):
        """exploration_multiplier should default to 1.0."""
        config = ModeConfig(tool_budget=10, max_iterations=30)
        assert config.exploration_multiplier == 1.0


class TestModeDefinition:
    """Tests for ModeDefinition dataclass."""

    def test_mode_definition_defaults(self):
        """ModeDefinition should have sensible defaults."""
        mode = ModeDefinition(
            name="test",
            tool_budget=10,
            max_iterations=30,
        )
        assert mode.name == "test"
        assert mode.tool_budget == 10
        assert mode.max_iterations == 30
        assert mode.temperature == 0.7
        assert mode.description == ""
        assert mode.exploration_multiplier == 1.0
        assert mode.allowed_stages == []
        assert mode.priority_tools == []
        assert mode.allowed_tools is None

    def test_mode_definition_to_dict(self):
        """to_dict should return all fields."""
        mode = ModeDefinition(
            name="test",
            tool_budget=10,
            max_iterations=30,
            temperature=0.8,
            description="Test mode",
            exploration_multiplier=2.0,
        )
        d = mode.to_dict()
        assert d["name"] == "test"
        assert d["tool_budget"] == 10
        assert d["temperature"] == 0.8
        assert d["description"] == "Test mode"
        assert d["exploration_multiplier"] == 2.0

    def test_mode_definition_to_mode_config(self):
        """to_mode_config should convert to ModeConfig correctly."""
        mode = ModeDefinition(
            name="test",
            tool_budget=15,
            max_iterations=50,
            exploration_multiplier=2.5,
            allowed_tools={"read_file"},
        )
        config = mode.to_mode_config()
        assert isinstance(config, ModeConfig)
        assert config.tool_budget == 15
        assert config.max_iterations == 50
        assert config.exploration_multiplier == 2.5
        assert config.allowed_tools == {"read_file"}


class TestDefaultModes:
    """Tests for default mode configurations."""

    def test_default_modes_exist(self):
        """Default modes should be defined."""
        assert "quick" in DEFAULT_MODES
        assert "standard" in DEFAULT_MODES
        assert "comprehensive" in DEFAULT_MODES
        assert "default" in DEFAULT_MODES

    def test_default_modes_have_expected_modes(self):
        """DEFAULT_MODES should have fast, thorough, explore, plan modes."""
        assert "fast" in DEFAULT_MODES
        assert "thorough" in DEFAULT_MODES
        assert "explore" in DEFAULT_MODES
        assert "plan" in DEFAULT_MODES

    def test_default_modes_have_valid_values(self):
        """Default modes should have valid configurations."""
        for name, mode in DEFAULT_MODES.items():
            assert mode.tool_budget > 0
            assert mode.max_iterations > 0
            assert 0.0 <= mode.temperature <= 1.0
            assert mode.exploration_multiplier >= 0.0

    def test_default_modes_have_exploration_multipliers(self):
        """Default modes should have exploration_multiplier defined."""
        # Explore mode should have higher multiplier for more exploration
        assert DEFAULT_MODES["explore"].exploration_multiplier > 1.0
        # Plan mode should have multiplier for planning
        assert DEFAULT_MODES["plan"].exploration_multiplier > 1.0
        # Fast should have lower multiplier
        assert DEFAULT_MODES["fast"].exploration_multiplier <= 1.0

    def test_mode_budget_progression(self):
        """Tool budgets should increase with mode complexity."""
        assert DEFAULT_MODES["quick"].tool_budget < DEFAULT_MODES["standard"].tool_budget
        assert DEFAULT_MODES["standard"].tool_budget < DEFAULT_MODES["comprehensive"].tool_budget


class TestDefaultTaskBudgets:
    """Tests for default task budgets."""

    def test_task_budgets_exist(self):
        """Default task budgets should be defined."""
        assert "general" in DEFAULT_TASK_BUDGETS
        assert "analyze" in DEFAULT_TASK_BUDGETS
        assert "edit" in DEFAULT_TASK_BUDGETS

    def test_task_budgets_positive(self):
        """All task budgets should be positive."""
        for task, budget in DEFAULT_TASK_BUDGETS.items():
            assert budget > 0, f"Task {task} has non-positive budget"


class TestModeConfigRegistry:
    """Tests for ModeConfigRegistry class."""

    def test_singleton_pattern(self, registry):
        """Registry should be a singleton."""
        registry2 = ModeConfigRegistry.get_instance()
        assert registry is registry2

    def test_reset_instance(self, registry):
        """reset_instance should create new instance."""
        ModeConfigRegistry.reset_instance()
        registry2 = ModeConfigRegistry.get_instance()
        assert registry is not registry2

    def test_get_mode_default(self, registry):
        """get_mode should return default modes."""
        mode = registry.get_mode(None, "quick")
        assert mode is not None
        assert mode.name == "quick"

    def test_get_mode_with_alias(self, registry):
        """get_mode should resolve aliases."""
        mode = registry.get_mode(None, "fast")
        assert mode is not None
        assert mode.name == "fast"  # Alias points to quick

    def test_get_mode_not_found(self, registry):
        """get_mode should return None for unknown modes."""
        mode = registry.get_mode(None, "nonexistent")
        assert mode is None

    def test_register_vertical(self, registry):
        """register_vertical should add vertical config."""
        registry.register_vertical(
            "test_vertical",
            modes={"custom": ModeDefinition(name="custom", tool_budget=50, max_iterations=100)},
            task_budgets={"custom_task": 25},
        )
        assert "test_vertical" in registry.list_verticals()

    def test_get_mode_vertical_override(self, registry):
        """Vertical modes should override defaults."""
        registry.register_vertical(
            "coding",
            modes={"quick": ModeDefinition(name="quick", tool_budget=8, max_iterations=15)},
        )
        mode = registry.get_mode("coding", "quick")
        assert mode.tool_budget == 8

    def test_get_mode_fallback_to_default(self, registry):
        """Unknown vertical modes should fall back to defaults."""
        registry.register_vertical("coding", modes={})
        mode = registry.get_mode("coding", "comprehensive")
        assert mode is not None
        assert mode.tool_budget == DEFAULT_MODES["comprehensive"].tool_budget

    def test_get_tool_budget_from_mode(self, registry):
        """get_tool_budget should return mode budget."""
        budget = registry.get_tool_budget(None, mode_name="quick")
        assert budget == DEFAULT_MODES["quick"].tool_budget

    def test_get_tool_budget_from_task(self, registry):
        """get_tool_budget should return task-based budget."""
        budget = registry.get_tool_budget(None, task_type="analyze")
        assert budget == DEFAULT_TASK_BUDGETS["analyze"]

    def test_get_tool_budget_vertical_task(self, registry):
        """get_tool_budget should use vertical task budgets."""
        registry.register_vertical(
            "devops",
            task_budgets={"kubernetes": 20},
        )
        budget = registry.get_tool_budget("devops", task_type="kubernetes")
        assert budget == 20

    def test_get_tool_budget_default(self, registry):
        """get_tool_budget should return 10 as global default."""
        budget = registry.get_tool_budget(None)
        assert budget == 10

    def test_get_mode_configs_includes_defaults(self, registry):
        """get_mode_configs should include default modes."""
        configs = registry.get_mode_configs()
        assert "quick" in configs
        assert "standard" in configs

    def test_get_mode_configs_with_vertical_overrides(self, registry):
        """get_mode_configs should include vertical overrides."""
        registry.register_vertical(
            "coding",
            modes={
                "architect": ModeDefinition(name="architect", tool_budget=40, max_iterations=100)
            },
        )
        configs = registry.get_mode_configs("coding")
        assert "architect" in configs
        assert "quick" in configs  # Still includes defaults

    def test_get_max_iterations(self, registry):
        """get_max_iterations should return correct value."""
        iters = registry.get_max_iterations(None, "comprehensive")
        assert iters == DEFAULT_MODES["comprehensive"].max_iterations

    def test_get_default_mode(self, registry):
        """get_default_mode should return correct default."""
        assert registry.get_default_mode() == "default"

    def test_get_default_mode_vertical(self, registry):
        """get_default_mode should use vertical default."""
        registry.register_vertical(
            "devops",
            default_mode="standard",
        )
        assert registry.get_default_mode("devops") == "standard"

    def test_list_modes(self, registry):
        """list_modes should return all mode names."""
        modes = registry.list_modes()
        assert "quick" in modes
        assert "standard" in modes
        assert "comprehensive" in modes

    def test_add_mode_alias(self, registry):
        """add_mode_alias should add new alias."""
        registry.add_mode_alias("super_fast", "quick")
        mode = registry.get_mode(None, "super_fast")
        assert mode is not None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_mode_config_function(self):
        """get_mode_config should work as convenience function."""
        ModeConfigRegistry.reset_instance()
        mode = get_mode_config("quick")
        assert mode is not None
        assert mode.name == "quick"

    def test_get_tool_budget_function(self):
        """get_tool_budget should work as convenience function."""
        ModeConfigRegistry.reset_instance()
        budget = get_tool_budget(mode_name="standard")
        assert budget == DEFAULT_MODES["standard"].tool_budget

    def test_register_vertical_modes_function(self):
        """register_vertical_modes should register correctly."""
        ModeConfigRegistry.reset_instance()
        register_vertical_modes(
            "test",
            modes={"custom": ModeDefinition(name="custom", tool_budget=100, max_iterations=200)},
        )
        registry = ModeConfigRegistry.get_instance()
        assert "test" in registry.list_verticals()


class TestGetModesAndRegisterModes:
    """Tests for get_modes() and register_modes() methods."""

    def test_get_modes_returns_default_modes(self, registry):
        """get_modes() should return default modes when no vertical specified."""
        modes = registry.get_modes()
        assert "fast" in modes
        assert "thorough" in modes
        assert "explore" in modes
        assert "plan" in modes
        # Check they are ModeConfig instances
        assert isinstance(modes["fast"], ModeConfig)

    def test_get_modes_returns_mode_config_objects(self, registry):
        """get_modes() should return ModeConfig objects with correct fields."""
        modes = registry.get_modes()
        fast_mode = modes["fast"]
        assert hasattr(fast_mode, "tool_budget")
        assert hasattr(fast_mode, "max_iterations")
        assert hasattr(fast_mode, "exploration_multiplier")
        assert hasattr(fast_mode, "allowed_tools")

    def test_get_modes_with_vertical_merges_modes(self, registry):
        """get_modes(vertical) should return merged modes with overrides."""
        # Register a vertical with custom mode
        registry.register_modes(
            "coding",
            {
                "architect": ModeConfig(
                    tool_budget=40, max_iterations=100, exploration_multiplier=1.5
                )
            },
        )
        modes = registry.get_modes("coding")
        # Should have both default modes and vertical modes
        assert "fast" in modes  # Default
        assert "architect" in modes  # Vertical-specific
        assert modes["architect"].tool_budget == 40

    def test_get_modes_vertical_overrides_default(self, registry):
        """get_modes(vertical) should override default modes."""
        # Override the fast mode for coding vertical
        registry.register_modes(
            "coding",
            {"fast": ModeConfig(tool_budget=8, max_iterations=15, exploration_multiplier=0.5)},
        )
        modes = registry.get_modes("coding")
        # The fast mode should be overridden
        assert modes["fast"].tool_budget == 8
        assert modes["fast"].max_iterations == 15

    def test_register_modes_adds_vertical_specific_modes(self, registry):
        """register_modes() should add vertical-specific modes."""
        registry.register_modes(
            "devops",
            {
                "kubernetes": ModeConfig(
                    tool_budget=25, max_iterations=60, exploration_multiplier=2.0
                ),
                "terraform": ModeConfig(tool_budget=20, max_iterations=50),
            },
        )
        # Check via get_modes
        modes = registry.get_modes("devops")
        assert "kubernetes" in modes
        assert "terraform" in modes
        assert modes["kubernetes"].tool_budget == 25
        assert modes["terraform"].exploration_multiplier == 1.0  # Default

    def test_register_modes_multiple_calls_update(self, registry):
        """register_modes() called multiple times should update modes."""
        registry.register_modes(
            "data",
            {"analysis": ModeConfig(tool_budget=15, max_iterations=40)},
        )
        registry.register_modes(
            "data",
            {"visualization": ModeConfig(tool_budget=10, max_iterations=30)},
        )
        modes = registry.get_modes("data")
        assert "analysis" in modes
        assert "visualization" in modes

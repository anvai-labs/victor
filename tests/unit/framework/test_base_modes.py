"""Tests for base mode configuration (WS-5)."""

from victor.framework.modes.base_modes import BaseModeConfig, ModeDefinition


class TestBaseModeConfig:

    def test_base_modes_exist(self):
        modes = BaseModeConfig.get_modes()
        assert "default" in modes
        assert "careful" in modes
        assert "fast" in modes
        assert "analysis" in modes

    def test_subclass_extends_modes(self):
        class CodingModeConfig(BaseModeConfig):
            @classmethod
            def get_vertical_modes(cls):
                return {
                    "architect": ModeDefinition(
                        name="architect",
                        tool_budget=40,
                        description="Architecture mode",
                    )
                }

        modes = CodingModeConfig.get_modes()
        assert "default" in modes  # base
        assert "architect" in modes  # vertical

    def test_complexity_mapping(self):
        assert BaseModeConfig.get_mode_for_complexity("trivial") == "fast"
        assert BaseModeConfig.get_mode_for_complexity("complex") == "careful"
        assert BaseModeConfig.get_mode_for_complexity("unknown") == "default"

    def test_task_budgets(self):
        budgets = BaseModeConfig.get_task_budgets()
        assert "code_generation" in budgets
        assert budgets["code_generation"] == 3

    def test_register_pushes_to_mode_config_registry(self):
        from victor.core.mode_config import ModeConfigRegistry

        class TestVerticalModes(BaseModeConfig):
            @classmethod
            def get_vertical_modes(cls):
                return {
                    "test_mode": ModeDefinition(
                        name="test_mode",
                        tool_budget=99,
                        description="Test mode",
                    )
                }

        TestVerticalModes.register("test_vertical")
        registry = ModeConfigRegistry.get_instance()
        mode = registry.get_mode("test_vertical", "test_mode")
        assert mode is not None

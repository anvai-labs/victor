"""Guard tests: config_registry.py must not hardcode vertical-specific configs.

The VerticalConfigRegistry should only contain a "default" fallback entry.
Vertical-specific provider hints and evaluation criteria should be registered
dynamically by external vertical packages during VictorPlugin.register().
"""

from victor.core.verticals.config_registry import VerticalConfigRegistry


class TestConfigRegistryDecoupling:
    """Ensure hardcoded vertical configs are removed from core."""

    def test_provider_hints_has_only_default_key(self):
        """The _provider_hints class dict should only contain 'default'.

        All vertical-specific hints (coding, research, devops, etc.) should
        come from entry points or register_vertical_config() at runtime.
        """
        allowed_keys = {"default"}
        actual_keys = set(VerticalConfigRegistry._provider_hints.keys())
        extra = actual_keys - allowed_keys
        assert not extra, (
            f"_provider_hints has hardcoded vertical keys: {extra}. "
            f"Move these to external vertical packages via register_vertical_config()."
        )

    def test_evaluation_criteria_has_only_default_key(self):
        """The _evaluation_criteria class dict should only contain 'default'."""
        allowed_keys = {"default"}
        actual_keys = set(VerticalConfigRegistry._evaluation_criteria.keys())
        extra = actual_keys - allowed_keys
        assert not extra, (
            f"_evaluation_criteria has hardcoded vertical keys: {extra}. "
            f"Move these to external vertical packages via register_vertical_config()."
        )

    def test_registered_config_overrides_default(self):
        """Register a vertical config and verify it takes precedence."""
        test_hints = {
            "preferred_providers": ["test-provider"],
            "min_context_window": 42,
        }
        test_criteria = ["test criterion 1", "test criterion 2"]

        # Register
        VerticalConfigRegistry.register_vertical_config(
            "test_vertical",
            provider_hints=test_hints,
            evaluation_criteria=test_criteria,
        )

        try:
            hints = VerticalConfigRegistry.get_provider_hints("test_vertical")
            criteria = VerticalConfigRegistry.get_evaluation_criteria("test_vertical")

            assert hints["preferred_providers"] == ["test-provider"]
            assert hints["min_context_window"] == 42
            assert criteria == test_criteria
        finally:
            # Cleanup: remove test registration
            VerticalConfigRegistry._registered_provider_hints.pop("test_vertical", None)
            VerticalConfigRegistry._registered_eval_criteria.pop("test_vertical", None)

    def test_unregistered_vertical_falls_back_to_default(self):
        """An unknown vertical should return default hints, not raise."""
        hints = VerticalConfigRegistry.get_provider_hints("nonexistent_vertical_xyz")
        default_hints = VerticalConfigRegistry.get_provider_hints("default")
        assert hints == default_hints

    def test_get_evaluation_criteria_falls_back_to_default(self):
        """An unknown vertical should return default criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("nonexistent_vertical_xyz")
        default_criteria = VerticalConfigRegistry.get_evaluation_criteria("default")
        assert criteria == default_criteria

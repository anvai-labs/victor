"""Tests for ExtensionModuleResolver."""

from unittest.mock import MagicMock

from victor.core.verticals.extension_module_resolver import ExtensionModuleResolver


class TestExtensionModuleResolver:
    def setup_method(self):
        self.monitor = MagicMock()
        self.monitor.record_missing_module.return_value = True
        self.resolver = ExtensionModuleResolver(pressure_monitor=self.monitor)

    def test_resolve_candidates_returns_list(self):
        candidates = self.resolver.resolve_candidates("coding", "safety")
        assert isinstance(candidates, list)

    def test_resolve_candidates_empty_suffix(self):
        assert self.resolver.resolve_candidates("coding", "") == []

    def test_resolve_candidates_empty_name(self):
        assert self.resolver.resolve_candidates("", "safety") == []

    def test_is_available_empty_path(self):
        assert self.resolver.is_available("") is False

    def test_is_available_nonexistent_module(self):
        assert self.resolver.is_available("nonexistent.module.path") is False

    def test_is_available_real_module(self):
        assert self.resolver.is_available("victor.core.verticals") is True

    def test_auto_generate_class_name(self):
        name = self.resolver.auto_generate_class_name("CodingAssistant", "safety_extension")
        assert name == "CodingSafetyExtension"

    def test_auto_generate_class_name_no_assistant_suffix(self):
        name = self.resolver.auto_generate_class_name("MyVertical", "mode_config")
        assert name == "MyVerticalModeConfig"

    def test_load_attribute_success(self):
        result = self.resolver.load_attribute(
            "victor.core.verticals.extension_module_resolver", "ExtensionModuleResolver"
        )
        assert result is ExtensionModuleResolver

    def test_load_attribute_missing_raises(self):
        import pytest

        with pytest.raises((ImportError, AttributeError)):
            self.resolver.load_attribute(
                "victor.core.verticals.extension_module_resolver", "NonExistent"
            )

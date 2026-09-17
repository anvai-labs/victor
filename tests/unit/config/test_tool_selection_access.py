# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the canonical tool-pruning decision accessor."""

import pytest

from victor.config.tool_selection_access import is_tool_selection_enabled


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VICTOR_TOOL_SELECTION", raising=False)
    monkeypatch.delenv("VICTOR_TOOLS__TOOL_SELECTION_ENABLED", raising=False)


class TestDefaults:
    def test_defaults_off_without_settings_or_env(self):
        assert is_tool_selection_enabled(None) is False
        assert is_tool_selection_enabled() is False


class TestFlatLegacyEnv:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "True", "ON"])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("VICTOR_TOOL_SELECTION", value)
        assert is_tool_selection_enabled(None) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "junk", "  "])
    def test_falsy_values_stay_off(self, monkeypatch, value):
        monkeypatch.setenv("VICTOR_TOOL_SELECTION", value)
        assert is_tool_selection_enabled(None) is False


class TestSettingsField:
    def _settings(self):
        from victor.config.settings import Settings
        from victor.config.tool_settings import ToolSettings

        return Settings(tools=ToolSettings(tool_selection_enabled=True))

    def test_nested_env_enables(self, monkeypatch):
        monkeypatch.setenv("VICTOR_TOOLS__TOOL_SELECTION_ENABLED", "true")
        settings = self._settings()
        assert is_tool_selection_enabled(settings) is True

    def test_settings_field_true_enables_even_without_env(self, monkeypatch):
        settings = self._settings()
        assert is_tool_selection_enabled(settings) is True

    def test_settings_field_false_with_env_unset_stays_off(self):
        from victor.config.settings import Settings
        from victor.config.tool_settings import ToolSettings

        settings = Settings(tools=ToolSettings(tool_selection_enabled=False))
        assert is_tool_selection_enabled(settings) is False


class TestConfigOverride:
    def test_override_false_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("VICTOR_TOOL_SELECTION", "1")
        assert is_tool_selection_enabled(None, config_override=False) is False

    def test_override_true_wins_without_env(self):
        assert is_tool_selection_enabled(None, config_override=True) is True

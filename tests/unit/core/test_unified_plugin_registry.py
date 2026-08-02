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

"""TDD tests for unified plugin registry API.

These tests define the expected behavior of list_all_with_type()
which provides a single view of ALL plugin types: verticals,
external manifest plugins, and entry-point plugins.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestListAllWithType:
    """Tests for PluginRegistry.list_all_with_type() unified view."""

    def _make_registry(self):
        from victor.core.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        return registry

    def test_returns_list_of_dicts(self):
        """list_all_with_type() should return a list of dicts."""
        registry = self._make_registry()
        result = registry.list_all_with_type()
        assert isinstance(result, list)

    def test_each_entry_has_required_keys(self):
        """Each entry must have: name, type, version, enabled, source."""
        registry = self._make_registry()
        # Add a mock plugin
        mock_plugin = MagicMock()
        mock_plugin.name = "test-plugin"
        mock_plugin.health_check.return_value = {"healthy": True}
        registry._plugins["test-plugin"] = mock_plugin
        registry._discovered = True

        result = registry.list_all_with_type()
        assert len(result) >= 1

        for entry in result:
            assert "name" in entry
            assert "type" in entry
            assert "version" in entry
            assert "enabled" in entry

    def test_external_plugins_have_type_external(self):
        """External manifest plugins should have type='external'."""
        from victor.core.plugins.registry import PluginRegistry, _ExternalPluginAdapter

        registry = PluginRegistry()

        mock_registered = MagicMock()
        mock_registered.plugin_id = "my-tool@external"
        mock_registered.enabled = True
        mock_registered.kind.value = "external"
        mock_registered.version = "0.1.0"
        mock_registered.manifest.tools = []

        adapter = _ExternalPluginAdapter(mock_registered)
        registry._plugins["my-tool@external"] = adapter
        registry._discovered = True

        result = registry.list_all_with_type()
        ext_entry = next((e for e in result if e["name"] == "my-tool@external"), None)
        assert ext_entry is not None
        assert ext_entry["type"] == "external"

    def test_regular_plugins_have_type_plugin(self):
        """Regular VictorPlugin instances should have type='plugin'."""
        from victor.core.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        mock_plugin = MagicMock()
        mock_plugin.name = "my-custom-plugin"
        registry._plugins["my-custom-plugin"] = mock_plugin
        registry._discovered = True

        result = registry.list_all_with_type()
        entry = next((e for e in result if e["name"] == "my-custom-plugin"), None)
        assert entry is not None
        assert entry["type"] == "plugin"

    def test_no_duplicates(self):
        """Each plugin should appear exactly once."""
        from victor.core.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        mock_plugin = MagicMock()
        mock_plugin.name = "coding"
        registry._plugins["coding"] = mock_plugin
        registry._discovered = True

        result = registry.list_all_with_type()
        names = [e["name"] for e in result]
        assert len(names) == len(set(names)), f"Duplicates found: {names}"

    def test_empty_when_not_discovered(self):
        """Returns empty list when discovery hasn't run."""
        registry = self._make_registry()
        result = registry.list_all_with_type()
        assert result == []

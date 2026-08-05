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

import unittest
from unittest.mock import MagicMock, patch
import typer

from victor.core.plugins.registry import PluginRegistry
from victor.core.plugins.protocol import VictorPlugin
from victor.core.container import ServiceContainer


class MockPlugin(VictorPlugin):
    def __init__(self, name="mock"):
        self._name = name
        self.registered = False
        self.cli_app = None

    @property
    def name(self) -> str:
        return self._name

    def register(self, container: ServiceContainer) -> None:
        self.registered = True

    def get_cli_app(self) -> typer.Typer:
        if self.cli_app is None:
            self.cli_app = typer.Typer(name=self._name)
        return self.cli_app


class TestPluginSystem(unittest.TestCase):
    def setUp(self):
        self.registry = PluginRegistry()
        # Reset singleton for testing if needed, or just use a fresh instance

    @patch("victor.core.plugins.registry.get_entry_point_values")
    def test_plugin_discovery(self, mock_get_entry_point_values):
        mock_get_entry_point_values.return_value = {
            "test_plugin": "mock_pkg:MockPlugin",
        }

        # Patch _load_plugin_from_value to return our MockPlugin instance
        with patch.object(
            self.registry,
            "_load_plugin_from_value",
            return_value=MockPlugin(name="test_plugin"),
        ):
            plugins = self.registry.discover(force=True)

        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0].name, "test_plugin")
        # discover() scans victor.plugins
        mock_get_entry_point_values.assert_called_once_with("victor.plugins", force=True)

    def test_plugin_registration(self):
        container = ServiceContainer()
        plugin = MockPlugin(name="reg_test")

        self.registry._plugins["reg_test"] = plugin
        self.registry.register_all(container)

        self.assertTrue(plugin.registered)

    def test_cli_app_retrieval(self):
        plugin = MockPlugin(name="cli_test")
        app = plugin.get_cli_app()

        self.assertIsInstance(app, typer.Typer)
        self.assertEqual(app.info.name, "cli_test")

    def test_list_plugins(self):
        self.registry._plugins = {"p1": MockPlugin("p1"), "p2": MockPlugin("p2")}

        plugins = self.registry.list_plugins()
        self.assertEqual(len(plugins), 2)
        names = [p.name for p in plugins]
        self.assertIn("p1", names)
        self.assertIn("p2", names)

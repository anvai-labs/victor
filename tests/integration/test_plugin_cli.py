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
from typer.testing import CliRunner
import typer

from victor.ui.cli import app
from victor.core.plugins.protocol import VictorPlugin


class MockPluginWithCLI(VictorPlugin):
    @property
    def name(self) -> str:
        return "testplugin"

    def register(self, container) -> None:
        pass

    def get_cli_app(self) -> typer.Typer:
        sub_app = typer.Typer(name="testplugin")

        @sub_app.command("hello")
        def hello():
            print("Hello from plugin!")

        return sub_app


class TestPluginCLIIntegration(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("victor.core.plugins.registry.PluginRegistry.discover")
    def test_cli_discovers_plugin_command(self, mock_discover):
        # Setup mock plugin
        plugin = MockPluginWithCLI()
        mock_discover.return_value = [plugin]

        # We need to re-run the registration because cli.py runs it at import time
        # For testing, we can manually call the registration helper
        from victor.ui.cli import _register_plugin_commands

        _register_plugin_commands()

        # Run help to see if 'testplugin' is listed
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("testplugin", result.output)

        # Run the plugin command
        result = self.runner.invoke(app, ["testplugin", "hello"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello from plugin!", result.output)

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Plugin registration tests - runnable with only victor-contracts installed."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from victor_security import (
    SecretPatternScanTool,
    SecurityAssistant,
    SecurityPlugin,
    plugin,
)

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


class FakePluginContext:
    """Minimal PluginContext double capturing plugin registrations."""

    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.verticals: list[Any] = []

    def register_tool(self, tool_instance: Any) -> None:
        self.tools.append(tool_instance)

    def register_vertical(self, vertical_class: Any) -> None:
        self.verticals.append(vertical_class)


def test_plugin_registers_vertical_and_tool() -> None:
    context = FakePluginContext()

    SecurityPlugin().register(context)

    assert context.verticals == [SecurityAssistant]
    assert [tool.name for tool in context.tools] == ["secret_pattern_scan"]


def test_registered_tool_matches_duck_typed_tool_shape() -> None:
    context = FakePluginContext()
    SecurityPlugin().register(context)
    (tool,) = context.tools

    assert isinstance(tool, SecretPatternScanTool)
    assert isinstance(tool.name, str) and tool.name
    assert isinstance(tool.description, str) and tool.description
    assert isinstance(tool.parameters, dict)
    assert tool.parameters["type"] == "object"
    assert set(tool.parameters["properties"]) == {"text", "paths"}
    assert inspect.iscoroutinefunction(tool.execute)


def test_entry_points_match_pyproject_metadata() -> None:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    entry_points = data["project"]["entry-points"]

    assert entry_points["victor.plugins"]["security"] == "victor_security:plugin"
    assert entry_points["victor.mode_configs"]["security"] == (
        "victor_security.mode_config:SecurityModeConfigProvider"
    )
    assert plugin.name == "security"
    assert data["project"]["version"] == SecurityAssistant.version


async def test_secret_pattern_scan_finds_and_masks_secrets(tmp_path: Path) -> None:
    secret_file = tmp_path / "config.py"
    secret_file.write_text(
        'AWS_KEY = "AKIAABCDEFGHIJKLMNOP"\n' 'password = "hunter2hunter2"\n' "safe_value = 42\n",
        encoding="utf-8",
    )

    result = await SecretPatternScanTool().execute(
        text="token = xoxb-123456789012-abcdefABCDEF",
        paths=[str(secret_file)],
    )

    assert result["success"] is True
    assert result["errors"] == []
    assert result["scanned_sources"] == ["<text>", str(secret_file)]

    patterns = {finding["pattern"] for finding in result["findings"]}
    assert "slack_token" in patterns
    assert "aws_access_key_id" in patterns
    assert "hardcoded_credential" in patterns

    for finding in result["findings"]:
        assert "*" in finding["match"], "matches must be masked"
        assert "hunter2hunter2" not in finding["match"]
        assert "AKIAABCDEFGHIJKLMNOP" not in finding["match"]


async def test_secret_pattern_scan_reports_unreadable_paths() -> None:
    result = await SecretPatternScanTool().execute(paths=["/nonexistent/creds.env"])

    assert result["success"] is True
    assert result["findings"] == []
    assert result["scanned_sources"] == []
    assert len(result["errors"]) == 1
    assert "/nonexistent/creds.env" in result["errors"][0]

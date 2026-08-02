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

"""Tests for the config-show CLI command."""

from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from victor.ui.cli import app

runner = CliRunner()


def test_config_show_uses_global_victor_dir_by_default(tmp_path):
    """config show should resolve config files from centralized Victor paths."""
    config_dir = tmp_path / "custom-victor-config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "profiles.yaml").write_text("profiles:\n  default:\n    provider: ollama\n")
    (config_dir / "settings.yaml").write_text("log_level: INFO\n")

    mock_settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="ollama",
            default_model="qwen3.5:27b-q4_K_M",
            default_temperature=0.7,
            default_max_tokens=4096,
        ),
        tools=SimpleNamespace(
            fallback_max_tools=5,
            tool_selection_cache_enabled=True,
            enable_tool_deduplication=True,
        ),
    )

    with (
        patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
        patch(
            "victor.ui.commands.config.get_project_paths",
            return_value=SimpleNamespace(global_victor_dir=config_dir),
        ),
    ):
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    rendered_output = result.output.replace("\n", "")
    assert "Config directory:" in result.output
    assert config_dir.name in rendered_output
    assert "settings.yaml" in result.output
    assert "profiles.yaml" in result.output

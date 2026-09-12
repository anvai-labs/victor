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

"""Tests for `victor mcp add`."""

import asyncio

import yaml
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from victor.integrations.mcp.registry import MCPRegistry
from victor.ui.commands.mcp import mcp_app

runner = CliRunner()


def _mock_client(connect_result: bool = True, tools=None):
    """Build a fake MCPClient matching test_mcp_registry.py's mock shape."""
    client = MagicMock()
    client.connect = AsyncMock(return_value=connect_result)
    client.tools = tools if tools is not None else []
    client.resources = []
    return client


class TestMcpAddHappyPath:
    def test_writes_list_shaped_yaml(self, tmp_path):
        with patch(
            "victor.integrations.mcp.registry.MCPClient",
            return_value=_mock_client(tools=[MagicMock(name="t1"), MagicMock(name="t2")]),
        ):
            result = runner.invoke(
                mcp_app,
                ["add", "myserver", "python", "-m", "myserver", "--config-dir", str(tmp_path)],
            )

        assert result.exit_code == 0, result.output
        assert "2 tool(s) discovered" in result.output

        data = yaml.safe_load((tmp_path / "mcp.yaml").read_text())
        assert isinstance(data["servers"], list)
        assert data["servers"][0]["name"] == "myserver"
        assert data["servers"][0]["command"] == ["python", "-m", "myserver"]

    def test_round_trips_through_from_config(self, tmp_path):
        with patch(
            "victor.integrations.mcp.registry.MCPClient",
            return_value=_mock_client(tools=[MagicMock()]),
        ):
            result = runner.invoke(
                mcp_app, ["add", "myserver", "echo", "hi", "--config-dir", str(tmp_path)]
            )
        assert result.exit_code == 0, result.output

        registry = MCPRegistry.from_config(tmp_path / "mcp.yaml")
        assert registry.list_servers() == ["myserver"]
        assert registry._servers["myserver"].config.command == ["echo", "hi"]


class TestMcpAddEnv:
    def test_env_pairs_persist(self, tmp_path):
        with patch("victor.integrations.mcp.registry.MCPClient", return_value=_mock_client()):
            result = runner.invoke(
                mcp_app,
                [
                    "add",
                    "myserver",
                    "echo",
                    "--env",
                    "FOO=bar",
                    "--env",
                    "BAZ=qux",
                    "--config-dir",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        data = yaml.safe_load((tmp_path / "mcp.yaml").read_text())
        assert data["servers"][0]["env"] == {"FOO": "bar", "BAZ": "qux"}

    def test_malformed_env_rejected_without_write(self, tmp_path):
        result = runner.invoke(
            mcp_app,
            ["add", "myserver", "echo", "--env", "NOVALUE", "--config-dir", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert not (tmp_path / "mcp.yaml").exists()


class TestMcpAddScope:
    def test_config_dir_overrides_scope(self, tmp_path):
        with patch("victor.integrations.mcp.registry.MCPClient", return_value=_mock_client()):
            result = runner.invoke(
                mcp_app,
                [
                    "add",
                    "myserver",
                    "echo",
                    "--scope",
                    "global",
                    "--config-dir",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "mcp.yaml").exists()

    def test_invalid_scope_rejected(self, tmp_path):
        result = runner.invoke(mcp_app, ["add", "myserver", "echo", "--scope", "nonsense"])
        assert result.exit_code != 0


class TestMcpAddUpsert:
    def test_same_name_updates_in_place(self, tmp_path):
        with patch("victor.integrations.mcp.registry.MCPClient", return_value=_mock_client()):
            runner.invoke(
                mcp_app, ["add", "myserver", "echo", "one", "--config-dir", str(tmp_path)]
            )
            result = runner.invoke(
                mcp_app, ["add", "myserver", "echo", "two", "--config-dir", str(tmp_path)]
            )
        assert result.exit_code == 0, result.output
        data = yaml.safe_load((tmp_path / "mcp.yaml").read_text())
        assert len(data["servers"]) == 1
        assert data["servers"][0]["command"] == ["echo", "two"]


class TestMcpAddValidationFailure:
    def test_failure_does_not_persist_by_default(self, tmp_path):
        with patch(
            "victor.integrations.mcp.registry.MCPClient",
            return_value=_mock_client(connect_result=False),
        ):
            result = runner.invoke(
                mcp_app, ["add", "myserver", "badcmd", "--config-dir", str(tmp_path)]
            )
        assert result.exit_code != 0
        assert not (tmp_path / "mcp.yaml").exists()

    def test_force_persists_despite_failure(self, tmp_path):
        with patch(
            "victor.integrations.mcp.registry.MCPClient",
            return_value=_mock_client(connect_result=False),
        ):
            result = runner.invoke(
                mcp_app,
                ["add", "myserver", "badcmd", "--force", "--config-dir", str(tmp_path)],
            )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "mcp.yaml").exists()

    def test_timeout_treated_as_failure(self, tmp_path):
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(10)
            return True

        client = MagicMock()
        client.connect = _hang
        client.tools = []
        client.resources = []

        with patch("victor.integrations.mcp.registry.MCPClient", return_value=client):
            result = runner.invoke(
                mcp_app,
                [
                    "add",
                    "myserver",
                    "slowcmd",
                    "--timeout",
                    "0.05",
                    "--config-dir",
                    str(tmp_path),
                ],
            )
        assert result.exit_code != 0
        assert not (tmp_path / "mcp.yaml").exists()


class TestMcpAddPermissions:
    def test_file_permissions_are_0600(self, tmp_path):
        with patch("victor.integrations.mcp.registry.MCPClient", return_value=_mock_client()):
            result = runner.invoke(
                mcp_app, ["add", "myserver", "echo", "--config-dir", str(tmp_path)]
            )
        assert result.exit_code == 0, result.output
        mode = (tmp_path / "mcp.yaml").stat().st_mode & 0o777
        assert mode == 0o600

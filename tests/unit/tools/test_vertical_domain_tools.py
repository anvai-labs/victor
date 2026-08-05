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

"""Tests for the vertical-driven domain dispatchers (docker / cicd / db).

Each is a bash-style dispatcher that delegates to its vertical implementation
when present and falls back otherwise. Advertised per-vertical via
``vertical_tools.yaml`` (Phase 1) so they cost no base-schema tokens elsewhere.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestDockerDispatcher:
    @pytest.mark.asyncio
    async def test_delegates_to_devops_docker(self):
        from victor.tools.unified.docker_tool import docker_tool

        mock_docker = AsyncMock(return_value={"success": True, "output": "CONTAINER ID ..."})
        with patch(
            "victor.tools.unified.docker_tool.resolve_vertical_callable",
            return_value=(mock_docker, "victor_devops.tools.docker_tool"),
        ):
            result = await docker_tool("docker ps")
        mock_docker.assert_awaited_once_with(operation="ps")
        assert "CONTAINER ID" in result

    @pytest.mark.asyncio
    async def test_logs_delegation_maps_resource_id(self):
        from victor.tools.unified.docker_tool import docker_tool

        mock_docker = AsyncMock(return_value={"success": True, "output": "logs"})
        with patch(
            "victor.tools.unified.docker_tool.resolve_vertical_callable",
            return_value=(mock_docker, "victor_devops.tools.docker_tool"),
        ):
            await docker_tool("docker logs myapp")
        mock_docker.assert_awaited_once_with(operation="logs", resource_id="myapp")

    @pytest.mark.asyncio
    async def test_shell_fallback_when_devops_absent(self):
        from victor.tools.unified.docker_tool import docker_tool

        mock_shell = AsyncMock(return_value={"success": True, "stdout": "## container"})
        with (
            patch(
                "victor.tools.unified.docker_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await docker_tool("docker ps")
        assert mock_shell.call_args.kwargs["readonly"] is True
        assert "container" in result


class TestCicdDispatcher:
    @pytest.mark.asyncio
    async def test_delegates_to_devops_cicd(self):
        from victor.tools.unified.cicd_tool import cicd_tool

        mock_cicd = AsyncMock(return_value={"success": True, "output": "generated"})
        with patch(
            "victor.tools.unified.cicd_tool.resolve_vertical_callable",
            return_value=(mock_cicd, "victor_devops.tools.cicd_tool"),
        ):
            result = await cicd_tool("cicd generate --workflow python-test")
        mock_cicd.assert_awaited_once()
        assert mock_cicd.call_args.kwargs["operation"] == "generate"
        assert mock_cicd.call_args.kwargs["workflow"] == "python-test"
        assert "generated" in result

    @pytest.mark.asyncio
    async def test_absent_devops_returns_graceful_message(self):
        from victor.tools.unified.cicd_tool import cicd_tool

        with patch(
            "victor.tools.unified.cicd_tool.resolve_vertical_callable",
            return_value=(None, None),
        ):
            result = await cicd_tool("cicd list")
        assert "### ❌ ERROR" in result
        assert "victor-devops" in result


class TestDbDispatcher:
    @pytest.mark.asyncio
    async def test_query_without_connection_uses_shell_fallback(self):
        from victor.tools.unified.db_tool import db_tool

        mock_shell = AsyncMock(return_value={"success": True, "stdout": "1"})
        with patch("victor.tools.bash.shell", mock_shell):
            result = await db_tool('db query "SELECT 1"')
        # Stateless query -> shell sqlite3, not the database() tool.
        assert "sqlite3" in mock_shell.call_args.kwargs["cmd"]
        assert "1" in result

    @pytest.mark.asyncio
    async def test_query_with_connection_delegates_to_database(self):
        from victor.tools.unified.db_tool import db_tool

        mock_db = AsyncMock(return_value={"success": True, "rows": [["1"]]})
        with patch("victor.tools.database_tool.database", mock_db):
            await db_tool('db query "SELECT 1" --connection c1')
        assert mock_db.call_args.kwargs["action"] == "query"
        assert mock_db.call_args.kwargs["connection_id"] == "c1"

    @pytest.mark.asyncio
    async def test_tables_delegates(self):
        from victor.tools.unified.db_tool import db_tool

        mock_db = AsyncMock(return_value={"success": True, "tables": ["users"]})
        with patch("victor.tools.database_tool.database", mock_db):
            result = await db_tool("db tables --connection c1")
        assert mock_db.call_args.kwargs["action"] == "tables"
        assert "users" in result

    @pytest.mark.asyncio
    async def test_tool_registered_names(self):
        from victor.tools.unified.cicd_tool import cicd_tool
        from victor.tools.unified.db_tool import db_tool
        from victor.tools.unified.docker_tool import docker_tool

        assert docker_tool.Tool.name == "docker"
        assert cicd_tool.Tool.name == "cicd"
        assert db_tool.Tool.name == "db"

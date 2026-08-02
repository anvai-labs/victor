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

"""Tests for tool observability follow-up propagation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.services.tool_observability import ToolObservabilityHandler
from victor.observability.request_correlation import request_correlation_id


def test_tool_observability_handler_is_service_owned() -> None:
    """ToolObservabilityHandler should be imported from services module."""
    # Verify the handler is available from the services module
    from victor.agent.services.tool_observability import ToolObservabilityHandler

    assert ToolObservabilityHandler is not None


@pytest.mark.asyncio
async def test_on_tool_complete_emits_follow_up_suggestions() -> None:
    """tool.complete events should include normalized follow-up suggestions."""
    handler = ToolObservabilityHandler(MagicMock())
    metrics_collector = MagicMock()
    bus = MagicMock()
    bus.emit = AsyncMock()

    result = SimpleNamespace(
        tool_id="tool-7",
        tool_name="code_search",
        success=True,
        result={
            "success": True,
            "metadata": {
                "follow_up_suggestions": [
                    {
                        "command": 'graph(mode="trace", node="main", depth=3)',
                        "description": "Trace execution starting from main.",
                    },
                    {"description": "missing command"},
                ]
            },
        },
        error=None,
        arguments={"query": "main entry point"},
        execution_time_ms=125.0,
    )

    with patch("victor.core.events.get_observability_bus", return_value=bus):
        handler.on_tool_complete(result, metrics_collector)
        await asyncio.sleep(0)

    metrics_collector.on_tool_complete.assert_called_once_with(result)
    bus.emit.assert_awaited_once()
    emitted_data = bus.emit.await_args.kwargs["data"]
    assert emitted_data["tool_id"] == "tool-7"
    assert emitted_data["tool_name"] == "code_search"
    assert emitted_data["follow_up_suggestions"] == [
        {
            "command": 'graph(mode="trace", node="main", depth=3)',
            "description": "Trace execution starting from main.",
        }
    ]


@pytest.mark.asyncio
async def test_on_tool_complete_emits_request_correlation_id() -> None:
    """tool.complete events should inherit the active request correlation ID."""
    handler = ToolObservabilityHandler(MagicMock())
    metrics_collector = MagicMock()
    bus = MagicMock()
    bus.emit = AsyncMock()

    result = SimpleNamespace(
        tool_name="graph",
        success=True,
        result={"success": True},
        error=None,
        arguments={"mode": "trace", "node": "main"},
        execution_time_ms=25.0,
    )

    with patch("victor.core.events.get_observability_bus", return_value=bus):
        with request_correlation_id("chat_test_123"):
            handler.on_tool_complete(result, metrics_collector)
            await asyncio.sleep(0)

    assert bus.emit.await_args.kwargs["data"]["tool_id"] == "tool-0"
    assert bus.emit.await_args.kwargs["correlation_id"] == "chat_test_123"


@pytest.mark.asyncio
async def test_on_tool_complete_tracks_legacy_read_alias() -> None:
    """Legacy read aliases should still populate read_files_session via canonicalization."""
    handler = ToolObservabilityHandler(MagicMock())
    metrics_collector = MagicMock()
    bus = MagicMock()
    bus.emit = AsyncMock()
    read_files_session: set[str] = set()

    result = SimpleNamespace(
        tool_id="tool-8",
        tool_name="read_file",
        success=True,
        result="file contents",
        error=None,
        arguments={"path": "/tmp/example.py"},
        execution_time_ms=15.0,
    )

    with patch("victor.core.events.get_observability_bus", return_value=bus):
        handler.on_tool_complete(
            result,
            metrics_collector,
            read_files_session=read_files_session,
        )
        await asyncio.sleep(0)

    assert "/tmp/example.py" in read_files_session

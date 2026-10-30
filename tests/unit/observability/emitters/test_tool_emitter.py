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

"""Characterisation tests for ToolEventEmitter typed helpers."""

from __future__ import annotations

import pytest

from victor.observability.emitters import IToolEventEmitter, ToolEventEmitter

from .conftest import FakeBus, SyncRecorder


def test_implements_protocol() -> None:
    assert isinstance(ToolEventEmitter(), IToolEventEmitter)


async def test_async_helpers(fake_bus: FakeBus) -> None:
    e = ToolEventEmitter(bus=fake_bus)
    assert await e.tool_start_async("read", {"p": 1}, agent_id="a") is True
    await e.tool_end_async("read", 12.5, result="x" * 1001, agent_id="a")
    await e.tool_end_async("read", 1.0)
    await e.tool_failure_async("read", 3.0, KeyError("k"), agent_id="a")
    assert fake_bus.emitted == [
        (
            "tool.start",
            {"tool_name": "read", "arguments": {"p": 1}, "agent_id": "a", "category": "tool"},
        ),
        (
            "tool.result",
            {
                "tool_name": "read",
                "success": True,
                "duration_ms": 12.5,
                "result": "x" * 1000 + "... (truncated)",
                "agent_id": "a",
                "category": "tool",
            },
        ),
        (
            "tool.result",
            {
                "tool_name": "read",
                "success": True,
                "duration_ms": 1.0,
                "result": None,
                "category": "tool",
            },
        ),
        (
            "tool.error",
            {
                "tool_name": "read",
                "success": False,
                "duration_ms": 3.0,
                "error": "'k'",
                "error_type": "KeyError",
                "agent_id": "a",
                "category": "tool",
            },
        ),
    ]


def test_sync_helpers(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    e = ToolEventEmitter(bus=fake_bus)
    e.tool_start("read", {"p": 1})
    e.tool_end("read", 2.0, result=123)
    e.tool_failure("read", 3.0, ValueError("bad"))
    assert [(c["topic"], c["data"], c["source"]) for c in sync_recorder.calls] == [
        ("tool.start", {"tool_name": "read", "arguments": {"p": 1}}, "ToolEventEmitter"),
        (
            "tool.result",
            {"tool_name": "read", "success": True, "duration_ms": 2.0, "result": "123"},
            "ToolEventEmitter",
        ),
        (
            "tool.error",
            {
                "tool_name": "read",
                "success": False,
                "duration_ms": 3.0,
                "error": "bad",
                "error_type": "ValueError",
            },
            "ToolEventEmitter",
        ),
    ]


def test_track_tool_success(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    e = ToolEventEmitter(bus=fake_bus)
    with e.track_tool("grep", {"q": "x"}, agent_id="a"):
        pass
    topics = [c["topic"] for c in sync_recorder.calls]
    assert topics == ["tool.start", "tool.result"]
    end = sync_recorder.calls[1]["data"]
    assert end["result"] is None and end["agent_id"] == "a" and end["duration_ms"] >= 0


def test_track_tool_failure_reraises(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    e = ToolEventEmitter(bus=fake_bus)
    with pytest.raises(RuntimeError):
        with e.track_tool("grep", {}):
            raise RuntimeError("x")
    assert [c["topic"] for c in sync_recorder.calls] == ["tool.start", "tool.error"]
    assert sync_recorder.calls[1]["data"]["error_type"] == "RuntimeError"

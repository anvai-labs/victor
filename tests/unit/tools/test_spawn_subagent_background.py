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

"""FOU-A1 phase-2 unit suite: background subagent mode + companion tools.

Covers the increment's gates:

- the background handle returned IMMEDIATELY by ``spawn_subagent(background=True)``
  (the substrate runner detaches the child; the tool never blocks on it);
- the bounded, thread-safe ``SubAgentRunRegistry`` (run_id -> status /
  child_session_id / timestamps / SubAgentResult fields, max 100 runs,
  oldest evicted);
- the ``subagent_status`` / ``subagent_result`` companion tools: running vs
  terminal payloads, hallucinated ids -> structured not-found (the registry
  is the ONLY source of run ids), unwired sessions refuse;
- terminal notifications on the event stream, mirroring the
  ``BackgroundAgentManager`` conventions (event callback + observability bus);
- the workspace-write permission tier for the companions, both directions;
- the FOREGROUND path unchanged (no registry entries, same mapped result).

The substrate runner is stubbed here; the REAL production wiring (factory
runner detaching the child through ``SubAgentOrchestrator.spawn``, depth-1 +
intersection applied to background children, end-to-end executor context) is
pinned in ``tests/unit/agent/test_spawn_subagent_background_wiring.py``.
"""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from victor.agent.subagents.run_registry import SubAgentRunRegistry, SubAgentRunStatus
from victor.security.permissions import (
    DEFAULT_TOOL_PERMISSIONS,
    PermissionMode,
    PermissionPolicy,
)
from victor.tools.spawn_subagent import (
    SPAWN_SUBAGENT_TOOL_NAME,
    SUBAGENT_RESULT_TOOL_NAME,
    SUBAGENT_STATUS_TOOL_NAME,
    spawn_subagent,
    subagent_result,
    subagent_status,
)

# A SubAgentResult.to_dict() shape, as the real substrate returns it.
_SUBAGENT_RESULT_DICT = {
    "success": True,
    "summary": "Found 3 auth middleware call sites in victor/agent.",
    "details": {"role": "researcher", "child_session_id": "sess-abc-spawn-1"},
    "tool_calls_used": 6,
    "context_size": 12000,
    "duration_seconds": 4.2,
    "error": None,
}


def _started_result(child_session_id="sess-abc-spawn-1") -> dict:
    return {
        **_SUBAGENT_RESULT_DICT,
        "details": {"role": "researcher", "child_session_id": child_session_id},
    }


async def _wait_terminal(registry: SubAgentRunRegistry, subagent_id: str, timeout: float = 5.0):
    """Poll the registry until the run reaches a terminal state."""

    async def _poll():
        while True:
            record = registry.get(subagent_id)
            if record is not None and record.is_terminal:
                return record
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_poll(), timeout=timeout)


# =============================================================================
# Registry: lifecycle and bounded retention
# =============================================================================


def test_start_run_records_running_entry():
    registry = SubAgentRunRegistry()
    record = registry.start_run(
        prompt="find auth call sites",
        child_session_id="sess-p-spawn-1",
        parent_session_id="sess-p",
    )

    assert record.subagent_id
    assert record.child_session_id == "sess-p-spawn-1"
    assert record.prompt == "find auth call sites"
    assert record.parent_session_id == "sess-p"
    assert record.status is SubAgentRunStatus.RUNNING
    assert not record.is_terminal
    assert len(registry) == 1
    assert registry.get(record.subagent_id) is record


def test_complete_run_transitions_running_to_completed_with_result():
    registry = SubAgentRunRegistry()
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    finished = registry.complete_run(record.subagent_id, _started_result())

    assert finished is not None
    assert finished.status is SubAgentRunStatus.COMPLETED
    assert finished.is_terminal
    assert finished.finished_at is not None
    assert finished.finished_at >= finished.started_at
    assert finished.result["summary"] == _SUBAGENT_RESULT_DICT["summary"]


def test_failed_subagent_result_maps_to_failed_status():
    """A child that finished unsuccessfully is FAILED, not COMPLETED."""
    registry = SubAgentRunRegistry()
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    finished = registry.complete_run(
        record.subagent_id,
        {**_SUBAGENT_RESULT_DICT, "success": False, "error": "Timeout after 300s"},
    )

    assert finished.status is SubAgentRunStatus.FAILED
    assert finished.error == "Timeout after 300s"


def test_fail_run_marks_failed_with_error():
    registry = SubAgentRunRegistry()
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    finished = registry.fail_run(record.subagent_id, "RuntimeError: substrate boom")

    assert finished.status is SubAgentRunStatus.FAILED
    assert "substrate boom" in finished.error
    assert finished.finished_at is not None


def test_cancel_run_marks_cancelled():
    registry = SubAgentRunRegistry()
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    finished = registry.cancel_run(record.subagent_id)

    assert finished.status is SubAgentRunStatus.CANCELLED
    assert finished.is_terminal


def test_terminal_transition_is_one_way():
    """A second terminal report must not overwrite the first result."""
    registry = SubAgentRunRegistry()
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")
    registry.complete_run(record.subagent_id, _started_result())

    second = registry.complete_run(
        record.subagent_id,
        {**_SUBAGENT_RESULT_DICT, "success": False, "error": "late failure"},
    )

    assert second is None
    assert registry.get(record.subagent_id).status is SubAgentRunStatus.COMPLETED


def test_registry_evicts_oldest_run_beyond_max():
    registry = SubAgentRunRegistry(max_runs=3)
    ids = [
        registry.start_run(prompt=f"t{i}", child_session_id=f"sess-{i}").subagent_id
        for i in range(4)
    ]

    assert len(registry) == 3
    assert registry.get(ids[0]) is None  # oldest evicted
    assert registry.get(ids[3]) is not None  # newest retained


def test_registry_is_thread_safe_under_concurrent_start_run():
    registry = SubAgentRunRegistry(max_runs=100)
    ids: list = []
    lock = threading.Lock()

    def _start(i):
        record = registry.start_run(prompt=f"t{i}", child_session_id=f"sess-{i}")
        with lock:
            ids.append(record.subagent_id)

    threads = [threading.Thread(target=_start, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ids) == 20
    assert len(set(ids)) == 20  # unique ids under concurrency
    assert len(registry) == 20
    assert all(registry.get(sid) is not None for sid in ids)


# =============================================================================
# Registry: terminal notification (BackgroundAgentManager conventions)
# =============================================================================


def test_terminal_notification_via_callback_completed():
    events: list = []
    registry = SubAgentRunRegistry(event_callback=lambda e, d: events.append((e, d)))
    record = registry.start_run(
        prompt="t",
        child_session_id="sess-p-spawn-1",
        parent_session_id="sess-p",
    )

    registry.complete_run(record.subagent_id, _started_result())

    assert len(events) == 1
    event, data = events[0]
    assert event == "subagent_completed"
    assert data["subagent_id"] == record.subagent_id
    assert data["child_session_id"] == "sess-p-spawn-1"
    assert data["parent_session_id"] == "sess-p"
    assert data["status"] == "completed"
    assert "auth middleware" in data["summary"]


def test_terminal_notification_via_callback_failed():
    events: list = []
    registry = SubAgentRunRegistry(event_callback=lambda e, d: events.append((e, d)))
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    registry.fail_run(record.subagent_id, "boom")

    event, data = events[0]
    assert event == "subagent_failed"
    assert data["status"] == "failed"
    assert "boom" in data["error"]


def test_event_callback_exception_does_not_break_bookkeeping():
    def _boom(event, data):
        raise RuntimeError("callback blew up")

    registry = SubAgentRunRegistry(event_callback=_boom)
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    finished = registry.complete_run(record.subagent_id, _started_result())

    assert finished is not None
    assert finished.status is SubAgentRunStatus.COMPLETED


async def test_terminal_event_reaches_observability_bus():
    """The terminal event is emitted on the existing observability event
    stream (topic ``subagent.terminal``), not only the direct callback."""
    from victor.core.events.backends import ObservabilityBus

    bus = ObservabilityBus()
    seen: list = []

    async def _handler(event):
        seen.append(event)

    await bus.subscribe("subagent.terminal", _handler)

    registry = SubAgentRunRegistry(event_bus=bus)
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")
    registry.complete_run(record.subagent_id, _started_result())

    await asyncio.sleep(0.2)  # fire-and-forget bus task

    assert seen, "terminal event never reached the observability bus"
    data = seen[0].data
    assert data["subagent_id"] == record.subagent_id
    assert data["status"] == "completed"


# =============================================================================
# Companion tools: subagent_status / subagent_result
# =============================================================================


@pytest.fixture
def registry() -> SubAgentRunRegistry:
    return SubAgentRunRegistry()


@pytest.fixture
def companion_ctx(registry) -> dict:
    return {"subagent_run_registry": registry}


async def test_status_known_running_run(companion_ctx, registry):
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    result = await subagent_status(subagent_id=record.subagent_id, _exec_ctx=companion_ctx)

    assert result["success"] is True
    assert result["found"] is True
    assert result["status"] == "running"
    assert result["child_session_id"] == "sess-p-spawn-1"
    assert result["finished_at"] is None


async def test_status_after_completion_reports_terminal_metadata(companion_ctx, registry):
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")
    registry.complete_run(record.subagent_id, _started_result())

    result = await subagent_status(subagent_id=record.subagent_id, _exec_ctx=companion_ctx)

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["finished_at"] is not None
    assert result["summary"] == _SUBAGENT_RESULT_DICT["summary"]


async def test_result_when_still_running_is_structured(companion_ctx, registry):
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    result = await subagent_result(subagent_id=record.subagent_id, _exec_ctx=companion_ctx)

    assert result["found"] is True
    assert result["result_ready"] is False
    assert result["status"] == "running"
    assert "still running" in result["message"]
    assert "result" not in result


async def test_result_terminal_returns_subagent_fields(companion_ctx, registry):
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")
    registry.complete_run(record.subagent_id, _started_result())

    result = await subagent_result(subagent_id=record.subagent_id, _exec_ctx=companion_ctx)

    assert result["success"] is True
    assert result["found"] is True
    assert result["result_ready"] is True
    assert result["status"] == "completed"
    payload = result["result"]
    assert payload["success"] is True
    assert payload["content"] == _SUBAGENT_RESULT_DICT["summary"]
    assert payload["tool_calls_used"] == 6
    assert payload["duration_seconds"] == 4.2
    assert payload["child_session_id"] == "sess-p-spawn-1"


async def test_result_failed_child_reports_error(companion_ctx, registry):
    record = registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")
    registry.fail_run(record.subagent_id, "Timeout after 300s")

    result = await subagent_result(subagent_id=record.subagent_id, _exec_ctx=companion_ctx)

    assert result["result_ready"] is True
    assert result["status"] == "failed"
    assert result["result"]["success"] is False
    assert "Timeout" in result["result"]["error"]


# =============================================================================
# Companion tools: hallucinated ids -> structured not-found (registry is the
# ONLY source of run ids)
# =============================================================================


async def test_status_hallucinated_id_not_found(companion_ctx, registry):
    registry.start_run(prompt="t", child_session_id="sess-p-spawn-1")

    result = await subagent_status(subagent_id="subagent-does-not-exist", _exec_ctx=companion_ctx)

    assert result["success"] is False
    assert result["found"] is False
    assert "unknown subagent id" in result["error"].lower()
    assert "status" not in result  # no fabricated metadata


async def test_result_hallucinated_id_not_found(companion_ctx, registry):
    result = await subagent_result(subagent_id="abc123", _exec_ctx=companion_ctx)

    assert result["success"] is False
    assert result["found"] is False
    assert "unknown subagent id" in result["error"].lower()


# =============================================================================
# Dormancy guard: companions refuse when the registry is not wired
# =============================================================================


async def test_companions_without_registry_refuse():
    status = await subagent_status(subagent_id="x", _exec_ctx={})
    result = await subagent_result(subagent_id="x", _exec_ctx={})

    assert status["success"] is False
    assert "registry" in status["error"].lower()
    assert result["success"] is False
    assert "registry" in result["error"].lower()


async def test_companions_empty_id_refused(companion_ctx):
    status = await subagent_status(subagent_id="   ", _exec_ctx=companion_ctx)
    result = await subagent_result(subagent_id="", _exec_ctx=companion_ctx)

    assert status["success"] is False
    assert "subagent_id" in status["error"]
    assert result["success"] is False
    assert "subagent_id" in result["error"]


# =============================================================================
# Permission gating (both directions), mirroring spawn_subagent's tier
# =============================================================================


def test_companions_declared_workspace_write_tier():
    """Both companions are registered in DEFAULT_TOOL_PERMISSIONS at the
    spawn tool's tier and their metadata agrees."""
    assert DEFAULT_TOOL_PERMISSIONS[SUBAGENT_STATUS_TOOL_NAME] == PermissionMode.WORKSPACE_WRITE
    assert DEFAULT_TOOL_PERMISSIONS[SUBAGENT_RESULT_TOOL_NAME] == PermissionMode.WORKSPACE_WRITE
    assert DEFAULT_TOOL_PERMISSIONS[SPAWN_SUBAGENT_TOOL_NAME] == PermissionMode.WORKSPACE_WRITE
    assert subagent_status.Tool.name == SUBAGENT_STATUS_TOOL_NAME
    assert subagent_result.Tool.name == SUBAGENT_RESULT_TOOL_NAME


def test_read_only_parent_cannot_poll_companions():
    policy = PermissionPolicy(active_mode=PermissionMode.READ_ONLY)
    assert not policy.authorize(SUBAGENT_STATUS_TOOL_NAME).allowed
    assert not policy.authorize(SUBAGENT_RESULT_TOOL_NAME).allowed


def test_workspace_write_parent_can_poll_companions():
    policy = PermissionPolicy(active_mode=PermissionMode.WORKSPACE_WRITE)
    assert policy.authorize(SUBAGENT_STATUS_TOOL_NAME).allowed
    assert policy.authorize(SUBAGENT_RESULT_TOOL_NAME).allowed


def test_full_access_parent_can_poll_companions():
    policy = PermissionPolicy(active_mode=PermissionMode.DANGER_FULL_ACCESS)
    assert policy.authorize(SUBAGENT_STATUS_TOOL_NAME).allowed
    assert policy.authorize(SUBAGENT_RESULT_TOOL_NAME).allowed


# =============================================================================
# spawn_subagent(background=True): the tool-side contract
# =============================================================================


def _background_handle(subagent_id="subagent-abcd1234", child_session_id="sess-p-spawn-1"):
    return {
        "success": True,
        "background": True,
        "subagent_id": subagent_id,
        "child_session_id": child_session_id,
        "status": "running",
    }


@pytest.fixture
def bg_runner() -> AsyncMock:
    return AsyncMock(return_value=_background_handle())


@pytest.fixture
def bg_ctx(bg_runner, registry) -> dict:
    return {
        "spawn_subagent_runner": bg_runner,
        "spawn_subagent_budget": None,
        "subagent_run_registry": registry,
    }


async def test_background_mode_returns_handle_without_blocking(bg_ctx, bg_runner):
    result = await spawn_subagent(prompt="bg task", background=True, _exec_ctx=bg_ctx)

    assert result == _background_handle()
    bg_runner.assert_awaited_once()


async def test_background_mode_asks_runner_to_detach(bg_ctx, bg_runner):
    await spawn_subagent(prompt="bg task", background=True, _exec_ctx=bg_ctx)

    assert bg_runner.await_args.kwargs["background"] is True


async def test_foreground_mode_does_not_touch_registry(bg_ctx, registry, bg_runner):
    """The foreground path is UNCHANGED: substrate result mapped as before,
    no run registered in the background registry."""
    bg_runner.return_value = _started_result()

    result = await spawn_subagent(prompt="fg task", _exec_ctx=bg_ctx)

    assert result["success"] is True
    assert result["content"] == _SUBAGENT_RESULT_DICT["summary"]
    assert bg_runner.await_args.kwargs["background"] is False
    assert len(registry) == 0


async def test_background_without_registry_refuses(bg_runner):
    ctx = {"spawn_subagent_runner": bg_runner, "spawn_subagent_budget": None}

    result = await spawn_subagent(prompt="bg task", background=True, _exec_ctx=ctx)

    assert result["success"] is False
    assert "registry" in result["error"].lower()
    bg_runner.assert_not_awaited()


async def test_background_with_disabled_orchestration_refuses(bg_ctx, bg_runner):
    bg_runner.return_value = None

    result = await spawn_subagent(prompt="bg task", background=True, _exec_ctx=bg_ctx)

    assert result["success"] is False
    assert "orchestration" in result["error"].lower()


async def test_background_runner_crash_is_structured_failure(bg_ctx, bg_runner):
    bg_runner.side_effect = RuntimeError("substrate boom")

    result = await spawn_subagent(prompt="bg task", background=True, _exec_ctx=bg_ctx)

    assert result["success"] is False
    assert "substrate boom" in result["error"]


async def test_background_runner_non_handle_result_refused(bg_ctx, bg_runner):
    """Fail closed: a runner that neither detaches nor refuses (e.g. an old
    foreground-shaped result) must never be presented as a background handle."""
    bg_runner.return_value = {"success": True, "content": "done"}  # not detached

    result = await spawn_subagent(prompt="bg task", background=True, _exec_ctx=bg_ctx)

    assert result["success"] is False
    assert "detach" in result["error"].lower()


async def test_background_refuses_depth_beyond_one(bg_ctx, bg_runner):
    result = await spawn_subagent(prompt="t", background=True, max_depth=2, _exec_ctx=bg_ctx)

    assert result["success"] is False
    assert "depth" in result["error"].lower()
    bg_runner.assert_not_awaited()

"""Focused tests for ToolRetryExecutor cache invalidation behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.tool_retry import ToolRetryExecutor
from victor.agent.tool_pipeline import ToolCallResult
from victor.tools.enums import AccessMode


class _Config:
    retry_enabled = True
    max_retry_attempts = 1
    retry_base_delay = 0.0
    retry_max_delay = 0.0


def _make_executor(cache: MagicMock) -> ToolRetryExecutor:
    pipeline = MagicMock()
    pipeline._execute_single_call = AsyncMock(
        return_value=SimpleNamespace(success=True, error=None)
    )
    cache.get.return_value = None
    return ToolRetryExecutor(config=_Config(), pipeline=pipeline, cache=cache)


@pytest.mark.asyncio
async def test_execute_tool_with_retry_invalidates_paths_for_canonical_write():
    cache = MagicMock()
    executor = _make_executor(cache)

    result, success, error = await executor.execute_tool_with_retry(
        "write",
        {"path": "/tmp/example.py"},
        {},
    )

    assert success is True
    assert error is None
    assert result is not None
    cache.invalidate_paths.assert_called_once_with(["/tmp/example.py"])
    cache.clear_namespaces.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_with_retry_invalidates_paths_for_create_file_alias():
    cache = MagicMock()
    executor = _make_executor(cache)

    await executor.execute_tool_with_retry(
        "create_file",
        {"path": "/tmp/example.py"},
        {},
    )

    cache.invalidate_paths.assert_called_once_with(["/tmp/example.py"])


@pytest.mark.asyncio
async def test_execute_tool_with_retry_clears_canonical_namespaces_for_shell():
    cache = MagicMock()
    executor = _make_executor(cache)

    await executor.execute_tool_with_retry(
        "shell",
        {"cmd": "pytest"},
        {},
    )

    cache.clear_namespaces.assert_called_once_with(["read", "ls"])


@pytest.mark.parametrize("custom", [False, True])
async def test_uncertain_effect_exception_never_replays(custom):
    executor = _make_executor(MagicMock())
    executor._pipeline.tools.get.return_value = SimpleNamespace(access_mode=AccessMode.WRITE)
    committed = []

    async def submit(*args):
        committed.append("receipt")
        raise TimeoutError("response lost")

    executor._pipeline._execute_single_call = submit
    result, success, error = await executor.execute_tool_with_retry(
        "submit",
        {},
        {},
        tool_executor=submit if custom else None,
        retry_config={"max_attempts": 3},
    )
    assert committed == ["receipt"]
    assert result is None and success is False
    assert "Reconcile" in error


@pytest.mark.parametrize("veto", [False, True])
async def test_read_retry_honors_structured_veto(veto):
    executor = _make_executor(MagicMock())
    executor._pipeline.tools.get.return_value = SimpleNamespace(access_mode=AccessMode.READONLY)
    failure = ToolCallResult(
        "read", {}, False, error="transient", retryable=False if veto else None
    )
    executor._pipeline._execute_single_call.side_effect = [
        failure,
        ToolCallResult("read", {}, True, result="ok"),
    ]
    result, success, _ = await executor.execute_tool_with_retry(
        "read", {}, {}, retry_config={"max_attempts": 3}
    )
    assert executor._pipeline._execute_single_call.await_count == (1 if veto else 2)
    assert success is (not veto)


@pytest.mark.parametrize("failing_component", ["cache", "callback"])
async def test_post_success_failure_never_repeats_action(failing_component):
    cache = MagicMock()
    executor = _make_executor(cache)
    executor._pipeline.tools.get.return_value = SimpleNamespace(access_mode=AccessMode.WRITE)
    callback = MagicMock()
    if failing_component == "cache":
        cache.set.side_effect = OSError("storage failed")
    else:
        callback.side_effect = RuntimeError("observer failed")
    result, success, error = await executor.execute_tool_with_retry(
        "submit", {}, {}, on_success=callback, retry_config={"max_attempts": 3}
    )
    executor._pipeline._execute_single_call.assert_awaited_once()
    assert result.success is True  # preserve evidence of the completed execution
    assert success is False and "do not repeat" in error

"""Spawn owns the member cache lifetime, but borrows the parent provider."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.lifecycle_manager import LifecycleManager
from victor.agent.runtime.bootstrapper import AgentRuntimeBootstrapper
from victor.agent.subagents.base import SubAgentResult
from victor.agent.subagents.orchestrator import SubAgentOrchestrator
from victor.core.shared_types import SubAgentRole
from victor.providers.base import StreamChunk
from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tool_cache import ToolCache


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "timeout", "stream_close"])
async def test_spawn_releases_only_member_cache(tmp_path, outcome):
    import sqlite3

    parent = MagicMock()
    parent.settings.subagent_default_tool_budget = 10
    parent.settings.subagent_default_context_limit = 1000
    parent_cache = ToolCache(60, ["read"], CacheConfig(disk_path=tmp_path / "parent"), MagicMock())
    cache = ToolCache(60, ["read"], CacheConfig(disk_path=tmp_path / "member"), MagicMock())
    db = cache.cache._disk_cache._db
    child = MagicMock()
    child.tool_cache = cache
    child.provider = parent.provider
    child._background_tasks = set()
    child._lifecycle_manager = LifecycleManager(MagicMock())
    AgentRuntimeBootstrapper.wire_lifecycle(child)
    member = MagicMock(orchestrator=child)
    result = SubAgentResult(True, "ready", {}, 0, 0, 0)
    if outcome in {"cancelled", "timeout"}:
        error = asyncio.CancelledError() if outcome == "cancelled" else asyncio.TimeoutError()
        member.execute = AsyncMock(side_effect=error)
    else:
        result.success = outcome != "failure"
        member.execute = AsyncMock(return_value=result)

    async def stream():
        yield StreamChunk(content="partial")
        await asyncio.Event().wait()

    member.stream_execute.side_effect = stream
    coordinator = SubAgentOrchestrator(parent)
    try:
        with patch("victor.agent.subagents.orchestrator.SubAgent", return_value=member):
            if outcome == "stream_close":
                iterator = coordinator.stream_spawn(SubAgentRole.EXECUTOR, "task")
                await anext(iterator)
                await iterator.aclose()
            elif outcome == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await coordinator.spawn(SubAgentRole.EXECUTOR, "task")
            else:
                returned = await coordinator.spawn(SubAgentRole.EXECUTOR, "task")
                assert returned.success is (outcome == "success")
        assert coordinator.get_active_count() == 0
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            db.execute("SELECT 1")
        assert parent_cache.cache._disk_cache._db.execute("SELECT 1").fetchone() == (1,)
        parent.provider.close.assert_not_called()
        child.close.assert_not_called()
    finally:
        # Explicitly clean the regression fixture, including on the pre-fix red run.
        cache.cache.close()
        parent_cache.cache.close()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "raised"])
async def test_member_cleanup_failure_retains_primary_outcome(outcome):
    parent = MagicMock()
    member = MagicMock()
    member.orchestrator._lifecycle_manager.close_tool_cache.side_effect = OSError(
        "disk close failed"
    )
    result = SubAgentResult(
        outcome == "success",
        "original",
        {},
        0,
        0,
        0,
        error="original failure" if outcome == "failure" else None,
    )
    member.execute = AsyncMock(return_value=result)
    if outcome in {"cancelled", "raised"}:
        member.execute.side_effect = (
            asyncio.CancelledError() if outcome == "cancelled" else RuntimeError("primary failure")
        )
    coordinator = SubAgentOrchestrator(parent)
    with patch("victor.agent.subagents.orchestrator.SubAgent", return_value=member):
        if outcome in {"cancelled", "raised"}:
            error_type = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
            with pytest.raises(error_type) as error:
                await coordinator.spawn(SubAgentRole.EXECUTOR, "task")
            assert "cache cleanup failed" in " ".join(error.value.__notes__)
        else:
            returned = await coordinator.spawn(SubAgentRole.EXECUTOR, "task")
            assert returned is result
            assert not returned.success
            assert returned.details["cache_cleanup_error"] == "OSError"
            if outcome == "failure":
                assert returned.error == "original failure"
    assert coordinator.get_active_count() == 0


@pytest.mark.parametrize("caller_handling_error", [False, True])
async def test_stream_cleanup_failure_cannot_attach_to_unrelated_caller_error(
    caller_handling_error,
):
    member = MagicMock()
    member.orchestrator._lifecycle_manager.close_tool_cache.side_effect = OSError(
        "disk close failed"
    )

    async def stream():
        yield StreamChunk(content="ready", is_final=True)

    member.stream_execute.side_effect = stream
    coordinator = SubAgentOrchestrator(MagicMock())

    async def consume():
        with patch("victor.agent.subagents.orchestrator.SubAgent", return_value=member):
            with pytest.raises(OSError, match="disk close failed"):
                async for _ in coordinator.stream_spawn(SubAgentRole.EXECUTOR, "task"):
                    pass

    if caller_handling_error:
        try:
            raise ValueError("unrelated caller error")
        except ValueError as previous:
            await consume()
            assert not hasattr(previous, "__notes__")
    else:
        await consume()
    assert coordinator.get_active_count() == 0


async def test_stream_close_error_survives_cache_close_error():
    member = MagicMock()
    member.orchestrator._lifecycle_manager.close_tool_cache.side_effect = OSError(
        "disk close failed"
    )

    async def stream():
        try:
            yield StreamChunk(content="partial")
        finally:
            raise RuntimeError("stream close failed")

    member.stream_execute.side_effect = stream
    coordinator = SubAgentOrchestrator(MagicMock())
    with patch("victor.agent.subagents.orchestrator.SubAgent", return_value=member):
        iterator = coordinator.stream_spawn(SubAgentRole.EXECUTOR, "task")
        await anext(iterator)
        with pytest.raises(RuntimeError, match="stream close failed") as error:
            await iterator.aclose()
        assert "cache cleanup failed" in " ".join(error.value.__notes__)
    assert coordinator.get_active_count() == 0

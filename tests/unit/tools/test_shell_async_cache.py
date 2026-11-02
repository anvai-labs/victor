"""Read caching must not replace async subprocess ownership or cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.tools.bash import shell
from victor.tools.shell_command_cache import ShellCommandCache


@pytest.fixture
def cache():
    instance = ShellCommandCache()
    with patch("victor.tools.shell_command_cache.get_shell_cache", return_value=instance):
        yield instance


async def test_read_miss_runs_managed_once_and_only_then_becomes_a_cache_hit(cache):
    command = "echo managed-read"
    with (
        patch(
            "victor.tools.shell_command_cache.execute_with_cache", return_value=(0, "legacy", "")
        ) as sync,
        patch(
            "victor.tools.subprocess_executor.run_managed_process",
            new=AsyncMock(return_value=(b"managed-read\\n", b"", 0, False, False)),
        ) as managed,
    ):
        first = await shell(cmd=command, readonly=True)
        assert first["success"] is True
        assert first.get("cached", False) is False
        managed.assert_awaited_once()
        sync.assert_not_called()
        second = await shell(cmd=command, readonly=True)
        assert second["success"] is True
        assert second["cached"] is True
        assert second["stdout"] == first["stdout"]
        managed.assert_awaited_once()


async def test_read_cache_hit_never_starts_a_process(cache):
    command = "echo cached-read"
    cache.set(command, (0, "cached-read", ""), ".")
    with (
        patch(
            "victor.tools.shell_command_cache.execute_with_cache", return_value=(0, "legacy", "")
        ) as sync,
        patch("victor.tools.subprocess_executor.run_managed_process", new=AsyncMock()) as managed,
    ):
        result = await shell(cmd=command, readonly=True)
        assert result["stdout"] == "cached-read"
        assert result["cached"] is True
        managed.assert_not_awaited()
        sync.assert_not_called()


async def test_cancelled_read_miss_joins_the_managed_runner(cache):
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def execute(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with (
        patch(
            "victor.tools.shell_command_cache.execute_with_cache", return_value=(0, "legacy", "")
        ),
        patch(
            "victor.tools.subprocess_executor.run_managed_process",
            new=AsyncMock(side_effect=execute),
        ),
    ):
        task = asyncio.create_task(shell(cmd="echo cancellable-read", readonly=True))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cleaned.is_set()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_cache_publication_failure_cannot_replay_execution(cache):
    cache.set = Mock(side_effect=RuntimeError("cache unavailable"))
    with (
        patch(
            "victor.tools.shell_command_cache.execute_with_cache", return_value=(0, "legacy", "")
        ) as sync,
        patch(
            "victor.tools.subprocess_executor.run_managed_process",
            new=AsyncMock(return_value=(b"done", b"", 0, False, False)),
        ) as managed,
    ):
        result = await shell(cmd="echo one-execution", readonly=True)
        assert result["success"] is True
        managed.assert_awaited_once()
        sync.assert_not_called()


async def test_sandboxed_read_cannot_reuse_another_execution_contexts_host_cache(cache):
    from types import SimpleNamespace

    command = "echo context-read"
    cache.set(command, (0, "host-private-result", ""), ".")
    sandbox = SimpleNamespace(
        type_name="isolated", wrap_argv=Mock(return_value=["sandboxed-command"])
    )
    with (
        patch("victor.tools.subprocess_executor._resolve_default_sandbox", return_value=sandbox),
        patch(
            "victor.tools.subprocess_executor.run_managed_process",
            new=AsyncMock(return_value=(b"sandbox-result", b"", 0, False, False)),
        ) as managed,
    ):
        result = await shell(cmd=command, readonly=True)
        assert result["stdout"] == "sandbox-result"
        assert result["cached"] is False
        assert cache.get(command, ".") == (0, "host-private-result", "")
        sandbox.wrap_argv.assert_called_once()
        managed.assert_awaited_once()
        assert managed.call_args.kwargs["argv"] == ["sandboxed-command"]


@pytest.mark.parametrize("command", ["cat project.txt", "git status --short", "gh pr view 1"])
async def test_file_repository_and_remote_status_reads_cannot_reuse_stale_results(cache, command):
    cache.set(command, (0, "stale-result", ""), ".")
    with patch(
        "victor.tools.subprocess_executor.run_managed_process",
        new=AsyncMock(
            side_effect=[(b"fresh-one", b"", 0, False, False), (b"fresh-two", b"", 0, False, False)]
        ),
    ) as managed:
        first = await shell(cmd=command, readonly=True)
        second = await shell(cmd=command, readonly=True)
        assert first["stdout"] == "fresh-one"
        assert second["stdout"] == "fresh-two"
        assert first["cached"] is second["cached"] is False
        assert managed.await_count == 2

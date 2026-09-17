"""Workflow process execution respects actual MCP text I/O and cleanup."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.isolation import IsolationConfig
from victor.workflows.sandbox_executor import SandboxedExecutor


async def test_process_workflow_round_trips_text_through_real_backend():
    result = await SandboxedExecutor(docker_available=False)._execute_process(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        IsolationConfig(sandbox_type="process", network_allowed=True),
        working_dir=None,
        env=None,
        input_data="payload",
    )
    assert result.success, result.error
    assert result.output == "PAYLOAD\n"
    assert result.error == ""


async def test_process_workflow_cannot_pretend_network_denial_is_enforced():
    with patch("victor.integrations.mcp.sandbox.subprocess.Popen") as launch:
        result = await SandboxedExecutor(docker_available=False)._execute_process(
            ["ignored"],
            IsolationConfig(sandbox_type="process", network_allowed=False),
            working_dir=None,
            env=None,
            input_data=None,
        )
    assert not result.success
    assert "cannot enforce" in result.error
    launch.assert_not_called()


@pytest.mark.parametrize(
    "failure", [asyncio.TimeoutError(), RuntimeError("broken pipe"), asyncio.CancelledError()]
)
async def test_workflow_process_is_reaped_after_communication_failure(failure):
    process = MagicMock()
    backend = MagicMock()
    backend.start = AsyncMock(return_value=process)
    backend.communicate = AsyncMock(side_effect=failure)
    backend.terminate = AsyncMock()
    caller_env = {"CUSTOM": "value"}
    with patch("victor.integrations.mcp.sandbox.SandboxedProcess", return_value=backend):
        call = SandboxedExecutor(docker_available=False)._execute_process(
            ["ignored"],
            IsolationConfig(sandbox_type="process", network_allowed=True),
            working_dir=None,
            env=caller_env,
            input_data="text",
        )
        if isinstance(failure, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await call
        else:
            result = await call
            assert not result.success
    backend.communicate.assert_awaited_once_with(process, "text")
    backend.terminate.assert_awaited_once_with(process)
    assert caller_env == {"CUSTOM": "value"}

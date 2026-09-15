"""Every factory connection uses the configured process policy and cleanup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.integrations.mcp.sandbox import SandboxConfig, create_sandboxed_mcp_client


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("failure", [None, "start", "initialize_false", "initialize_error"])
async def test_factory_preserves_policy_and_cleans_failed_connections(reconnect, failure):
    config = SandboxConfig(max_memory_mb=64)
    client = create_sandboxed_mcp_client(config)
    client._health_check_interval = 0
    client._reconnect_delay = 0
    client._command = ["server"]
    process = MagicMock()
    process.poll.return_value = None
    wrapper = MagicMock()
    wrapper.start = AsyncMock(return_value=process)
    wrapper.terminate_all = AsyncMock()
    if failure == "start":
        wrapper.start.side_effect = RuntimeError("resource setup failed")
    client.initialize = AsyncMock(return_value=failure != "initialize_false")
    if failure == "initialize_error":
        client.initialize.side_effect = RuntimeError("handshake failed")
    with (
        patch("victor.integrations.mcp.sandbox.SandboxedProcess", return_value=wrapper) as factory,
        patch("subprocess.Popen") as plain,
    ):
        ok = await client._try_reconnect() if reconnect else await client.connect(["server"])
        assert ok is (failure is None)
        factory.assert_called_once_with(config)
        wrapper.start.assert_awaited_once_with(["server"])
        plain.assert_not_called()
        if failure:
            wrapper.terminate_all.assert_awaited_once()
            assert client.process is None
        else:
            await client._cleanup_process_async()
            wrapper.terminate_all.assert_awaited_once()


def test_default_factory_still_enables_resource_limits():
    client = create_sandboxed_mcp_client()
    assert isinstance(client._sandbox_config, SandboxConfig)

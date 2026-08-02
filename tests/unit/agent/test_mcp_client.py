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

"""Tests for MCP client module."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.protocol import MCPTool, MCPResource, MCPToolCallResult


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_init_defaults(self):
        """Test MCPClient with default values."""
        client = MCPClient()

        assert client.name == "Victor MCP Client"
        assert client.version == "1.0.0"
        assert client.server_info is None
        assert client.tools == []
        assert client.resources == []
        assert client.process is None
        assert client.initialized is False
        assert client._health_check_interval == 30
        assert client._auto_reconnect is True
        assert client._max_reconnect_attempts == 3
        assert client._reconnect_delay == 5

    def test_init_custom_values(self):
        """Test MCPClient with custom values."""
        client = MCPClient(
            name="Custom Client",
            version="2.0.0",
            health_check_interval=60,
            auto_reconnect=False,
            max_reconnect_attempts=5,
            reconnect_delay=10,
        )

        assert client.name == "Custom Client"
        assert client.version == "2.0.0"
        assert client._health_check_interval == 60
        assert client._auto_reconnect is False
        assert client._max_reconnect_attempts == 5
        assert client._reconnect_delay == 10

    def test_init_client_info(self):
        """Test client_info is properly initialized."""
        client = MCPClient(name="Test", version="1.0")

        assert client.client_info is not None
        assert client.client_info.name == "Test"
        assert client.client_info.version == "1.0"


class TestMCPClientCallbacks:
    """Tests for MCPClient callback management."""

    def test_on_connect_callback(self):
        """Test on_connect callback registration."""
        client = MCPClient()
        callback = MagicMock()

        client._on_connect_callbacks.append(callback)

        assert len(client._on_connect_callbacks) == 1
        assert callback in client._on_connect_callbacks

    def test_on_disconnect_callback(self):
        """Test on_disconnect callback registration."""
        client = MCPClient()
        callback = MagicMock()

        client._on_disconnect_callbacks.append(callback)

        assert len(client._on_disconnect_callbacks) == 1

    def test_on_health_change_callback(self):
        """Test on_health_change callback registration."""
        client = MCPClient()
        callback = MagicMock()

        client._on_health_change_callbacks.append(callback)

        assert len(client._on_health_change_callbacks) == 1


class TestMCPClientRefreshTools:
    """Tests for tool refreshing."""

    @pytest.mark.asyncio
    async def test_refresh_tools_not_initialized(self):
        """Test refresh_tools returns empty when not initialized."""
        client = MCPClient()
        client.initialized = False

        result = await client.refresh_tools()

        assert result == []

    @pytest.mark.asyncio
    async def test_refresh_tools_success(self):
        """Test refresh_tools parses tools correctly."""
        client = MCPClient()
        client.initialized = True

        mock_response = {
            "result": {
                "tools": [
                    {"name": "tool1", "description": "Test tool 1"},
                    {"name": "tool2", "description": "Test tool 2"},
                ]
            }
        }

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            result = await client.refresh_tools()

            assert len(result) == 2
            assert result[0].name == "tool1"
            assert result[1].name == "tool2"
            assert client.tools == result

    @pytest.mark.asyncio
    async def test_refresh_tools_no_response(self):
        """Test refresh_tools handles no response."""
        client = MCPClient()
        client.initialized = True

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            result = await client.refresh_tools()

            assert result == []


class TestMCPClientRefreshResources:
    """Tests for resource refreshing."""

    @pytest.mark.asyncio
    async def test_refresh_resources_not_initialized(self):
        """Test refresh_resources returns empty when not initialized."""
        client = MCPClient()
        client.initialized = False

        result = await client.refresh_resources()

        assert result == []

    @pytest.mark.asyncio
    async def test_refresh_resources_success(self):
        """Test refresh_resources parses resources correctly."""
        client = MCPClient()
        client.initialized = True

        mock_response = {
            "result": {
                "resources": [
                    {
                        "uri": "file:///test1.txt",
                        "name": "Test 1",
                        "description": "Test file 1",
                    },
                    {
                        "uri": "file:///test2.txt",
                        "name": "Test 2",
                        "description": "Test file 2",
                    },
                ]
            }
        }

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            result = await client.refresh_resources()

            assert len(result) == 2
            assert result[0].uri == "file:///test1.txt"
            assert result[1].uri == "file:///test2.txt"
            assert client.resources == result


class TestMCPClientCallTool:
    """Tests for tool calling."""

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self):
        """Test call_tool returns error when not initialized."""
        client = MCPClient()
        client.initialized = False

        result = await client.call_tool("test_tool", arg1="value")

        assert isinstance(result, MCPToolCallResult)
        assert result.success is False
        assert "not initialized" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test call_tool returns result on success."""
        client = MCPClient()
        client.initialized = True

        mock_response = {
            "result": {
                "tool_name": "test_tool",
                "success": True,
                "result": {"data": "test_data"},
            }
        }

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            result = await client.call_tool("test_tool", arg1="value")

            assert result.success is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        """Test call_tool handles error response."""
        client = MCPClient()
        client.initialized = True

        mock_response = {"error": {"message": "Tool execution failed"}}

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            result = await client.call_tool("test_tool")

            assert result.success is False
            assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_no_response(self):
        """Test call_tool handles no response."""
        client = MCPClient()
        client.initialized = True

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            result = await client.call_tool("test_tool")

            assert result.success is False
            assert "No response" in result.error


class TestMCPClientReadResource:
    """Tests for reading resources."""

    @pytest.mark.asyncio
    async def test_read_resource_not_initialized(self):
        """Test read_resource returns None when not initialized."""
        client = MCPClient()
        client.initialized = False

        result = await client.read_resource("file:///test.txt")

        assert result is None

    @pytest.mark.asyncio
    async def test_read_resource_success(self):
        """Test read_resource returns content on success."""
        client = MCPClient()
        client.initialized = True

        mock_response = {"result": {"content": "File content here"}}

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            result = await client.read_resource("file:///test.txt")

            assert result == "File content here"


class TestMCPClientInitialize:
    """Tests for initialization."""

    @pytest.mark.asyncio
    async def test_initialize_no_process(self):
        """Test initialize returns False when no process."""
        client = MCPClient()
        client.process = None

        result = await client.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test initialize succeeds with valid response."""
        client = MCPClient()
        client.process = MagicMock()

        mock_response = {"result": {"serverInfo": {"name": "Test Server", "version": "1.0"}}}

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response
            with patch.object(client, "refresh_tools", new_callable=AsyncMock):
                with patch.object(client, "refresh_resources", new_callable=AsyncMock):
                    result = await client.initialize()

        assert result is True
        assert client.initialized is True
        assert client.server_info is not None

    @pytest.mark.asyncio
    async def test_initialize_no_response(self):
        """Test initialize returns False when no response."""
        client = MCPClient()
        client.process = MagicMock()

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            result = await client.initialize()

        assert result is False


class TestMCPClientConnect:
    """Tests for connection."""

    @pytest.mark.asyncio
    async def test_connect_stores_command(self):
        """Test connect stores command for reconnection."""
        client = MCPClient()

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process

            with patch.object(client, "initialize", new_callable=AsyncMock) as mock_init:
                mock_init.return_value = False

                await client.connect(["python", "server.py"])

                assert client._command == ["python", "server.py"]

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        client = MCPClient(health_check_interval=0)
        callback = MagicMock()
        client._on_connect_callbacks.append(callback)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process

            with patch.object(client, "initialize", new_callable=AsyncMock) as mock_init:
                mock_init.return_value = True

                result = await client.connect(["python", "server.py"])

                assert result is True
                assert client._running is True
                assert client._consecutive_failures == 0
                callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        client = MCPClient()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = Exception("Process failed")

            result = await client.connect(["python", "server.py"])

            assert result is False
            assert client._consecutive_failures == 1


class TestMCPClientDisconnect:
    """Tests for disconnection."""

    def test_disconnect(self):
        """Test disconnect cleans up properly."""
        client = MCPClient()
        mock_process = MagicMock()
        client.process = mock_process
        client.initialized = True
        client._running = True

        callback = MagicMock()
        client._on_disconnect_callbacks.append(callback)

        client.disconnect()

        assert client.initialized is False
        assert client._running is False
        callback.assert_called_once()
        mock_process.terminate.assert_called_once()

    def test_disconnect_with_reason(self):
        """Test disconnect passes reason to callbacks."""
        client = MCPClient()
        mock_process = MagicMock()
        client.process = mock_process

        callback = MagicMock()
        client._on_disconnect_callbacks.append(callback)

        client.disconnect(reason="Manual disconnect")

        callback.assert_called_once_with("Manual disconnect")

    def test_disconnect_cancels_health_task(self):
        """Test disconnect cancels health monitoring task."""
        client = MCPClient()
        client.process = MagicMock()
        mock_task = MagicMock()
        client._health_task = mock_task

        client.disconnect()

        mock_task.cancel.assert_called_once()
        assert client._health_task is None

    def test_disconnect_uses_shared_sync_bridge_for_sandboxed_process(self):
        """Test disconnect bridges sandbox cleanup with run_sync outside event loops."""
        client = MCPClient()
        sandboxed_process = MagicMock()
        coro = object()
        sandboxed_process.terminate.return_value = coro
        client._sandboxed_process = sandboxed_process

        with (
            patch(
                "victor.integrations.mcp.client.asyncio.get_running_loop",
                side_effect=RuntimeError,
            ),
            patch("victor.integrations.mcp.client.run_sync", return_value=None) as mock_run_sync,
        ):
            client.disconnect()

        sandboxed_process.terminate.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)
        assert client._sandboxed_process is None

    def test_cleanup_process_uses_shared_sync_bridge_for_sandboxed_process(self):
        """Test _cleanup_process bridges sandbox cleanup with run_sync outside loops."""
        client = MCPClient()
        sandboxed_process = MagicMock()
        coro = object()
        sandboxed_process.terminate.return_value = coro
        client._sandboxed_process = sandboxed_process

        with (
            patch(
                "victor.integrations.mcp.client.asyncio.get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(client, "_cleanup_process_sync") as mock_cleanup_sync,
            patch("victor.integrations.mcp.client.run_sync", return_value=None) as mock_run_sync,
        ):
            client._cleanup_process()

        sandboxed_process.terminate.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)
        mock_cleanup_sync.assert_called_once_with()
        assert client._sandboxed_process is None


class TestMCPClientPing:
    """Tests for ping/health checking."""

    @pytest.mark.asyncio
    async def test_ping_success(self):
        """Test ping succeeds with response."""
        client = MCPClient()

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"result": "pong"}

            result = await client.ping()

            assert result is True
            assert client._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_ping_failure(self):
        """Test ping fails with no response."""
        client = MCPClient()

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            result = await client.ping()

            assert result is False
            assert client._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_ping_exception(self):
        """Test ping handles exception."""
        client = MCPClient()

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Network error")

            result = await client.ping()

            assert result is False
            assert client._consecutive_failures == 1


class TestMCPClientHelpers:
    """Tests for helper methods."""

    def test_get_tool_by_name_found(self):
        """Test get_tool_by_name returns tool when found."""
        client = MCPClient()
        client.tools = [
            MCPTool(name="tool1", description="Test 1"),
            MCPTool(name="tool2", description="Test 2"),
        ]

        result = client.get_tool_by_name("tool2")

        assert result is not None
        assert result.name == "tool2"

    def test_get_tool_by_name_not_found(self):
        """Test get_tool_by_name returns None when not found."""
        client = MCPClient()
        client.tools = [MCPTool(name="tool1", description="Test")]

        result = client.get_tool_by_name("nonexistent")

        assert result is None

    def test_get_resource_by_uri_found(self):
        """Test get_resource_by_uri returns resource when found."""
        client = MCPClient()
        client.resources = [
            MCPResource(uri="file:///test1.txt", name="Test 1", description="Test file 1"),
            MCPResource(uri="file:///test2.txt", name="Test 2", description="Test file 2"),
        ]

        result = client.get_resource_by_uri("file:///test2.txt")

        assert result is not None
        assert result.uri == "file:///test2.txt"

    def test_get_resource_by_uri_not_found(self):
        """Test get_resource_by_uri returns None when not found."""
        client = MCPClient()
        client.resources = [
            MCPResource(uri="file:///test.txt", name="Test", description="Test file")
        ]

        result = client.get_resource_by_uri("file:///nonexistent.txt")

        assert result is None

    def test_get_status(self):
        """Test get_status returns status dict."""
        client = MCPClient()
        client.initialized = True
        client.tools = [MCPTool(name="tool1", description="Test")]
        client.resources = [
            MCPResource(uri="file:///test.txt", name="Test", description="Test file")
        ]

        status = client.get_status()

        assert status["connected"] is True
        assert status["tools_count"] == 1
        assert status["resources_count"] == 1
        assert "health" in status

    def test_reset_connection(self):
        """Test reset_connection resets failure count."""
        client = MCPClient()
        client._consecutive_failures = 5

        client.reset_connection()

        assert client._consecutive_failures == 0


class TestMCPClientCallbackRegistration:
    """Tests for callback registration methods."""

    def test_on_connect(self):
        """Test on_connect registers callback."""
        client = MCPClient()
        callback = MagicMock()

        client.on_connect(callback)

        assert callback in client._on_connect_callbacks

    def test_on_disconnect(self):
        """Test on_disconnect registers callback."""
        client = MCPClient()
        callback = MagicMock()

        client.on_disconnect(callback)

        assert callback in client._on_disconnect_callbacks

    def test_on_health_change(self):
        """Test on_health_change registers callback."""
        client = MCPClient()
        callback = MagicMock()

        client.on_health_change(callback)

        assert callback in client._on_health_change_callbacks


class TestMCPClientReconnect:
    """Tests for reconnection."""

    @pytest.mark.asyncio
    async def test_try_reconnect_no_command(self):
        """Test _try_reconnect fails without stored command."""
        client = MCPClient()
        client._command = None

        result = await client._try_reconnect()

        assert result is False

    @pytest.mark.asyncio
    async def test_try_reconnect_max_attempts_exceeded(self):
        """Test _try_reconnect gives up after max attempts."""
        client = MCPClient(max_reconnect_attempts=3)
        client._command = ["python", "server.py"]
        client._consecutive_failures = 3

        result = await client._try_reconnect()

        assert result is False
        assert client._running is False


class TestMCPClientResourceManagement:
    """Tests for resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_method_exists(self):
        """Test that cleanup() method exists as public API."""
        client = MCPClient()
        assert hasattr(client, "cleanup")
        assert callable(client.cleanup)

    @pytest.mark.asyncio
    async def test_cleanup_closes_process(self):
        """Test cleanup properly terminates subprocess."""
        client = MCPClient()
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.wait = MagicMock()
        client.process = mock_process
        client.initialized = True
        client._running = True

        await client.cleanup()

        assert client.process is None
        assert client.initialized is False
        assert client._running is False
        mock_process.stdin.close.assert_called_once()
        mock_process.stdout.close.assert_called_once()
        mock_process.stderr.close.assert_called_once()
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_cancels_health_task(self):
        """Test cleanup cancels health monitoring task."""
        client = MCPClient()

        # Create a real asyncio task that we can cancel
        async def dummy_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_task())
        client._health_task = task

        await client.cleanup()

        assert task.cancelled() or task.done()
        assert client._health_task is None

    @pytest.mark.asyncio
    async def test_cleanup_emits_disconnect_callback(self):
        """Test cleanup emits disconnect callback with reason."""
        client = MCPClient()
        mock_process = MagicMock()
        mock_process.wait = MagicMock()
        client.process = mock_process

        callback = MagicMock()
        client._on_disconnect_callbacks.append(callback)

        await client.cleanup(reason="test_reason")

        callback.assert_called_once_with("test_reason")

    @pytest.mark.asyncio
    async def test_cleanup_handles_exception_gracefully(self):
        """Test cleanup handles exceptions without raising."""
        client = MCPClient()
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.close.side_effect = Exception("Close failed")
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.wait = MagicMock()
        client.process = mock_process

        # Should not raise
        await client.cleanup()

        assert client.process is None

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(self):
        """Test cleanup can be called multiple times safely."""
        client = MCPClient()
        mock_process = MagicMock()
        mock_process.wait = MagicMock()
        client.process = mock_process

        # First cleanup
        await client.cleanup()
        assert client.process is None

        # Second cleanup should not raise
        await client.cleanup()

    @pytest.mark.asyncio
    async def test_close_is_alias_for_cleanup(self):
        """Test close() method works as alias for cleanup()."""
        client = MCPClient()
        mock_process = MagicMock()
        mock_process.wait = MagicMock()
        client.process = mock_process
        client.initialized = True
        client._running = True

        await client.close()

        assert client.process is None
        assert client.initialized is False
        assert client._running is False


class TestMCPClientAsyncContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_without_command(self):
        """Test context manager works without auto-connect command."""
        async with MCPClient() as client:
            assert client is not None
            assert not client.initialized
            # Should be able to connect manually if needed

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exit(self):
        """Test context manager calls cleanup on exit."""
        client = MCPClient()
        with patch.object(client, "cleanup", new_callable=AsyncMock) as mock_cleanup:
            async with client:
                pass
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test context manager calls cleanup even when exception occurs."""
        client = MCPClient()
        with patch.object(client, "cleanup", new_callable=AsyncMock) as mock_cleanup:
            with pytest.raises(ValueError):
                async with client:
                    raise ValueError("Test error")
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_command_success(self):
        """Test context manager auto-connects when command provided."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.wait = MagicMock()  # For cleanup
            mock_popen.return_value = mock_process

            client = MCPClient(command=["python", "server.py"], health_check_interval=0)

            with patch.object(client, "initialize", new_callable=AsyncMock) as mock_init:
                mock_init.return_value = True

                async with client:
                    assert client._running is True
                    # initialized is set by initialize() which is mocked
                    # The mock returns True so connect() succeeds
                    mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_command_failure_raises(self):
        """Test context manager raises ConnectionError on connect failure."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = Exception("Process failed")

            client = MCPClient(command=["python", "server.py"])

            with pytest.raises(ConnectionError) as exc_info:
                async with client:
                    pass

            assert "Failed to connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_manager_stores_command_for_reconnect(self):
        """Test that command provided in __init__ is stored for reconnection."""
        command = ["python", "server.py"]
        client = MCPClient(command=command)

        assert client._command == command
        assert client._auto_connect_command == command

    def test_init_with_command_parameter(self):
        """Test MCPClient accepts command parameter in __init__."""
        command = ["python", "server.py"]
        client = MCPClient(command=command)

        assert client._command == command


class TestMCPClientDestructor:
    """Tests for destructor behavior."""

    def test_destructor_cleanup_on_garbage_collection(self):
        """Test __del__ attempts cleanup when process still active."""
        client = MCPClient()
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.wait = MagicMock()
        client.process = mock_process

        # Manually call __del__ to simulate garbage collection
        client.__del__()

        # Process should be cleaned up
        assert client.process is None

    def test_destructor_safe_when_no_process(self):
        """Test __del__ is safe when no process exists."""
        client = MCPClient()
        client.process = None
        client._sandboxed_process = None

        # Should not raise
        client.__del__()

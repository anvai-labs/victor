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

"""MCP Client implementation for Victor.

Connects to external MCP servers to access their tools and resources,
extending Victor's capabilities with third-party integrations.

Features:
- Stdio-based communication with MCP servers
- Health monitoring with configurable intervals
- Automatic reconnection on connection loss
- Tool and resource caching
- Optional subprocess sandboxing with resource limits
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.config.timeouts import McpTimeouts
from victor.core.async_utils import run_sync

if TYPE_CHECKING:
    from victor.integrations.mcp.sandbox import SandboxConfig, SandboxedProcess
from victor.integrations.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    MCPClientInfo,
    MCPMessage,
    MCPMessageType,
    MCPParameter,
    MCPParameterType,
    MCPResource,
    MCPServerInfo,
    MCPTool,
    MCPToolCallResult,
)

logger = logging.getLogger(__name__)


def _translate_input_schema_to_parameters(
    tool_name: str,
    input_schema: Dict[str, Any],
) -> List["MCPParameter"]:
    """Convert standard MCP inputSchema to Victor's legacy parameters format.

    This bridges AgentBrowser's modern MCP protocol (JSON Schema Draft 2020-12)
    with Victor's legacy parameter format. Post-transformation, the inputSchema
    key is deleted from the tool dict to prevent Pydantic from silently ignoring it.

    Args:
        tool_name: Name of the tool (for logging)
        input_schema: JSON Schema inputSchema from MCP tools/list response

    Returns:
        List of MCPParameter objects in Victor's legacy format

    Example:
        AgentBrowser returns:
        {
            "name": "browser_navigate",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "..."}},
                "required": ["url"]
            }
        }

        This function translates to:
        [
            MCPParameter(
                name="url",
                type=MCPParameterType.STRING,
                description="...",
                required=True,
                default=None
            )
        ]
    """
    if not input_schema or input_schema.get("type") != "object":
        logger.warning(f"Invalid inputSchema for tool {tool_name}: {input_schema}")
        return []

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    params = []
    for param_name, prop_def in properties.items():
        param_type_str = prop_def.get("type", "string")

        # Map JSON Schema types to MCPParameterType enum
        try:
            param_type = MCPParameterType(param_type_str)
        except ValueError:
            logger.debug(
                f"Unknown type '{param_type_str}' for param '{param_name}' "
                f"in tool '{tool_name}', defaulting to STRING"
            )
            param_type = MCPParameterType.STRING

        params.append(
            MCPParameter(
                name=param_name,
                type=param_type,
                description=prop_def.get("description", ""),
                required=param_name in required,
                default=prop_def.get("default"),
            )
        )

    return params


class MCPClient:
    """MCP client that connects to external MCP servers.

    This client can discover and use tools from MCP-compatible servers,
    allowing Victor to integrate with external tools and data sources.

    Features:
    - Health monitoring with configurable intervals
    - Automatic reconnection on connection loss
    - Event callbacks for connection state changes
    - Tool and resource caching
    """

    def __init__(
        self,
        name: str = "Victor MCP Client",
        version: str = "1.0.0",
        health_check_interval: int = 30,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: int = 5,
        sandbox_config: Optional["SandboxConfig"] = None,
        command: Optional[List[str]] = None,
    ):
        """Initialize MCP client.

        Args:
            name: Client name
            version: Client version
            health_check_interval: Seconds between health checks (0 to disable)
            auto_reconnect: Whether to automatically reconnect on failure
            max_reconnect_attempts: Maximum reconnection attempts before giving up
            reconnect_delay: Seconds to wait between reconnection attempts
            sandbox_config: Optional sandboxing config for subprocess resource limits
            command: Optional command to auto-connect when used as async context manager.
                     If provided, __aenter__ will call connect(command) automatically.

        Example with command parameter for context manager usage:
            async with MCPClient(command=["python", "server.py"]) as client:
                # Client is automatically connected
                tools = await client.refresh_tools()
            # Client is automatically cleaned up
        """
        self.name = name
        self.version = version
        self.client_info = MCPClientInfo(name=name, version=version)

        self.server_info: Optional[MCPServerInfo] = None
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []

        self.process: Optional[subprocess.Popen] = None
        self.initialized = False

        # Health monitoring configuration
        self._health_check_interval = health_check_interval
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay

        # Sandboxing configuration
        self._sandbox_config = sandbox_config
        self._sandboxed_process: Optional["SandboxedProcess"] = None

        # Connection state
        self._command: Optional[List[str]] = command  # Store for reconnection and context manager
        self._last_health_check: float = 0.0
        self._consecutive_failures: int = 0
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._auto_connect_command: Optional[List[str]] = command  # For context manager

        # Event callbacks
        self._on_connect_callbacks: List[Callable[[], None]] = []
        self._on_disconnect_callbacks: List[Callable[[Optional[str]], None]] = []
        self._on_health_change_callbacks: List[Callable[[bool], None]] = []

    async def connect(self, command: List[str]) -> bool:
        """Connect to MCP server via stdio.

        Args:
            command: Command to start MCP server (e.g., ["python", "server.py"])

        Returns:
            True if connection successful
        """
        self._command = command  # Store for reconnection

        try:
            # Start server process (with optional sandboxing)
            if self._sandbox_config is not None:
                # Use sandboxed process for resource limits and isolation
                try:
                    from victor.integrations.mcp.sandbox import SandboxedProcess

                    self._sandboxed_process = SandboxedProcess(self._sandbox_config)
                    self.process = await self._sandboxed_process.start(command)
                    logger.info(
                        f"Started sandboxed MCP server with limits: "
                        f"memory={self._sandbox_config.max_memory_mb}MB, "
                        f"timeout={self._sandbox_config.timeout_seconds}s"
                    )
                except Exception as e:
                    logger.warning(f"Sandboxed process failed, falling back to regular: {e}")
                    self.process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                    )
            else:
                # Standard subprocess without sandboxing
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

            # Initialize connection
            success = await self.initialize()

            if success:
                self._consecutive_failures = 0
                self._last_health_check = time.time()
                self._running = True

                # Emit connect event
                for callback in self._on_connect_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Connect callback error: {e}")

                # Start health monitoring if enabled
                if self._health_check_interval > 0:
                    self._health_task = asyncio.create_task(self._health_monitor_loop())

                return True

            # Cleanup on initialization failure
            self._cleanup_process()
            return False

        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            self._consecutive_failures += 1
            # Cleanup on exception
            self._cleanup_process()
            return False

    async def _cleanup_process_async(self) -> None:
        """Clean up subprocess and its resources asynchronously."""
        # Clean up sandboxed process if used
        if self._sandboxed_process is not None:
            try:
                await self._sandboxed_process.terminate()
            except Exception as e:
                logger.debug(f"Error terminating sandboxed process: {e}")
            self._sandboxed_process = None

        self._cleanup_process_sync()

    def _cleanup_process_sync(self) -> None:
        """Synchronously clean up subprocess resources (for use in sync contexts)."""
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception as e:
                logger.debug(f"Error closing process pipes during cleanup: {e}")

            try:
                self.process.terminate()
                self.process.wait(timeout=McpTimeouts.TERMINATE)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    logger.debug(f"Error killing process during cleanup: {e}")
            except Exception as e:
                logger.debug(f"Error terminating process during cleanup: {e}")

            self.process = None
            self.initialized = False

    def _cleanup_sandboxed_process_sync(self) -> None:
        """Clean up sandboxed process from sync code paths."""
        if self._sandboxed_process is None:
            return

        sandboxed_process = self._sandboxed_process
        self._sandboxed_process = None

        try:
            asyncio.get_running_loop()
            asyncio.create_task(sandboxed_process.terminate())
        except RuntimeError:
            try:
                run_sync(sandboxed_process.terminate())
            except Exception as e:
                logger.debug(f"Error terminating sandboxed process: {e}")

    def _cleanup_process(self) -> None:
        """Clean up subprocess and its resources (sync wrapper for backwards compatibility)."""
        self._cleanup_sandboxed_process_sync()
        self._cleanup_process_sync()

    async def initialize(self) -> bool:
        """Initialize MCP connection.

        Returns:
            True if initialization successful
        """
        if not self.process:
            return False

        # Send initialize message with protocol version (required by MCP spec)
        response = await self._send_request(
            MCPMessageType.INITIALIZE,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": self.client_info.model_dump(),
                "capabilities": self.client_info.capabilities.model_dump(),
            },
        )

        if response and "result" in response:
            result = response["result"]
            self.server_info = MCPServerInfo(**result.get("serverInfo", {}))
            self.initialized = True

            # Fetch available tools and resources
            await self.refresh_tools()
            await self.refresh_resources()

            return True

        return False

    async def refresh_tools(self) -> List[MCPTool]:
        """Refresh list of available tools from server.

        Returns:
            List of available tools
        """
        if not self.initialized:
            return []

        response = await self._send_request(MCPMessageType.LIST_TOOLS, {})

        if response and "result" in response:
            tools_data = response["result"].get("tools", [])

            # Translate inputSchema → parameters for compatibility with modern MCP servers
            # (e.g., AgentBrowser uses standard JSON Schema inputSchema, Victor uses legacy format)
            mcp_tools = []
            for tool_dict in tools_data:
                if "inputSchema" in tool_dict:
                    # Modern MCP server: translate JSON Schema to legacy parameters
                    tool_dict["parameters"] = _translate_input_schema_to_parameters(
                        tool_dict["name"], tool_dict["inputSchema"]
                    )
                    del tool_dict["inputSchema"]  # Clean up after translation
                mcp_tools.append(MCPTool(**tool_dict))

            self.tools = mcp_tools
            return self.tools

        return []

    async def refresh_resources(self) -> List[MCPResource]:
        """Refresh list of available resources from server.

        Returns:
            List of available resources
        """
        if not self.initialized:
            return []

        response = await self._send_request(MCPMessageType.LIST_RESOURCES, {})

        if response and "result" in response:
            resources_data = response["result"].get("resources", [])
            self.resources = [MCPResource(**r) for r in resources_data]
            return self.resources

        return []

    async def call_tool(self, tool_name: str, **arguments: Any) -> MCPToolCallResult:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of tool to call
            **arguments: Tool arguments

        Returns:
            Tool call result
        """
        if not self.initialized:
            return MCPToolCallResult(
                tool_name=tool_name,
                success=False,
                error="Client not initialized",
            )

        response = await self._send_request(
            MCPMessageType.CALL_TOOL, {"name": tool_name, "arguments": arguments}
        )

        if response and "result" in response:
            result_data = response["result"]
            # Standard MCP format per modelcontextprotocol.io specification:
            # {"content": [{"type": "text", "text": "..."}], "isError": false}
            content_blocks = result_data.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "image":
                    # Handle image content blocks
                    text_parts.append(f"[image: {block.get('mimeType', 'image')}]")
                elif isinstance(block, str):
                    text_parts.append(block)
            combined_result = "\n".join(text_parts) if text_parts else str(result_data)
            is_error = result_data.get("isError", False)
            return MCPToolCallResult(
                tool_name=tool_name,
                success=not is_error,
                result=combined_result,
                error=combined_result if is_error else None,
            )
        elif response and "error" in response:
            error = response["error"]
            return MCPToolCallResult(
                tool_name=tool_name,
                success=False,
                error=error.get("message", "Unknown error"),
            )

        return MCPToolCallResult(
            tool_name=tool_name, success=False, error="No response from server"
        )

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read content from a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content if successful
        """
        if not self.initialized:
            return None

        response = await self._send_request(MCPMessageType.READ_RESOURCE, {"uri": uri})

        if response and "result" in response:
            return response["result"].get("content")

        return None

    async def _send_request(
        self, method: MCPMessageType, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send request to MCP server.

        Args:
            method: Request method
            params: Request parameters

        Returns:
            Response dictionary or None
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            return None

        msg_id = str(uuid.uuid4())
        message = MCPMessage(id=msg_id, method=method, params=params)

        try:
            # Send request - use asyncio to avoid blocking
            request_json = message.model_dump_json(exclude_none=True)
            (request_json + "\n").encode()

            # Write asynchronously using run_in_executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: (
                    self.process.stdin.write(request_json + "\n"),
                    self.process.stdin.flush(),
                ),
            )

            # Read response with timeout to avoid indefinite blocking
            try:
                response_line = await asyncio.wait_for(
                    loop.run_in_executor(None, self.process.stdout.readline),
                    timeout=McpTimeouts.RESPONSE,
                )
            except asyncio.TimeoutError:
                logger.warning(f"MCP request timeout for method: {method}")
                return None

            if not response_line:
                return None

            response = json.loads(response_line)

            # Verify response ID matches
            if response.get("id") != msg_id:
                logger.warning(f"Response ID mismatch: {response.get('id')} != {msg_id}")

            return response

        except Exception as e:
            logger.error(f"Error sending MCP request: {e}")
            return None

    async def ping(self) -> bool:
        """Ping the MCP server.

        Returns:
            True if server responds
        """
        try:
            response = await self._send_request(MCPMessageType.PING, {})
            healthy = response is not None and "result" in response

            if healthy:
                self._last_health_check = time.time()
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            return healthy
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            self._consecutive_failures += 1
            return False

    def disconnect(self, reason: Optional[str] = None) -> None:
        """Disconnect from MCP server (synchronous).

        Args:
            reason: Optional reason for disconnection
        """
        self._running = False

        # Cancel health monitoring
        if self._health_task:
            self._health_task.cancel()
            self._health_task = None

        self._cleanup_sandboxed_process_sync()

        if self.process:
            # Close file handles to prevent resource leaks
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception as e:
                logger.debug(f"Error closing process pipes: {e}")

            try:
                self.process.terminate()
                self.process.wait(timeout=McpTimeouts.TERMINATE)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    logger.debug(f"Error killing process: {e}")
            except Exception as e:
                logger.debug(f"Error terminating process: {e}")

            self.process = None
            self.initialized = False

            # Emit disconnect event
            for callback in self._on_disconnect_callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")

    async def cleanup(self, reason: Optional[str] = None) -> None:
        """Clean up all resources properly (async).

        This is the recommended method for cleaning up the MCP client.
        It properly handles:
        - Cancelling background health monitoring tasks
        - Terminating sandboxed processes
        - Closing subprocess file handles (stdin, stdout, stderr)
        - Terminating the subprocess
        - Emitting disconnect callbacks

        Args:
            reason: Optional reason for cleanup/disconnection

        Example:
            client = MCPClient()
            try:
                await client.connect(["python", "server.py"])
                # ... use client ...
            finally:
                await client.cleanup()

            # Or use as async context manager:
            async with MCPClient() as client:
                await client.connect(["python", "server.py"])
                # ... use client ...
            # cleanup() called automatically on exit
        """
        await self._cleanup_internal(reason)

    async def close(self, reason: Optional[str] = None) -> None:
        """Disconnect from MCP server (async version with proper cleanup).

        This is an alias for cleanup() for backward compatibility.

        Args:
            reason: Optional reason for disconnection
        """
        await self._cleanup_internal(reason)

    async def _cleanup_internal(self, reason: Optional[str] = None) -> None:
        """Internal cleanup implementation.

        Args:
            reason: Optional reason for cleanup
        """
        self._running = False

        # Cancel and await health monitoring task
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        # Clean up sandboxed process if used
        if self._sandboxed_process is not None:
            try:
                await self._sandboxed_process.terminate()
            except Exception as e:
                logger.debug(f"Error terminating sandboxed process: {e}")
            self._sandboxed_process = None

        if self.process:
            # Close file handles to prevent resource leaks
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception as e:
                logger.debug(f"Error closing process pipes: {e}")

            try:
                self.process.terminate()
                # Use asyncio to wait non-blocking
                loop = asyncio.get_running_loop()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self.process.wait),
                        timeout=McpTimeouts.TERMINATE,
                    )
                except asyncio.TimeoutError:
                    self.process.kill()
                    await loop.run_in_executor(None, self.process.wait)
            except Exception as e:
                logger.debug(f"Error terminating process: {e}")

            self.process = None
            self.initialized = False

            # Emit disconnect event
            for callback in self._on_disconnect_callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        last_healthy = True

        while self._running:
            await asyncio.sleep(self._health_check_interval)

            if not self._running:
                break

            healthy = await self.ping()

            # Emit health change event if state changed
            if healthy != last_healthy:
                for callback in self._on_health_change_callbacks:
                    try:
                        callback(healthy)
                    except Exception as e:
                        logger.error(f"Health change callback error: {e}")
                last_healthy = healthy

            # Attempt reconnection if unhealthy
            if not healthy and self._auto_reconnect:
                await self._try_reconnect()

    async def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the server.

        Returns:
            True if reconnection successful
        """
        if not self._command:
            logger.error("Cannot reconnect: no command stored")
            return False

        if self._consecutive_failures >= self._max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts ({self._max_reconnect_attempts}) exceeded, " "giving up"
            )
            await self.close("max_retries_exceeded")
            return False

        logger.info(
            f"Attempting reconnection "
            f"(attempt {self._consecutive_failures + 1}/{self._max_reconnect_attempts})"
        )

        # Clean up current connection properly (close pipes first)
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception as e:
                logger.debug(f"Error closing process pipes during reconnect: {e}")

            try:
                self.process.terminate()
                self.process.wait(timeout=McpTimeouts.KILL)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    logger.debug(f"Error killing process during reconnect: {e}")
            except Exception as e:
                logger.debug(f"Error terminating process during reconnect: {e}")

            self.process = None
            self.initialized = False

        await asyncio.sleep(self._reconnect_delay)

        # Reconnect
        try:
            self.process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            success = await self.initialize()
            if success:
                self._consecutive_failures = 0
                logger.info("Reconnection successful")

                # Emit connect event
                for callback in self._on_connect_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Connect callback error: {e}")

                return True

            return False

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self._consecutive_failures += 1
            return False

    def reset_connection(self) -> None:
        """Reset connection state to allow fresh reconnection attempts."""
        self._consecutive_failures = 0

    # Event callback registration

    def on_connect(self, callback: Callable[[], None]) -> None:
        """Register callback for connection events.

        Args:
            callback: Callback function (no arguments)
        """
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[Optional[str]], None]) -> None:
        """Register callback for disconnection events.

        Args:
            callback: Callback function (reason: Optional[str])
        """
        self._on_disconnect_callbacks.append(callback)

    def on_health_change(self, callback: Callable[[bool], None]) -> None:
        """Register callback for health state changes.

        Args:
            callback: Callback function (is_healthy: bool)
        """
        self._on_health_change_callbacks.append(callback)

    def get_tool_by_name(self, name: str) -> Optional[MCPTool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool definition or None
        """
        return next((t for t in self.tools if t.name == name), None)

    def get_resource_by_uri(self, uri: str) -> Optional[MCPResource]:
        """Get resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource definition or None
        """
        return next((r for r in self.resources if r.uri == uri), None)

    def get_status(self) -> Dict[str, Any]:
        """Get client status.

        Returns:
            Status dictionary
        """
        return {
            "connected": self.initialized,
            "server": self.server_info.model_dump() if self.server_info else None,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "health": {
                "last_check": self._last_health_check,
                "consecutive_failures": self._consecutive_failures,
                "health_monitoring": self._health_check_interval > 0,
                "auto_reconnect": self._auto_reconnect,
            },
        }

    @property
    def is_healthy(self) -> bool:
        """Check if client connection is healthy.

        Returns:
            True if connected and no recent failures
        """
        return self.initialized and self._consecutive_failures == 0

    @property
    def last_health_check(self) -> float:
        """Get timestamp of last health check.

        Returns:
            Unix timestamp of last health check
        """
        return self._last_health_check

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry.

        If a command was provided in __init__, this will automatically
        call connect(command). Otherwise, you must call connect() manually
        after entering the context.

        Returns:
            self

        Raises:
            ConnectionError: If auto-connect was configured but connection failed

        Example:
            # With command - auto-connects
            async with MCPClient(command=["python", "server.py"]) as client:
                tools = client.tools  # Already connected

            # Without command - manual connect required
            async with MCPClient() as client:
                await client.connect(["python", "server.py"])
                tools = client.tools
        """
        if self._auto_connect_command is not None:
            success = await self.connect(self._auto_connect_command)
            if not success:
                # Clean up any partial state before raising
                await self.cleanup()
                raise ConnectionError(
                    f"Failed to connect to MCP server with command: {self._auto_connect_command}"
                )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper async cleanup.

        This method ensures all resources are properly cleaned up:
        - Background tasks are cancelled
        - Subprocess is terminated
        - File handles are closed

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            None (exceptions are not suppressed)
        """
        await self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection.

        This is a safety net - prefer using context manager or calling close() explicitly.
        """
        if self.process is not None or self._sandboxed_process is not None:
            logger.debug("MCPClient being garbage collected with active process - cleaning up")
            # Use sync cleanup as we can't await in __del__
            self._cleanup_process_sync()
            if self._sandboxed_process is not None:
                # Can't await here, but at least log the issue
                logger.warning(
                    "MCPClient: sandboxed process not properly closed before garbage collection"
                )
                self._sandboxed_process = None

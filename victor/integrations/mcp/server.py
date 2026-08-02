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

"""MCP Server implementation for Victor.

Exposes Victor's tools and resources through the Model Context Protocol,
allowing other MCP clients to use Victor's capabilities.
"""

import asyncio
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

from victor.core.async_utils import run_sync
from victor.integrations.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    MCPCapabilities,
    MCPMessage,
    MCPMessageType,
    MCPParameter,
    MCPParameterType,
    MCPResource,
    MCPResourceContent,
    MCPServerInfo,
    MCPTool,
    MCPToolCallResult,
)
from victor.tools.base import BaseTool
from victor.tools.registry import ToolRegistry


class MCPServer:
    """MCP server that exposes Victor's tools and resources.

    This server implements the Model Context Protocol, allowing external
    clients to discover and use Victor's tools through a standardized interface.
    """

    def __init__(
        self,
        name: str = "Victor MCP Server",
        version: str = "1.0.0",
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """Initialize MCP server.

        Args:
            name: Server name
            version: Server version
            tool_registry: Victor's tool registry
        """
        self.name = name
        self.version = version
        self.tool_registry = tool_registry or ToolRegistry()

        self.info = MCPServerInfo(
            name=name,
            version=version,
            capabilities=MCPCapabilities(
                tools={},  # Empty object means supported
                resources={},  # Empty object means supported
                prompts=None,  # None means not supported (omitted from output)
                sampling=None,  # None means not supported (omitted from output)
            ),
        )

        self.initialized = False
        self.resources: List[MCPResource] = []
        self._running = False  # For graceful shutdown of stdio server
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer_transport: Optional[asyncio.WriteTransport] = None

    def register_resource(self, resource: MCPResource) -> None:
        """Register a resource with the MCP server.

        Args:
            resource: Resource to register
        """
        self.resources.append(resource)

    def _tool_to_mcp(self, tool: BaseTool) -> MCPTool:
        """Convert Victor tool to MCP tool definition.

        Args:
            tool: Victor tool

        Returns:
            MCP tool definition
        """
        mcp_params = []

        # Handle both formats: List[ToolParameter] and Dict (JSON Schema)
        tool_params = tool.parameters

        if isinstance(tool_params, list):
            # New format: List[ToolParameter]
            for param in tool_params:
                type_map = {
                    "string": MCPParameterType.STRING,
                    "number": MCPParameterType.NUMBER,
                    "integer": MCPParameterType.NUMBER,
                    "boolean": MCPParameterType.BOOLEAN,
                    "object": MCPParameterType.OBJECT,
                    "array": MCPParameterType.ARRAY,
                }

                mcp_type = type_map.get(param.type.lower(), MCPParameterType.STRING)

                mcp_params.append(
                    MCPParameter(
                        name=param.name,
                        type=mcp_type,
                        description=param.description,
                        required=param.required,
                    )
                )

        elif isinstance(tool_params, dict):
            # Old format: JSON Schema dict
            properties = tool_params.get("properties", {})
            required_list = tool_params.get("required", [])

            for param_name, param_def in properties.items():
                type_map = {
                    "string": MCPParameterType.STRING,
                    "number": MCPParameterType.NUMBER,
                    "integer": MCPParameterType.NUMBER,
                    "boolean": MCPParameterType.BOOLEAN,
                    "object": MCPParameterType.OBJECT,
                    "array": MCPParameterType.ARRAY,
                }

                param_type = param_def.get("type", "string")
                mcp_type = type_map.get(param_type.lower(), MCPParameterType.STRING)

                mcp_params.append(
                    MCPParameter(
                        name=param_name,
                        type=mcp_type,
                        description=param_def.get("description", ""),
                        required=param_name in required_list,
                    )
                )

        return MCPTool(
            name=tool.name,
            description=tool.description,
            parameters=mcp_params,
        )

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP message.

        Args:
            message: MCP message dictionary

        Returns:
            Response message dictionary
        """
        try:
            mcp_msg = MCPMessage(**message)
            msg_id = mcp_msg.id or str(uuid.uuid4())

            if mcp_msg.method == MCPMessageType.INITIALIZE:
                return await self._handle_initialize(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.LIST_TOOLS:
                return await self._handle_list_tools(msg_id)

            elif mcp_msg.method == MCPMessageType.CALL_TOOL:
                return await self._handle_call_tool(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.LIST_RESOURCES:
                return await self._handle_list_resources(msg_id)

            elif mcp_msg.method == MCPMessageType.READ_RESOURCE:
                return await self._handle_read_resource(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.PING:
                return self._create_response(msg_id, {"pong": True})

            else:
                return self._create_error(msg_id, -32601, f"Method not found: {mcp_msg.method}")

        except Exception as e:
            return self._create_error(None, -32700, f"Parse error: {str(e)}")

    async def _handle_initialize(self, msg_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request.

        Args:
            msg_id: Message ID
            params: Request parameters

        Returns:
            Initialize response
        """
        self.initialized = True

        return self._create_response(
            msg_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": self.info.model_dump(),
                "capabilities": self.info.capabilities.model_dump(),
            },
        )

    async def _handle_list_tools(self, msg_id: str) -> Dict[str, Any]:
        """Handle list tools request.

        Args:
            msg_id: Message ID

        Returns:
            Tools list response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        tools = []
        for tool in self.tool_registry.list_tools():
            mcp_tool = self._tool_to_mcp(tool)
            tools.append(mcp_tool.model_dump())

        return self._create_response(msg_id, {"tools": tools})

    async def _handle_call_tool(self, msg_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call tool request.

        Args:
            msg_id: Message ID
            params: Tool call parameters

        Returns:
            Tool call response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if not tool_name:
            return self._create_error(msg_id, -32602, "Missing tool name")

        try:
            # Create minimal context for tool execution
            context: Dict[str, Any] = {}
            result = await self.tool_registry.execute(tool_name, context, **tool_args)

            # Return standard MCP format per modelcontextprotocol.io specification:
            # {"content": [{"type": "text", "text": "..."}], "isError": false}
            if result.success:
                output_text = str(result.output) if result.output is not None else ""
                response_data = {
                    "content": [{"type": "text", "text": output_text}],
                    "isError": False,
                }
            else:
                error_text = str(result.error) if result.error else "Unknown error"
                response_data = {
                    "content": [{"type": "text", "text": error_text}],
                    "isError": True,
                }

            return self._create_response(msg_id, response_data)

        except Exception as e:
            return self._create_error(msg_id, -32603, f"Tool execution error: {str(e)}")

    async def _handle_list_resources(self, msg_id: str) -> Dict[str, Any]:
        """Handle list resources request.

        Args:
            msg_id: Message ID

        Returns:
            Resources list response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        resources = [r.model_dump() for r in self.resources]
        return self._create_response(msg_id, {"resources": resources})

    async def _handle_read_resource(self, msg_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read resource request.

        Args:
            msg_id: Message ID
            params: Request parameters

        Returns:
            Resource content response
        """
        import asyncio

        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        uri = params.get("uri")
        if not uri:
            return self._create_error(msg_id, -32602, "Missing resource URI")

        # Find resource
        resource = next((r for r in self.resources if r.uri == uri), None)
        if not resource:
            return self._create_error(msg_id, -32001, f"Resource not found: {uri}")

        # For file:// URIs, read the file
        if uri.startswith("file://"):
            try:
                from pathlib import Path

                file_path = uri.replace("file://", "")
                file_obj = Path(file_path)

                # Use run_in_executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(None, file_obj.read_text)

                resource_content = MCPResourceContent(
                    uri=uri,
                    mime_type=resource.mime_type or "text/plain",
                    content=content,
                    metadata=resource.metadata,
                )

                return self._create_response(msg_id, resource_content.model_dump())

            except Exception as e:
                return self._create_error(msg_id, -32603, f"Error reading resource: {str(e)}")

        return self._create_error(msg_id, -32001, "Resource type not supported")

    def _create_response(self, msg_id: str, result: Any) -> Dict[str, Any]:
        """Create success response.

        Args:
            msg_id: Message ID
            result: Response result

        Returns:
            Response message
        """
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _create_error(self, msg_id: Optional[str], code: int, message: str) -> Dict[str, Any]:
        """Create error response.

        Args:
            msg_id: Message ID
            code: Error code
            message: Error message

        Returns:
            Error response message
        """
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

    async def _setup_async_stdio(
        self,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Set up async stdin/stdout streams.

        Returns:
            Tuple of (reader, writer) for async I/O.

        Raises:
            OSError: If unable to set up async stdio streams.
        """
        loop = asyncio.get_running_loop()

        # Set up async stdin reader
        reader = asyncio.StreamReader(loop=loop)
        protocol = asyncio.StreamReaderProtocol(reader, loop=loop)

        try:
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        except (OSError, ValueError) as e:
            raise OSError(f"Failed to set up async stdin reader: {e}") from e

        # Set up async stdout writer
        try:
            writer_transport, writer_protocol = await loop.connect_write_pipe(
                lambda: asyncio.streams.FlowControlMixin(loop=loop),
                sys.stdout,
            )
            writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)
        except (OSError, ValueError) as e:
            raise OSError(f"Failed to set up async stdout writer: {e}") from e

        # Store references for cleanup
        self._reader = reader
        self._writer_transport = writer_transport

        return reader, writer

    async def _write_response(self, writer: asyncio.StreamWriter, response: Dict[str, Any]) -> None:
        """Write a JSON response to the output stream.

        Args:
            writer: Async stream writer
            response: Response dictionary to serialize and write
        """
        try:
            response_json = json.dumps(response) + "\n"
            writer.write(response_json.encode("utf-8"))
            await writer.drain()
        except (ConnectionError, BrokenPipeError) as e:
            print(f"Error writing response: {e}", file=sys.stderr)
            raise

    async def start_stdio_server(self) -> None:
        """Start MCP server on stdio (for MCP clients).

        This allows the server to communicate via stdin/stdout,
        which is the standard MCP transport mechanism.

        Uses native async I/O with asyncio.StreamReader/StreamWriter
        to avoid blocking the event loop.
        """
        print("MCP Server started on stdio", file=sys.stderr)
        print(f"Server: {self.name} v{self.version}", file=sys.stderr)
        print("Waiting for messages...", file=sys.stderr)

        try:
            reader, writer = await self._setup_async_stdio()
        except OSError as e:
            print(f"Failed to initialize async stdio: {e}", file=sys.stderr)
            # Fallback to executor-based I/O for compatibility
            await self._start_stdio_server_fallback()
            return

        self._running = True

        try:
            while self._running:
                try:
                    # Read line with timeout using native async readline
                    try:
                        line = await asyncio.wait_for(
                            reader.readline(),
                            timeout=300.0,  # 5 minute timeout for server idle
                        )
                    except asyncio.TimeoutError:
                        # No message received, continue waiting
                        continue

                    if not line:
                        # EOF reached (stdin closed)
                        break

                    # Decode and parse message
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    message = json.loads(line_str)
                    response = await self.handle_message(message)

                    # Write response using native async write
                    await self._write_response(writer, response)

                except json.JSONDecodeError as e:
                    error_response = self._create_error(None, -32700, f"Parse error: {str(e)}")
                    await self._write_response(writer, error_response)

                except (ConnectionError, BrokenPipeError):
                    # Connection lost, stop server
                    break

                except Exception as e:
                    error_response = self._create_error(None, -32603, f"Internal error: {str(e)}")
                    try:
                        await self._write_response(writer, error_response)
                    except (ConnectionError, BrokenPipeError):
                        break

        finally:
            # Clean up resources
            self._cleanup_stdio()

    async def _start_stdio_server_fallback(self) -> None:
        """Fallback stdio server using executor-based I/O.

        Used when native async stdio setup fails (e.g., on some platforms
        or when stdin/stdout are redirected in incompatible ways).
        """
        print("Using fallback executor-based I/O", file=sys.stderr)

        loop = asyncio.get_running_loop()
        self._running = True

        while self._running:
            try:
                # Read JSON-RPC message from stdin using run_in_executor
                try:
                    line = await asyncio.wait_for(
                        loop.run_in_executor(None, sys.stdin.readline),
                        timeout=300.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not line:
                    break

                message = json.loads(line)
                response = await self.handle_message(message)

                # Write response to stdout
                response_json = json.dumps(response)
                await loop.run_in_executor(None, lambda: print(response_json, flush=True))

            except json.JSONDecodeError as e:
                error_response = self._create_error(None, -32700, f"Parse error: {str(e)}")
                await loop.run_in_executor(
                    None, lambda r=error_response: print(json.dumps(r), flush=True)
                )

            except Exception as e:
                error_response = self._create_error(None, -32603, f"Internal error: {str(e)}")
                await loop.run_in_executor(
                    None, lambda r=error_response: print(json.dumps(r), flush=True)
                )

    def _cleanup_stdio(self) -> None:
        """Clean up stdio stream resources."""
        if self._writer_transport is not None:
            try:
                self._writer_transport.close()
            except Exception:
                pass
            self._writer_transport = None
        self._reader = None

    def stop(self) -> None:
        """Stop the stdio server gracefully."""
        self._running = False
        self._cleanup_stdio()

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information.

        Returns:
            Server info dictionary
        """
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.info.capabilities.model_dump(),
            "tools_count": len(self.tool_registry.list_tools()),
            "resources_count": len(self.resources),
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in MCP format.

        Returns:
            List of MCP tool definitions as dictionaries
        """
        tools = []
        for tool in self.tool_registry.list_tools():
            mcp_tool = self._tool_to_mcp(tool)
            tools.append(mcp_tool.model_dump())
        return tools

    @classmethod
    def create_with_default_tools(cls, name: str = "Victor MCP Server") -> "MCPServer":
        """Create MCP server with Victor's default tools.

        Args:
            name: Server name

        Returns:
            Configured MCPServer instance
        """
        from victor.config.settings import Settings

        # Create minimal orchestrator to get tools
        Settings()
        registry = ToolRegistry()

        # We need to register tools manually without full orchestrator
        # This is a simplified version for MCP exposure
        server = cls(name=name, tool_registry=registry)
        return server


def create_mcp_server_from_orchestrator(
    orchestrator: "AgentOrchestrator",
    name: str = "Victor MCP Server",
) -> MCPServer:
    """Create MCP server from an existing orchestrator.

    This allows exposing the orchestrator's registered tools via MCP.

    Args:
        orchestrator: AgentOrchestrator instance
        name: Server name

    Returns:
        Configured MCPServer
    """
    return MCPServer(
        name=name,
        version="1.0.0",
        tool_registry=orchestrator.tools,
    )


async def run_mcp_server_stdio() -> None:
    """Run MCP server in stdio mode.

    This is the main entry point for running Victor as an MCP server.
    Can be invoked via: python -m victor.mcp.server
    """
    import importlib
    import inspect
    import os
    import sys

    from victor.config.settings import Settings
    from victor.tools.base import BaseTool
    from victor.tools.registry import ToolRegistry

    # Initialize settings
    Settings()
    registry = ToolRegistry()

    # Dynamic tool discovery (same as orchestrator)
    tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
    excluded_files = {
        "__init__.py",
        "base.py",
        "decorators.py",
        "semantic_selector.py",
        "common.py",
    }
    registered_count = 0

    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and filename not in excluded_files:
            module_name = f"victor.tools.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _name, obj in inspect.getmembers(module):
                    # Register @tool decorated functions
                    if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                        registry.register(obj)
                        registered_count += 1
                    # Register BaseTool class instances
                    elif (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and hasattr(obj, "name")
                    ):
                        try:
                            tool_instance = obj()
                            registry.register(tool_instance)
                            registered_count += 1
                        except Exception:
                            pass  # Skip tools that need special initialization
            except Exception as e:
                print(f"Warning: Could not load {module_name}: {e}", file=sys.stderr)

    server = MCPServer(
        name="Victor MCP Server",
        version="1.0.0",
        tool_registry=registry,
    )

    print(f"Starting MCP server with {registered_count} tools", file=sys.stderr)
    await server.start_stdio_server()


def main() -> None:
    """CLI entry point for MCP server."""
    run_sync(run_mcp_server_stdio())


if __name__ == "__main__":
    main()

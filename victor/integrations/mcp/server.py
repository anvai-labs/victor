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
import inspect
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from victor.integrations.mcp.context import MCPToolContext
    from victor.tools.registry import ToolRegistry

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
)
from victor.tools.base import BaseTool
from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

GRAPH_INDEX_RESOURCE_URI = "victor://graph/index-status"
GRAPH_INDEX_RESOURCE_NAME = "Codegraph index status"


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
        exec_context: Optional["MCPToolContext"] = None,
        admission_refusal: Optional[str] = None,
    ):
        """Initialize MCP server.

        Args:
            name: Server name
            version: Server version
            tool_registry: Victor's tool registry
            exec_context: Launch-time sandbox admission (FEP-0049-followup
                B2). When present, every tools/call injects the admitted
                policy snapshot into the execution context; when ``None``
                the historical empty context is preserved (backward compat).
            admission_refusal: Set when launch-time admission FAILED: the
                server still starts, but every tools/call returns an isError
                refusal carrying this text (fail-closed, zero execution).
        """
        self.name = name
        self.version = version
        self.tool_registry = tool_registry or ToolRegistry()
        self.exec_context = exec_context
        self.admission_refusal = admission_refusal
        self._refusal_count = 0

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
        self._resource_read_handlers: Dict[str, Callable[[], Any]] = {}
        self._running = False  # For graceful shutdown of stdio server
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer_transport: Optional[asyncio.WriteTransport] = None

    def register_resource(
        self, resource: MCPResource, read_handler: Optional[Callable[[], Any]] = None
    ) -> None:
        """Register a resource with the MCP server.

        Args:
            resource: Resource to register
            read_handler: Optional callable returning the resource content
                (a JSON-serializable object or a string). When provided,
                reads of this URI are served by the handler instead of the
                ``file://`` path.
        """
        self.resources.append(resource)
        if read_handler is not None:
            self._resource_read_handlers[resource.uri] = read_handler

    def _tool_to_mcp(self, tool: BaseTool) -> MCPTool:
        """Convert Victor tool to MCP tool definition.

        Delegates to ``tool.to_json_schema()`` — the ONE schema source
        (Code Mode assessment: schema consolidation). Falls back to
        ``tool.parameters`` directly for the legacy List[ToolParameter]
        format that to_json_schema may not fully convert.
        """
        type_map = {
            "string": MCPParameterType.STRING,
            "number": MCPParameterType.NUMBER,
            "integer": MCPParameterType.NUMBER,
            "boolean": MCPParameterType.BOOLEAN,
            "object": MCPParameterType.OBJECT,
            "array": MCPParameterType.ARRAY,
        }

        schema = tool.to_json_schema()
        func = schema.get("function", schema)
        params_schema = func.get("parameters", {})

        mcp_params = []
        if isinstance(params_schema, dict) and params_schema.get("properties"):
            properties = params_schema.get("properties", {})
            required_set = set(params_schema.get("required", []))
            for name, prop in properties.items():
                if not isinstance(prop, dict):
                    mcp_params.append(
                        MCPParameter(
                            name=name,
                            type=MCPParameterType.STRING,
                            description="",
                            required=name in required_set,
                        )
                    )
                    continue
                prop_type = prop.get("type", "string")
                if isinstance(prop_type, list):
                    prop_type = prop_type[0] if prop_type else "string"
                mcp_type = type_map.get(prop_type.lower(), MCPParameterType.STRING)
                mcp_params.append(
                    MCPParameter(
                        name=name,
                        type=mcp_type,
                        description=prop.get("description", ""),
                        required=name in required_set,
                    )
                )
        elif isinstance(tool.parameters, list):
            for param in tool.parameters:
                mcp_type = type_map.get(param.type.lower(), MCPParameterType.STRING)
                mcp_params.append(
                    MCPParameter(
                        name=param.name,
                        type=mcp_type,
                        description=param.description,
                        required=param.required,
                    )
                )

        return MCPTool(
            name=tool.name,
            description=func.get("description", tool.description),
            parameters=mcp_params,
        )

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming MCP message.

        Notifications (id-less messages) are processed for state effects
        and return ``None`` — JSON-RPC spec: notifications are never
        answered, even for errors. Transports decide what ``None`` means
        (stdio: no write; HTTP: 202 Accepted).

        Args:
            message: MCP message dictionary

        Returns:
            Response message dictionary, or ``None`` for notifications.
        """
        is_notification = "id" not in message or message["id"] is None
        try:
            mcp_msg = MCPMessage(**message)

            if mcp_msg.method == MCPMessageType.INITIALIZE:
                msg_id = mcp_msg.id or str(uuid.uuid4())
                return await self._handle_initialize(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.LIST_TOOLS:
                if is_notification:
                    return None
                return await self._handle_list_tools(mcp_msg.id)

            elif mcp_msg.method == MCPMessageType.CALL_TOOL:
                if is_notification:
                    return None
                return await self._handle_call_tool(mcp_msg.id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.LIST_RESOURCES:
                if is_notification:
                    return None
                return await self._handle_list_resources(mcp_msg.id)

            elif mcp_msg.method == MCPMessageType.READ_RESOURCE:
                if is_notification:
                    return None
                return await self._handle_read_resource(mcp_msg.id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.PING:
                if is_notification:
                    return None
                return self._create_response(mcp_msg.id, {"pong": True})

            else:
                # Notifications are never answered (JSON-RPC spec) — even
                # unknown methods get no response, just a parse error if
                # the notification carries a valid id.
                if is_notification:
                    return None
                return self._create_error(None, -32601, f"Method not found: {mcp_msg.method}")

        except Exception as e:
            if is_notification:
                return None
            return self._create_error(None, -32700, f"Parse error: {str(e)}")

    def for_session(self, session_id: str) -> "MCPServer":
        """Create a per-session view sharing immutable state.

        Shares: tool_registry, exec_context, admission_refusal, resources,
        _resource_read_handlers (all registered at launch, read-only after).
        Copies (per-session): ``initialized = False``, ``_refusal_count = 0``,
        and the stdio transport fields stay only on the root instance.

        Concurrency: ``tool_registry.execute`` is already invoked concurrently
        by the agent loop; MCP tool calls across sessions hit the same
        registry — no new lock needed. ``resources`` list mutation after
        launch would race; ``register_resource`` is launch-time only.
        """
        view = MCPServer(
            name=self.name,
            version=self.version,
            tool_registry=self.tool_registry,
            exec_context=self.exec_context,
            admission_refusal=self.admission_refusal,
        )
        view.initialized = self.initialized
        view._refusal_count = self._refusal_count
        view.resources = self.resources
        view._resource_read_handlers = self._resource_read_handlers
        return view

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

        # Launch-time admission failure: refuse EVERY call without executing
        # (fail-closed). The server stays up so the client sees typed refusals.
        if self.admission_refusal is not None and self.exec_context is None:
            self._refusal_count += 1
            print(
                f"MCP tools/call '{tool_name}' refused "
                f"(#{self._refusal_count}): {self.admission_refusal}",
                file=sys.stderr,
            )
            return self._create_response(
                msg_id,
                {
                    "content": [{"type": "text", "text": self.admission_refusal}],
                    "isError": True,
                },
            )

        # Build the tool execution context from the admitted launch context.
        # An unenforceable required policy refuses here, before any execution.
        context: Dict[str, Any] = {}
        if self.exec_context is not None:
            if self.exec_context.backend_decision.refuse:
                refusal_text = self.exec_context.refusal_text()
                self._refusal_count += 1
                print(
                    f"MCP tools/call '{tool_name}' refused "
                    f"(#{self._refusal_count}): {refusal_text}",
                    file=sys.stderr,
                )
                return self._create_response(
                    msg_id,
                    {"content": [{"type": "text", "text": refusal_text}], "isError": True},
                )
            context = self.exec_context.to_exec_context()

        try:
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

        # Handler-backed resources (e.g. victor:// graph status) first
        handler = self._resource_read_handlers.get(uri)
        if handler is not None:
            try:
                content = handler()
                if inspect.iscoroutine(content):
                    content = await content
                payload = content if isinstance(content, str) else json.dumps(content, default=str)
                resource_content = MCPResourceContent(
                    uri=uri,
                    mime_type=resource.mime_type or "application/json",
                    content=payload,
                    metadata=resource.metadata,
                )
                return self._create_response(msg_id, resource_content.model_dump())
            except Exception as e:
                return self._create_error(msg_id, -32603, f"Error reading resource: {e}")

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

                    # Notifications (handle_message → None) are never
                    # written — JSON-RPC spec: no response for notifications.
                    if response is not None:
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


class ToolsProvider(Protocol):
    """Anything exposing a tool registry — the only surface the MCP server needs.

    Structural stand-in for the agent's runtime object, so this inbound
    integration surface never depends on (or names) it directly.
    """

    @property
    def tools(self) -> "ToolRegistry": ...


def create_mcp_server_from_orchestrator(
    orchestrator: "ToolsProvider",
    name: str = "Victor MCP Server",
) -> MCPServer:
    """Create MCP server from an existing orchestrator.

    This allows exposing the orchestrator's registered tools via MCP.

    Args:
        orchestrator: any object exposing a ``tools`` registry attribute
        name: Server name

    Returns:
        Configured MCPServer
    """
    return MCPServer(
        name=name,
        version="1.0.0",
        tool_registry=orchestrator.tools,
    )


def apply_graph_tool_flag(no_graph_tools: bool) -> None:
    """Enable graph tools for this MCP server process (FEP-0049-followup B3).

    Explicit ``victor mcp`` launch IS the consent; the orchestrator's own
    loop keeps its evidence-gated default (ADR-036), so this is process-
    scoped. Runtime overrides are the TOP-precedence flag mechanism
    (feature_flags.py: runtime > env > config > default), so:

    - the enable here beats even ``VICTOR_USE_GRAPH_QUERY_TOOL=false`` —
      a separate process observing the global default is unaffected;
    - ``--no-graph-tools`` must call ``disable_feature`` because the runtime
      override is the ONLY mechanism that beats ``...=true`` in the env.

    Args:
        no_graph_tools: True when the caller passed ``--no-graph-tools``.
    """
    from victor.core.feature_flags import FeatureFlag, disable_feature, enable_feature

    if no_graph_tools:
        disable_feature(FeatureFlag.USE_GRAPH_QUERY_TOOL)
    else:
        enable_feature(FeatureFlag.USE_GRAPH_QUERY_TOOL)


def register_graph_index_resource(server: "MCPServer", project_path: Path) -> None:
    """Expose codegraph index status as an MCP resource.

    The store is built LAZILY inside the read handler (bounded latency —
    a resource read never triggers an index build) and any failure degrades
    to a ``{"status": "unavailable"}`` payload, never an open refusal of the
    whole resource surface.
    """
    from victor.integrations.mcp.protocol import MCPResource

    async def _read_index_status() -> dict[str, Any]:
        try:
            from victor.storage.graph import create_graph_store

            store = create_graph_store(name="auto", project_path=project_path)
            raw = await store.stats()
            return {
                "status": "ok",
                "project": str(project_path),
                "store_path": raw.get("path"),
                "nodes": raw.get("nodes", 0),
                "edges": raw.get("edges", 0),
                "files_tracked": raw.get("indexed_files", 0),
            }
        except Exception as exc:
            logger.debug("Graph index status unavailable: %s", exc)
            return {
                "status": "unavailable",
                "project": str(project_path),
                "error": str(exc),
            }

    server.register_resource(
        MCPResource(
            uri=GRAPH_INDEX_RESOURCE_URI,
            name=GRAPH_INDEX_RESOURCE_NAME,
            description=(
                "Codegraph index status for the launch directory: store path, "
                "node count, files tracked. Read-only; never builds an index."
            ),
            mime_type="application/json",
        ),
        read_handler=_read_index_status,
    )


def _admit_or_refuse(workspace_root: Path) -> tuple[Optional["MCPToolContext"], Optional[str]]:
    """Admit the launch execution context; on failure return the refusal.

    The client layer builds the runtime policy snapshot (contextvar wins,
    else ``RunPolicySnapshot.from_settings``) and INJECTS it into the
    infrastructure-layer admission — the layering ratchet's
    "inject the dependency" guidance. The server still starts on admission
    failure — every tools/call then returns an isError refusal
    (fail-closed, zero execution).
    """
    from victor.integrations.mcp.context import (
        SANDBOX_ESCAPE_HATCH,
        admit_mcp_execution_context,
    )

    try:
        from victor.runtime.policy_snapshot import (
            RunPolicySnapshot,
            current_run_policy_snapshot,
        )

        from victor.config.settings import get_settings

        snapshot = current_run_policy_snapshot() or RunPolicySnapshot.from_settings(get_settings())
    except Exception as exc:
        refusal = (
            f"Sandbox policy could not be established ({exc}); refusing MCP "
            f"tool execution (fail-closed). {SANDBOX_ESCAPE_HATCH}".strip()
        )
        print(f"MCP launch admission failed: {refusal}", file=sys.stderr)
        return None, refusal
    try:
        return admit_mcp_execution_context(workspace_root, snapshot), None
    except Exception as exc:
        print(f"MCP launch admission failed: {exc}", file=sys.stderr)
        return None, str(exc)


async def run_mcp_server_stdio(no_graph_tools: bool = False) -> None:
    """Run MCP server in stdio mode.

    This is the main entry point for running Victor as an MCP server.
    Can be invoked via: python -m victor.mcp.server (or the
    ``victor-mcp-server`` console script).
    """
    import os

    from victor.integrations.mcp.assembly import create_mcp_server

    workspace_root = Path(os.getcwd())
    assembly = create_mcp_server(workspace_root, no_graph_tools=no_graph_tools)
    await assembly.server.start_stdio_server()


def main() -> None:
    """CLI entry point for the ``victor-mcp-server`` console script."""
    if "--no-graph-tools" in sys.argv:
        run_sync(run_mcp_server_stdio(no_graph_tools=True))
    else:
        run_sync(run_mcp_server_stdio())


if __name__ == "__main__":
    main()

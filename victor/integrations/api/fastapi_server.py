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

"""FastAPI-based HTTP API Server for Victor.

Provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
and external tool access. This is the modern replacement for the aiohttp server.

Features:
- REST API for chat, completions, code search
- WebSocket support for real-time updates
- Server-Sent Events (SSE) for streaming
- Optional rate limiting and API key authentication
- CORS support for browser-based clients
- Auto-generated OpenAPI documentation
"""

import asyncio
from functools import partial
import json
import logging
import secrets
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from victor.integrations.search_types import CodeSearchResult

from victor.integrations.api.change_tracker_ops import (
    apply_patch_request,
    change_history,
    create_patch_request,
    redo_last_change,
    undo_last_change,
)
from victor.integrations.api.event_bridge import EventBridge, EventBroadcaster
from victor.integrations.api.graph_export import (
    export_graph_schema,
    get_execution_state,
    WorkflowExecutionState,
)
from victor.integrations.api.router_plugins import load_fastapi_router_registrations
from victor.integrations.api.workflow_event_bridge import WorkflowEventBridge
from victor.core.events import get_observability_bus
from victor.observability.request_correlation import request_correlation_id
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.session_config import SessionConfig


# =============================================================================
# Pydantic Models for Request/Response validation
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.5.1"


class StatusResponse(BaseModel):
    """Server status response."""

    connected: bool
    mode: str
    provider: str
    model: str
    workspace: str
    capabilities: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request payload."""

    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    """Chat response payload.

    FEP-0029: when a turn durably pauses on a policy ASK, ``status`` is ``"awaiting_approval"`` and
    ``run_id`` + ``approval_request`` are set so the caller can approve later via ``/chat/resume``.
    """

    role: str = "assistant"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    status: str = "ok"
    run_id: Optional[str] = None
    approval_request: Optional[Dict[str, Any]] = None


class ResumeRequest(BaseModel):
    """Resume a durably-paused chat turn with a human approval decision (FEP-0029)."""

    run_id: str
    approved: bool
    response: Optional[str] = None
    responder: Optional[str] = None


def _new_chat_request_id() -> str:
    """Create a stable request identifier for chat API calls."""
    return f"chat_{uuid.uuid4().hex[:12]}"


class CompletionPosition(BaseModel):
    """Cursor position for completions."""

    line: int
    character: int


class CompletionRequest(BaseModel):
    """Code completion request with FIM (Fill-in-the-Middle) support."""

    prompt: str  # Code before cursor (prefix)
    suffix: Optional[str] = None  # Code after cursor (for FIM)
    file: Optional[str] = None
    language: Optional[str] = None
    position: Optional[CompletionPosition] = None
    context: Optional[str] = None  # Additional file context (imports, etc.)
    max_tokens: int = Field(default=128, ge=1, le=512)  # Limit completion length
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)  # Deterministic by default
    stop_sequences: Optional[List[str]] = None  # Custom stop sequences


class CompletionResponse(BaseModel):
    """Code completion response."""

    completions: List[str]
    error: Optional[str] = None
    latency_ms: Optional[float] = None  # Track performance


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str
    max_results: int = Field(default=10, ge=1, le=100)


class CodeSearchRequest(BaseModel):
    """Code search request payload."""

    query: str = Field(..., min_length=1, max_length=1000)
    regex: bool = False
    case_sensitive: bool = True
    file_pattern: str = Field(default="*", max_length=200)

    @field_validator("file_pattern")
    @classmethod
    def validate_file_pattern(cls, v: str) -> str:
        """Validate file pattern to prevent shell injection."""
        # Disallow shell metacharacters except glob patterns
        dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "<", ">", "\\"]
        for char in dangerous_chars:
            if char in v:
                raise ValueError(f"Invalid character '{char}' in file pattern")
        return v


class APISearchResult(BaseModel):
    """Single search result for API responses.

    For HTTP API search result serialization.
    Renamed from SearchResult to be semantically distinct from other search types.
    """

    file: str
    line: int
    content: str
    score: float

    @classmethod
    def from_code_result(cls, result: CodeSearchResult) -> "APISearchResult":
        return cls(
            file=result.file,
            line=result.line,
            content=result.content,
            score=result.score,
        )


# Backward compatibility alias
class SearchResponse(BaseModel):
    """Search response payload."""

    results: List[APISearchResult]
    error: Optional[str] = None


class SwitchModelRequest(BaseModel):
    """Model switch request."""

    provider: str
    model: str


class SwitchModeRequest(BaseModel):
    """Mode switch request."""

    mode: str


class SwitchProfileRequest(BaseModel):
    """Profile switch request."""

    profile: str


class PatchApplyRequest(BaseModel):
    """Patch apply request."""

    patch: str
    dry_run: bool = False


class PatchCreateRequest(BaseModel):
    """Patch create request."""

    file_path: str
    new_content: str


class GitCommitRequest(BaseModel):
    """Git commit request."""

    message: Optional[str] = Field(default=None, max_length=5000)
    use_ai: bool = False
    files: Optional[List[str]] = None

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate file paths to prevent path traversal."""
        if v is None:
            return v
        for file_path in v:
            # Disallow absolute paths and path traversal
            if file_path.startswith("/") or file_path.startswith("\\"):
                raise ValueError("Absolute paths not allowed")
            if ".." in file_path:
                raise ValueError("Path traversal not allowed")
            # Disallow shell metacharacters
            dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "<", ">"]
            for char in dangerous_chars:
                if char in file_path:
                    raise ValueError(f"Invalid character '{char}' in file path")
        return v


class MCPConnectRequest(BaseModel):
    """MCP connect request."""

    server: str
    endpoint: Optional[str] = None


class RLExploreRequest(BaseModel):
    """RL exploration rate request."""

    rate: float = Field(ge=0.0, le=1.0)


class RLStrategyRequest(BaseModel):
    """RL strategy request."""

    strategy: str


class ToolApprovalRequest(BaseModel):
    """Tool approval request."""

    approval_id: str
    approved: bool = False


class TerminalCommandRequest(BaseModel):
    """Terminal command execution request."""

    command: str = Field(..., min_length=1, max_length=5000)
    working_dir: Optional[str] = None
    timeout: int = Field(default=60, ge=1, le=300)
    require_approval: bool = True

    @field_validator("working_dir")
    @classmethod
    def validate_working_dir(cls, v: Optional[str]) -> Optional[str]:
        """Validate working_dir to prevent path traversal."""
        if v is None:
            return None
        from pathlib import Path

        if ".." in Path(v).parts:
            raise ValueError("working_dir must not contain '..' components")
        return str(Path(v).resolve())

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command to prevent obviously dangerous operations."""
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "> /dev/sd",
            "mkfs.",
            "dd if=",
            ":(){:|:&};:",
        ]
        cmd_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                raise ValueError(f"Command contains dangerous pattern: {pattern}")
        return v


class TerminalCommandResponse(BaseModel):
    """Terminal command execution response."""

    command_id: str
    command: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    output: Optional[str] = None
    exit_code: Optional[int] = None
    is_dangerous: bool = False
    requires_approval: bool = True


class _LegacyOrchestratorClientAdapter:
    """Small compatibility adapter for tests and legacy injected orchestrators."""

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator

    async def initialize(self) -> Any:
        return self._orchestrator

    async def chat(self, message: str) -> Any:
        from victor.runtime.chat_runtime import resolve_chat_runtime

        return await resolve_chat_runtime(self._orchestrator).chat(message)

    async def stream_chat(self, message: str) -> AsyncIterator[Any]:
        from victor.runtime.chat_runtime import resolve_chat_runtime

        async for chunk in resolve_chat_runtime(self._orchestrator).stream_chat(message):
            yield chunk

    async def close(self) -> None:
        shutdown = getattr(self._orchestrator, "graceful_shutdown", None)
        if shutdown is not None:
            await shutdown()


# =============================================================================
# FastAPI Server Class
# =============================================================================


def collect_route_paths(app: Any) -> set[str]:
    """Collect all concrete route paths from a FastAPI/Starlette app.

    Robust across Starlette versions. Newer Starlette stores lazily-included
    routers as ``_IncludedRouter`` wrapper objects in ``app.routes`` (resolved
    at request time) instead of eagerly flattening their sub-routes; a naive
    ``{route.path for route in app.routes}`` then misses every included path
    (and capability detection silently reports nothing). This recurses into
    included routers / mounts so it works on both old (flat) and new (lazy)
    Starlette.
    """
    paths: set[str] = set()
    seen: set[int] = set()

    def _walk(routes: Any) -> None:
        for route in routes or []:
            if id(route) in seen:
                continue
            seen.add(id(route))
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                paths.add(path)
            # Newer Starlette wraps included routers; the real router (with the
            # concrete sub-routes) is exposed as ``original_router``.
            included = getattr(route, "original_router", None)
            sub = getattr(included, "routes", None) if included is not None else None
            if sub is None:
                # Mounts / sub-apps expose nested routes directly.
                sub = getattr(route, "routes", None)
            if sub:
                _walk(sub)

    _walk(getattr(app, "routes", None))
    return paths


class VictorFastAPIServer:
    """FastAPI-based HTTP API Server for Victor."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        workspace_root: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        api_keys: Optional[Dict[str, str]] = None,
        enable_cors: bool = True,
        enable_hitl: bool = False,
        hitl_auth_token: Optional[str] = None,
        hitl_persistent: bool = True,
        enable_graphql: bool = True,
        session_config: Optional["SessionConfig"] = None,
    ):
        """Initialize the FastAPI server.

        Args:
            host: Host to bind to
            port: Port to listen on
            workspace_root: Root directory of the workspace
            rate_limit_rpm: Optional requests per minute limit (None = no limit)
            api_keys: Optional dict of {api_key: client_id} for authentication
            enable_cors: Enable CORS headers (default: True)
            enable_hitl: Enable HITL (Human-in-the-Loop) endpoints (default: False)
            hitl_auth_token: Optional auth token for HITL endpoints
            hitl_persistent: Use SQLite for persistent HITL storage (default: True)
            enable_graphql: Enable GraphQL endpoint at /graphql (default: True)
        """
        self.host = host
        self.port = port
        self.workspace_root = workspace_root or str(Path.cwd())
        self.rate_limit_rpm = rate_limit_rpm
        self.api_keys = api_keys or {}
        self.enable_cors = enable_cors
        self.enable_hitl = enable_hitl
        self.hitl_auth_token = hitl_auth_token
        self.hitl_persistent = hitl_persistent
        self._enable_graphql = enable_graphql

        from victor.config.settings import load_settings
        from victor.core.bootstrap import ensure_bootstrapped
        from victor.framework.session_config import SessionConfig
        from victor.framework.session_runner import (
            FrameworkSessionRunner,
            create_victor_client,
        )

        self._settings = load_settings()
        self._session_config = session_config or SessionConfig()
        self._container = ensure_bootstrapped(self._settings)
        self._session_runner = FrameworkSessionRunner(
            self._settings,
            self._session_config,
            client_factory=partial(create_victor_client, container=self._container),
        )

        self._orchestrator = None
        self._victor_client = None
        self._ws_clients: List[WebSocket] = []
        self._pending_tool_approvals: Dict[str, Dict[str, Any]] = {}
        self._hitl_store = None
        self._event_bridge: Optional[EventBridge] = None
        self._event_clients: List[WebSocket] = []
        self._workflow_event_bridge: Optional[WorkflowEventBridge] = None
        self._workflow_executions: Dict[str, Dict[str, Any]] = {}
        self._shutting_down = False

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="Victor API",
            description="AI Coding Assistant API for IDE integrations",
            version="0.5.1",
            lifespan=self._lifespan,
        )

        # Configure CORS with secure defaults
        # Allow localhost for development and VS Code webview origins
        if enable_cors:
            cors_origins = [
                "http://localhost:*",
                "http://127.0.0.1:*",
                "https://localhost:*",
                "https://127.0.0.1:*",
                "vscode-webview://*",
            ]
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_origin_regex=r"^(http://localhost:\d+|http://127\.0\.0\.1:\d+|vscode-webview://[a-z0-9-]+)$",
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=[
                    "Content-Type",
                    "Accept",
                    "Authorization",
                    "X-Requested-With",
                ],
            )

        # Setup routes
        self._setup_routes()
        self._setup_router_plugins()

        # Setup GraphQL endpoint if enabled
        if self._enable_graphql:
            try:
                from victor.integrations.api.graphql_schema import create_graphql_schema
                from strawberry.fastapi import GraphQLRouter

                schema = create_graphql_schema(self)
                graphql_router = GraphQLRouter(schema)
                self.app.include_router(graphql_router, prefix="/graphql")
                logger.info("GraphQL endpoint enabled at /graphql")
            except ImportError:
                logger.debug("strawberry-graphql not installed, GraphQL disabled")

        # Setup HITL routes if enabled
        if self.enable_hitl:
            self._setup_hitl_routes()

    async def _verify_api_key(self, request: Request) -> Optional[str]:
        """Verify API key if authentication is configured. Returns client_id or None."""
        if not self.api_keys:
            return None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            if api_key in self.api_keys:
                return self.api_keys[api_key]
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage server lifespan."""
        logger.info(f"Starting Victor FastAPI server on {self.host}:{self.port}")

        # Initialize EventBridge for real-time event streaming
        event_bus = get_observability_bus()
        self._event_bridge = EventBridge(event_bus)
        await self._event_bridge.async_start()
        logger.info("EventBridge started for real-time event streaming")

        # Initialize WorkflowEventBridge for workflow visualization
        self._workflow_event_bridge = WorkflowEventBridge(event_bus)
        await self._workflow_event_bridge.start()
        logger.info("WorkflowEventBridge started for workflow visualization")

        yield

        # Cleanup
        if self._event_bridge:
            await self._event_bridge.async_stop()
        if self._workflow_event_bridge:
            await self._workflow_event_bridge.stop()
        if self._orchestrator:
            await self._orchestrator.graceful_shutdown()
        # Close WebSocket connections
        for ws in self._ws_clients + self._event_clients:
            try:
                await ws.close()
            except Exception:
                pass
        logger.info("Victor FastAPI server shutdown complete")

    def _setup_routes(self) -> None:
        """Set up API routes by including extracted APIRouter modules."""
        from victor.integrations.api.routes import create_all_routers

        for router in create_all_routers(self):
            self.app.include_router(router)

    def _setup_router_plugins(self) -> None:
        """Load optional FastAPI routers from vertical packages.

        Routers are discovered via `victor.api_routers` entry points.
        """
        registrations = load_fastapi_router_registrations(workspace_root=self.workspace_root)
        for registration in registrations:
            try:
                self.app.include_router(registration.router, prefix=registration.prefix)
            except Exception as exc:
                logger.warning(
                    "Failed to include router from entry point '%s' (%s): %s",
                    registration.entry_point_name,
                    registration.entry_point_value,
                    exc,
                )

    def _detect_capabilities(self) -> List[str]:
        """Detect capabilities from currently mounted API routes."""
        route_paths = collect_route_paths(self.app)
        capabilities: set[str] = set()

        if "/chat" in route_paths:
            capabilities.add("chat")
        if "/completions" in route_paths:
            capabilities.add("completions")
        if "/search/semantic" in route_paths or "/search/code" in route_paths:
            capabilities.add("search")
        if any(path.startswith("/lsp/") for path in route_paths):
            capabilities.add("lsp")
        if "/agents/start" in route_paths:
            capabilities.add("agents")
        if "/plans" in route_paths:
            capabilities.add("plans")
        if "/workflows/execute" in route_paths:
            capabilities.add("workflows")
        if "/teams" in route_paths:
            capabilities.add("teams")
        if "/tools" in route_paths:
            capabilities.add("tools")
        if "/mcp/servers" in route_paths:
            capabilities.add("mcp")

        return sorted(capabilities)

    # =========================================================================
    # HITL (Human-in-the-Loop) Routes
    # =========================================================================

    def _setup_hitl_routes(self) -> None:
        """Set up HITL (Human-in-the-Loop) endpoints for workflow approvals.

        This enables the unified server to handle both VS Code/IDE requests
        and workflow approval requests on the same port.

        Endpoints added:
            GET /hitl/requests - List pending approval requests
            GET /hitl/requests/{id} - Get specific request
            POST /hitl/respond/{id} - Submit approval/rejection
            GET /hitl/ui - Web-based approval UI
            GET /hitl/history - Approval history (SQLite only)
        """
        try:
            from victor.workflows.hitl_api import (
                HITLStore,
                SQLiteHITLStore,
                create_hitl_router,
            )

            # Create HITL store - SQLite for persistence, in-memory otherwise
            if self.hitl_persistent:
                self._hitl_store = SQLiteHITLStore()
                logger.info(f"HITL using SQLite store: {self._hitl_store.db_path}")
            else:
                self._hitl_store = HITLStore()
                logger.info("HITL using in-memory store")

            # Create and include the HITL router
            hitl_router = create_hitl_router(
                store=self._hitl_store,
                require_auth=bool(self.hitl_auth_token),
                auth_token=self.hitl_auth_token,
            )
            self.app.include_router(hitl_router, prefix="/hitl")

            logger.info("HITL endpoints enabled at /hitl/*")

        except ImportError as e:
            logger.warning(f"HITL endpoints not available: {e}")

    def get_hitl_store(self) -> Optional[Any]:
        """Get the HITL store for this server instance.

        Returns:
            HITLStore if HITL is enabled, None otherwise
        """
        return self._hitl_store

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_orchestrator(self) -> Any:
        """Get or create the orchestrator."""
        if self._orchestrator is None:
            client = await self._get_victor_client()
            agent = await client.initialize()
            if hasattr(agent, "get_orchestrator"):
                self._orchestrator = agent.get_orchestrator()
            else:
                self._orchestrator = agent

        return self._orchestrator

    async def _get_victor_client(self) -> Any:
        """Get or create the framework-managed client for API conversation access."""
        if self._victor_client is None:
            if self._orchestrator is not None:
                self._victor_client = _LegacyOrchestratorClientAdapter(self._orchestrator)
            else:
                self._victor_client = self._session_runner.create_client(self._session_config)
                await self._session_runner.initialize_client(self._victor_client)

        return self._victor_client

    async def update_session_config(self, session_config: "SessionConfig") -> None:
        """Replace runtime session config and reset cached runtime owners."""
        from victor.framework.session_runner import create_victor_client

        self._session_config = session_config
        self._session_runner = type(self._session_runner)(
            self._settings,
            self._session_config,
            client_factory=partial(create_victor_client, container=self._container),
        )
        if self._victor_client is not None:
            try:
                await self._victor_client.close()
            except Exception:
                logger.debug("Failed to close VictorClient during config switch", exc_info=True)
            self._victor_client = None
        if self._orchestrator is not None:
            try:
                await self._orchestrator.graceful_shutdown()
            except Exception:
                logger.debug(
                    "Failed to shutdown orchestrator during config switch",
                    exc_info=True,
                )
            self._orchestrator = None

    async def reset_conversation(self) -> None:
        """Reset conversation history using VictorClient (service layer)."""
        client = await self._get_victor_client()
        await client.reset_conversation()

    async def get_conversation_messages(
        self, limit: Optional[int] = None, role: Optional[str] = None
    ) -> List[Any]:
        """Get conversation messages using VictorClient (service layer).

        Args:
            limit: Maximum number of messages to return
            role: Optional filter by message role

        Returns:
            List of message objects
        """
        client = await self._get_victor_client()
        return await client.get_messages(limit=limit, role=role)

    async def _record_rl_feedback(self) -> None:
        """Record RL feedback for the current session."""
        if self._orchestrator is None:
            return

        try:
            from victor.framework.rl.base import RLOutcome
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return

            provider = self._orchestrator.provider
            if provider is None:
                return

            msg_count = 0
            if hasattr(self._orchestrator, "message_count"):
                msg_count = self._orchestrator.message_count
            elif hasattr(self._orchestrator, "get_messages"):
                messages = self._orchestrator.get_messages()
                msg_count = len(messages) if messages else 0

            if msg_count == 0:
                return

            metrics = {}
            if hasattr(self._orchestrator, "get_session_metrics"):
                metrics = self._orchestrator.get_session_metrics() or {}

            # Compute quality score based on session metrics
            latency = metrics.get("total_latency", 0)
            token_count = metrics.get("total_tokens", 0)
            tool_calls = metrics.get("tool_calls", 0)

            # Base quality: 1.0 for successful session
            quality_score = 1.0

            # Latency penalty: -0.1 per 30s over 30s threshold (max -0.5)
            if latency > 30:
                quality_score -= min(0.1 * (latency - 30) / 30, 0.5)

            # Tool usage bonus: +0.05 per tool (max +0.2)
            if tool_calls > 0:
                quality_score += min(0.05 * tool_calls, 0.2)

            outcome = RLOutcome(
                provider=provider.name,
                model=getattr(provider, "model", "unknown"),
                task_type="chat",  # API sessions are general chat
                success=True,
                quality_score=max(0.0, min(1.0, quality_score)),
                metadata={
                    "session_id": str(uuid.uuid4())[:8],
                    "latency_seconds": latency,
                    "token_count": token_count,
                    "tool_calls_made": tool_calls,
                    "message_count": msg_count,
                },
                vertical="coding",
            )

            coordinator.record_outcome("model_selector", outcome, "coding")

            # Get updated Q-value for logging
            rankings = learner.get_provider_rankings()
            provider_ranking = next((r for r in rankings if r["provider"] == provider.name), None)
            new_q = provider_ranking["q_value"] if provider_ranking else 0.0

            logger.info(
                f"RL API session feedback: {provider.name} "
                f"({msg_count} messages, {tool_calls} tools) → Q={new_q:.3f}"
            )

        except Exception as e:
            logger.debug(f"RL feedback recording skipped: {e}")

    async def _handle_ws_message(self, ws: WebSocket, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
            messages = data.get("messages", [])
            if not messages:
                await ws.send_json({"type": "error", "message": "No messages"})
                return

            client = await self._get_victor_client()

            # FEP-0020 attribution join: the client_id resolved by the WebSocket
            # `auth` message is bound for this turn so downstream cost/usage
            # records carry the authenticated subject.
            from victor.core.context import bind_attribution

            ws_client_id = getattr(getattr(ws, "state", None), "client_id", None)

            try:
                with bind_attribution(subject_id=ws_client_id):
                    async for chunk in client.stream_chat(messages[-1].get("content", "")):
                        content = getattr(chunk, "content", "")
                        metadata = getattr(chunk, "metadata", {}) or {}
                        if content:
                            await ws.send_json({"type": "content", "content": content})
                        elif "tool_start" in metadata:
                            await ws.send_json(
                                {"type": "tool_call", "tool_call": metadata["tool_start"]}
                            )
                        elif "tool_result" in metadata:
                            await ws.send_json(
                                {
                                    "type": "tool_result",
                                    "tool_result": metadata["tool_result"],
                                }
                            )

                await ws.send_json({"type": "done"})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

        elif msg_type == "auth":
            api_key = data.get("api_key", "")
            if api_key and self.api_keys:
                matched = next(
                    (k for k in self.api_keys if secrets.compare_digest(k, api_key)),
                    None,
                )
                if matched:
                    if not hasattr(ws, "state"):
                        ws.state = type("State", (), {})()
                    ws.state.authenticated = True
                    ws.state.client_id = self.api_keys[matched]
                    logger.debug("WebSocket client authenticated via message")
                    await ws.send_json({"type": "auth_success"})
                else:
                    await ws.send_json({"type": "auth_failed", "message": "Invalid API key"})
            elif not self.api_keys:
                await ws.send_json({"type": "auth_success"})
            else:
                await ws.send_json({"type": "auth_failed", "message": "No API key provided"})

        elif msg_type == "subscribe":
            channel = data.get("channel", "")
            await ws.send_json({"type": "subscribed", "channel": channel})

    async def _broadcast_ws(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSocket clients."""
        for ws in self._ws_clients:
            try:
                await ws.send_json(message)
            except Exception:
                pass

    async def _broadcast_agent_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast agent events to all connected WebSocket clients.

        Used by BackgroundAgentManager to send real-time updates.
        """
        message = {
            "type": "agent_event",
            "event": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        await self._broadcast_ws(message)

    def _format_messages_markdown(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as markdown."""
        lines = ["# Conversation Export\n"]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"## {role}\n")
            lines.append(f"{content}\n")
        return "\n".join(lines)

    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool based on its name."""
        categories = {
            "filesystem": ["read", "write", "ls", "edit", "glob", "overview"],
            "search": ["search", "grep", "semantic_code_search", "code_search"],
            "git": ["git", "commit_msg", "conflicts", "pr", "merge"],
            "shell": ["shell", "bash", "sandbox"],
            "refactor": ["extract", "inline", "organize_imports", "rename", "refactor"],
            "code_intelligence": ["symbol", "refs", "lsp"],
            "web": ["fetch", "http", "web", "web_search", "web_fetch"],
            "docker": ["docker"],
            "database": ["database", "db"],
            "testing": ["test", "pytest"],
            "documentation": ["docs", "documentation", "docs_coverage"],
            "analysis": ["code_review", "metrics", "scan", "audit", "iac"],
            "infrastructure": ["cicd", "pipeline", "dependency"],
            "batch": ["batch"],
            "cache": ["cache"],
            "mcp": ["mcp"],
            "workflow": ["workflow", "scaffold"],
            "patch": ["patch"],
        }

        for category, keywords in categories.items():
            if any(kw in tool_name.lower() for kw in keywords):
                return category.replace("_", " ").title()

        return "Other"

    def _is_dangerous_tool(self, tool_name: str) -> bool:
        """Check if a tool is considered dangerous."""
        dangerous_tools = {
            "shell",
            "bash",
            "sandbox",
            "docker",
            "write",
            "edit",
            "delete",
            "rm",
            "database",
        }
        return tool_name.lower() in dangerous_tools

    def run(self) -> None:
        """Run the server synchronously."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    async def start_async(self) -> "VictorFastAPIServer":
        """Start the server asynchronously and return self for cleanup."""
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        asyncio.create_task(self._server.serve())
        logger.info(f"Victor FastAPI server running on {self.host}:{self.port}")
        return self

    async def shutdown(self) -> None:
        """Shutdown the server gracefully.

        This uses uvicorn's built-in shutdown mechanism to properly terminate
        the server without affecting the asyncio event loop or causing system issues.
        """
        if hasattr(self, "_server") and self._server is not None:
            # Signal the server to exit gracefully
            self._server.should_exit = True

            # Use uvicorn's built-in shutdown if available
            if hasattr(self._server, "shutdown"):
                try:
                    await asyncio.wait_for(self._server.shutdown(), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Timeout is acceptable - server will exit on its own
                    pass
                except Exception as e:
                    logger.warning(f"Server shutdown encountered error: {e}")

            # Small delay to ensure port is released (not for event loop stability)
            await asyncio.sleep(0.1)

    async def _delayed_shutdown(self) -> None:
        """Delayed shutdown helper for /shutdown endpoint.

        This method is called asynchronously after the shutdown response is sent,
        allowing the HTTP response to complete before the server actually shuts down.
        """
        # Check if shutdown is already in progress
        if self._shutting_down:
            return

        # Small delay to ensure response is sent
        await asyncio.sleep(0.5)

        # Perform actual shutdown
        if hasattr(self, "_server") and self._server is not None:
            if hasattr(self._server, "shutdown"):
                try:
                    await asyncio.wait_for(self._server.shutdown(), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                    logger.warning(f"Delayed shutdown encountered error: {e}")


def create_fastapi_app(workspace_root: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application."""
    server = VictorFastAPIServer(workspace_root=workspace_root)
    return server.app

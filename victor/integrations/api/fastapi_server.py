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
import json
from collections import deque
from dataclasses import dataclass
from victor import __version__
from functools import partial
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute, APIWebSocketRoute
from pydantic import BaseModel, Field, field_validator

from victor.integrations.search_types import CodeSearchResult

from victor.integrations.api.event_bridge import EventBridge
from victor.integrations.api.operator_posture import is_multi_principal, refusal_payload
from victor.integrations.api.router_plugins import load_fastapi_router_registrations
from victor.integrations.api.workflow_event_bridge import WorkflowEventBridge
from victor.core.events import get_observability_bus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.session_config import SessionConfig

# =============================================================================
# Bounded per-client WebSocket delivery (VAS-06/10, TD-03)
# =============================================================================

# Defaults when the server settings group is unavailable (tests construct the
# server directly; production reads ServerSettings).
DEFAULT_WS_DELIVERY_QUEUE_SIZE = 256
DEFAULT_WS_SEND_DEADLINE_SECONDS = 10.0

# The only agent_event variants declared coalescible under queue pressure
# (TD-03: "coalesce only DECLARED nonessential progress"). The default for
# any agent_event not listed here — including the terminal variants emitted
# by BackgroundAgentManager (agent_completed/agent_error/agent_cancelled),
# workflow/team completion, and any FUTURE event type — is ESSENTIAL: a
# closed progress allowlist cannot silently shed final state. Non-agent_event
# broadcasts (tool_approval_resolved, tool_cancelled, request/response
# frames) are essential unconditionally.
_NONESSENTIAL_AGENT_EVENTS = frozenset(
    {
        "agent_started",
        "agent_progress",
        "agent_running",
        "agent_tool_call",
        "agent_tool_result",
        "task_update",
        "workflow_started",
        "team_created",
        "team_started",
    }
)


def _is_nonessential_progress(message: Dict[str, Any]) -> bool:
    """True when *message* is declared-coalescible progress (TD-03).

    Only ``agent_event`` broadcasts whose ``event`` is in the declared
    progress allowlist may be shed from a full outbound queue; terminal,
    approval and unknown future state must survive every overflow policy.
    """
    if message.get("type") != "agent_event":
        return False
    return message.get("event") in _NONESSENTIAL_AGENT_EVENTS


class _ClientOutboundQueue:
    """Bounded FIFO for one client's outbound broadcast messages.

    A plain ``asyncio.Queue`` cannot express the TD-03 overflow policy (shed
    the oldest *nonessential* progress first; never drop terminal or approval
    state), so the queue keeps insertion order in a deque and overflow
    handling is explicit at the producer. The sender task awaits :meth:`get`.
    """

    __slots__ = ("_items", "_maxsize", "_wakeup")

    def __init__(self, maxsize: int) -> None:
        self._items: "deque[Dict[str, Any]]" = deque()
        self._maxsize = maxsize
        self._wakeup = asyncio.Event()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def qsize(self) -> int:
        return len(self._items)

    def full(self) -> bool:
        return len(self._items) >= self._maxsize

    def put(self, message: Dict[str, Any]) -> None:
        """Enqueue without awaiting. Callers apply the overflow policy first
        (see ``_enqueue_ws_delivery``) when :meth:`full`."""
        self._items.append(message)
        self._wakeup.set()

    def evict_oldest_nonessential(self) -> bool:
        """Remove the oldest declared-nonessential progress event.

        Eviction is by POSITION (``del``), not ``deque.remove`` by value:
        two value-equal dicts would otherwise make remove() ambiguous, and
        position is what "oldest" means. Returns True when something was
        shed (making room), False when the queue holds only essential state.
        """
        for index, item in enumerate(self._items):
            if _is_nonessential_progress(item):
                del self._items[index]
                return True
        return False

    def snapshot(self) -> List[Dict[str, Any]]:
        """Insertion-ordered copy (introspection/tests)."""
        return list(self._items)

    async def get(self) -> Dict[str, Any]:
        """Await the next message (sender-task drain)."""
        while True:
            if self._items:
                item = self._items.popleft()
                if not self._items:
                    self._wakeup.clear()
                return item
            await self._wakeup.wait()
            self._wakeup.clear()


@dataclass
class _WSClientDelivery:
    """Per-socket bounded delivery state (VAS-06/10, TD-03)."""

    ws: Any
    queue: _ClientOutboundQueue
    sender_task: Optional["asyncio.Task[None]"] = None
    # Coalescible progress events shed from this client's queue under
    # overflow — the per-client measure of retention loss.
    dropped_nonessential: int = 0


# =============================================================================
# Pydantic Models for Request/Response validation
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = __version__


class StatusResponse(BaseModel):
    """Server status response."""

    connected: bool
    mode: str
    provider: str
    model: str
    workspace: str
    capabilities: List[str] = Field(default_factory=list)
    sandbox_posture: str = "unknown"
    # Effective-posture surface (Codex finding 22): what the running server
    # actually enforces per advertised setting.
    effective_posture: Optional[Dict[str, Any]] = None


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
    resume_thread_id: Optional[str] = None


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
        allow_unauthenticated: Optional[bool] = None,
        single_operator_mode: Optional[bool] = None,
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
            allow_unauthenticated: Explicit opt-in for unauthenticated mode.
                When False (default) and no api_keys configured, raises ValueError.
                Set to True or env VICTOR_ALLOW_UNAUTHENTICATED=1 to allow.
            single_operator_mode: Explicit operator declaration that every
                configured API key belongs to ONE trusted operator. Default
                (None/False) is the SAFE posture: with more than one
                configured credential, the shared-state surfaces (chat
                conversation addressing, /conversation/export,
                /conversation/reset, /memory) refuse with the typed
                ``MultiPrincipalRefused`` payload, because they resolve one
                server-wide client/store with no owner in the lookup (Codex
                findings 3/4; ownership itself is the paused A4a+ design).
                Env alias: ``VICTOR_SINGLE_OPERATOR_MODE=1``.
        """
        self.host = host
        self.port = port
        self.workspace_root = workspace_root or str(Path.cwd())
        self.rate_limit_rpm = rate_limit_rpm
        self.api_keys = api_keys or {}
        # Wire the advertised singular server key at its actual owner (Codex
        # finding 22): ``settings.server.server_api_key`` is a credential for
        # the FastAPI auth gate, so a configured value joins the credential
        # map instead of being validated-then-ignored. Sources are tracked
        # for the effective-posture surface.
        from victor.config.settings import load_settings

        self._settings = load_settings()
        self._api_key_sources: List[str] = ["parameter"] if api_keys else []
        _server_settings = getattr(self._settings, "server", None)
        _configured_key = getattr(_server_settings, "server_api_key", None)
        _configured_key_value = (
            _configured_key.get_secret_value() if _configured_key is not None else None
        )
        if _configured_key_value:
            self._api_key_sources.append("settings.server.server_api_key")
            self.api_keys.setdefault(_configured_key_value, "server")
        env_single_operator = os.environ.get("VICTOR_SINGLE_OPERATOR_MODE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self.single_operator_mode = bool(single_operator_mode) or env_single_operator
        if is_multi_principal(self.api_keys, self.single_operator_mode):
            logger.warning(
                "Multi-principal posture: %d API credentials configured without a "
                "single-trusted-operator declaration. Shared-state surfaces (chat "
                "conversation addressing, /conversation/export, /conversation/reset, "
                "/memory) will refuse (MultiPrincipalRefused) — one shared client/store "
                "resolves them with no owner in the lookup. Full per-principal "
                "isolation is the paused VAS-06 / FOU-A4a design.",
                len(self.api_keys),
            )
        # Sticky principal that first claimed the shared conversation/memory
        # state under the multi-credential posture (see claim_shared_state).
        self._shared_state_principal: Optional[str] = None
        self.enable_cors = enable_cors
        self.enable_hitl = enable_hitl
        self.hitl_auth_token = hitl_auth_token
        self.hitl_persistent = hitl_persistent
        self._enable_graphql = enable_graphql

        # Security: require explicit opt-in for unauthenticated mode
        env_allow_unauth = os.environ.get("VICTOR_ALLOW_UNAUTHENTICATED", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if allow_unauthenticated is None:
            allow_unauthenticated = env_allow_unauth

        if not self.api_keys and not allow_unauthenticated:
            raise ValueError(
                "Server requires api_keys or explicit allow_unauthenticated=True. "
                "Running without authentication exposes all endpoints (chat, tools, "
                "shell execution, HITL approvals) to any network client. "
                "Set allow_unauthenticated=True or VICTOR_ALLOW_UNAUTHENTICATED=1 for local dev."
            )

        self._allow_unauthenticated = allow_unauthenticated

        from victor.core.bootstrap import ensure_bootstrapped
        from victor.framework.session_config import SessionConfig
        from victor.framework.session_runner import (
            FrameworkSessionRunner,
            create_victor_client,
        )

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
        # Bounded per-client delivery (VAS-06/10, TD-03): every /ws socket
        # registered by the route gets a bounded outbound queue drained by
        # its own sender task, so broadcast producers never await a slow
        # socket and one wedged client cannot stall every recipient.
        self._ws_deliveries: Dict[Any, "_WSClientDelivery"] = {}
        # Count of disconnect-on-essential-overflow events: a terminal or
        # approval broadcast that could not enqueue behind an all-essential
        # queue. The stream protocol has no resume cursor yet (VAS-10 owns
        # the replay/cursor contract); the gap counter plus the structured
        # disconnect log is the honest record until then.
        self._ws_delivery_gaps: int = 0
        server_settings = getattr(self._settings, "server", None)
        self._ws_delivery_queue_size = int(
            getattr(
                server_settings,
                "server_ws_delivery_queue_size",
                DEFAULT_WS_DELIVERY_QUEUE_SIZE,
            )
        )
        self._ws_send_deadline = float(
            getattr(
                server_settings,
                "server_ws_send_deadline_seconds",
                DEFAULT_WS_SEND_DEADLINE_SECONDS,
            )
        )
        # Inbound WebSocket payload cap (Codex finding 22: the advertised
        # ``server.server_max_message_bytes`` control is now enforced at its
        # actual owner — the /ws and /ws/events receive paths).
        self._ws_max_message_bytes = int(
            getattr(server_settings, "server_max_message_bytes", 0) or 0
        )
        # Seconds a /ws socket may sit unauthenticated on a keyed server
        # before the route closes it (VAS-05h bounded handshake).
        self._ws_auth_deadline: float = 30.0
        self._pending_tool_approvals: Dict[str, Dict[str, Any]] = {}
        self._hitl_store = None
        self._event_bridge: Optional[EventBridge] = None
        self._event_clients: List[WebSocket] = []
        self._workflow_event_bridge: Optional[WorkflowEventBridge] = None
        self._workflow_executions: Dict[str, Dict[str, Any]] = {}
        self._shutting_down = False

        # Create FastAPI app with lifespan. Auth is enforced per-route in
        # _setup_routes/_setup_router_plugins (co-design review U7-F1..F3:
        # auth was previously opt-in per handler — only /chat* verified, so
        # git/config/agents/terminal were open). An app-level dependency is
        # NOT usable: it also runs for websocket routes, where FastAPI's
        # solver cannot supply a Request parameter (it crashes the socket).
        # On keyed servers the docs surfaces are disabled — they disclose the
        # full route tree and add nothing for authenticated clients.
        self.app = FastAPI(
            title="Victor API",
            description="AI Coding Assistant API for IDE integrations",
            version=__version__,
            lifespan=self._lifespan,
            **({"docs_url": None, "redoc_url": None, "openapi_url": None} if self.api_keys else {}),
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
                # HTTP queries/mutations get the API-key dependency;
                # subscriptions (websocket) get the WS guard.
                self._include_with_auth(graphql_router, prefix="/graphql")
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

    def claim_shared_state(
        self, client_id: Optional[str], surface: str
    ) -> Optional[Dict[str, Any]]:
        """Sticky single-principal claim over the shared-state surfaces
        (Codex findings 3/4).

        Chat conversation addressing, export, reset and /memory resolve ONE
        server-wide client/store with no owner in the lookup, so under the
        enforced posture (more than one configured credential, no explicit
        ``single_operator_mode`` declaration) the FIRST authenticated
        principal to use a shared surface becomes its sticky owner and every
        DIFFERENT credential is refused with the ``MultiPrincipalRefused``
        payload — key B can never address key A's conversation or memory.
        The ``client_id`` LABEL is the principal identity: keys aliased to
        the same label are treated as one principal.

        Returns:
            The structured 409 refusal payload to translate into an HTTP
            response, or ``None`` when the caller may proceed. This is a
            posture statement, not an authorization decision: real
            ownership is the paused VAS-06 / FOU-A4a design.
        """
        if not is_multi_principal(self.api_keys, self.single_operator_mode):
            return None
        if client_id is None:
            # Posture active but the caller resolved no principal: fail closed.
            return refusal_payload(surface)
        if self._shared_state_principal is None:
            self._shared_state_principal = client_id
            logger.info(
                "Shared-state principal bound to %r for this server session "
                "(multi-credential posture; other credentials are refused on "
                "chat conversation addressing, /conversation/export, "
                "/conversation/reset and /memory).",
                client_id,
            )
            return None
        if client_id != self._shared_state_principal:
            return refusal_payload(surface)
        return None

    # Paths reachable without an API key: health/status probes. The /ws chat
    # socket authenticates at the message level (auth frame, gated in
    # _handle_ws_message); HITL keeps its own optional token auth.
    AUTH_EXEMPT_PATHS = frozenset({"/health", "/status"})

    def effective_posture(self) -> Dict[str, Any]:
        """What THIS running server actually enforces per advertised setting.

        Codex finding 22: operators could configure convincing-looking
        controls (a singular server key, a generated session secret, session
        TTL/max-session caps, a rate limit) that no serving path consumed.
        This surface reports the enforced state of every advertised control
        — logged at startup (see ``_lifespan``) and attached to ``/status``
        — so a dead control cannot masquerade as protection.

        Returns:
            Mapping with ``auth``, ``websocket`` and
            ``declared_but_not_enforced`` sections.
        """
        enforced_session_keys = [
            "server_ws_delivery_queue_size",
            "server_ws_send_deadline_seconds",
            "server_max_message_bytes",
        ]
        # Session-token controls (Codex finding 22 correction): the web UI
        # serving path (web/server/main.py) consumes the session secret
        # (HMAC token signing), TTL and max-session cap. An auto-generated
        # secret is a per-process dev/test default — visible here so a
        # multi-worker deployment cannot silently fail token verification
        # across workers.
        _server_settings = getattr(self._settings, "server", None)
        secret_auto = bool(getattr(_server_settings, "server_session_secret_auto_generated", False))
        return {
            "auth": {
                "enforced": bool(self.api_keys),
                "credentials": len(self.api_keys),
                "sources": self._api_key_sources,
                "single_operator_mode": self.single_operator_mode,
                "multi_principal_refusal_active": is_multi_principal(
                    self.api_keys, self.single_operator_mode
                ),
            },
            "websocket": {
                "delivery_queue_size": self._ws_delivery_queue_size,
                "send_deadline_seconds": self._ws_send_deadline,
                "auth_deadline_seconds": self._ws_auth_deadline,
                "inbound_max_message_bytes": self._ws_max_message_bytes,
                "enforced_settings": enforced_session_keys,
            },
            "session_token": {
                "enforced_by": "web UI serving path (web/server/main.py)",
                "secret_configured": bool(getattr(_server_settings, "server_session_secret", None)),
                "secret_source": (
                    "auto-generated per-process (set server.server_session_secret "
                    "for multi-worker deployments; tokens will not verify across "
                    "workers)"
                    if secret_auto
                    else "explicit operator configuration"
                ),
                "ttl_seconds": getattr(_server_settings, "server_session_ttl_seconds", None),
                "max_sessions": getattr(_server_settings, "server_max_sessions", None),
            },
            "declared_but_not_enforced": {
                "rate_limit_rpm": (
                    f"configured to {self.rate_limit_rpm} but the FastAPI server "
                    "mounts no rate limiter (the aiohttp RateLimiter is deprecated "
                    "and unmounted)"
                    if self.rate_limit_rpm
                    else "not configured"
                ),
            },
        }

    def _make_auth_dependency(self):
        """Build the HTTP-route auth dependency (closure over this server)."""

        async def _enforce_api_auth(request: Request) -> None:
            """Auth gate for HTTP routes: a valid API key is required when
            keys are configured. Keyless servers (local dev, most tests) are
            unaffected — the gate short-circuits without keys.

            On success the resolved principal (client_id) is recorded on
            ``request.state`` so broadcast call sites can bind event delivery
            to the owning principal."""
            if not self.api_keys:
                return
            if request.url.path in self.AUTH_EXEMPT_PATHS:
                return
            client_id = await self._verify_api_key(request)
            request.state.client_id = client_id

        return _enforce_api_auth

    def _make_ws_auth_dependency(self):
        """Build the websocket-route auth dependency (closure over server)."""

        async def _enforce_ws_auth(websocket: WebSocket) -> None:
            """Auth gate for websocket routes: Bearer header or ?api_key=
            query parameter. The /ws chat socket is exempt (message-level
            auth); this guards event streams (/ws/events, workflow streams,
            GraphQL subscriptions).

            On success the resolved principal is stamped onto
            ``websocket.state`` so the event transports can bind it as the
            verified owner of the connection's subscription records (Codex
            finding 9): authenticating the KEY must also identify the
            PRINCIPAL, or every authenticated user shares one broadcast.
            """
            if not self.api_keys:
                return
            header = websocket.headers.get("Authorization", "")
            matched = None
            if header.startswith("Bearer "):
                candidate = header[7:]
                matched = next(
                    (k for k in self.api_keys if secrets.compare_digest(k, candidate)),
                    None,
                )
            if matched is None:
                query_key = websocket.query_params.get("api_key")
                if query_key is not None:
                    matched = next(
                        (k for k in self.api_keys if secrets.compare_digest(k, query_key)),
                        None,
                    )
            if matched is None:
                # In websocket scope FastAPI closes the connection when a
                # dependency raises HTTPException.
                raise HTTPException(status_code=403, detail="Not authenticated")
            if not hasattr(websocket, "state"):
                websocket.state = type("State", (), {})()
            websocket.state.authenticated = True
            websocket.state.client_id = self.api_keys[matched]

        return _enforce_ws_auth

    def _include_with_auth(
        self, router: APIRouter, *, prefix: str = "", guard_websockets: bool = True
    ) -> None:
        """Include ``router`` with auth attached per route class, WITHOUT
        mutating the (possibly module-shared) router object.

        HTTP routes get the API-key dependency; websocket routes get the WS
        guard, except the message-authenticated /ws chat socket. Splitting
        into fresh child routers is load-bearing twice over:

        - A router-level include dependency applies to websocket routes too,
          and a Request-taking dependency crashes websocket scope — so
          mixed routers (workflow_routes: one WS stream + nine HTTP routes
          incl. POST /workflows/execute; GraphQL subscriptions) need HTTP
          and WS handled separately.
        - Appending to ``route.dependencies`` of a module-level shared
          router (the observability router is a singleton) binds THIS
          server's key-closure onto every later server that includes it —
          cross-server/cross-test auth pollution (found by adversarial
          review). include_router copies routes; fresh child routers keep
          those copies per-server.
        """
        http_dep = Depends(self._make_auth_dependency())
        ws_dep = Depends(self._make_ws_auth_dependency())
        http_routes, plain_ws_routes, guarded_ws_routes = [], [], []
        for route in router.routes:
            if isinstance(route, APIRoute):
                http_routes.append(route)
            elif isinstance(route, APIWebSocketRoute):
                if not guard_websockets or route.path == "/ws":
                    plain_ws_routes.append(route)  # /ws: message-level auth
                else:
                    guarded_ws_routes.append(route)

        if http_routes:
            child = APIRouter()
            child.routes.extend(http_routes)
            self.app.include_router(child, prefix=prefix, dependencies=[http_dep])
        if plain_ws_routes:
            child = APIRouter()
            child.routes.extend(plain_ws_routes)
            self.app.include_router(child, prefix=prefix)
        if guarded_ws_routes:
            child = APIRouter()
            child.routes.extend(guarded_ws_routes)
            self.app.include_router(child, prefix=prefix, dependencies=[ws_dep])
        if not (http_routes or plain_ws_routes or guarded_ws_routes):
            self.app.include_router(router, prefix=prefix)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage server lifespan."""
        logger.info(f"Starting Victor FastAPI server on {self.host}:{self.port}")
        # Effective-posture check (Codex finding 22): log what this running
        # server ACTUALLY enforces per advertised setting, so a configured
        # control that no serving path consumes is visible at startup.
        try:
            logger.info(
                "Effective server posture: %s",
                json.dumps(self.effective_posture(), default=str),
            )
        except Exception as e:  # pragma: no cover - posture logging must not block startup
            logger.debug("Effective-posture logging failed: %s", e)

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
        # Close WebSocket connections (and drop their bounded delivery state)
        for ws in self._ws_clients + self._event_clients:
            try:
                await ws.close()
            except Exception:
                pass
            self._teardown_ws_delivery(ws, reason="shutdown")
        logger.info("Victor FastAPI server shutdown complete")

    def _setup_routes(self) -> None:
        """Set up API routes by including extracted APIRouter modules.

        Auth attaches per-route (see _attach_route_auth) so mixed
        HTTP+websocket routers get HTTP gating without breaking their
        websocket routes.
        """
        from victor.integrations.api.routes import create_all_routers

        for router in create_all_routers(self):
            self._include_with_auth(router)

    def _setup_router_plugins(self) -> None:
        """Load optional FastAPI routers from vertical packages.

        Routers are discovered via `victor.api_routers` entry points.
        """
        registrations = load_fastapi_router_registrations(workspace_root=self.workspace_root)
        for registration in registrations:
            try:
                self._include_with_auth(registration.router, prefix=registration.prefix)
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
            if self.api_keys and not self.hitl_auth_token:
                # API keys configured but no dedicated HITL token: the HITL
                # routes must not stay open (approving agent tool gates
                # unauthenticated — adversarial-review finding). Gate them
                # with the API key.
                self._include_with_auth(hitl_router, prefix="/hitl", guard_websockets=False)
            else:
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
        # A config switch drops the runtime owners — the sticky shared-state
        # claim binds to the replaced client/store and must not carry over.
        self._shared_state_principal = None
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
        """Handle incoming WebSocket messages.

        All response frames flow through :meth:`_send_ws_frame` — for a
        route-registered socket that means the per-client bounded queue, so
        the socket's sender task is its ONLY writer. Direct ``send_json``
        here would overlap a queued broadcast send on the same connection
        (uvicorn raises on concurrent sends; the sender task would tear the
        healthy client down mid-chat). It also makes frame ordering FIFO:
        a broadcast enqueued before a pong is delivered before it.
        """
        msg_type = data.get("type", "")

        # Inbound payload cap (Codex finding 22): the advertised
        # ``server.server_max_message_bytes`` control is enforced here —
        # oversized frames are refused with a typed error, not processed.
        if self._ws_max_message_bytes:
            payload_bytes = len(json.dumps(data, separators=(",", ":")).encode("utf-8"))
            if payload_bytes > self._ws_max_message_bytes:
                await self._send_ws_frame(
                    ws,
                    {
                        "type": "error",
                        "message": (
                            f"Message exceeds server_max_message_bytes "
                            f"({payload_bytes} > {self._ws_max_message_bytes})"
                        ),
                    },
                )
                return

        if msg_type == "chat":
            messages = data.get("messages", [])
            if not messages:
                await self._send_ws_frame(ws, {"type": "error", "message": "No messages"})
                return

            # Auth gate: when API keys are configured, chat turns require a
            # prior successful `auth` message. Previously the flag was set
            # but never enforced — an unauthenticated WS client got full
            # agent execution (co-design review U7-F1).
            if self.api_keys and not getattr(getattr(ws, "state", None), "authenticated", False):
                await self._send_ws_frame(ws, {"type": "error", "message": "Not authenticated"})
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
                            await self._send_ws_frame(ws, {"type": "content", "content": content})
                        elif "tool_start" in metadata:
                            await self._send_ws_frame(
                                ws, {"type": "tool_call", "tool_call": metadata["tool_start"]}
                            )
                        elif "tool_result" in metadata:
                            await self._send_ws_frame(
                                ws,
                                {
                                    "type": "tool_result",
                                    "tool_result": metadata["tool_result"],
                                },
                            )

                await self._send_ws_frame(ws, {"type": "done"})
            except Exception as e:
                await self._send_ws_frame(ws, {"type": "error", "message": str(e)})

        elif msg_type == "ping":
            await self._send_ws_frame(ws, {"type": "pong"})

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
                    # Keyed servers deferred delivery registration to the
                    # handshake: an authenticated socket can now receive
                    # broadcasts, so it needs its bounded queue + sender task.
                    self._ensure_ws_delivery(ws)
                    await self._send_ws_frame(ws, {"type": "auth_success"})
                else:
                    await self._send_ws_frame(
                        ws, {"type": "auth_failed", "message": "Invalid API key"}
                    )
            elif not self.api_keys:
                await self._send_ws_frame(ws, {"type": "auth_success"})
            else:
                await self._send_ws_frame(
                    ws, {"type": "auth_failed", "message": "No API key provided"}
                )

        elif msg_type == "subscribe":
            channel = data.get("channel", "")
            await self._send_ws_frame(ws, {"type": "subscribed", "channel": channel})

    async def _send_ws_frame(self, ws: Any, message: Dict[str, Any]) -> None:
        """Send a request/response frame to a /ws socket.

        Route-registered sockets carry a bounded delivery queue and ALL
        their writes flow through it (VAS-06/10, TD-03): the per-client
        sender task is the socket's only writer, so chat/pong/auth frames
        can never overlap a queued broadcast send. Frames that are not
        declared-coalescible progress are essential — a full queue
        disconnects the slow client instead of dropping the frame.
        """
        delivery = self._ws_deliveries.get(ws)
        if delivery is not None:
            self._enqueue_ws_delivery(delivery, message, nonessential=False)
            return
        # No per-client delivery (keyed-unauthenticated socket, whose only
        # outbound frames are the auth handshake and one error): bounded
        # direct send. No sender task exists for this socket, so nothing
        # can overlap this write.
        try:
            await asyncio.wait_for(ws.send_json(message), timeout=self._ws_send_deadline)
        except Exception:
            pass

    def _owner_authorized(self, resource_owner: Optional[str], principal: Optional[str]) -> bool:
        """Whether ``principal`` may access a resource stored with
        ``resource_owner`` (HB-01 / ADR-L01 semantics).

        Keyless servers (explicit single-user opt-in) authorize everything.
        On keyed servers the stored owner must exist and equal the
        server-resolved principal: unknown ownership is quarantined, not
        shared, and caller-supplied identity fields grant nothing.
        """
        if not self.api_keys:
            return True
        return bool(resource_owner) and resource_owner == principal

    def _owned_entry(
        self,
        store: "Dict[str, Dict[str, Any]]",
        resource_id: str,
        principal: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Return the stored entry only when it exists and the principal owns
        it (HB-01); ``None`` otherwise, so callers return an identical
        not-found shape without confirming another owner's existence."""
        entry = store.get(resource_id)
        if entry is None or not self._owner_authorized(entry.get("owner_client_id"), principal):
            return None
        return entry

    def register_pending_approval(
        self,
        approval_id: str,
        tool_name: str,
        owner_client_id: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Register a pending tool approval with its owning principal.

        The stored ``owner_client_id`` drives list filtering, mutation
        authorization and owner-scoped event delivery. Producers must set it
        from the server-resolved principal at creation time.
        """
        self._pending_tool_approvals[approval_id] = {
            "tool_name": tool_name,
            "owner_client_id": owner_client_id,
            "approved": False,
            "resolved": False,
            **metadata,
        }

    async def _broadcast_ws(
        self, message: Dict[str, Any], owner_client_id: Optional[str] = None
    ) -> None:
        """Broadcast a message to WebSocket clients admitted to the event.

        Recipient admission (VAS-05h): ``/ws`` registers sockets before
        message-level authentication, so on a keyed server a socket that has
        not completed the ``auth`` handshake is excluded from protected-event
        delivery. Keyless servers (explicit allow_unauthenticated=True) treat
        every connected socket as authorized, as before.

        Owner scoping: when ``owner_client_id`` is provided (the resolved
        principal of the triggering request, recorded by the auth dependency),
        only sockets authenticated as that principal receive the event —
        cross-owner delivery is denied. ``None`` keeps the historical
        all-authenticated behavior for broadcasts with no bound request
        principal (e.g. internal agent events).

        Bounded delivery (VAS-06/10, TD-03): producers never await a slow
        socket. Each route-registered client owns a bounded outbound queue
        drained by its own sender task (send deadline enforced), so a client
        that never reads cannot block the broadcast loop for every
        recipient. Overflow sheds only DECLARED nonessential progress events
        (agent_event allowlist); terminal/approval state is never dropped —
        if an essential event cannot enqueue, the slow client is
        disconnected and the gap is recorded (see ``_ws_delivery_gaps``;
        cursor-based replay is the VAS-10 contract).
        """
        nonessential = _is_nonessential_progress(message)
        for ws in list(self._ws_clients):
            state = getattr(ws, "state", None)
            if self.api_keys:
                if not getattr(state, "authenticated", False):
                    continue
                if (
                    owner_client_id is not None
                    and getattr(state, "client_id", None) != owner_client_id
                ):
                    continue
            delivery = self._ws_deliveries.get(ws)
            if delivery is None:
                # Socket without per-client delivery (custom registration,
                # or the tiny window before teardown removes it): keep
                # delivery working via a SCHEDULED bounded send — the
                # producer never awaits a slow socket, even here (TD-03).
                try:
                    asyncio.get_running_loop().create_task(self._bounded_direct_send(ws, message))
                except RuntimeError:
                    pass
                continue
            self._enqueue_ws_delivery(delivery, message, nonessential=nonessential)

    async def _bounded_direct_send(self, ws: Any, message: Dict[str, Any]) -> None:
        """Best-effort send for sockets without per-client delivery."""
        try:
            await asyncio.wait_for(ws.send_json(message), timeout=self._ws_send_deadline)
        except Exception:
            pass

    def _register_ws_delivery(self, ws: Any) -> "_WSClientDelivery":
        """Create the bounded delivery state + sender task for a /ws socket.

        Called by the route on accept (keyless) or on auth success (keyed),
        on the running loop.
        """
        delivery = _WSClientDelivery(
            ws=ws, queue=_ClientOutboundQueue(self._ws_delivery_queue_size)
        )
        delivery.sender_task = asyncio.get_running_loop().create_task(
            self._ws_sender_loop(delivery)
        )
        self._ws_deliveries[ws] = delivery
        return delivery

    def _ensure_ws_delivery(self, ws: Any) -> "_WSClientDelivery":
        """Return the socket's delivery state, creating it if absent."""
        delivery = self._ws_deliveries.get(ws)
        if delivery is None:
            delivery = self._register_ws_delivery(ws)
        return delivery

    def _teardown_ws_delivery(self, ws: Any, reason: str) -> None:
        """Drop a client's delivery state, stop its sender task and close the
        socket. Safe to call from the sender task itself, the route's finally
        block or a broadcast producer."""
        delivery = self._ws_deliveries.pop(ws, None)
        if delivery is None:
            return
        task = delivery.sender_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        if ws in self._ws_clients:
            self._ws_clients.remove(ws)
        # Make the disconnect visible to the client (1013 try_again_later):
        # bounded-deadline disconnects must not look like a silent halt.
        # Scheduled, not awaited: teardown also runs on producer paths, and
        # during shutdown a task scheduled onto a dying loop is harmless —
        # _close_ws_quietly swallows the resulting failure.
        try:
            asyncio.get_running_loop().create_task(
                self._close_ws_quietly(ws, code=1013, reason=reason)
            )
        except RuntimeError:
            pass

    async def _close_ws_quietly(self, ws: Any, code: int, reason: str) -> None:
        try:
            await ws.close(code=code, reason=reason)
        except Exception:
            pass

    def _enqueue_ws_delivery(
        self, delivery: "_WSClientDelivery", message: Dict[str, Any], *, nonessential: bool
    ) -> None:
        """Apply the TD-03 overflow policy for one client's queue.

        Nonessential progress may be shed (oldest first, then the newcomer);
        essential (terminal/approval) state never is. An essential event that
        cannot fit behind an all-essential queue would silently lose final
        state, so the slow client is disconnected and the gap recorded.
        """
        queue = delivery.queue
        if not queue.full():
            queue.put(message)
            return
        if queue.evict_oldest_nonessential():
            delivery.dropped_nonessential += 1
            queue.put(message)
            return
        if nonessential:
            # Nothing sheddable and the newcomer is coalescible: drop the
            # newcomer, never queued essential state, never the connection.
            delivery.dropped_nonessential += 1
            return
        self._ws_delivery_gaps += 1
        logger.warning(
            "WS client overflowed with essential state queued; disconnecting slow client "
            "(gap recorded; dropped_progress=%d)",
            delivery.dropped_nonessential,
        )
        self._teardown_ws_delivery(delivery.ws, reason="essential_overflow")

    async def _ws_sender_loop(self, delivery: "_WSClientDelivery") -> None:
        """Own one client's socket: drain its queue with a per-send deadline.

        The broadcast producer never awaits this socket — a wedged or slow
        client only stalls its own sender task, which the deadline tears
        down (VAS-06/10, TD-03).
        """
        ws = delivery.ws
        while True:
            message = await delivery.queue.get()
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=self._ws_send_deadline)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("WS delivery send failed (%s); disconnecting slow client", exc)
                self._teardown_ws_delivery(ws, reason="send_deadline_or_error")
                return

    async def _broadcast_agent_event(
        self, event_type: str, data: Dict[str, Any], owner_client_id: Optional[str] = None
    ) -> None:
        """Broadcast agent events to admitted WebSocket clients.

        Used by BackgroundAgentManager to send real-time updates.
        """
        message = {
            "type": "agent_event",
            "event": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        await self._broadcast_ws(message, owner_client_id=owner_client_id)

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

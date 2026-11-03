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
from victor.integrations.api.router_plugins import load_fastapi_router_registrations
from victor.integrations.api.workflow_event_bridge import WorkflowEventBridge
from victor.core.events import get_observability_bus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.session_config import SessionConfig


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
        allow_unauthenticated: Optional[bool] = None,
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

        # Track WebSocket auth failures for rate limiting
        self._ws_auth_failures: Dict[str, int] = {}
        self._ws_auth_failure_lock = (
            asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        )

        # Create FastAPI app with lifespan. Auth is enforced per-route in
        # _setup_routes/_setup_router_plugins (co-design review U7-F1..F3:
        # auth was previously opt-in per handler — only /chat* verified, so
        # git/config/agents/terminal were open). An app-level dependency is
        # NOT usable: it also runs for websocket routes, where FastAPI's
        # solver cannot supply a Request parameter (it crashes the socket).
        # Security: Always disable docs by default"} to prevent information disclosure

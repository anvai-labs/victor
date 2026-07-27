import base64
import hashlib
import hmac
import logging
import secrets
import asyncio
import uuid
import subprocess
import tempfile
import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel
from fastapi import (
    Depends,
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Body,
    Response,
    HTTPException,
    Query,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from victor.config.settings import load_settings
from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from web.server.session_store import (
    InMemorySessionStore,
    SessionLimitReached,
    SessionStore,
    WebSession,
    set_session_store,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Victor AI Assistant API", version="2.0.0")

# CORS configuration for cross-origin requests
# Set CORS_ORIGINS environment variable for production (comma-separated)
# Example: CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    # Default to localhost for development
    allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

logger.info(f"CORS enabled for origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load settings once at startup
try:
    settings = load_settings()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.critical(f"FATAL: Failed to load settings: {e}")
    import sys

    sys.exit(1)  # Fail fast - don't start with broken config

# Security and limits (configurable via settings / env vars)
API_KEY = settings.server_api_key
SESSION_SECRET = settings.server_session_secret or secrets.token_hex(32)
SESSION_TTL = settings.server_session_ttl_seconds
MAX_SESSIONS = settings.server_max_sessions
MAX_MESSAGE_BYTES = settings.server_max_message_bytes
RENDER_MAX_BYTES = settings.render_max_payload_bytes
RENDER_TIMEOUT = settings.render_timeout_seconds
RENDER_SEMAPHORE = asyncio.Semaphore(settings.render_max_concurrency)

# Session management: typed, self-locking, injectable store (P0-B).
# Swap backends (service-backed, Redis) via set_session_store() at startup.
SESSION_STORE: SessionStore = set_session_store(InMemorySessionStore(MAX_SESSIONS))

# Configuration constants
HEARTBEAT_INTERVAL = 30  # seconds
SESSION_IDLE_TIMEOUT = 3600  # 1 hour in seconds
CLEANUP_INTERVAL = 300  # 5 minutes
MESSAGE_TIMEOUT = 300  # 5 minutes for receive timeout

# Track background tasks for graceful shutdown
_background_tasks = []


def _get_bearer_token(raw_header: Optional[str]) -> Optional[str]:
    if not raw_header:
        return None
    if raw_header.lower().startswith("bearer "):
        return raw_header.split(" ", 1)[1].strip()
    return raw_header.strip()


async def _require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce API key when configured."""
    if not API_KEY:
        return  # Auth disabled (development)

    header_token = _get_bearer_token(request.headers.get("Authorization"))
    query_token = request.query_params.get("api_key")
    token = header_token or query_token

    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _issue_session_token(session_id: str) -> str:
    """Create an HMAC-signed session token."""
    issued_at = int(time.time())
    payload = f"{session_id}:{issued_at}"
    signature = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token_bytes = f"{payload}:{signature}".encode()
    return base64.urlsafe_b64encode(token_bytes).decode()


def _parse_session_token(token: str) -> Optional[Tuple[str, int]]:
    """Validate a session token and return (session_id, issued_at)."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        session_id, issued_at_str, signature = decoded.split(":")
        expected_sig = hmac.new(
            SESSION_SECRET.encode(),
            f"{session_id}:{issued_at_str}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        issued_at = int(issued_at_str)
        if time.time() - issued_at > SESSION_TTL:
            return None
        return session_id, issued_at
    except Exception:
        return None


def _validate_render_payload(payload: str) -> None:
    if len(payload.encode()) > RENDER_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Render payload too large (max {RENDER_MAX_BYTES} bytes)",
        )


async def _render_with_limits(render_fn, payload: str) -> str:
    """Apply size, concurrency, and timeout limits to renderers."""
    _validate_render_payload(payload)
    async with RENDER_SEMAPHORE:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: render_fn(payload, RENDER_TIMEOUT))


class SessionTokenRequest(BaseModel):
    """Request model for issuing session tokens."""

    session_id: Optional[str] = None


def _render_plantuml_svg(source: str, timeout: Optional[int] = None) -> str:
    """Render PlantUML text to SVG using local plantuml CLI."""
    try:
        proc = subprocess.run(
            ["plantuml", "-tsvg", "-pipe"],
            input=source.encode(),
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return proc.stdout.decode()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"PlantUML render failed: {exc.stderr.decode()}"
        ) from exc


def _render_mermaid_svg(source: str, timeout: Optional[int] = None) -> str:
    """Render Mermaid text to SVG using local mmdc CLI."""
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".mmd", delete=True) as fin,
            tempfile.NamedTemporaryFile(mode="r", suffix=".svg", delete=True) as fout,
        ):
            fin.write(source)
            fin.flush()
            cmd = ["mmdc", "-i", fin.name, "-o", fout.name]
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            fout.seek(0)
            return fout.read()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"Mermaid render failed: {exc.stderr.decode()}"
        ) from exc


def _render_drawio_svg(source: str, timeout: Optional[int] = None) -> str:
    """Render Draw.io (or Lucid-style XML) to SVG using local drawio CLI."""
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".drawio", delete=True) as fin,
            tempfile.NamedTemporaryFile(mode="r", suffix=".svg", delete=True) as fout,
        ):
            fin.write(source)
            fin.flush()
            # drawio CLI flags: -x (export), -f svg (format), -o output
            cmd = ["drawio", "-x", "-f", "svg", "-o", fout.name, fin.name]
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            fout.seek(0)
            return fout.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="drawio CLI not found. Please install draw.io desktop/CLI.",
        ) from None
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"Draw.io render failed: {exc.stderr.decode()}"
        ) from exc


@app.post("/render/plantuml")
async def render_plantuml(
    payload: str = Body(..., media_type="text/plain"),
    _: None = Depends(_require_api_key),
) -> Response:
    svg = await _render_with_limits(_render_plantuml_svg, payload)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/render/mermaid")
async def render_mermaid(
    payload: str = Body(..., media_type="text/plain"),
    _: None = Depends(_require_api_key),
) -> Response:
    svg = await _render_with_limits(_render_mermaid_svg, payload)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/render/drawio")
async def render_drawio(
    payload: str = Body(..., media_type="text/plain"),
    _: None = Depends(_require_api_key),
) -> Response:
    svg = await _render_with_limits(_render_drawio_svg, payload)
    return Response(content=svg, media_type="image/svg+xml")


def _render_graphviz_svg(source: str, engine: str = "dot", timeout: Optional[int] = None) -> str:
    """Render Graphviz DOT to SVG using local graphviz CLI.

    Supported engines: dot, neato, fdp, circo, twopi, sfdp
    - dot: Hierarchical/layered graphs (default)
    - neato: Spring model layouts
    - fdp: Force-directed placement
    - circo: Circular layout
    - twopi: Radial layouts
    - sfdp: Large graph layouts (scalable force-directed)
    """
    valid_engines = {"dot", "neato", "fdp", "circo", "twopi", "sfdp"}
    if engine not in valid_engines:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid engine '{engine}'. Must be one of: {', '.join(valid_engines)}",
        )

    try:
        proc = subprocess.run(
            [engine, "-Tsvg"],
            input=source.encode(),
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return proc.stdout.decode()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Graphviz engine '{engine}' not found. Install: brew install graphviz (macOS) or apt-get install graphviz (Linux)",
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"Graphviz render failed: {exc.stderr.decode()}"
        )


def _render_d2_svg(source: str, timeout: Optional[int] = None) -> str:
    """Render D2 diagram to SVG using local d2 CLI.

    D2 is a modern diagram scripting language with features:
    - Auto-layout algorithms
    - Built-in themes
    - Icons and shapes
    - Connections and relationships
    """
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".d2", delete=True) as fin,
            tempfile.NamedTemporaryFile(mode="r", suffix=".svg", delete=True) as fout,
        ):
            fin.write(source)
            fin.flush()
            cmd = ["d2", fin.name, fout.name]
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            fout.seek(0)
            return fout.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="d2 CLI not found. Install: curl -fsSL https://d2lang.com/install.sh | sh -s --",
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"D2 render failed: {exc.stderr.decode()}")


@app.post("/render/graphviz")
async def render_graphviz(
    payload: str = Body(..., media_type="text/plain"),
    engine: str = "dot",
    _: None = Depends(_require_api_key),
) -> Response:
    """Render Graphviz DOT diagram to SVG.

    Query param 'engine' can be: dot (default), neato, fdp, circo, twopi, sfdp
    """
    svg = await _render_with_limits(
        lambda text, timeout=RENDER_TIMEOUT: _render_graphviz_svg(text, engine, timeout),
        payload,
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/render/d2")
async def render_d2(
    payload: str = Body(..., media_type="text/plain"),
    _: None = Depends(_require_api_key),
) -> Response:
    """Render D2 diagram to SVG."""
    svg = await _render_with_limits(_render_d2_svg, payload)
    return Response(content=svg, media_type="image/svg+xml")


async def _shutdown_session_agent(session: WebSession) -> None:
    """Best-effort agent/provider teardown for a removed session."""
    try:
        agent = session.agent
        if hasattr(agent, "shutdown"):
            await agent.shutdown()
        if hasattr(agent, "provider"):
            await agent.provider.close()
    except Exception as e:
        logger.warning(f"Error shutting down agent for session {session.session_id}: {e}")


async def heartbeat_loop(websocket: WebSocket, session_id: str) -> None:
    """Send periodic ping messages to keep connection alive and detect dead connections."""
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_text("[ping]")
                # Update last activity on successful ping
                await SESSION_STORE.touch(session_id)
            except Exception as e:
                logger.warning(f"Heartbeat failed for session {session_id}: {e}")
                break
    except asyncio.CancelledError:
        logger.debug(f"Heartbeat loop cancelled for session {session_id}")


async def cleanup_idle_sessions() -> None:
    """Background task to clean up idle sessions to prevent memory leaks."""
    logger.info("Starting idle session cleanup task")
    try:
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)

                # Removal (with token revocation) is atomic in the store; agent
                # shutdown happens out here so a slow shutdown can't block
                # connects/heartbeats the way the old inline lock did.
                expired = await SESSION_STORE.pop_idle(SESSION_IDLE_TIMEOUT)
                for session in expired:
                    logger.info(f"Cleaning up idle session: {session.session_id}")
                    await _shutdown_session_agent(session)

                if expired:
                    logger.info(f"Cleaned up {len(expired)} idle sessions")

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Idle session cleanup task cancelled gracefully")
        raise  # Re-raise to properly handle task cancellation


@app.on_event("startup")
async def startup_event() -> None:
    """Start background tasks on server startup."""
    logger.info("Starting background tasks...")
    task = asyncio.create_task(cleanup_idle_sessions())
    _background_tasks.append(task)
    logger.info("Server started successfully with background tasks tracked")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up background tasks and sessions on server shutdown."""
    logger.info("Shutting down background tasks...")

    # Cancel all background tasks gracefully
    for task in _background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Clean up all active sessions (state cleared atomically; agents shut down
    # outside the store's critical section).
    for session in await SESSION_STORE.pop_all():
        await _shutdown_session_agent(session)
        logger.info(f"Closed session {session.session_id} during shutdown")

    logger.info("Shutdown complete - all tasks and sessions cleaned up")


@app.get("/health")
async def health_check(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": await SESSION_STORE.count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/session/token")
async def issue_session_token(
    payload: SessionTokenRequest, _: None = Depends(_require_api_key)
) -> Dict[str, str]:
    """Issue a signed session token for WebSocket reuse."""
    # Enforce session cap early (advisory; add() is the authoritative gate)
    if not payload.session_id and not await SESSION_STORE.has_capacity():
        raise HTTPException(status_code=429, detail="Session limit reached")

    # Re-issue token for an existing session
    if payload.session_id:
        if not await SESSION_STORE.contains(payload.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        token = _issue_session_token(payload.session_id)
        await SESSION_STORE.bind_token(token, payload.session_id)
        return {"session_token": token, "session_id": payload.session_id}

    # New logical session (agent will be created on first WS connect)
    session_id = str(uuid.uuid4())
    token = _issue_session_token(session_id)
    await SESSION_STORE.bind_token(token, session_id)
    return {"session_token": token, "session_id": session_id}


class ChatStreamRequest(BaseModel):
    """Request model for the typed SSE chat stream (UX P1)."""

    message: str
    session_id: Optional[str] = None


@app.post("/chat/stream")
async def chat_stream(
    payload: ChatStreamRequest, _: None = Depends(_require_api_key)
) -> StreamingResponse:
    """Versioned JSON-over-SSE event stream for one chat turn (UX P1).

    Emits the framework's typed events on the shared v1 wire contract
    (victor/framework/wire_events.py): thinking → tool_call → tool_result →
    content → stream_end, with in-stream failures surfaced as error events.
    Debuggable with a plain ``curl -N``.
    """
    from victor.framework.wire_events import stream_sse

    session = None
    if payload.session_id:
        session = await SESSION_STORE.acquire_connection(payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        agent = session.agent
        session_id = payload.session_id
    else:
        if not await SESSION_STORE.has_capacity():
            raise HTTPException(status_code=429, detail="Session limit reached")
        agent = VictorClient(SessionConfig())
        await agent.initialize()
        session_id = str(uuid.uuid4())
        try:
            session, created = await SESSION_STORE.add(
                WebSession(session_id=session_id, agent=agent, connection_count=1)
            )
            if not created:
                await _shutdown_session_agent(WebSession(session_id=session_id, agent=agent))
                agent = session.agent
                await SESSION_STORE.acquire_connection(session_id)
        except SessionLimitReached as exc:
            await _shutdown_session_agent(WebSession(session_id=session_id, agent=agent))
            raise HTTPException(status_code=429, detail="Session limit reached") from exc

    async def event_source():
        try:
            async for frame in stream_sse(agent, payload.message):
                yield frame
            await SESSION_STORE.touch(session_id)
        finally:
            await SESSION_STORE.release_connection(session_id)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id, "Cache-Control": "no-cache"},
    )


# --- Minimal compatibility REST endpoints for VS Code client ---


@app.get("/history")
async def get_history(
    limit: int = Query(20, ge=1, le=200), _: None = Depends(_require_api_key)
) -> Dict[str, Any]:
    """Return an empty history list (placeholder to satisfy VS Code client)."""
    return {"history": [], "limit": limit}


@app.get("/credentials/get")
async def get_credentials(
    provider: str = Query(...), _: None = Depends(_require_api_key)
) -> Dict[str, Any]:
    """Placeholder credentials endpoint."""
    return {"provider": provider, "api_key": None}


@app.get("/models")
async def list_models(_: None = Depends(_require_api_key)) -> Dict[str, Any]:
    """Placeholder models endpoint."""
    return {"models": []}


@app.get("/providers")
async def list_providers(_: None = Depends(_require_api_key)) -> Dict[str, Any]:
    """Placeholder providers endpoint."""
    return {"providers": []}


@app.get("/rl/stats")
async def rl_stats(_: None = Depends(_require_api_key)) -> Dict[str, Any]:
    """Placeholder RL stats endpoint."""
    return {"success": True, "stats": {}}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Enhanced WebSocket endpoint with heartbeat, timeout, and error handling."""
    session_id = None
    session_token: Optional[str] = None
    heartbeat_task = None
    session_initialized = False  # Track if we successfully initialized session

    try:
        # API key validation (FastAPI deps not available for websockets)
        if API_KEY:
            header_token = _get_bearer_token(websocket.headers.get("authorization"))
            query_token = websocket.query_params.get("api_key")
            api_token = header_token or query_token
            if api_token != API_KEY:
                await websocket.close(code=4401, reason="Unauthorized")
                return

        await websocket.accept()
        logger.info("WebSocket connection established.")

        # Session-aware agent reuse with signed tokens
        incoming_token = websocket.query_params.get("session_token")
        parsed = _parse_session_token(incoming_token) if incoming_token else None

        if parsed:
            candidate_session_id, _issued = parsed
            if await SESSION_STORE.contains(candidate_session_id):
                session_id = candidate_session_id
                session_token = incoming_token

        if not session_id:
            # Enforce max sessions before creating a new one (advisory pre-check;
            # SESSION_STORE.add() enforces the cap atomically below)
            if not await SESSION_STORE.has_capacity():
                await websocket.send_text("[error] Session limit reached, try later.")
                await websocket.close(code=1013, reason="Session limit reached")
                return
            session_id = str(uuid.uuid4())
            session_token = _issue_session_token(session_id)

        try:
            # Send session token as both legacy text and JSON for client compatibility
            if session_token:
                await websocket.send_text(f"[session] {session_token}")
                await websocket.send_json({"type": "session", "token": session_token})
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected before session handshake.")
            return

        # Create or retrieve the session. Agent construction happens OUTSIDE
        # the store's critical section — a cold VictorClient.initialize() must
        # never block other connects or heartbeats (the old inline lock did).
        session = await SESSION_STORE.acquire_connection(session_id)
        if session is not None:
            agent = session.agent
            session_token = session.session_token or session_token
            logger.info(
                f"Reusing existing agent for session {session_id} "
                f"(connection #{session.connection_count})"
            )
            session_initialized = True
        else:
            try:
                agent = VictorClient(SessionConfig())
                await agent.initialize()

                # Trigger background embedding preload so first user turn
                # does not pay the warm-up cost.
                await agent.start_embedding_preload()

                session, created = await SESSION_STORE.add(
                    WebSession(
                        session_id=session_id,
                        agent=agent,
                        session_token=session_token,
                        connection_count=1,
                    )
                )
                if not created:
                    # A concurrent connect for the same session won the race —
                    # use its agent and discard our duplicate.
                    await _shutdown_session_agent(WebSession(session_id=session_id, agent=agent))
                    agent = session.agent
                    await SESSION_STORE.acquire_connection(session_id)
                logger.info(
                    f"Created new VictorClient for session {session_id} with preload started."
                )
                session_initialized = True
            except SessionLimitReached:
                await _shutdown_session_agent(WebSession(session_id=session_id, agent=agent))
                await websocket.send_text("[error] Session limit reached, try later.")
                await websocket.close(code=1013, reason="Session limit reached")
                return
            except Exception as e:
                logger.error(f"Failed to create VictorClient: {e}", exc_info=True)
                await websocket.send_text(f"[error] Could not initialize agent: {str(e)}")
                await websocket.close(code=1011, reason="Agent initialization failed")
                return

        # Start heartbeat loop
        heartbeat_task = asyncio.create_task(heartbeat_loop(websocket, session_id))

        # Main message loop with timeout
        while True:
            try:
                # Receive message with timeout to prevent hanging
                user_message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=MESSAGE_TIMEOUT
                )

                logger.info(f"Session {session_id}: Received message")

                # Enforce message size limits
                if len(user_message.encode()) > MAX_MESSAGE_BYTES:
                    await websocket.send_text(
                        f"[error] Message too large (max {MAX_MESSAGE_BYTES} bytes)"
                    )
                    continue

                # Update last activity
                await SESSION_STORE.touch(session_id)

                # Handle special commands
                if user_message.strip() == "__reset_session__":
                    await agent.reset_conversation()
                    await websocket.send_text("[session] reset")
                    logger.info(f"Session {session_id}: Conversation reset")
                    continue

                # Stream the agent's response back to the client
                try:
                    async for chunk in agent.stream_chat(user_message):
                        if chunk.content:
                            await websocket.send_text(chunk.content)

                    # Send final empty chunk to signal completion
                    await websocket.send_text("")

                except Exception as e:
                    logger.error(
                        f"Session {session_id}: Error during agent response: {e}",
                        exc_info=True,
                    )
                    await websocket.send_text(
                        f"[error] An error occurred while processing your request: {str(e)}"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"Session {session_id}: Timeout waiting for message")
                await websocket.send_text("[error] Connection timeout. Please refresh.")
                break

            except WebSocketDisconnect:
                logger.info(f"Session {session_id}: WebSocket disconnected by client")
                break

    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: WebSocket connection disconnected.")
    except Exception as e:
        logger.error(f"Session {session_id}: Unexpected error in WebSocket: {e}", exc_info=True)
        try:
            await websocket.send_text(f"[error] Server error: {str(e)}")
        except (WebSocketDisconnect, RuntimeError) as send_error:
            logger.debug(f"Failed to send error message to client: {send_error}")
    finally:
        # Cancel heartbeat task
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Update connection count (only if we successfully initialized)
        if session_id and session_initialized:
            await SESSION_STORE.release_connection(session_id)
            logger.debug(f"Session {session_id}: Connection released")

        logger.info(f"Session {session_id}: WebSocket connection closed and cleaned up.")

"""VictorClient — unified facade making CLI/Web first-class framework clients.

This is the SINGLE entry point that the UI layer (CLI, Web server) uses
to interact with the framework. It properly delegates to SERVICES layer
instead of bypassing to orchestrator.

Architectural Alignment:
    - Uses ServiceAccessor to access ChatService, ToolService, etc.
    - Does NOT bypass to orchestrator directly
    - Enforces proper service boundaries
    - SessionConfig for CLI/runtime overrides (not settings mutation)

Usage in CLI (victor/ui/commands/chat.py):
    from victor.framework.client import VictorClient
    from victor.framework.session_config import SessionConfig

    config = SessionConfig.from_cli_flags(tool_budget=50, enable_smart_routing=True)
    client = VictorClient(config)
    result = await client.chat("Hello")
    # OR streaming:
    async for event in client.stream("Hello"):
        print(event)

Architecture:
    CLI/Web  →  VictorClient  →  Services (Chat, Tool, Session, etc.)  →  Providers
                         ↓
               ExecutionContext (with ServiceAccessor)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from victor.framework.message_execution import (
    execute_message,
    iter_runtime_stream_events,
    stream_message_events,
)
from victor.framework.task import TaskResult

if TYPE_CHECKING:
    from victor.framework.session_config import SessionConfig
    from victor.framework.agent import Agent
    from victor.framework.agent_components import AgentSession
    from victor.runtime.context import RuntimeExecutionContext
    from victor.runtime.context import ResolvedRuntimeServices
    from victor.core.container import ServiceContainer

logger = logging.getLogger(__name__)

# Max seconds close() waits for in-flight stream() calls to drain before forcing
# shutdown. Module-level so tests can shorten it.
_CLOSE_DRAIN_TIMEOUT_SECONDS = 10.0


class _StreamEvent:
    """Concrete StreamEventProtocol implementation."""

    def __init__(
        self,
        event_type: Any,
        *,
        content: Optional[str] = None,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._type = event_type
        self._content = content
        self._tool_name = tool_name
        self._arguments = arguments or {}
        self._result = result
        self._success = success
        self._metadata = metadata or {}

    @property
    def event_type(self) -> str:
        return getattr(self._type, "value", str(self._type))

    @property
    def type(self) -> Any:
        return self._type

    @property
    def content(self) -> Optional[str]:
        return self._content

    @property
    def tool_name(self) -> Optional[str]:
        return self._tool_name

    @property
    def arguments(self) -> Dict[str, Any]:
        return self._arguments

    @property
    def result(self) -> Any:
        return self._result

    @property
    def success(self) -> bool:
        return self._success

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


@dataclass
class _RenderChunk:
    """Chunk adapter consumed by existing CLI renderer helpers."""

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _session_metadata(config: "SessionConfig") -> Dict[str, Any]:
    """Build stable metadata for a client-managed chat session."""
    return {
        "tool_budget": config.tool_budget,
        "smart_routing": config.smart_routing.enabled,
    }


def _enrich_task_result(result: TaskResult, config: "SessionConfig") -> TaskResult:
    """Attach client session metadata without changing the canonical result type."""
    return replace(
        result,
        metadata={
            **(result.metadata or {}),
            "tool_budget": config.tool_budget,
            "smart_routing": config.smart_routing.enabled,
        },
    )


def _to_stream_event(event: Any) -> _StreamEvent:
    """Convert a framework event into the client stream-event surface."""
    from victor.framework.events import EventType

    if event.type == EventType.CONTENT:
        return _StreamEvent(
            EventType.CONTENT,
            content=event.content,
            metadata=getattr(event, "metadata", {}),
        )
    if event.type == EventType.THINKING:
        return _StreamEvent(
            EventType.THINKING,
            content=event.content,
            metadata=getattr(event, "metadata", {}),
        )
    if event.type == EventType.TOOL_CALL:
        return _StreamEvent(
            EventType.TOOL_CALL,
            tool_name=getattr(event, "tool_name", None),
            content=str(getattr(event, "arguments", "")),
            arguments=getattr(event, "arguments", {}),
            metadata={
                **getattr(event, "metadata", {}),
                "arguments": getattr(event, "arguments", {}),
            },
        )
    if event.type == EventType.TOOL_RESULT:
        event_metadata = getattr(event, "metadata", {}) or {}
        # The producer nests the rich payload under ``metadata["tool_result"]``
        # (name, success, elapsed, arguments, error, follow_up_suggestions,
        # was_pruned, result, original_result). Flatten it to the top level so every
        # consumer — the Rich renderer and the Chainlit event mapping — reads one
        # predictable, flat shape instead of digging through nested dicts.
        nested = event_metadata.get("tool_result")
        result_payload: Dict[str, Any] = {**event_metadata}
        if isinstance(nested, dict):
            result_payload.update(nested)
        result_payload.setdefault("result", getattr(event, "result", None))
        result_payload["success"] = result_payload.get("success", getattr(event, "success", True))
        result_payload.setdefault("arguments", getattr(event, "arguments", {}) or {})
        return _StreamEvent(
            EventType.TOOL_RESULT,
            tool_name=getattr(event, "tool_name", None),
            content=getattr(event, "result", None) or getattr(event, "content", None),
            arguments=result_payload.get("arguments", {}),
            result=result_payload,
            success=bool(result_payload.get("success", True)),
            metadata=result_payload,
        )
    if event.type == EventType.ERROR:
        error_text = getattr(event, "content", None) or getattr(event, "error", None) or str(event)
        return _StreamEvent(
            EventType.ERROR,
            content=str(error_text),
            metadata={
                **getattr(event, "metadata", {}),
                "error": str(error_text),
                "recoverable": getattr(event, "recoverable", None),
            },
        )
    return _StreamEvent(
        event.type,
        content=getattr(event, "content", None),
        metadata={"raw_event": event},
    )


class VictorClient:
    """Unified client facade — the ONLY interface the UI layer uses.

    This class makes the CLI a first-class client of the framework by:
    1. Accepting SessionConfig for CLI/runtime overrides (not settings mutation)
    2. Creating Agent via Agent.create() with session_config
    3. Accessing services through ServiceAccessor (not orchestrator bypass)
    4. Providing chat(), stream(), create_session(), run_workflow()

    Architectural Guarantees:
        - NEVER accesses orchestrator directly
        - ALWAYS uses services (ChatService, ToolService, etc.)
        - ENFORCES proper service boundaries

    Example:
        from victor.framework.client import VictorClient
        from victor.framework.session_config import SessionConfig

        config = SessionConfig.from_cli_flags(tool_budget=50)
        async with VictorClient(config) as client:
            result = await client.chat("Write a hello world")
            print(result.content)
    """

    def __init__(
        self,
        config: "SessionConfig",
        *,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        """Initialize client with session config and optional DI container.

        Args:
            config: SessionConfig with CLI/runtime overrides (immutable)
            container: Optional pre-built DI container. If None, one is
                       bootstrapped lazily on first use.
        """
        self._config = config
        self._container = container
        self._agent: Optional["Agent"] = None
        self._context: Optional["RuntimeExecutionContext"] = None
        self._initialized = False
        # Close-guard: keep close() from tearing down the agent/provider while a
        # stream() is still iterating. Without this, a UI on_chat_end firing on a
        # WebSocket disconnect mid-run closes the provider's httpx client and the
        # in-flight stream fails with "Cannot send a request, as the client has been
        # closed." close() drains active streams (bounded) before shutting down.
        self._active_streams = 0
        self._closing = False
        # Per-session policy ASK approval handler. Stored here and registered into the
        # session's DI container during _ensure_initialized — AFTER _bootstrap_container()
        # (which swaps/disposes the global container) and BEFORE Agent.create() builds the
        # policy middleware. Registering into the global container earlier (e.g. from a UI
        # on_chat_start) is lost when bootstrap disposes that container, which made every
        # ASK-gated tool hit ask_fallback (deny) in the chat UI.
        self._approval_handler: Optional[Any] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> "Agent":
        """Lazily initialize agent via Agent.create() with SessionConfig."""
        if self._agent is not None:
            return self._agent

        from victor.framework.agent import Agent
        from victor.config.settings import load_settings

        # Bootstrap DI container if not provided
        if self._container is None:
            self._container = self._bootstrap_container()

        # Register the per-session policy ASK approval handler into THIS container, after
        # bootstrap (which may have swapped/disposed the previous global container) and
        # before Agent.create() builds the policy middleware. This is the only point where
        # the handler is guaranteed to land in the same container the middleware reads from.
        if self._approval_handler is not None:
            try:
                from victor.framework.policies import register_policy_approval_handler

                register_policy_approval_handler(self._approval_handler, container=self._container)
            except Exception:  # approval is best-effort; chat still works without it
                logger.debug("approval handler registration skipped", exc_info=True)

        # Load base settings
        settings = load_settings()

        # Apply SessionConfig overrides to settings (ONLY place where settings is mutated)
        self._config.apply_to_settings(settings)

        provider_override = getattr(self._config, "provider_override", None)
        provider_name = None
        model_name = None

        if provider_override is not None:
            provider_name = provider_override.provider
            model_name = provider_override.model

        if self._config.agent_profile is None:
            provider_settings = getattr(settings, "provider", None)
            if provider_name is None:
                provider_name = getattr(provider_settings, "default_provider", None)
            if model_name is None:
                model_name = getattr(provider_settings, "default_model", None)

        # Create agent with SessionConfig (including agent_profile if specified)
        self._agent = await Agent.create(
            profile=self._config.agent_profile,  # Pass agent_profile from SessionConfig
            provider=provider_name,
            model=model_name,
            session_config=self._config,  # Pass SessionConfig
        )
        execution_context = getattr(self._agent, "execution_context", None)
        if execution_context is not None:
            self._context = execution_context
        elif hasattr(self._agent, "get_orchestrator"):
            from victor.runtime.context import resolve_execution_context

            orchestrator = self._agent.get_orchestrator()
            self._context = resolve_execution_context(orchestrator)
        self._initialized = True

        logger.info(
            "VictorClient initialized (SessionConfig applied: agent_profile=%s, tool_budget=%s, smart_routing=%s)",
            self._config.agent_profile,
            self._config.tool_budget,
            self._config.smart_routing.enabled,
        )
        return self._agent

    async def initialize(self) -> "Agent":
        """Public initialization hook for UI surfaces."""
        return await self._ensure_initialized()

    def set_approval_handler(self, handler: Any) -> None:
        """Register a policy ASK approval handler for this session.

        The handler (an async elicitation callable, e.g. a Chainlit Approve/Reject prompt)
        is stored on the client and registered into the session's DI container during
        initialization — after the container is bootstrapped and before the agent builds
        its policy middleware — so ASK-gated tools resolve interactively instead of falling
        back to ``ask_fallback``. Must be called before the first ``chat()``/``stream()``.

        Idempotent: the latest handler set before initialization wins. Safe to call on
        every UI ``on_chat_start`` (including reconnects), since registration is deferred
        to ``_ensure_initialized``.
        """
        self._approval_handler = handler
        # If the agent is already initialized (handler set late), register immediately into
        # the live container so it still takes effect for subsequent turns.
        if self._initialized and self._container is not None and handler is not None:
            try:
                from victor.framework.policies import register_policy_approval_handler

                register_policy_approval_handler(handler, container=self._container)
            except Exception:
                logger.debug("late approval handler registration skipped", exc_info=True)

    @property
    def provider_name(self) -> Optional[str]:
        """Active provider name for the current session (e.g. 'zai', 'ollama')."""
        agent = self._agent
        if agent is None:
            return None
        orchestrator = getattr(agent, "_orchestrator", None)
        return getattr(orchestrator, "provider_name", None) or getattr(
            getattr(orchestrator, "provider", None), "name", None
        )

    @property
    def model(self) -> Optional[str]:
        """Active model name for the current session (e.g. 'glm-5.1')."""
        agent = self._agent
        if agent is None:
            return None
        orchestrator = getattr(agent, "_orchestrator", None)
        return getattr(orchestrator, "model", None)

    @property
    def provider_base_url(self) -> Optional[str]:
        """Active provider base_url (non-empty when a non-default endpoint is used)."""
        agent = self._agent
        if agent is None:
            return None
        orchestrator = getattr(agent, "_orchestrator", None)
        provider = getattr(orchestrator, "provider", None)
        return getattr(provider, "base_url", None)

    async def start_embedding_preload(self) -> None:
        """Warm embedding-dependent runtime state when supported."""
        agent = await self._ensure_initialized()
        if hasattr(agent, "start_embedding_preload"):
            agent.start_embedding_preload()

    async def get_session_metrics(self) -> Dict[str, Any]:
        """Return session-level runtime metrics when available."""
        agent = await self._ensure_initialized()
        if hasattr(agent, "get_session_metrics"):
            metrics = agent.get_session_metrics()
            if isinstance(metrics, dict):
                return metrics
        return {}

    def get_last_turn_cost(self) -> Dict[str, Any]:
        """Return the most recent per-turn cost/latency record (the C0 TaskExecutionReport).

        Surfaces the canonical per-turn record — tokens, cost, duration, request count, cache
        hit rate — so UI surfaces can render a cost/latency footer without reaching into the
        orchestrator directly (UI-layer mandate). Returns ``{}`` when no turn has completed or
        the record is unavailable.
        """
        agent = self._agent
        if agent is None:
            return {}
        orchestrator = getattr(agent, "_orchestrator", None)
        getter = getattr(orchestrator, "get_last_task_report", None)
        if callable(getter):
            try:
                report = getter()
                if isinstance(report, dict):
                    return report
            except Exception:
                logger.debug("get_last_turn_cost failed", exc_info=True)
        return {}

    def _bootstrap_container(self) -> "ServiceContainer":
        """Bootstrap the DI container with core services."""
        try:
            # Bootstrap core services
            try:
                from victor.core.bootstrap import bootstrap_container

                return bootstrap_container()
            except ImportError:
                logger.debug("bootstrap_container not available, using minimal DI")

            from victor.core.container import ServiceContainer

            return ServiceContainer()
        except Exception as e:
            logger.warning(f"DI container bootstrap failed, continuing without: {e}")
            from victor.core.container import ServiceContainer

            return ServiceContainer()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — the UI layer uses ONLY these methods
    # ─────────────────────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        *,
        stream: bool = False,
    ) -> TaskResult:
        """Send a single message and get a response.

        Uses the shared framework message-execution surface so UI callers get
        the same service-first runtime resolution and output normalization as Agent.

        Args:
            message: User's message
            stream: If True, use streaming internally but return final result

        Returns:
            TaskResult with response content, tool calls, and metadata
        """
        agent = await self._ensure_initialized()

        if hasattr(agent, "get_orchestrator"):
            result = await execute_message(
                orchestrator=agent.get_orchestrator(),
                execution_context=self._context,
                user_message=message,
                stream=stream,
                forward_stream_option=True,
            )
        else:
            result = await agent.run(message)

        return _enrich_task_result(result, self._config)

    async def stream(
        self,
        message: str,
    ) -> AsyncIterator[_StreamEvent]:
        """Send a message and yield streaming events.

        Uses the shared framework runtime-event iterator so chat services and
        Agent wrappers present the same event contract to UI callers.

        Args:
            message: User's message

        Yields:
            StreamEvent instances (content, thinking, tool_call, etc.)
        """
        if self._closing:
            raise RuntimeError("VictorClient is closing; cannot start a new stream")

        agent = await self._ensure_initialized()
        self._active_streams += 1
        try:
            if hasattr(agent, "get_orchestrator"):
                async for event in stream_message_events(
                    orchestrator=agent.get_orchestrator(),
                    execution_context=self._context,
                    user_message=message,
                ):
                    yield _to_stream_event(event)
                return

            async for event in iter_runtime_stream_events(agent, message):
                yield _to_stream_event(event)
        finally:
            self._active_streams = max(0, self._active_streams - 1)

    async def stream_chat(self, message: str) -> AsyncIterator[_RenderChunk]:
        """Yield renderer-compatible chunks for legacy UI streaming helpers."""
        from victor.framework.events import EventType

        async for event in self.stream(message):
            metadata = dict(event.metadata or {})

            if event.event_type == "thinking":
                if event.content:
                    metadata["reasoning_content"] = event.content
                yield _RenderChunk(metadata=metadata)
                continue

            if event.event_type == "tool_call":
                metadata["tool_start"] = {
                    "name": event.tool_name or "unknown",
                    "arguments": metadata.get("arguments", {}),
                }
                yield _RenderChunk(metadata=metadata)
                continue

            if event.event_type == "tool_result":
                result_payload = event.result if isinstance(event.result, dict) else metadata
                tool_result: Dict[str, Any] = {
                    "name": event.tool_name or "unknown",
                    "result": event.content or result_payload.get("result", ""),
                    "success": result_payload.get("success", True),
                    "arguments": result_payload.get("arguments", {}),
                }
                # Preserve the telemetry the Rich renderer expects rather than
                # cherry-picking 4 keys — see StreamRenderer.on_tool_result.
                for key in (
                    "elapsed",
                    "error",
                    "follow_up_suggestions",
                    "was_pruned",
                    "original_result",
                ):
                    if key in result_payload:
                        tool_result[key] = result_payload[key]
                metadata["tool_result"] = tool_result
                yield _RenderChunk(metadata=metadata)
                continue

            if event.event_type == EventType.ERROR:
                error_message = event.content or metadata.get("error", "Unknown streaming error")
                metadata["error"] = error_message
                yield _RenderChunk(content=f"Error: {error_message}\n", metadata=metadata)
                continue

            yield _RenderChunk(content=event.content or "", metadata=metadata)

    async def create_session(
        self,
        initial_prompt: Optional[str] = None,
    ) -> "AgentSession":
        """Create an interactive multi-turn chat session.

        Delegates to the canonical framework ``AgentSession`` so chat/session
        behavior stays on one framework-owned path.

        Returns:
            AgentSession for multi-turn conversation
        """
        agent = await self._ensure_initialized()
        metadata = _session_metadata(self._config)

        create_session = getattr(agent, "create_session", None)
        if callable(create_session):
            session = create_session(
                initial_prompt,
                metadata=metadata,
                container=self._container,
            )
        else:
            from victor.framework.agent_components import AgentSession

            session = AgentSession(
                agent,
                initial_prompt,
                metadata=metadata,
                container=self._container,
            )

        await session.initialize()
        return session

    async def run_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a named workflow via the Agent's workflow API.

        Args:
            workflow_name: Registered workflow name
            inputs: Workflow input parameters

        Returns:
            Workflow execution result
        """
        agent = await self._ensure_initialized()
        return await agent.run_workflow(workflow_name, inputs=inputs or {})

    def get_available_workflows(self) -> List[str]:
        """List available workflow names."""
        if self._agent:
            return self._agent.get_available_workflows()
        return []

    def get_available_verticals(self) -> List[str]:
        """List available vertical names."""
        try:
            from victor.core.verticals import list_verticals

            return list_verticals()
        except Exception:
            return []

    def get_available_providers(self) -> List[str]:
        """List available provider names."""
        try:
            from victor.providers import list_available_providers

            return list_available_providers()
        except Exception:
            return ["anthropic", "openai", "ollama", "google"]

    # ─────────────────────────────────────────────────────────────────────────
    # Conversation Management
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_runtime_services(self) -> "ResolvedRuntimeServices":
        """Resolve the canonical runtime service bundle for client helpers."""
        from victor.runtime.context import (
            ResolvedRuntimeServices,
            resolve_runtime_services,
        )

        if self._context is None:
            return ResolvedRuntimeServices()

        runtime_owner = None
        if self._agent is not None and hasattr(self._agent, "get_orchestrator"):
            runtime_owner = self._agent.get_orchestrator()

        return resolve_runtime_services(runtime_owner, self._context)

    async def reset_conversation(self) -> None:
        """Reset conversation history and state.

        Uses ChatService to clear conversation context while preserving
        system prompts and session configuration.

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._initialized or not self._context:
            raise RuntimeError("VictorClient not initialized. Call initialize() first.")

        services = self._resolve_runtime_services()
        if services.chat is not None:
            await services.chat.reset_conversation()
        else:
            logger.warning("ChatService not available for conversation reset")

    async def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List[Any]:
        """Get conversation messages.

        Args:
            limit: Maximum number of messages to return (most recent first)
            role: Optional filter by message role (e.g., "user", "assistant")

        Returns:
            List of message objects (type depends on Message implementation)

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._initialized or not self._context:
            raise RuntimeError("VictorClient not initialized. Call initialize() first.")

        services = self._resolve_runtime_services()
        if services.context is not None:
            return await services.context.get_messages(limit=limit, role=role)
        else:
            logger.warning("ContextService not available for get_messages")
            return []

    def list_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent stored sessions for a resume picker (cross-visit).

        Reads the SQLite session persistence directly — a read-only lookup that
        does not require an initialized agent, so a surface can render the
        picker before a client is bound to a session.

        Args:
            limit: Maximum number of sessions to return (most recent first).

        Returns:
            Session summary dicts (session_id, title, message_count,
            last_activity, …); empty on any lookup failure.
        """
        try:
            from victor.agent.sqlite_session_persistence import (
                get_sqlite_session_persistence,
            )

            return get_sqlite_session_persistence().list_sessions(limit=limit)
        except Exception as exc:  # a picker must never take the surface down
            logger.warning("list_recent_sessions failed: %s", exc)
            return []

    async def resume_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Resume a stored session so the agent recalls its prior turns.

        Hydrates the live conversation from the ConversationStore via
        ``ChatService.resume_session`` (which repopulates both message stores),
        then stamps the resumed ``session_id`` (and conversation state, when
        present) onto the orchestrator so continuation writes append to the
        same session.

        Args:
            session_id: The stored session to resume.

        Returns:
            The session metadata (title, message_count, …) on success, or
            ``None`` when the session is not found.

        Raises:
            RuntimeError: If the client is not initialized.
        """
        if not self._initialized or not self._context:
            raise RuntimeError("VictorClient not initialized. Call initialize() first.")

        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        session_data = get_sqlite_session_persistence().load_session(session_id)
        if not session_data:
            return None

        services = self._resolve_runtime_services()
        if services.chat is None:
            logger.warning("ChatService not available for resume_session")
            return None
        metadata = services.chat.resume_session(session_data)

        # Stamp the resumed id/state on the orchestrator so continuation turns
        # append to the same session rather than minting a fresh one.
        agent = self._agent
        orchestrator = getattr(agent, "_orchestrator", None) if agent is not None else None
        if orchestrator is not None:
            orchestrator.active_session_id = session_id
            state_dict = session_data.get("conversation_state")
            if state_dict:
                try:
                    from victor.agent.conversation_state import ConversationStateMachine

                    orchestrator.conversation_state = ConversationStateMachine.from_dict(state_dict)
                except Exception as exc:
                    logger.debug("Could not restore conversation state: %s", exc)
        return metadata

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up agent and container resources.

        Waits (bounded) for any in-flight ``stream()`` to finish before tearing down
        the agent/provider, so a close() racing an active stream (e.g. a UI on_chat_end
        on a mid-run WebSocket disconnect) does not close the provider out from under it.
        """
        self._closing = True

        if self._active_streams > 0:
            deadline = time.monotonic() + _CLOSE_DRAIN_TIMEOUT_SECONDS
            while self._active_streams > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            if self._active_streams > 0:
                logger.warning(
                    "VictorClient.close(): %d stream(s) still active after drain timeout; "
                    "proceeding with shutdown",
                    self._active_streams,
                )

        if self._agent is not None:
            await self._agent.close()
            self._agent = None

        if self._container is not None:
            try:
                await self._container.adispose()
            except Exception as e:
                logger.debug(f"Container dispose error: {e}")
            self._container = None

        self._context = None
        self._initialized = False

    async def __aenter__(self) -> "VictorClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "pending"
        return (
            f"VictorClient(tool_budget={self._config.tool_budget}, "
            f"smart_routing={self._config.smart_routing.enabled}, status={status})"
        )


__all__ = [
    "VictorClient",
    "_StreamEvent",
]

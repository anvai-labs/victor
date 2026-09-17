"""ExecutionContext — explicit context passing to replace global singletons.

Provides a single context object that carries all runtime dependencies
needed during agent execution. Replaces scattered get_global_manager(),
get_instance() calls with explicit parameter passing.

Usage:
    # Create at agent initialization
    ctx = ExecutionContext.create(settings, container)

    # Pass to coordinators, services, workflows
    result = await coordinator.process(ctx, user_message)

    # Access services via context (not globals)
    state = await ctx.state.get("key", scope=StateScope.CONVERSATION)
    metrics = ctx.services.context.get_context_metrics()

Migration:
    # Before (global singleton)
    from victor.state import get_global_manager
    state = get_global_manager()
    value = await state.get("key")

    # After (explicit context)
    value = await ctx.state.get("key")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.core.container import ServiceContainer
    from victor.state.global_state_manager import GlobalStateManager

logger = logging.getLogger(__name__)


@dataclass
class ServiceAccessor:
    """Typed accessor for resolved service instances.

    Provides attribute-based access to services resolved from the DI container.
    Services are resolved lazily on first access and cached.
    """

    _container: Any = field(repr=False)
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def chat(self) -> Any:
        """ChatServiceProtocol instance."""
        return self._resolve("chat", "ChatServiceProtocol")

    @property
    def tool(self) -> Any:
        """ToolServiceProtocol instance."""
        return self._resolve("tool", "ToolServiceProtocol")

    @property
    def session(self) -> Any:
        """SessionServiceProtocol instance."""
        return self._resolve("session", "SessionServiceProtocol")

    @property
    def context(self) -> Any:
        """ContextServiceProtocol instance."""
        return self._resolve("context", "ContextServiceProtocol")

    @property
    def provider(self) -> Any:
        """ProviderServiceProtocol instance."""
        return self._resolve("provider", "ProviderServiceProtocol")

    @property
    def recovery(self) -> Any:
        """RecoveryServiceProtocol instance."""
        return self._resolve("recovery", "RecoveryServiceProtocol")

    def _resolve(self, key: str, protocol_name: str) -> Any:
        """Lazily resolve a service from the container."""
        if key not in self._cache:
            try:
                from victor.agent.services import protocols as svc_protocols

                proto = getattr(svc_protocols, protocol_name, None)
                if proto and self._container:
                    self._cache[key] = self._container.get_optional(proto)
                else:
                    self._cache[key] = None
            except Exception:
                self._cache[key] = None
        return self._cache[key]


@dataclass(frozen=True)
class ResolvedRuntimeServices:
    """Canonical runtime-owned view of resolved service instances."""

    chat: Any = None
    tool: Any = None
    session: Any = None
    context: Any = None
    provider: Any = None
    recovery: Any = None

    def has_any(self) -> bool:
        """Whether any service handle is populated."""
        return any(
            service is not None
            for service in (
                self.chat,
                self.tool,
                self.session,
                self.context,
                self.provider,
                self.recovery,
            )
        )


def _services_from_accessor(accessor: Any) -> ResolvedRuntimeServices:
    """Build a resolved runtime service bundle from a typed accessor."""

    def _service_attr(name: str) -> Any:
        if isinstance(accessor, ServiceAccessor):
            return getattr(accessor, name, None)

        accessor_dict = getattr(accessor, "__dict__", {})
        if name in accessor_dict or hasattr(type(accessor), name):
            return getattr(accessor, name, None)
        return None

    return ResolvedRuntimeServices(
        chat=_service_attr("chat"),
        tool=_service_attr("tool"),
        session=_service_attr("session"),
        context=_service_attr("context"),
        provider=_service_attr("provider"),
        recovery=_service_attr("recovery"),
    )


def _services_from_runtime_state(
    runtime_state: Dict[str, Any],
) -> ResolvedRuntimeServices:
    """Build a resolved runtime service bundle from owner attributes."""
    return ResolvedRuntimeServices(
        chat=runtime_state.get("_chat_service"),
        tool=runtime_state.get("_tool_service"),
        session=runtime_state.get("_session_service"),
        context=runtime_state.get("_context_service"),
        provider=runtime_state.get("_provider_service"),
        recovery=runtime_state.get("_recovery_service"),
    )


def resolve_execution_context(
    runtime_owner: Any,
    execution_context: Any = None,
) -> Any:
    """Resolve the explicit runtime execution context when available."""
    if execution_context is not None:
        return execution_context

    runtime_state = getattr(runtime_owner, "__dict__", {})
    return runtime_state.get("_execution_context")


def resolve_runtime_services(
    runtime_owner: Any,
    execution_context: Any = None,
) -> ResolvedRuntimeServices:
    """Resolve the canonical runtime service bundle from explicit runtime state."""
    runtime_state = getattr(runtime_owner, "__dict__", {})
    runtime_context = resolve_execution_context(runtime_owner, execution_context)

    services = getattr(runtime_context, "services", None) if runtime_context is not None else None
    if services is not None:
        resolved = _services_from_accessor(services)
        if resolved.has_any():
            return resolved

    context_container = (
        getattr(runtime_context, "container", None) if runtime_context is not None else None
    )
    if context_container is not None:
        accessor = ServiceAccessor(_container=context_container)
        resolved = _services_from_accessor(accessor)
        if resolved.has_any():
            return resolved

    resolved = _services_from_runtime_state(runtime_state)
    if resolved.has_any():
        return resolved

    container = runtime_state.get("_container")
    if container is not None:
        accessor = ServiceAccessor(_container=container)
        return _services_from_accessor(accessor)

    return ResolvedRuntimeServices()


def register_runtime_services(container: Any, services: ResolvedRuntimeServices) -> None:
    """Register the canonical runtime service bundle into the DI container."""
    if container is None:
        return

    from victor.agent.services.protocols import (
        ChatServiceProtocol,
        ContextServiceProtocol,
        ProviderServiceProtocol,
        RecoveryServiceProtocol,
        SessionServiceProtocol,
        ToolServiceProtocol,
    )
    from victor.core.container import ServiceLifetime

    for protocol, service in (
        (ChatServiceProtocol, services.chat),
        (ToolServiceProtocol, services.tool),
        (SessionServiceProtocol, services.session),
        (ContextServiceProtocol, services.context),
        (ProviderServiceProtocol, services.provider),
        (RecoveryServiceProtocol, services.recovery),
    ):
        if service is not None:
            container.register_or_replace(
                protocol,
                lambda c, runtime_service=service: runtime_service,
                ServiceLifetime.SINGLETON,
            )


@dataclass
class RuntimeExecutionContext:
    """Explicit context object replacing global singletons.

    Carries all runtime dependencies needed during agent execution.
    Created once at agent initialization, passed to all components.

    Attributes:
        session_id: Current session identifier
        settings: Application settings (frozen after creation)
        state: State manager (replaces get_global_manager())
        container: DI service container
        services: Typed accessor for resolved services
        metadata: Extensible metadata dict for runtime info
    """

    session_id: str
    settings: Any  # Settings
    state: Any  # GlobalStateManager
    container: Any  # ServiceContainer
    services: ServiceAccessor
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cleanup_hooks: list = field(default_factory=list, repr=False)

    @property
    def prompt_orchestrator(self) -> Any:
        """Typed access to the active prompt orchestrator when configured."""
        return self.metadata.get("prompt_orchestrator")

    @classmethod
    def create(
        cls,
        settings: Any,
        container: Any,
        session_id: str = "",
        state_manager: Optional[Any] = None,
    ) -> RuntimeExecutionContext:
        """Create an ExecutionContext from application settings and container.

        Args:
            settings: Application settings
            container: DI service container
            session_id: Session identifier (empty string if not yet assigned)
            state_manager: Optional explicit state manager. If None, resolves
                          from the factory (transitional — avoids breaking
                          existing code during migration).

        Returns:
            Fully initialized RuntimeExecutionContext
        """
        if state_manager is None:
            try:
                from victor.state.factory import get_global_manager

                state_manager = get_global_manager()
            except Exception:
                state_manager = None

        return cls(
            session_id=session_id,
            settings=settings,
            state=state_manager,
            container=container,
            services=ServiceAccessor(_container=container),
            metadata={},
        )

    def with_session(self, session_id: str) -> RuntimeExecutionContext:
        """Create a new context with a different session ID.

        Returns a shallow copy — services and state are shared.
        """
        return RuntimeExecutionContext(
            session_id=session_id,
            settings=self.settings,
            state=self.state,
            container=self.container,
            services=self.services,
            metadata=dict(self.metadata),
        )

    def with_metadata(self, **kwargs: Any) -> RuntimeExecutionContext:
        """Create a new context with additional metadata."""
        new_meta = dict(self.metadata)
        new_meta.update(kwargs)
        return RuntimeExecutionContext(
            session_id=self.session_id,
            settings=self.settings,
            state=self.state,
            container=self.container,
            services=self.services,
            metadata=new_meta,
        )

    def with_prompt_orchestrator(self, prompt_orchestrator: Any) -> RuntimeExecutionContext:
        """Create a new context with a typed prompt-orchestrator binding."""
        if self.prompt_orchestrator is prompt_orchestrator:
            return self
        return self.with_metadata(prompt_orchestrator=prompt_orchestrator)

    # ── Cleanup Lifecycle (GS-4) ──────────────────────────────────────

    def register_cleanup(self, hook: Any) -> None:
        """Register a cleanup hook to run when the context is disposed.

        Hooks are callables (sync or async) invoked in reverse registration
        order during cleanup(). Use for releasing resources held by
        long-running sessions (connections, caches, temp files).

        Args:
            hook: A callable (sync function, async coroutine function,
                  or object with close()/cleanup() method)
        """
        self._cleanup_hooks.append(hook)

    async def cleanup(self) -> int:
        """Run all registered cleanup hooks in reverse order.

        Returns the number of hooks that executed successfully.
        Logs but does not raise on individual hook failures.
        """
        import asyncio
        import inspect

        success_count = 0
        for hook in reversed(self._cleanup_hooks):
            try:
                if inspect.iscoroutinefunction(hook):
                    await hook()
                elif callable(hook):
                    result = hook()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(hook, "cleanup"):
                    result = hook.cleanup()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(hook, "close"):
                    result = hook.close()
                    if inspect.isawaitable(result):
                        await result
                success_count += 1
            except Exception as e:
                logger.warning("Cleanup hook %s failed: %s", hook, e)
        self._cleanup_hooks.clear()
        return success_count


# =============================================================================
# Backward Compatibility Alias
# =============================================================================


# Alias for backward compatibility (renamed 2026-04-19)
ExecutionContext = RuntimeExecutionContext
"""Deprecated: Use RuntimeExecutionContext instead.

This alias will be removed in v0.11.0. Migration:
  OLD: from victor.runtime.context import ExecutionContext
  NEW: from victor.runtime.context import RuntimeExecutionContext
"""

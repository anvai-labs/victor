"""Service-owned KV tool-ordering policy for the orchestration runtime."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolStrategyRuntime:
    """Own KV-prefix tool ordering while the host retains session state.

    The collaborator is deliberately small: it owns only deterministic ordering
    and configuration interpretation. The orchestrator remains the compatibility
    surface and keeps the per-session cache because its lifetime is the session.
    """

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    def resolve_kv_strategy_setting(self) -> str:
        """Read ``kv_tool_strategy`` from settings, defaulting to context-aware."""
        try:
            settings = getattr(self._runtime, "settings", None)
            context = getattr(settings, "context", None)
            if context is not None:
                return getattr(context, "kv_tool_strategy", "context_aware")
        except Exception:
            pass
        return "context_aware"

    def should_stabilize_tool_order(self) -> bool:
        """Return whether this turn needs byte-stable tool ordering."""
        runtime = self._runtime
        if runtime._kv_optimization_enabled:
            return True
        strategy = self.resolve_kv_strategy_setting()
        if strategy in ("session_stable", "additive"):
            return True
        if strategy != "context_aware":
            return False
        return self.context_aware_profile_session_locked()

    def context_aware_profile_session_locked(self) -> bool:
        """Whether provider economics favor a session-stable tool prefix."""
        runtime = self._runtime
        try:
            from victor.config.tool_tiers import resolve_tool_supply_profile

            context_window = runtime._get_context_window(runtime.provider, runtime.model)
            fallback_max_tools = getattr(getattr(runtime.settings, "tools", None), "budget", None)
            profile = resolve_tool_supply_profile(
                runtime.provider,
                context_window,
                fallback_max_tools=int(fallback_max_tools or 8),
            )
            return profile.session_lock != "off"
        except Exception:
            return False

    def is_tool_strategy_v2_enabled(self) -> bool:
        """Return the single feature-flag authority for strategy v2."""
        try:
            from victor.core.feature_flags import is_enabled

            return is_enabled("tool_strategy_v2")
        except Exception:
            return False

    def get_context_window(self, provider: Any, model: str) -> int:
        """Return provider context capacity, with a conservative fallback."""
        if hasattr(provider, "context_window"):
            return provider.context_window(model)
        logger.warning(
            "Provider %s does not support context_window(), using default 8192",
            provider.name,
        )
        return 8192

    def estimate_tool_tokens(
        self, tool: Any, provider_category: str | None = None, _use_cache: bool = True
    ) -> int:
        """Delegate schema-token estimation to the canonical ToolService."""
        return self._runtime._tool_service.estimate_tool_tokens(
            tool,
            provider_category=provider_category,
            _use_cache=_use_cache,
        )

    def semantic_select_tools(
        self,
        tools: Any,
        max_tokens: int,
        provider_category: str | None = None,
    ) -> list[Any]:
        """Delegate semantic selection to the canonical ToolService."""
        return self._runtime._tool_service.semantic_select_tools(
            tools,
            max_tokens,
            provider_category=provider_category,
        )

    def apply_context_aware_strategy(self, tools: Any) -> Any:
        """Select context-aware tools and record the resulting strategy event."""
        runtime = self._runtime
        result = runtime._tool_service.apply_context_aware_strategy(
            tools,
            provider=runtime.provider,
            model=runtime.model,
        )
        try:
            from victor.config.tool_tiers import get_provider_category

            context_window = runtime._get_context_window(runtime.provider, runtime.model)
            provider_category = get_provider_category(context_window)
            runtime._emit_tool_strategy_event(
                strategy="context_aware",
                tool_count=len(result),
                tool_tokens=sum(
                    runtime._estimate_tool_tokens(tool, provider_category) for tool in result
                ),
                context_window=context_window,
                provider=runtime.provider,
                reason="kv_context_aware",
                tools=result,
            )
        except Exception:
            pass
        return result

    def apply_kv_tool_strategy(self, tools: Any) -> Any:
        """Apply the configured strategy and retain any session-stable selection."""
        runtime = self._runtime
        if runtime._is_tool_strategy_v2_enabled():
            result = runtime._tool_service.apply_context_aware_strategy(
                tools,
                provider=runtime.provider,
                model=runtime.model,
                session_semantic_tools=getattr(runtime, "_session_semantic_tools", None),
            )
            if runtime._context_aware_profile_session_locked():
                runtime._session_semantic_tools = result
            return result

        strategy = runtime._resolve_kv_strategy_setting()
        result = runtime._tool_service.apply_kv_tool_strategy(
            tools,
            kv_optimization_enabled=runtime._kv_optimization_enabled,
            provider=runtime.provider,
            model=runtime.model,
            session_semantic_tools=getattr(runtime, "_session_semantic_tools", None),
            kv_tool_strategy=strategy,
        )
        if strategy in ("session_stable", "additive") or (
            strategy == "context_aware" and runtime._context_aware_profile_session_locked()
        ):
            runtime._session_semantic_tools = result
        return result

    def emit_tool_strategy_event(
        self,
        strategy: str,
        tool_count: int,
        tool_tokens: int,
        context_window: int,
        provider: Any,
        reason: str,
        tools: Any = None,
    ) -> None:
        """Emit strategy telemetry through the runtime metrics coordinator."""
        runtime = self._runtime
        try:
            runtime._metrics_coordinator.emit_tool_strategy_event(
                strategy=strategy,
                tool_count=tool_count,
                tool_tokens=tool_tokens,
                context_window=context_window,
                provider=provider,
                model=runtime.model,
                reason=reason,
                tools=tools,
                v2_enabled=runtime._is_tool_strategy_v2_enabled(),
            )
        except Exception as exc:
            logger.debug("Failed to emit tool strategy event: %s", exc)

    def sort_tools_for_kv_stability(self, tools: Any) -> Any:
        """Sort tools once per named set and retain the session-local result."""
        if tools is None:
            return None
        if not self.should_stabilize_tool_order():
            return tools

        runtime = self._runtime
        current_names = frozenset(tool.name for tool in tools)
        if (
            getattr(runtime, "_last_sorted_tool_names", None) == current_names
            and getattr(runtime, "_last_sorted_tools", None) is not None
        ):
            return runtime._last_sorted_tools

        sorted_tools = runtime._tool_service.sort_tools_for_kv_stability(
            tools,
            kv_optimization_enabled=True,
        )
        runtime._last_sorted_tool_names = current_names
        runtime._last_sorted_tools = sorted_tools
        return sorted_tools

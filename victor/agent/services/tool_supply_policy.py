# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tool-supply policy decisions shared by ToolService and the turn runtimes.

Lives outside ToolService so the hotspot file stays under its size ratchet
and the pruning-economics decisions are testable without constructing the
service. ToolService retains thin delegations for interface stability.
"""

from __future__ import annotations

from typing import Any, Optional


def pruning_disabled_for(config: Any) -> bool:
    """True when per-turn pruning is explicitly disabled (the default).

    Defensive against ``__new__``-constructed services (tests) that skip
    ``__init__``: a missing config means legacy behavior (semantic selection
    active).
    """
    enabled = getattr(config, "tool_selection_enabled", True)
    return not enabled


def pruning_disabled(service: Any) -> bool:
    """Instance-bound form of ``pruning_disabled_for`` (reads service config)."""
    return pruning_disabled_for(getattr(service, "_config", None))


def fallback_max_full_tools(service: Any) -> int:
    """Configured full-schema head size for supply-profile resolution.

    Group precedence: ``tool_selection.fallback_max_tools`` (where the flat
    ``VICTOR_FALLBACK_MAX_TOOLS`` legacy mapping and profiles land), then
    ``tools.fallback_max_tools``, else the 8 default.
    """
    try:
        settings = getattr(service, "_settings", None)
        if settings is None:
            from victor.config.settings import load_settings

            settings = load_settings()
        for group_name in ("tool_selection", "tools"):
            group = getattr(settings, group_name, None)
            value = getattr(group, "fallback_max_tools", None)
            if value and int(value) > 0:
                return int(value)
    except Exception:
        pass
    return 8


def _get_tool_context_window(service: Any, provider: Any, model: Any) -> int:
    if hasattr(provider, "context_window"):
        return provider.context_window(model)
    service._logger.warning(
        f"Provider {getattr(provider, 'name', '?')} has no context_window(); using 8192"
    )
    return 8192


def _should_session_lock_tools(service: Any, provider: Any, context_window: int) -> bool:
    if hasattr(provider, "supports_prompt_caching") and provider.supports_prompt_caching():
        return True
    return context_window >= 32000


def _merge_session_tools(tools: list, cached: Optional[list]) -> list:
    """Additive session-stable merge (tool-supply P4): grow-only tool set.

    Returns the cached set plus any newly-selected tools not already in it.
    When nothing new is selected, returns the cached object unchanged so the
    downstream name-sort yields a byte-identical prefix (KV cache hit).
    """
    if not cached:
        return tools
    cached_names = {t.name for t in cached}
    additions = [t for t in tools if t.name not in cached_names]
    if not additions:
        return cached
    return list(cached) + additions


def _estimate_all(service: Any, tools: list, provider_category: Optional[str]) -> int:
    return sum(service.estimate_tool_tokens(t, provider_category=provider_category) for t in tools)


def apply_context_aware_strategy(
    service: Any,
    tools: list,
    *,
    provider: Any,
    model: Any,
    session_semantic_tools: Optional[list] = None,
) -> list:
    """Economy-first, context-window-aware tool selection.

    Decision tree (priority order):
    1. Resolve one provider-economics profile for cache/token tradeoffs.
    2. Demote/drop tools that exceed the profile's tool-schema budget.
    3. Apply additive session stability when the profile asks for it.
    4. For hard token-constrained profiles, semantic-select within budget.

    Does NOT emit tool-strategy events — that responsibility stays with the
    orchestrator shim to preserve ``AgentMetricsService`` ownership.
    """
    if pruning_disabled(service):
        # Pruning disabled (default): expose the registered toolset as-is.
        return list(tools)

    from victor.config.tool_tiers import get_provider_category
    from victor.config.tool_tiers import resolve_tool_supply_profile

    context_window = _get_tool_context_window(service, provider, model)
    max_full_tools = fallback_max_full_tools(service)
    profile = resolve_tool_supply_profile(
        provider,
        context_window,
        fallback_max_tools=max_full_tools,
    )
    provider_category = get_provider_category(context_window)
    tool_tokens = _estimate_all(service, tools, provider_category)
    max_tool_tokens = profile.budget_tokens or int(context_window * 0.25)

    if tool_tokens > max_tool_tokens:
        service._logger.warning(
            f"Tool tokens ({tool_tokens}) exceed 25% of context window ({context_window}). "
            "Demoting low-priority tools."
        )
        tools = _demote_tools_to_fit_budget(
            service, tools, max_tool_tokens, context_window, provider_category
        )

    if profile.cap_mode == "none" or _should_session_lock_tools(service, provider, context_window):
        if profile.session_lock == "additive":
            return _merge_session_tools(tools, session_semantic_tools)
        return tools

    # Route through the service method (not this module) so tests that patch
    # the instance seam keep working.
    tools = service.semantic_select_tools(
        tools, max_tool_tokens, provider_category=provider_category
    )
    if profile.session_lock == "additive":
        return _merge_session_tools(tools, session_semantic_tools)
    return tools


def semantic_select_tools(
    service: Any, tools: list, max_tokens: int, *, provider_category: Optional[str] = None
) -> list:
    """Select tools by semantic relevance within a token budget.

    CRITICAL-priority tools are always included first. Remaining tools are
    added in declaration order until the budget is 90 % consumed.
    """
    if pruning_disabled(service):
        return list(tools)

    from victor.tools.enums import Priority

    core_tools = [t for t in tools if hasattr(t, "priority") and t.priority == Priority.CRITICAL]
    core_tokens = sum(
        service.estimate_tool_tokens(t, provider_category=provider_category) for t in core_tools
    )

    if core_tokens > max_tokens:
        result, used = [], 0
        for tool in core_tools:
            cost = service.estimate_tool_tokens(tool)
            if used + cost <= max_tokens:
                result.append(tool)
                used += cost
        return result

    selected = core_tools.copy()
    for tool in (t for t in tools if t not in core_tools):
        cost = service.estimate_tool_tokens(tool, provider_category=provider_category)
        if core_tokens + cost <= max_tokens:
            selected.append(tool)
            core_tokens += cost
            if core_tokens >= max_tokens * 0.9:
                break

    return selected


def _demote_tools_to_fit_budget(
    service: Any,
    tools: list,
    max_tokens: int,
    context_window: int,
    provider_category: Optional[str] = None,
) -> list:
    from victor.tools.enums import Priority, SchemaLevel

    sorted_tools = sorted(
        tools,
        key=lambda t: (t.priority.value if hasattr(t, "priority") else 99, t.name),
    )
    result, used = [], 0
    for tool in sorted_tools:
        cost = service.estimate_tool_tokens(tool, provider_category=provider_category)
        if used + cost <= max_tokens:
            result.append(tool)
            used += cost
        elif hasattr(tool, "priority") and tool.priority == Priority.CRITICAL:
            try:
                original = getattr(tool, "_schema_level", None)
                tool._schema_level = SchemaLevel.STUB
                stub_cost = service.estimate_tool_tokens(tool, _use_cache=False)
                tool._schema_level = original
                if used + stub_cost <= max_tokens:
                    result.append(tool)
                    used += stub_cost
                    service._logger.debug(f"Demoted critical tool {tool.name} to STUB")
                else:
                    tool._schema_level = original
            except Exception:
                service._logger.debug(f"Demote check failed for {tool.name}", exc_info=True)
    return result

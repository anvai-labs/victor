# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Regression: a curated enabled-tools set must include registered-but-disabled tools.

Root cause this guards: ``set_enabled_tools()`` (tool_access_policy) updates the
policy + selector filter but NOT the registry's per-tool ``_tool_enabled`` map.
``select_keywords`` gathered from ``list_tools(only_enabled=True)``, which checks
that map — so curated tools that are registered-but-disabled (e.g. code/graph)
were silently dropped from the advertised schema, even though the benchmark
explicitly enabled them and the prompt advertised them. The fix gathers from
``list_tools(only_enabled=False)`` filtered by the curated set.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from victor.agent.tool_selection import ToolSelector
from victor.providers.base import ToolDefinition

_CURATED = {"code", "edit", "graph", "read", "shell", "write"}


def _make_registry():
    """Mock registry: only_enabled=True → 4 enabled; only_enabled=False → all 6.

    Mirrors the real bug condition: code/graph are registered but disabled in
    the registry's _tool_enabled map, while read/shell/edit/write are enabled.
    """
    enabled = [
        ToolDefinition(name=n, description="d", parameters={})
        for n in ("read", "shell", "edit", "write")
    ]
    disabled = [ToolDefinition(name=n, description="d", parameters={}) for n in ("code", "graph")]

    reg = MagicMock()

    def list_tools(only_enabled=True, include_folded=False):
        return list(enabled) if only_enabled else list(enabled) + list(disabled)

    reg.list_tools.side_effect = list_tools

    def _mock_tool(n):
        t = MagicMock()
        t.name = n
        t.description = f"{n} tool"
        t.parameters = {"type": "object", "properties": {}}
        return t

    reg.get.side_effect = lambda n: _mock_tool(n) if n in _CURATED else None
    return reg


def test_curated_set_includes_registered_but_disabled():
    """code/graph (registered, disabled) must be gathered when curated."""
    selector = ToolSelector(tools=_make_registry(), tool_selection_config={})
    selector.set_enabled_tools(set(_CURATED))

    result = selector.select_keywords("Fix the bug by editing the source code.")
    names = {t.name for t in result}

    assert {"code", "graph"} <= names, f"curated code/graph dropped: {sorted(names)}"
    assert names <= _CURATED, f"non-curated tools leaked: {sorted(names - _CURATED)}"


def test_curated_set_respects_filter_when_all_enabled():
    """When all curated tools are registry-enabled, behavior is unchanged."""
    reg = MagicMock()
    all_tools = [ToolDefinition(name=n, description="d", parameters={}) for n in _CURATED]
    reg.list_tools.side_effect = lambda only_enabled=True, include_folded=False: list(all_tools)

    selector = ToolSelector(tools=reg, tool_selection_config={})
    selector.set_enabled_tools(set(_CURATED))
    result = selector.select_keywords("Fix the bug.")
    assert {t.name for t in result} <= _CURATED


def test_union_curated_enabled_restores_semantic_drop():
    """The semantic path can drop code/graph; _union_curated_enabled restores them.

    Mirrors the real bug: semantic selection gathers via list_tools(only_enabled=True),
    which excludes registered-but-disabled tools. The curated set (code/graph) must
    still reach the LLM, so select_tools unions them back in.
    """
    from unittest.mock import MagicMock

    from victor.agent.tool_selection import ToolSelector

    enabled = [
        ToolDefinition(name=n, description="d", parameters={})
        for n in ("read", "shell", "edit", "write")
    ]
    disabled = [ToolDefinition(name=n, description="d", parameters={}) for n in ("code", "graph")]
    reg = MagicMock()
    reg.list_tools.side_effect = lambda only_enabled=True, include_folded=False: (
        list(enabled) if only_enabled else list(enabled) + list(disabled)
    )

    sel = ToolSelector.__new__(ToolSelector)
    sel.tools = reg
    sel._enabled_tools = set(_CURATED)

    # Semantic path returned only the 4 enabled (dropped code/graph)
    dropped = list(enabled)
    restored = sel._union_curated_enabled(dropped)
    assert {t.name for t in restored} == _CURATED
    assert {"code", "graph"} <= {t.name for t in restored}


def test_union_curated_enabled_noop_without_curated_set():
    """No curated set (common auto-selected case) → union is a no-op."""
    from unittest.mock import MagicMock

    from victor.agent.tool_selection import ToolSelector

    reg = MagicMock()
    sel = ToolSelector.__new__(ToolSelector)
    sel.tools = reg
    sel._enabled_tools = None
    tools = [ToolDefinition(name="read", description="d", parameters={})]
    assert sel._union_curated_enabled(tools) is tools


async def test_select_tools_short_circuits_to_stable_curated_set():
    """select_tools (the path the benchmark actually calls) returns the stable
    6-tool curated set immediately, bypassing semantic/keyword selection that
    drops code/graph (#353 fix — the short-circuit previously lived only in
    select_tools_for_turn, which the benchmark never calls)."""
    selector = ToolSelector(tools=_make_registry(), tool_selection_config={})
    selector.set_enabled_tools(set(_CURATED))

    result = await selector.select_tools("Fix the bug by editing the source code.")
    names = {t.name for t in result}

    assert names == _CURATED, f"expected stable 6-tool curated set, got {sorted(names)}"


async def test_select_tools_no_curated_set_uses_normal_selection():
    """Without a curated set, select_tools falls through to normal selection
    (the short-circuit is a no-op — doesn't change auto-selected behavior)."""
    selector = ToolSelector(tools=_make_registry(), tool_selection_config={})
    # No set_enabled_tools → _enabled_tools is None → no short-circuit.
    result = await selector.select_tools("Fix the bug by editing the source code.")
    # Falls to select_keywords (no semantic_selector) → returns some tools.
    assert isinstance(result, list)

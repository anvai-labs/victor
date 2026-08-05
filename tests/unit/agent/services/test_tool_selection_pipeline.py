# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""tool-supply P8 — the ACT pipeline is an explicit ordered stage list (byte-identical).

Empirical evidence that the refactor of the procedural pile into a declared pipeline
preserves stage order, applies every transform, and records each stage in the trace.
Shipped on by default (no flag): the existing order/arg/output-pinning runtime suite
plus these tests are the parity gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.tool_selection_runtime import (
    ToolSelectionRuntime,
    _SelectionStage,
)
from victor.tools.tool_supply_trace import ToolSupplyTrace

# Canonical ACT stage order — a regression here means the pile was reordered/dropped.
EXPECTED_STAGES = [
    "stage_priority",
    "intent_filter",
    "ensure_write",
    "explicit_db",
    "e3tir_rerank",
    "kv_strategy",
    "kv_sort",
]


class _T:
    def __init__(self, name: str) -> None:
        self.name = name


def test_run_act_pipeline_applies_stages_in_order_and_records_trace():
    trace = ToolSupplyTrace.begin(None)
    tools = [_T("a"), _T("b"), _T("c")]
    stages = [
        _SelectionStage("drop_c", lambda ts: [t for t in ts if t.name != "c"]),
        _SelectionStage("reverse", lambda ts: list(reversed(ts)), "rev"),
    ]
    out = ToolSelectionRuntime._run_act_pipeline(tools, stages, trace)
    assert [t.name for t in out] == ["b", "a"]
    # Trace recorded each stage in order, with the drop attributed to the right stage.
    assert [s.name for s in trace.stages] == ["drop_c", "reverse"]
    assert trace.stages[0].dropped == ("c",)
    assert trace.stages[1].dropped == ()
    assert trace.stages[1].reason == "rev"


def test_act_pipeline_declares_the_seven_canonical_stages_in_order():
    runtime = SimpleNamespace(
        tool_selector=MagicMock(),
        _tool_planner=MagicMock(),
        _apply_kv_tool_strategy=MagicMock(),
        _sort_tools_for_kv_stability=MagicMock(),
    )
    rt = ToolSelectionRuntime(runtime)
    stages = rt._act_pipeline_stages(runtime, "msg", current_intent=None)
    assert [s.name for s in stages] == EXPECTED_STAGES


@pytest.mark.asyncio
async def test_full_turn_emits_trace_with_all_seven_stages(monkeypatch):
    """End-to-end: a real select_tools_for_turn run records all 7 ACT stages in order."""
    captured = {}

    class _Bus:
        def emit_sync(self, **kwargs):
            captured["data"] = kwargs.get("data")

    import victor.core.events as events_mod

    monkeypatch.setattr(events_mod, "get_observability_bus", lambda: _Bus())

    provider = MagicMock()
    provider.supports_tools.return_value = True
    selected = [_T("read"), _T("write")]
    tool_selector = MagicMock()
    tool_selector.select_tools = AsyncMock(return_value=selected)
    tool_selector.prioritize_by_stage.side_effect = lambda _m, t: t
    tool_planner = MagicMock()
    tool_planner.filter_tools_by_intent.side_effect = lambda t, *_a, **_k: t
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _tool_skip_mode=MagicMock(return_value="tools"),
        _should_skip_tools_for_turn=MagicMock(return_value=False),
        observed_files=set(),
        _tool_planner=tool_planner,
        conversation=SimpleNamespace(message_count=MagicMock(return_value=1)),
        messages=[],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent=None,
        _current_user_message="inspect and edit app.py",
        _apply_kv_tool_strategy=MagicMock(side_effect=lambda t: t),
        _sort_tools_for_kv_stability=MagicMock(side_effect=lambda t: t),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn("inspect and edit app.py", goals=None)

    assert [t.name for t in result] == ["read", "write"]
    assert captured["data"] is not None
    assert [s["name"] for s in captured["data"]["stages"]] == EXPECTED_STAGES
    assert captured["data"]["dispatched"] == ["read", "write"]

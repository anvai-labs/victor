# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for per-turn tool-supply telemetry (P0).

These cover the pure trace builder and the best-effort emit helper. The
no-behavior-change guarantee for ``select_tools_for_turn`` is covered by the
existing ``test_tool_selection_runtime.py`` suite continuing to pass.
"""

from __future__ import annotations

from victor.tools.tool_supply_trace import (
    TOOL_SUPPLY_TOPIC,
    GateRecord,
    ToolSupplyTrace,
)


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


def _tools(*names: str):
    return [_Tool(n) for n in names]


def test_topic_constant():
    assert TOOL_SUPPLY_TOPIC == "tool.supply"


def test_record_computes_dropped_by_name():
    trace = ToolSupplyTrace.begin(_tools("read", "write", "edit"))
    before = _tools("read", "write", "edit")
    after = _tools("read")  # write + edit removed
    trace.record("intent_filter", before, after, reason="READ_ONLY")

    assert len(trace.stages) == 1
    rec = trace.stages[0]
    assert isinstance(rec, GateRecord)
    assert rec.name == "intent_filter"
    assert rec.in_count == 3
    assert rec.out_count == 1
    assert rec.dropped == ("write", "edit")
    assert rec.reason == "READ_ONLY"


def test_record_returns_after_unchanged():
    trace = ToolSupplyTrace.begin(None)
    after = _tools("read")
    assert trace.record("s", _tools("read", "x"), after) is after


def test_reorder_drops_nothing():
    trace = ToolSupplyTrace.begin(None)
    before = _tools("a", "b")
    after = _tools("b", "a")  # pure reorder (kv_sort)
    trace.record("kv_sort", before, after)
    assert trace.stages[0].dropped == ()
    assert trace.stages[0].in_count == trace.stages[0].out_count == 2


def test_full_funnel_payload():
    trace = ToolSupplyTrace.begin(_tools("read", "write", "edit", "grep", "shell"))
    cand = _tools("read", "write", "edit", "grep")
    trace.set_candidates(cand)
    after_intent = _tools("read", "grep")
    trace.record("intent_filter", cand, after_intent)
    trace.finalize(after_intent)

    payload = trace.to_payload()
    assert payload["registered_count"] == 5
    assert payload["candidate_count"] == 4
    assert payload["dispatched_count"] == 2
    assert payload["skipped"] is False
    assert payload["dispatched"] == ["read", "grep"]
    assert [s["name"] for s in payload["stages"]] == ["intent_filter"]
    # payload must be JSON-friendly (lists, not tuples)
    assert isinstance(payload["stages"][0]["dropped"], list)


def test_mark_skipped_clears_dispatched():
    trace = ToolSupplyTrace.begin(_tools("read", "write"))
    trace.set_candidates(_tools("read"))
    trace.mark_skipped("qa_necessity_gate")
    payload = trace.to_payload()
    assert payload["skipped"] is True
    assert payload["skip_reason"] == "qa_necessity_gate"
    assert payload["dispatched_count"] == 0
    assert payload["dispatched"] == []


def test_demoted_to_stub_passthrough():
    trace = ToolSupplyTrace.begin(None)
    before = _tools("a", "b", "c")
    after = _tools("a", "b", "c")  # nothing dropped, but two demoted
    trace.record("budget", before, after, demoted=["b", "c"])
    assert trace.stages[0].dropped == ()
    assert trace.stages[0].demoted_to_stub == ("b", "c")


def test_names_handle_strings_and_none():
    # begin(None) yields empty registered; record tolerates None sides
    trace = ToolSupplyTrace.begin(None)
    trace.record("s", None, None)
    assert trace.stages[0].in_count == 0
    assert trace.stages[0].out_count == 0


# --- emit helper -----------------------------------------------------------------


def test_emit_helper_emits_once(monkeypatch):
    import victor.core.events as events_mod
    from victor.agent.services import tool_selection_runtime as tsr

    calls = []

    class _Bus:
        def emit_sync(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(events_mod, "get_observability_bus", lambda: _Bus())

    trace = ToolSupplyTrace.begin(_tools("read"))
    trace.finalize(_tools("read"))
    tsr._emit_tool_supply_trace(trace)

    assert len(calls) == 1
    assert calls[0]["topic"] == TOOL_SUPPLY_TOPIC
    assert calls[0]["source"] == "ToolSelectionRuntime"
    assert calls[0]["data"]["dispatched"] == ["read"]


def test_emit_helper_noop_when_no_bus(monkeypatch):
    import victor.core.events as events_mod
    from victor.agent.services import tool_selection_runtime as tsr

    monkeypatch.setattr(events_mod, "get_observability_bus", lambda: None)
    # Must not raise.
    tsr._emit_tool_supply_trace(ToolSupplyTrace.begin(None))


def test_emit_helper_swallows_errors(monkeypatch):
    import victor.core.events as events_mod
    from victor.agent.services import tool_selection_runtime as tsr

    def _boom():
        raise RuntimeError("bus down")

    monkeypatch.setattr(events_mod, "get_observability_bus", _boom)
    # Telemetry must never propagate failures into selection.
    tsr._emit_tool_supply_trace(ToolSupplyTrace.begin(None))


# --- correlation spine (R1) ------------------------------------------------------


def test_trace_captures_correlation_from_context():
    from victor.core import context as ctx

    s = ctx.set_session_id("sess-trace")
    t = ctx.set_turn_id("turn-trace")
    r = ctx.set_request_id("req-trace")
    try:
        payload = ToolSupplyTrace.begin(_tools("read")).finalize(_tools("read")).to_payload()
        assert payload["session_id"] == "sess-trace"
        assert payload["turn_id"] == "turn-trace"
        assert payload["request_id"] == "req-trace"
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)
        ctx.request_id.reset(r)


def test_trace_correlation_empty_when_context_unset():
    from victor.core import context as ctx

    s = ctx.set_session_id("")
    t = ctx.set_turn_id("")
    try:
        payload = ToolSupplyTrace.begin(None).to_payload()
        assert payload["session_id"] == ""
        assert payload["turn_id"] == ""
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)

# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""tool-supply P3 — Q&A necessity gate returns read-core, not None."""

from __future__ import annotations

from victor.agent.orchestrator import AgentOrchestrator


class _Gate:
    """Minimal stand-in carrying the real gate logic (avoids full orchestrator build).

    The keyword-set constants now live in ``victor/agent/tool_supply_policy.py``
    (ADR-019 increment 2); ``_tool_skip_mode`` delegates there, so the stub only
    needs to supply the edge-check callback.
    """

    _tool_skip_mode = AgentOrchestrator._tool_skip_mode
    _should_skip_tools_for_turn = AgentOrchestrator._should_skip_tools_for_turn

    def __init__(self, edge_says_skip: bool = True):
        self._edge_says_skip = edge_says_skip

    def _check_tool_necessity_via_edge(self, context_msg, heuristic_conf):
        return self._edge_says_skip


def test_greeting_is_hard_skip():
    g = _Gate()
    assert g._tool_skip_mode("hi") == "skip"
    assert g._tool_skip_mode("thanks!") == "skip"


def test_short_command_gets_tools():
    assert _Gate()._tool_skip_mode("fix it") == "tools"
    assert _Gate()._tool_skip_mode("run tests") == "tools"


def test_borderline_qa_gets_read_core_not_skip():
    # The key P3 fix: a "how does X work?" turn the edge model would skip now keeps a
    # read-only core instead of having ALL tools removed.
    g = _Gate(edge_says_skip=True)
    assert g._tool_skip_mode("how does the auth flow work") == "read_core"
    assert g._tool_skip_mode("what does this module do") == "read_core"


def test_borderline_qa_gets_tools_when_edge_disagrees():
    g = _Gate(edge_says_skip=False)
    assert g._tool_skip_mode("how does the auth flow work") == "tools"


def test_multi_tool_signal_gets_tools():
    assert _Gate()._tool_skip_mode("read the file and fix the bug in auth") == "tools"


def test_should_skip_backcompat_true_for_skip_and_read_core():
    # Back-compat bool: both "skip" and "read_core" map to True (not "tools").
    assert _Gate(edge_says_skip=True)._should_skip_tools_for_turn("hi") is True
    assert (
        _Gate(edge_says_skip=True)._should_skip_tools_for_turn("how does the auth flow work")
        is True
    )
    assert _Gate()._should_skip_tools_for_turn("read the file and fix the bug in auth") is False

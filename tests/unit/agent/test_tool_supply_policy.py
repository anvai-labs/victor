"""Unit tests for the pure tool-supply policy (ADR-019 increment 2 extraction)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional

from victor.agent.tool_supply_policy import (
    classify_tool_supply,
    demote_tools_to_fit,
    should_skip_tools_via_edge,
)
from victor.tools.enums import Priority, SchemaLevel

# ── classify_tool_supply ──────────────────────────────────────────


def _never_skip(_msg: str, _conf: float) -> bool:
    return False


def _always_skip(_msg: str, _conf: float) -> bool:
    return True


def test_continuation_token_gets_read_core() -> None:
    assert classify_tool_supply("continue", edge_check=_never_skip) == "read_core"
    assert classify_tool_supply("apply it", edge_check=_never_skip) == "read_core"


def test_greeting_is_hard_skip() -> None:
    assert classify_tool_supply("hi", edge_check=_never_skip) == "skip"
    assert classify_tool_supply("thanks!", edge_check=_never_skip) == "skip"


def test_short_command_gets_tools() -> None:
    assert classify_tool_supply("fix it", edge_check=_never_skip) == "tools"
    assert classify_tool_supply("run tests", edge_check=_never_skip) == "tools"


def test_multi_tool_signal_gets_tools() -> None:
    assert (
        classify_tool_supply("read the file and fix the bug in auth", edge_check=_always_skip)
        == "tools"
    )


def test_borderline_qa_read_core_when_edge_skips() -> None:
    assert (
        classify_tool_supply("how does the auth flow work", edge_check=_always_skip) == "read_core"
    )
    assert classify_tool_supply("what does this module do", edge_check=_always_skip) == "read_core"


def test_borderline_qa_tools_when_edge_disagrees() -> None:
    assert classify_tool_supply("how does the auth flow work", edge_check=_never_skip) == "tools"


def test_edge_check_receives_message_and_confidence() -> None:
    seen: List[Any] = []

    def _spy(msg: str, conf: float) -> bool:
        seen.append((msg, conf))
        return True

    classify_tool_supply("how does the auth flow work", edge_check=_spy)
    assert seen == [("how does the auth flow work", 0.85)]


def test_edge_decision_falls_back_to_heuristic_without_service() -> None:
    assert should_skip_tools_via_edge("what is Victor", 0.85) is True
    assert should_skip_tools_via_edge("what is Victor", 0.6) is False


# ── demote_tools_to_fit ───────────────────────────────────────────


def _tool(name: str, priority: Priority, full: int, stub: int) -> Any:
    return SimpleNamespace(name=name, priority=priority, _schema_level=None, _full=full, _stub=stub)


def _estimator(tool: Any, provider_category: Optional[str] = None) -> int:
    return tool._stub if getattr(tool, "_schema_level", None) == SchemaLevel.STUB else tool._full


def _names(tools: List[Any]) -> List[str]:
    return [t.name for t in tools]


def test_all_tools_fit_within_budget() -> None:
    tools = [_tool("a", Priority.HIGH, 10, 5), _tool("b", Priority.LOW, 10, 5)]
    kept = demote_tools_to_fit(
        tools, max_tokens=100, context_window=8000, estimate_tokens=_estimator
    )
    assert _names(kept) == ["a", "b"]


def test_noncritical_tool_dropped_when_over_budget() -> None:
    tools = [_tool("crit", Priority.CRITICAL, 60, 20), _tool("opt", Priority.LOW, 60, 20)]
    kept = demote_tools_to_fit(
        tools, max_tokens=100, context_window=8000, estimate_tokens=_estimator
    )
    assert _names(kept) == [
        "crit"
    ]  # critical fits (60); non-critical (60) would overflow → dropped


def test_critical_tool_demoted_to_stub_to_fit() -> None:
    tools = [_tool("c1", Priority.CRITICAL, 60, 20), _tool("c2", Priority.CRITICAL, 60, 20)]
    kept = demote_tools_to_fit(
        tools, max_tokens=100, context_window=8000, estimate_tokens=_estimator
    )
    # c1 full (60) fits; c2 full (60) overflows but STUB (20) fits → 80 total.
    assert _names(kept) == ["c1", "c2"]


def test_critical_tool_dropped_when_stub_still_too_big() -> None:
    tools = [_tool("c1", Priority.CRITICAL, 60, 20), _tool("c2", Priority.CRITICAL, 60, 50)]
    kept = demote_tools_to_fit(
        tools, max_tokens=100, context_window=8000, estimate_tokens=_estimator
    )
    # c1 (60) fits; c2 STUB (50) → 110 > 100 → dropped.
    assert _names(kept) == ["c1"]

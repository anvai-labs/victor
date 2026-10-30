"""Tests for the shared keyword tool selector in victor.tools.selection_common."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from victor.tools import selection_common
from victor.tools.selection_common import select_tools_by_keywords

_MODULE = "victor.tools.selection_common"


def _select(message: str, names: set[str], *args, critical=None, matches=None, **kwargs):
    with (
        patch(f"{_MODULE}.get_critical_tools", return_value=set(critical or ())) as crit,
        patch(f"{_MODULE}.get_tools_from_message", return_value=set(matches or ())),
    ):
        return select_tools_by_keywords(message, names, *args, **kwargs), crit


def test_selects_critical_and_keyword_matches_filtered_to_available() -> None:
    selected, _ = _select(
        "read the file",
        {"read", "ls", "web"},
        critical={"read", "missing"},
        matches={"ls", "absent"},
    )
    assert selected == {"read", "ls"}


def test_small_model_limit_keeps_critical_tools() -> None:
    names = {"read", "a", "b", "c"}
    selected, _ = _select("x", names, True, 2, critical={"read"}, matches={"a", "b", "c"})
    assert "read" in selected
    assert len(selected) == 2


def test_registry_is_keyword_only_and_forwarded() -> None:
    registry = object()
    _, crit = _select("x", {"read"}, critical={"read"}, registry=registry)
    crit.assert_called_once_with(registry)


def test_third_positional_argument_is_is_small_not_registry() -> None:
    # Re-exported as victor.agent.tool_selection.select_tools_by_keywords, whose
    # historical signature is (message, names, is_small, max_tools_for_small).
    names = {"read", "a", "b", "c"}
    selected, crit = _select("x", names, True, 1, critical=set(), matches={"a", "b", "c"})
    assert len(selected) == 1
    crit.assert_called_once_with(None)


def test_registry_rejected_positionally() -> None:
    with pytest.raises(TypeError):
        select_tools_by_keywords("x", set(), False, 10, object())  # type: ignore[misc]


def test_agent_tool_selection_reexports_shared_function() -> None:
    from victor.agent.tool_selection import select_tools_by_keywords as reexported

    assert reexported is selection_common.select_tools_by_keywords

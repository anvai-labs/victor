"""Tests for the production TOON formatter and the shared TOON serializer."""

from __future__ import annotations

from typing import Any

import pytest

from victor.agent.format_strategies import ToolOutputFormat, ToonFormatStrategy, toon_serialize
from victor.agent.toon_production import ToonFormatter

_CASES: list[tuple[Any, str]] = [
    ("plain", "plain"),
    ("a|b", '"a|b"'),
    ("k: v", '"k: v"'),
    (3, "3"),
    (1.5, "1.5"),
    (True, "True"),
    (None, "None"),
    ([], "[]"),
    ({}, "{}"),
    (["x", 1, None], "x | 1 | None"),
    ({"a": 1, "b": "two"}, "a: 1\nb: two"),
    ({"outer": {"inner": [1, 2]}}, "outer:   inner: 1 | 2"),
    ([{"a": 1}, {"b": 2}], "  a: 1\n  b: 2"),
    ([[1, 2], ["y"]], "1 | 2\ny"),
    ((1, 2), "(1, 2)"),
]


@pytest.mark.parametrize(("data", "expected"), _CASES)
def test_toon_serialize_golden(data: Any, expected: str) -> None:
    assert toon_serialize(data) == expected


def test_toon_serialize_indents_nested_dict_values() -> None:
    assert toon_serialize({"a": 1}, indent=2) == "    a: 1"


@pytest.mark.parametrize(("data", "expected"), _CASES)
def test_formatter_format_toon_uses_shared_serializer(data: Any, expected: str) -> None:
    assert ToonFormatter()._format_toon(data) == expected


def test_strategy_format_output_uses_shared_serializer() -> None:
    data = {"files": ["a.py", "b.py"], "count": 2}
    strategy = ToonFormatStrategy(ToolOutputFormat(style="toon"))
    assert strategy.format("tool", {}, data) == toon_serialize(data)

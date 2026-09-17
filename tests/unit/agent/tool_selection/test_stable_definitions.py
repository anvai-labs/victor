# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the single stable-definitions builder (dedup of 4 copies)."""

from types import SimpleNamespace

from victor.agent.tool_selection.stable_definitions import (
    build_stable_definitions,
    normalize_registry_names,
    stable_all_definitions,
    stable_curated_definitions,
)
from victor.tools.enums import SchemaLevel


class FakeRegistry:
    """Dict-backed registry exposing get/list_tools like ToolRegistry."""

    def __init__(self, tools):
        self._tools = tools

    def get(self, name):
        return self._tools.get(name)

    def list_tools(self, only_enabled=True):
        return list(self._tools.values())


def _tool(name):
    return SimpleNamespace(name=name, description=f"desc {name}", parameters={})


def test_build_stable_definitions_sorted_full_schema():
    registry = FakeRegistry({"read": _tool("read"), "shell": _tool("shell")})
    out = build_stable_definitions(registry, ["shell", "read"], SchemaLevel.FULL)
    assert [d.name for d in out] == ["read", "shell"]  # sorted, not input order
    assert all(getattr(d, "parameters", None) is not None for d in out)


def test_build_stable_definitions_skips_missing():
    registry = FakeRegistry({"read": _tool("read")})
    out = build_stable_definitions(registry, ["read", "phantom"], SchemaLevel.FULL)
    assert [d.name for d in out] == ["read"]


def test_normalize_registry_names_handles_instances_and_strings():
    listed = [_tool("write"), "read"]
    assert normalize_registry_names(listed) == ["write", "read"]


def test_stable_all_definitions_cached_per_selector():
    registry = FakeRegistry({"read": _tool("read"), "write": _tool("write")})
    selector = SimpleNamespace(tools=registry, _enabled_tools=None)
    first = stable_all_definitions(selector)
    second = stable_all_definitions(selector)
    assert first is second  # same cached list object
    assert [d.name for d in first] == ["read", "write"]


def test_stable_all_definitions_rebuilds_on_registry_swap():
    selector = SimpleNamespace(tools=FakeRegistry({"read": _tool("read")}), _enabled_tools=None)
    first = stable_all_definitions(selector)
    selector.tools = FakeRegistry({"read": _tool("read"), "edit": _tool("edit")})
    second = stable_all_definitions(selector)
    assert [d.name for d in first] == ["read"]
    assert [d.name for d in second] == ["edit", "read"]  # registry swap rebuilt


def test_stable_curated_definitions_keyed_by_enabled_set():
    registry = FakeRegistry({"read": _tool("read"), "edit": _tool("edit"), "shell": _tool("shell")})
    selector = SimpleNamespace(tools=registry, _enabled_tools={"read"})
    curated = stable_curated_definitions(selector)
    assert [d.name for d in curated] == ["read"]
    selector._enabled_tools = {"edit", "shell"}
    curated2 = stable_curated_definitions(selector)
    assert [d.name for d in curated2] == ["edit", "shell"]


def test_stable_curated_definitions_none_without_enabled_set():
    selector = SimpleNamespace(tools=FakeRegistry({"read": _tool("read")}), _enabled_tools=None)
    assert stable_curated_definitions(selector) is None

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the vertical-tool delegation resolver.

Covers entry-point-based resolution (the primary path) and the direct-import
fallback, plus graceful absence handling.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from victor.tools.unified import _vertical_resolver as resolver
from victor.tools.unified._vertical_resolver import (
    TOOL_CALLABLES_GROUP,
    resolve_vertical_callable,
)


def _fake_ep(name, obj):
    """Build a minimal EntryPoint-like object that loads to ``obj``."""

    return SimpleNamespace(name=name, group=TOOL_CALLABLES_GROUP, load=lambda: obj)


def test_entry_point_resolution_is_preferred(monkeypatch):
    """When an entry point is registered for the key, it wins over the fallback."""
    sentinel = lambda: "ep"  # noqa: E731
    monkeypatch.setattr(
        resolver, "_entry_point_callables", lambda: {"git": _fake_ep("git", sentinel)}
    )

    fn, src = resolve_vertical_callable(
        "git", fallback_module="victor_devops.tools.git_tool", fallback_attr="git"
    )

    assert fn is sentinel
    assert src == "entry-point:git"


def test_direct_import_fallback_when_no_entry_point(monkeypatch):
    """With no entry point, the direct-import fallback is used."""
    monkeypatch.setattr(resolver, "_entry_point_callables", lambda: {})

    fn, src = resolve_vertical_callable(
        "git", fallback_module="victor.tools.unified.git_tool", fallback_attr="git_tool"
    )

    assert fn is not None and callable(fn)
    assert src == "victor.tools.unified.git_tool"


def test_absent_key_with_no_fallback_returns_none(monkeypatch):
    monkeypatch.setattr(resolver, "_entry_point_callables", lambda: {})
    fn, src = resolve_vertical_callable("does_not_exist")
    assert fn is None and src is None


def test_absent_package_fallback_returns_none_without_raising(monkeypatch):
    """A missing optional package degrades to (None, None), never raises."""
    monkeypatch.setattr(resolver, "_entry_point_callables", lambda: {})
    fn, src = resolve_vertical_callable(
        "git", fallback_module="victor_nonexistent.pkg", fallback_attr="git"
    )
    assert fn is None and src is None


def test_non_callable_entry_point_falls_through_to_import(monkeypatch):
    """An entry point that loads to a non-callable falls through to the fallback."""
    monkeypatch.setattr(
        resolver,
        "_entry_point_callables",
        lambda: {"git": _fake_ep("git", "not-callable")},
    )

    fn, src = resolve_vertical_callable(
        "git", fallback_module="victor.tools.unified.git_tool", fallback_attr="git_tool"
    )

    assert fn is not None and callable(fn)
    assert src == "victor.tools.unified.git_tool"  # used the fallback, not the bad EP


def test_entry_point_cache_is_resettable():
    """The cache can be cleared (used by tests / future hot-reload)."""
    resolver._entry_point_cache = {"stale": object()}
    resolver._reset_entry_point_cache()
    assert resolver._entry_point_cache is None

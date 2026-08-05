# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for vertical tool dependency provider resolution fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.core.tool_dependency_base import BaseToolDependencyProvider
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
from victor.core.tool_types import EmptyToolDependencyProvider
import victor.core.tool_dependency_loader as loader_mod


@pytest.fixture(autouse=True)
def _clear_tool_dependency_ep_cache() -> None:
    """Keep entry-point cache and telemetry isolated between tests."""
    loader_mod.clear_vertical_tool_dependency_provider_cache()
    loader_mod.clear_tool_dependency_entry_point_cache()
    loader_mod.reset_tool_dependency_resolution_stats(clear_entry_point_cache=False)
    yield
    loader_mod.clear_vertical_tool_dependency_provider_cache()
    loader_mod.clear_tool_dependency_entry_point_cache()
    loader_mod.reset_tool_dependency_resolution_stats(clear_entry_point_cache=False)


def test_entry_point_provider_has_priority_over_fallbacks(monkeypatch) -> None:
    """Entry point providers should short-circuit module/resource fallbacks."""
    sentinel = EmptyToolDependencyProvider("coding")

    class _FakeEntryPoint:
        name = "coding"

        @staticmethod
        def load():
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])

    def _unexpected_import(_module_name: str):
        raise AssertionError("fallback import should not execute when entry point resolves")

    monkeypatch.setattr(loader_mod, "import_module_with_fallback", _unexpected_import)

    provider = create_vertical_tool_dependency_provider("coding")

    assert provider is sentinel


def test_entry_point_alias_name_matches_normalized_vertical(monkeypatch) -> None:
    """Entry point names with alias spellings should resolve correctly."""
    sentinel = EmptyToolDependencyProvider("dataanalysis")

    class _FakeEntryPoint:
        name = "data-analysis"

        @staticmethod
        def load():
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])

    def _unexpected_import(_module_name: str):
        raise AssertionError("fallback import should not execute when entry point resolves")

    monkeypatch.setattr(loader_mod, "import_module_with_fallback", _unexpected_import)

    provider = create_vertical_tool_dependency_provider("data_analysis")

    assert provider is sentinel


def test_module_factory_fallback_is_used_when_entry_points_missing(monkeypatch) -> None:
    """Resolver should load module-level get_provider before YAML/resource fallbacks."""
    sentinel = EmptyToolDependencyProvider("coding")
    fake_module = SimpleNamespace(get_provider=lambda: sentinel)

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])

    calls: list[str] = []

    def _fake_import(module_name: str):
        calls.append(module_name)
        return fake_module, "victor_coding.tool_dependencies"

    monkeypatch.setattr(loader_mod, "import_module_with_fallback", _fake_import)

    provider = create_vertical_tool_dependency_provider("coding")

    assert provider is sentinel
    assert calls == ["victor.coding.tool_dependencies"]


@pytest.mark.parametrize(
    "vertical",
    ["coding", "devops", "research", "rag", "dataanalysis"],
)
def test_package_resource_yaml_fallback_for_known_verticals(monkeypatch, vertical: str) -> None:
    """Known verticals return a provider (possibly empty) when entry points unavailable.

    After contrib vertical removal, bundled YAML configs are only available
    if the external package (e.g., victor-coding) is installed. Otherwise
    the resolver gracefully returns an EmptyToolDependencyProvider.
    """
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider = create_vertical_tool_dependency_provider(vertical)

    # Provider should always be returned (not raise), even if empty
    assert isinstance(provider, (BaseToolDependencyProvider, EmptyToolDependencyProvider))


def test_unknown_vertical_returns_empty_provider(monkeypatch) -> None:
    """Unknown verticals should gracefully return an EmptyToolDependencyProvider."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider = create_vertical_tool_dependency_provider("mlops")
    assert isinstance(provider, EmptyToolDependencyProvider)


def test_known_vertical_returns_empty_provider_when_resource_lookup_fails(
    monkeypatch,
) -> None:
    """Known verticals should degrade to EmptyToolDependencyProvider on lookup failure."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    def _missing_resource(_package_name: str):
        raise ModuleNotFoundError("simulated missing package")

    monkeypatch.setattr(loader_mod, "files", _missing_resource)

    provider = create_vertical_tool_dependency_provider("coding")

    assert isinstance(provider, EmptyToolDependencyProvider)
    assert provider.vertical == "coding"


@pytest.mark.parametrize("alias_name", ["data-analysis", "data_analysis"])
def test_dataanalysis_aliases_resolve_to_supported_vertical(monkeypatch, alias_name: str) -> None:
    """Historical data-analysis spellings should resolve without raising."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider = create_vertical_tool_dependency_provider(alias_name)

    # Alias should resolve (not raise ValueError) even if empty
    assert isinstance(provider, (BaseToolDependencyProvider, EmptyToolDependencyProvider))


def test_tool_dependency_entry_point_queries_are_cached(monkeypatch) -> None:
    """Entry point scans should be cached across repeated resolution calls."""
    call_count = 0

    def _fake_entry_points(group: str):
        nonlocal call_count
        assert group == "victor.tool_dependencies"
        call_count += 1
        return []

    monkeypatch.setattr(loader_mod, "entry_points", _fake_entry_points)
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    create_vertical_tool_dependency_provider("mlops")
    create_vertical_tool_dependency_provider("mlops")

    assert call_count == 1


def test_clear_tool_dependency_entry_point_cache_forces_rescan(monkeypatch) -> None:
    """Cache clear helper should force a fresh entry-point query."""
    call_count = 0

    def _fake_entry_points(group: str):
        nonlocal call_count
        assert group == "victor.tool_dependencies"
        call_count += 1
        return []

    monkeypatch.setattr(loader_mod, "entry_points", _fake_entry_points)
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    create_vertical_tool_dependency_provider("mlops")
    loader_mod.clear_tool_dependency_entry_point_cache()
    loader_mod.clear_vertical_tool_dependency_provider_cache()
    create_vertical_tool_dependency_provider("mlops")

    assert call_count == 2


def test_resolution_stats_track_entry_point_path(monkeypatch) -> None:
    """Telemetry should record successful entry-point-based resolution."""
    sentinel = EmptyToolDependencyProvider("coding")

    class _FakeEntryPoint:
        name = "coding"

        @staticmethod
        def load():
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider = create_vertical_tool_dependency_provider("coding")
    stats = loader_mod.get_tool_dependency_resolution_stats()

    assert provider is sentinel
    assert stats["total_requests"] == 1
    assert stats["entry_point_resolutions"] == 1
    assert stats["module_factory_resolutions"] == 0
    assert stats["package_resource_resolutions"] == 0
    assert stats["empty_provider_returns"] == 0


def test_resolution_stats_track_module_factory_path(monkeypatch) -> None:
    """Telemetry should record successful module-factory resolution."""
    sentinel = EmptyToolDependencyProvider("coding")
    fake_module = SimpleNamespace(get_provider=lambda: sentinel)

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(
        loader_mod,
        "import_module_with_fallback",
        lambda module_name: (fake_module, "victor_coding.tool_dependencies"),
    )

    provider = create_vertical_tool_dependency_provider("coding")
    stats = loader_mod.get_tool_dependency_resolution_stats()

    assert provider is sentinel
    assert stats["total_requests"] == 1
    assert stats["entry_point_resolutions"] == 0
    assert stats["module_factory_resolutions"] == 1
    assert stats["package_resource_resolutions"] == 0


def test_resolution_stats_track_requests_and_cache_hits(monkeypatch) -> None:
    """Telemetry should record requests and provider cache usage."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    create_vertical_tool_dependency_provider("mlops")
    create_vertical_tool_dependency_provider("mlops")

    stats = loader_mod.get_tool_dependency_resolution_stats()
    assert stats["total_requests"] == 2
    assert stats["provider_cache_hits"] >= 1
    assert stats["provider_cache_misses"] >= 1


def test_provider_cache_reuses_resolved_provider(monkeypatch) -> None:
    """Provider cache should reuse previously resolved provider instances."""
    sentinel = EmptyToolDependencyProvider("coding")
    load_calls = 0

    class _FakeEntryPoint:
        name = "coding"

        @staticmethod
        def load():
            nonlocal load_calls
            load_calls += 1
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    p1 = create_vertical_tool_dependency_provider("coding")
    p2 = create_vertical_tool_dependency_provider("coding")
    stats = loader_mod.get_tool_dependency_resolution_stats()

    assert p1 is sentinel
    assert p2 is sentinel
    assert load_calls == 1
    assert stats["provider_cache_hits"] >= 1
    assert stats["provider_cache_misses"] >= 1
    assert stats["provider_cache_currsize"] >= 1


def test_clear_vertical_provider_cache_forces_reresolution(monkeypatch) -> None:
    """Clearing provider cache should force a fresh provider resolution path."""
    load_calls = 0

    class _FakeEntryPoint:
        name = "coding"

        @staticmethod
        def load():
            nonlocal load_calls
            load_calls += 1
            # Return a fresh provider each time to detect re-resolution
            return lambda: EmptyToolDependencyProvider("coding")

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    p1 = create_vertical_tool_dependency_provider("coding")
    stats_before_clear = loader_mod.get_tool_dependency_resolution_stats()
    loader_mod.clear_vertical_tool_dependency_provider_cache()
    stats_after_clear = loader_mod.get_tool_dependency_resolution_stats()
    p2 = create_vertical_tool_dependency_provider("coding")

    assert p1 is not p2
    assert load_calls == 2
    assert stats_before_clear["provider_cache_currsize"] >= 1
    assert stats_after_clear["provider_cache_currsize"] == 0


@pytest.mark.parametrize(
    ("vertical", "explicit_canonicalize"),
    [
        ("coding", True),  # default canonicalize=True
        ("devops", True),  # default canonicalize=True (no config registered)
    ],
)
def test_default_and_explicit_canonicalize_share_provider_cache_key(
    monkeypatch, vertical: str, explicit_canonicalize: bool
) -> None:
    """canonicalize=None and canonicalize=<default> should reuse the same cache entry."""
    load_calls = 0

    class _FakeEntryPoint:
        name = vertical

        @staticmethod
        def load():
            nonlocal load_calls
            load_calls += 1
            return lambda: EmptyToolDependencyProvider(vertical)

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider_default = create_vertical_tool_dependency_provider(vertical, canonicalize=None)
    provider_explicit = create_vertical_tool_dependency_provider(
        vertical, canonicalize=explicit_canonicalize
    )

    assert provider_default is provider_explicit
    assert load_calls == 1


def test_cache_clear_stats_track_clear_operations(monkeypatch) -> None:
    """Telemetry should track provider/entry-point cache clear operations."""
    sentinel = EmptyToolDependencyProvider("coding")

    class _FakeEntryPoint:
        name = "coding"

        @staticmethod
        def load():
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    create_vertical_tool_dependency_provider("coding")
    cleared = loader_mod.clear_vertical_tool_dependency_provider_cache()
    loader_mod.clear_tool_dependency_entry_point_cache()

    stats = loader_mod.get_tool_dependency_resolution_stats()
    assert cleared >= 1
    assert stats["provider_cache_clears"] == 1
    assert stats["provider_cache_entries_cleared"] >= 1
    assert stats["entry_point_cache_clears"] == 1

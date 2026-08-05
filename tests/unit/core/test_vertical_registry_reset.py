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

"""Tests for VerticalRegistry.reset_discovery behavior."""

from __future__ import annotations

from unittest.mock import Mock, call, patch

from victor.core.verticals.base import VerticalRegistry


def test_reset_discovery_clears_loader_and_entry_point_caches() -> None:
    """reset_discovery should clear registry state and all relevant entry-point caches."""
    VerticalRegistry._external_discovered = True
    loader = Mock()
    cache = Mock()
    framework_cache_clear = Mock()
    tool_dep_cache_clear = Mock()
    provider_cache_clear = Mock(return_value=0)

    with patch("victor.core.verticals.vertical_loader.get_vertical_loader", return_value=loader):
        with patch("victor.framework.module_loader.get_entry_point_cache", return_value=cache):
            with patch(
                "victor.framework.entry_point_loader.clear_entry_point_loader_cache",
                framework_cache_clear,
            ):
                with patch(
                    "victor.core.tool_dependency_loader.clear_tool_dependency_entry_point_cache",
                    tool_dep_cache_clear,
                ):
                    with patch(
                        "victor.core.tool_dependency_loader.clear_vertical_tool_dependency_provider_cache",
                        provider_cache_clear,
                    ):
                        VerticalRegistry.reset_discovery()

    assert VerticalRegistry._external_discovered is False
    loader.reset_discovery_state.assert_called_once()
    assert cache.invalidate.call_count == 2
    cache.invalidate.assert_has_calls(
        [
            call(VerticalRegistry.ENTRY_POINT_GROUP),
            call("victor.plugins"),
        ]
    )
    framework_cache_clear.assert_called_once()
    tool_dep_cache_clear.assert_called_once()
    provider_cache_clear.assert_called_once()


def test_reset_discovery_is_resilient_when_reset_hooks_fail() -> None:
    """reset_discovery should still clear registry flag when hook resets fail."""
    VerticalRegistry._external_discovered = True

    with patch(
        "victor.core.verticals.vertical_loader.get_vertical_loader",
        side_effect=RuntimeError("loader unavailable"),
    ):
        with patch(
            "victor.framework.module_loader.get_entry_point_cache",
            side_effect=RuntimeError("cache unavailable"),
        ):
            with patch(
                "victor.framework.entry_point_loader.clear_entry_point_loader_cache",
                side_effect=RuntimeError("framework cache unavailable"),
            ):
                with patch(
                    "victor.core.tool_dependency_loader.clear_tool_dependency_entry_point_cache",
                    side_effect=RuntimeError("tool dependency cache unavailable"),
                ):
                    with patch(
                        "victor.core.tool_dependency_loader.clear_vertical_tool_dependency_provider_cache",
                        side_effect=RuntimeError("provider cache unavailable"),
                    ):
                        VerticalRegistry.reset_discovery()

    assert VerticalRegistry._external_discovered is False

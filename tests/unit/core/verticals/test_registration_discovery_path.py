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

"""Tests for external vertical discovery path selection."""

from __future__ import annotations

from unittest.mock import Mock, patch

import victor.core.verticals as core_verticals
from victor.core.verticals.base import VerticalRegistry


def test_discover_external_verticals_uses_loader_primary_path() -> None:
    """Lazy registration should prefer VerticalLoader discovery path."""
    original_flag = VerticalRegistry._external_discovered
    VerticalRegistry._external_discovered = False

    mock_loader = Mock()

    try:
        with patch.object(core_verticals, "get_vertical_loader", return_value=mock_loader):
            core_verticals._discover_external_verticals()

        mock_loader.discover_verticals.assert_called_once_with(emit_event=False)
        assert VerticalRegistry._external_discovered is True
    finally:
        VerticalRegistry._external_discovered = original_flag


def test_discover_external_verticals_falls_back_to_registry() -> None:
    """Loader failures should fall back to legacy registry discovery."""
    with patch.object(
        core_verticals,
        "get_vertical_loader",
        side_effect=RuntimeError("loader unavailable"),
    ):
        with patch.object(VerticalRegistry, "discover_external_verticals") as discover_mock:
            core_verticals._discover_external_verticals()

    discover_mock.assert_called_once()

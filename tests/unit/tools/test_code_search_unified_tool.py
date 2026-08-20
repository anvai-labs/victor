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

"""Tests for the unified ``code`` tool's ``search`` subcommand.

``code search`` delegates to ``victor_coding.code_search`` when present and
falls back to literal ``grep_search`` otherwise (or for ``--mode literal``).
"""

from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.unified.code_tool import code_tool


class TestCodeSearchLiteral:
    """``--mode literal`` always uses grep_search (no delegation)."""

    @pytest.mark.asyncio
    async def test_literal_mode_uses_grep(self):
        mock_grep = AsyncMock(return_value=[{"file": "a.py", "line": 1, "content": "auth"}])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code search "auth" src --mode literal')
        mock_grep.assert_awaited_once_with(
            query="auth", path="src", regex=False, case_sensitive=False
        )
        assert "a.py:1" in result


class TestCodeSearchSemanticDelegation:
    """When victor-coding is present, semantic modes delegate to code_search."""

    @pytest.mark.asyncio
    async def test_semantic_delegates_to_coding(self):
        mock_search = AsyncMock(
            return_value={
                "success": True,
                "results": [
                    {
                        "file": "auth.py",
                        "line": 12,
                        "score": 0.9,
                        "snippet": "def login()",
                    }
                ],
            }
        )
        with patch(
            "victor.tools.unified._vertical_resolver.resolve_vertical_callable",
            return_value=(mock_search, "victor_coding.tools.code_search_tool"),
        ):
            result = await code_tool('code search "login flow" src --mode semantic --k 5')
        mock_search.assert_awaited_once_with(query="login flow", path="src", mode="semantic", k=5)
        assert "auth.py:12" in result
        assert "0.9" in result

    @pytest.mark.asyncio
    async def test_graph_mode_delegates(self):
        mock_search = AsyncMock(return_value={"success": True, "results": []})
        with patch(
            "victor.tools.unified._vertical_resolver.resolve_vertical_callable",
            return_value=(mock_search, "victor_coding.tools.code_search_tool"),
        ):
            result = await code_tool('code search "callers of foo" --mode graph')
        assert mock_search.call_args.kwargs["mode"] == "graph"
        assert "No matches found." in result

    @pytest.mark.asyncio
    async def test_delegation_failure_surfaced(self):
        """Backend error dicts degrade to literal grep; the reason stays visible."""
        mock_search = AsyncMock(return_value={"success": False, "error": "index missing"})
        mock_grep = AsyncMock(return_value=[])
        with (
            patch(
                "victor.tools.unified._vertical_resolver.resolve_vertical_callable",
                return_value=(mock_search, "victor_coding.tools.code_search_tool"),
            ),
            patch("victor.tools.unified._search_helpers.grep_search", mock_grep),
        ):
            result = await code_tool('code search "foo"')
        assert "SYSTEM HINT" in result
        assert "index missing" in result
        assert "### ❌ ERROR" not in result


class TestCodeSearchFallback:
    """When victor-coding is absent, semantic modes degrade to literal grep."""

    @pytest.mark.asyncio
    async def test_absent_coding_falls_back_to_literal(self):
        mock_grep = AsyncMock(return_value=[{"file": "a.py", "line": 1, "content": "auth"}])
        with (
            patch(
                "victor.tools.unified._vertical_resolver.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.unified._search_helpers.grep_search", mock_grep),
        ):
            result = await code_tool('code search "auth" src')
        mock_grep.assert_awaited_once_with(
            query="auth", path="src", regex=False, case_sensitive=False
        )
        assert "SYSTEM HINT" in result
        assert "literal" in result.lower()
        assert "a.py:1" in result

    @pytest.mark.asyncio
    async def test_backend_error_dict_falls_back_to_literal(self):
        """A resolved backend returning success:False degrades to literal grep
        with a SYSTEM HINT naming the reason (measured 20x live: Settings not
        available in tool context)."""
        mock_search = AsyncMock(
            return_value={"success": False, "error": "Settings not available in tool context."}
        )
        mock_grep = AsyncMock(return_value=[{"file": "a.py", "line": 1, "content": "auth"}])
        with (
            patch(
                "victor.tools.unified._vertical_resolver.resolve_vertical_callable",
                return_value=(mock_search, "victor_coding.tools.code_search_tool"),
            ),
            patch("victor.tools.unified._search_helpers.grep_search", mock_grep),
        ):
            result = await code_tool('code search "auth" src')
        mock_grep.assert_awaited_once()
        assert "SYSTEM HINT" in result
        assert "Settings not available in tool context." in result
        assert "a.py:1" in result
        assert "### ❌ ERROR" not in result

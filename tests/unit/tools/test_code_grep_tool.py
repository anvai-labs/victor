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

"""Tests for the unified ``code`` tool's ``grep`` subcommand.

``code grep`` is the canonical literal content-search surface and delegates to
the shared ``grep_search`` helper.
"""

from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.unified.code_tool import code_tool


class TestCodeGrep:
    """``code grep {query} {path}`` -> ``grep_search``."""

    @pytest.mark.asyncio
    async def test_grep_delegates_to_grep_search(self):
        mock_grep = AsyncMock(
            return_value=[
                {"file": "a.py", "line": 3, "content": "def foo():"},
                {"file": "b.py", "line": 7, "content": "def bar():"},
            ]
        )
        with patch("victor.tools.unified.code_tool.grep_search", mock_grep, create=True):
            # grep_search is imported lazily inside the handler; patch the source.
            with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
                result = await code_tool('code grep "def foo" src')
        mock_grep.assert_awaited_once_with(
            query="def foo", path="src", regex=False, case_sensitive=False
        )
        assert "a.py:3" in result
        assert "def foo():" in result

    @pytest.mark.asyncio
    async def test_grep_regex_and_case_sensitive_flags_forwarded(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep "^def " src --regex --case-sensitive')
        mock_grep.assert_awaited_once_with(
            query="^def ", path="src", regex=True, case_sensitive=True
        )
        assert "No matches found." in result

    @pytest.mark.asyncio
    async def test_grep_short_case_sensitive_flag(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep "Foo" src -C')
        assert mock_grep.call_args.kwargs["case_sensitive"] is True

    @pytest.mark.asyncio
    async def test_grep_defaults_path_to_cwd(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep "foo"')
        assert mock_grep.call_args.kwargs["path"] == "."

    @pytest.mark.asyncio
    async def test_grep_no_matches_message(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep "zzz" src')
        assert result == "No matches found."

    @pytest.mark.asyncio
    async def test_existing_code_subcommands_still_work(self):
        """Regression: metrics subcommand must still parse after adding grep/search."""
        with patch("victor.tools.unified.code_tool.analyze_metrics", AsyncMock()):
            # Just confirm it routes to metrics (not grep/search) without error.
            result = await code_tool("code metrics .")
            assert "### ❌ ERROR" not in result or "Metrics" in result


class TestGrepFlagAbsorb:
    """GNU-grep flags whose behavior is always-on are silently absorbed."""

    @pytest.mark.asyncio
    async def test_absorb_dash_n(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep -n "foo" src')
        mock_grep.assert_awaited_once()
        assert mock_grep.call_args.kwargs["query"] == "foo"
        assert mock_grep.call_args.kwargs["path"] == "src"
        assert "⚠️" not in result
        assert "❌" not in result

    @pytest.mark.asyncio
    async def test_absorb_combined_rn(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep -rn "foo" src')
        mock_grep.assert_awaited_once()
        assert mock_grep.call_args.kwargs["query"] == "foo"
        assert "⚠️" not in result
        assert "❌" not in result

    @pytest.mark.asyncio
    async def test_absorb_long_flags(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep --line-number --recursive --ignore-case "foo" src')
        mock_grep.assert_awaited_once()
        assert mock_grep.call_args.kwargs["query"] == "foo"
        assert "⚠️" not in result
        assert "❌" not in result


class TestGrepFlagMap:
    """Grep flags with a supported equivalent are mapped onto it."""

    @pytest.mark.asyncio
    async def test_short_E_maps_to_regex(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep -E "^def " src')
        assert mock_grep.call_args.kwargs["regex"] is True

    @pytest.mark.asyncio
    async def test_long_extended_regexp_maps_to_regex(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep --extended-regexp "^def " src')
        assert mock_grep.call_args.kwargs["regex"] is True


class TestGrepFilesOnly:
    """``-l`` / ``--files-with-matches`` -> ``--files-only`` output mode."""

    @pytest.mark.asyncio
    async def test_files_only_short_flag_shape(self):
        mock_grep = AsyncMock(
            return_value=[
                {"file": "a.py", "line": 3, "content": "def foo():"},
                {"file": "a.py", "line": 9, "content": "def foo2():"},
                {"file": "b.py", "line": 7, "content": "def bar():"},
            ]
        )
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep -l "def" src')
        lines = result.splitlines()
        assert lines == ["a.py", "b.py"]  # unique, first-seen order, no :line: parts

    @pytest.mark.asyncio
    async def test_files_with_matches_long_flag(self):
        mock_grep = AsyncMock(return_value=[{"file": "a.py", "line": 3, "content": "def foo():"}])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep --files-with-matches "def" src')
        assert result.splitlines() == ["a.py"]


class TestGrepInclude:
    """``--include GLOB`` is a real argument plumbed to the search engines."""

    @pytest.mark.asyncio
    async def test_include_equals_form(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep --include=*.py "foo" src')
        assert mock_grep.call_args.kwargs["include_glob"] == "*.py"

    @pytest.mark.asyncio
    async def test_include_space_form(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep --include *.py "foo" src')
        assert mock_grep.call_args.kwargs["include_glob"] == "*.py"

    @pytest.mark.asyncio
    async def test_live_combo_rn_include(self):
        """The exact combination measured live: code grep -rn --include=*.py "pattern" ."""
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep -rn --include=*.py "pattern" .')
        mock_grep.assert_awaited_once()
        assert mock_grep.call_args.kwargs["query"] == "pattern"
        assert mock_grep.call_args.kwargs["path"] == "."
        assert mock_grep.call_args.kwargs["include_glob"] == "*.py"
        assert "⚠️" not in result
        assert "❌" not in result

    @pytest.mark.asyncio
    async def test_include_glob_filters_python_walk(self, tmp_path):
        """Basename fnmatch semantics in the pure-Python walk engine."""
        from victor.tools.unified._search_helpers import grep_search

        (tmp_path / "match.py").write_text("needle here\n")
        (tmp_path / "match.txt").write_text("needle here\n")
        with patch("victor.tools.unified._search_helpers.shutil.which", return_value=None):
            results = await grep_search(query="needle", path=str(tmp_path), include_glob="*.py")
        assert [r["file"] for r in results] == [str(tmp_path / "match.py")]


class TestGrepRejections:
    """Unsupported grep flags get a corrective rejection, not a parse error."""

    @pytest.mark.asyncio
    async def test_unknown_long_flag_rejected(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep --invert-match "foo" src')
        mock_grep.assert_not_awaited()
        assert "UNSUPPORTED GREP FLAG" in result
        assert "--invert-match" in result
        assert "shell(cmd='grep" in result

    @pytest.mark.asyncio
    async def test_combined_run_with_unknown_letter_rejected(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep -rv "foo" src')
        mock_grep.assert_not_awaited()
        assert "UNSUPPORTED GREP FLAG" in result
        assert "-rv" in result
        assert "shell(cmd='grep" in result


class TestUnknownSubcommands:
    """Hallucinated subcommands redirect instead of dumping argparse noise."""

    @pytest.mark.asyncio
    async def test_ls_redirect(self):
        result = await code_tool("code ls -la")
        assert "UNKNOWN SUBCOMMAND" in result
        assert "`ls` tool" in result
        assert "search, grep, test, python, execute, metrics" in result

    @pytest.mark.asyncio
    async def test_find_redirect(self):
        result = await code_tool('code find . -name "*.py"')
        assert "UNKNOWN SUBCOMMAND" in result
        assert "find" in result
        assert "code grep" in result

    @pytest.mark.asyncio
    async def test_help_returns_usage(self):
        result = await code_tool("code help")
        assert "usage:" in result
        assert "grep" in result
        assert "❌" not in result


class TestGrepNormalizerRegressions:
    """The normalizer must not disturb supported syntax."""

    @pytest.mark.asyncio
    async def test_double_dash_protects_literal_dash_n_pattern(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep -- "-n" src')
        mock_grep.assert_awaited_once()
        assert mock_grep.call_args.kwargs["query"] == "-n"
        assert mock_grep.call_args.kwargs["path"] == "src"
        assert "⚠️" not in result

    @pytest.mark.asyncio
    async def test_regex_and_case_sensitive_still_forwarded(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep "^def " src --regex -C')
        assert mock_grep.call_args.kwargs["regex"] is True
        assert mock_grep.call_args.kwargs["case_sensitive"] is True

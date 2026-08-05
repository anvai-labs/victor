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

"""Guard tests for the literal-search helpers.

Covers the regression where ``grep_search`` text-scanned ``.victor/`` binary
index artifacts (project.db, LanceDB fragments) for minutes: skip-dir pruning,
size/binary guards, the ripgrep fast path, the literal-search timeout, and
progress-sink emission.
"""

import asyncio
import shutil
import time
from unittest.mock import AsyncMock, patch

import pytest

from victor.framework.tool_progress import clear_progress_sink, set_progress_sink
from victor.tools.unified import _search_helpers
from victor.tools.unified._search_helpers import grep_search
from victor.tools.unified.code_tool import code_tool

_RG_AVAILABLE = shutil.which("rg") is not None


def _make_tree(tmp_path):
    """A small tree with a match, a .victor binary blob, and guard-trip files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def needle_function():\n    pass\n")
    (tmp_path / "README.md").write_text("no match here\n")

    victor_dir = tmp_path / ".victor"
    victor_dir.mkdir(exist_ok=True)  # an autouse fixture may pre-create it
    # Binary blob that CONTAINS the needle — must never be scanned/matched.
    (victor_dir / "project.db").write_bytes(b"\x00\x01needle_function\x00" * 1000)

    # Binary file outside .victor — caught by the NUL sniff.
    (tmp_path / "blob.bin").write_bytes(b"\x00needle_function\x00")
    return tmp_path


@pytest.fixture
def python_walk_only():
    """Force the pure-Python engine so guards (not rg) are what's under test."""
    with patch.object(_search_helpers.shutil, "which", return_value=None):
        yield


class TestScanGuards:
    async def test_victor_dir_and_binaries_skipped(self, tmp_path, python_walk_only):
        root = _make_tree(tmp_path)
        results = await grep_search(query="needle_function", path=str(root))
        files = {r["file"] for r in results}
        assert files == {str(root / "src" / "app.py")}

    async def test_oversized_file_skipped(self, tmp_path, python_walk_only):
        root = _make_tree(tmp_path)
        big = root / "generated.txt"
        big.write_text("needle_function\n" * 200_000)  # > 2 MB
        results = await grep_search(query="needle_function", path=str(root))
        assert str(big) not in {r["file"] for r in results}

    async def test_large_binary_tree_is_fast(self, tmp_path, python_walk_only):
        """A fat .victor dir must not affect scan time (the 257s regression)."""
        root = _make_tree(tmp_path)
        for i in range(20):
            (root / ".victor" / f"frag{i}.lance").write_bytes(b"\x00" * 1_000_000)
        started = time.monotonic()
        results = await grep_search(query="needle_function", path=str(root))
        assert time.monotonic() - started < 5.0
        assert len(results) == 1

    async def test_single_file_path_still_works(self, tmp_path, python_walk_only):
        root = _make_tree(tmp_path)
        target = root / "src" / "app.py"
        results = await grep_search(query="needle_function", path=str(target))
        assert [r["file"] for r in results] == [str(target)]

    async def test_regex_and_case_flags(self, tmp_path, python_walk_only):
        root = _make_tree(tmp_path)
        assert await grep_search(query="NEEDLE_function", path=str(root)) != []
        assert await grep_search(query="NEEDLE_function", path=str(root), case_sensitive=True) == []
        regex_hits = await grep_search(query=r"def \w+_function", path=str(root), regex=True)
        assert len(regex_hits) == 1


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
class TestRipgrepParity:
    async def test_rg_and_python_walk_agree(self, tmp_path):
        root = _make_tree(tmp_path)
        rg_results = await _search_helpers._ripgrep_search(
            "needle_function", root, regex=False, case_sensitive=False
        )
        py_results = await _search_helpers._python_walk_search(
            "needle_function", root, regex=False, case_sensitive=False
        )
        assert rg_results is not None
        key = lambda r: (r["file"], r["line"], r["content"])  # noqa: E731
        assert sorted(map(key, rg_results)) == sorted(map(key, py_results))

    async def test_rg_result_shape(self, tmp_path):
        root = _make_tree(tmp_path)
        results = await _search_helpers._ripgrep_search(
            "needle_function", root, regex=False, case_sensitive=False
        )
        assert results and set(results[0]) == {"file", "line", "content"}
        assert isinstance(results[0]["line"], int)


class TestProgressAndTimeout:
    async def test_progress_sink_receives_heartbeats(self, tmp_path, python_walk_only):
        root = tmp_path
        for i in range(_search_helpers._PROGRESS_EVERY_FILES + 5):
            (root / f"f{i}.txt").write_text("nothing\n")
        calls = []
        set_progress_sink(lambda **kw: calls.append(kw))
        try:
            await grep_search(query="needle", path=str(root))
        finally:
            clear_progress_sink()
        assert calls and calls[0]["name"] == "code"
        assert "scanning" in calls[0]["stdout"]

    async def test_literal_search_timeout_message(self, monkeypatch):
        monkeypatch.setenv("VICTOR_TIMEOUT_LITERAL_SEARCH", "0.05")

        async def slow_search(**kwargs):
            await asyncio.sleep(1.0)

        with patch("victor.tools.unified._search_helpers.grep_search", slow_search):
            result = await code_tool('code grep "anything" .')
        assert "exceeded" in result and "VICTOR_TIMEOUT_LITERAL_SEARCH" in result

    async def test_search_literal_mode_timeout_message(self, monkeypatch):
        monkeypatch.setenv("VICTOR_TIMEOUT_LITERAL_SEARCH", "0.05")

        async def slow_search(**kwargs):
            await asyncio.sleep(1.0)

        with patch("victor.tools.unified._search_helpers.grep_search", slow_search):
            result = await code_tool('code search "anything" . --mode literal')
        assert "exceeded" in result

    async def test_timeout_not_triggered_on_fast_search(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VICTOR_TIMEOUT_LITERAL_SEARCH", "30")
        mock_grep = AsyncMock(return_value=[{"file": "a.py", "line": 1, "content": "x"}])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep "x" .')
        assert "a.py:1" in result

    async def test_bad_timeout_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("VICTOR_TIMEOUT_LITERAL_SEARCH", "not-a-number")
        from victor.tools.unified.code_tool import _literal_search_timeout

        assert _literal_search_timeout() == 30.0

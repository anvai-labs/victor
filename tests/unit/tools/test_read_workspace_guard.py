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

"""Workspace guard for ``read`` must treat linked git worktrees as in-workspace.

Live failure: reading a sibling git-worktree path (``../wt-web-tool/...``) was
rejected as "outside the current workspace" while ``edit``/``write`` accepted it,
forcing clunky ``sed`` fallbacks. Worktrees are the documented workflow, so the
guard now allows the project root plus every linked worktree root.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import victor.tools.filesystem as fs
from victor.tools.filesystem import (
    _discover_read_roots,
    _path_within_workspace,
    read,
)


@pytest.fixture(autouse=True)
def _clear_cache_and_enable_guard(monkeypatch):
    # The guard must be active (the shared filesystem test module disables it).
    monkeypatch.delenv("VICTOR_DISABLE_WORKSPACE_GUARD", raising=False)
    fs._ALLOWED_READ_ROOTS_CACHE.clear()
    yield
    fs._ALLOWED_READ_ROOTS_CACHE.clear()


class TestPathWithinWorkspace:
    def test_project_root_subpath_allowed(self):
        root = Path("/repo/main").resolve()
        with patch.object(fs, "_discover_read_roots", return_value=frozenset({str(root)})):
            assert _path_within_workspace(root / "victor" / "x.py", root) is True

    def test_worktree_path_allowed(self):
        root = Path("/repo/main").resolve()
        wt = Path("/repo/wt-feature").resolve()
        with patch.object(fs, "_discover_read_roots", return_value=frozenset({str(root), str(wt)})):
            assert _path_within_workspace(wt / "victor" / "tool.py", root) is True

    def test_outside_path_rejected(self):
        root = Path("/repo/main").resolve()
        with patch.object(fs, "_discover_read_roots", return_value=frozenset({str(root)})):
            assert _path_within_workspace(Path("/etc/passwd"), root) is False

    def test_sibling_prefix_not_falsely_matched(self):
        # "/repo/main-secrets" must NOT match root "/repo/main" (separator-aware).
        root = Path("/repo/main").resolve()
        with patch.object(fs, "_discover_read_roots", return_value=frozenset({str(root)})):
            assert _path_within_workspace(Path("/repo/main-secrets/x"), root) is False

    def test_refresh_picks_up_new_worktree(self):
        # First discovery lacks the worktree; a mid-session worktree appears on refresh.
        root = Path("/repo/main").resolve()
        wt = Path("/repo/wt-new").resolve()
        calls = [frozenset({str(root)}), frozenset({str(root), str(wt)})]
        with patch.object(fs, "_discover_read_roots", side_effect=calls):
            assert _path_within_workspace(wt / "f.py", root) is True


class TestDiscoverReadRoots:
    def test_parses_worktree_list_porcelain(self):
        root = Path("/repo/main").resolve()
        stdout = (
            f"worktree {root}\nHEAD abc\nbranch refs/heads/develop\n\n"
            "worktree /repo/wt-feature\nHEAD def\nbranch refs/heads/feat/x\n"
        )
        proc = MagicMock(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            roots = _discover_read_roots(root)
        assert str(root) in roots
        assert str(Path("/repo/wt-feature").resolve()) in roots

    def test_git_failure_falls_back_to_project_root(self):
        root = Path("/repo/main").resolve()
        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            roots = _discover_read_roots(root)
        assert roots == frozenset({str(root)})


class TestReadGuardEndToEnd:
    @pytest.mark.asyncio
    async def test_read_rejects_genuine_outside_path(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        outside = tmp_path / "elsewhere" / "secret.txt"
        outside.parent.mkdir()
        outside.write_text("nope")
        paths = MagicMock(project_root=str(project))
        with (
            patch("victor.config.settings.get_project_paths", return_value=paths),
            patch.object(fs, "_discover_read_roots", return_value=frozenset({str(project)})),
        ):
            result = await read(path=str(outside))
        assert "outside it" in result
        # The rejection must stay actionable: a guard the model cannot route
        # around reads as a missing capability and the session gives up.
        assert "shell(cmd=" in result
        assert "VICTOR_EXTRA_READ_ROOTS" in result


class TestOutOfWorkspaceMessage:
    """The suggested route must work for the kind of path that was rejected."""

    def test_directory_is_routed_to_unscoped_ls(self, tmp_path):
        target = tmp_path / "other-repo"
        target.mkdir()
        msg = fs._out_of_workspace_message(str(target), target, Path("/repo/main"))
        assert f"ls(path='{target}')" in msg
        assert "sed -n" not in msg

    def test_sqlite_file_is_routed_to_sqlite3(self, tmp_path):
        target = tmp_path / "victor.db"
        target.write_bytes(b"SQLite format 3\x00")
        msg = fs._out_of_workspace_message(str(target), target, Path("/repo/main"))
        # A text pager on a binary store is the advice that stalls sessions.
        assert "sqlite3" in msg
        assert "sed -n" not in msg

    def test_text_file_is_routed_to_a_pager(self, tmp_path):
        target = tmp_path / "notes.md"
        target.write_text("hello")
        msg = fs._out_of_workspace_message(str(target), target, Path("/repo/main"))
        assert "sed -n" in msg

    def test_declared_root_is_the_directory_not_the_file(self, tmp_path):
        target = tmp_path / "repo" / "file.py"
        target.parent.mkdir()
        target.write_text("x = 1")
        msg = fs._out_of_workspace_message(str(target), target, Path("/repo/main"))
        assert f"VICTOR_EXTRA_READ_ROOTS={target.parent}" in msg

    def test_message_states_that_only_read_is_scoped(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("x")
        msg = fs._out_of_workspace_message(str(target), target, Path("/repo/main"))
        assert "not a sandbox" in msg

    def test_no_literal_placeholder_is_suggested(self, tmp_path):
        # Guidance containing shell(cmd='...') is what taught models to send "...".
        target = tmp_path / "f.txt"
        target.write_text("x")
        msg = fs._out_of_workspace_message(str(target), target, Path("/repo/main"))
        assert "cmd='...'" not in msg

    @pytest.mark.asyncio
    async def test_read_allows_worktree_path(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        worktree = tmp_path / "wt-feature"
        worktree.mkdir()
        target = worktree / "hello.py"
        target.write_text("print('hi')\n")
        paths = MagicMock(project_root=str(project))
        with (
            patch("victor.config.settings.get_project_paths", return_value=paths),
            patch.object(
                fs,
                "_discover_read_roots",
                return_value=frozenset({str(project.resolve()), str(worktree.resolve())}),
            ),
        ):
            result = await read(path=str(target))
        assert "outside the current workspace" not in result
        assert "hi" in result


class TestExtraReadRoots:
    """VICTOR_EXTRA_READ_ROOTS opens read()/ls() onto a second repo.

    Cross-repo sessions (co-designing a wire contract in two codebases, or
    letting Victor inspect its own ~/.victor state) previously had no route
    through read() at all.
    """

    def test_unset_env_changes_nothing(self, monkeypatch):
        monkeypatch.delenv(fs.EXTRA_READ_ROOTS_ENV, raising=False)
        assert fs._extra_read_roots() == ()

    def test_pathsep_separated_roots_are_resolved(self, monkeypatch, tmp_path):
        a = tmp_path / "repo-a"
        b = tmp_path / "repo-b"
        a.mkdir()
        b.mkdir()
        monkeypatch.setenv(fs.EXTRA_READ_ROOTS_ENV, f"{a}{os.pathsep}{b}")
        assert fs._extra_read_roots() == tuple(sorted({str(a.resolve()), str(b.resolve())}))

    def test_blank_entries_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv(fs.EXTRA_READ_ROOTS_ENV, f"{os.pathsep}  {os.pathsep}{tmp_path}")
        assert fs._extra_read_roots() == (str(tmp_path.resolve()),)

    def test_extra_root_joins_discovered_roots(self, monkeypatch, tmp_path):
        root = tmp_path / "proj"
        other = tmp_path / "other-repo"
        root.mkdir()
        other.mkdir()
        monkeypatch.setenv(fs.EXTRA_READ_ROOTS_ENV, str(other))
        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            roots = _discover_read_roots(root)
        assert str(other.resolve()) in roots

    def test_cache_key_tracks_env_so_changes_apply_immediately(self, monkeypatch, tmp_path):
        # A TTL-only cache would keep rejecting the new root for up to 60s.
        root = tmp_path / "proj"
        other = tmp_path / "other-repo"
        root.mkdir()
        other.mkdir()
        monkeypatch.delenv(fs.EXTRA_READ_ROOTS_ENV, raising=False)
        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            assert _path_within_workspace(other / "f.py", root) is False
            monkeypatch.setenv(fs.EXTRA_READ_ROOTS_ENV, str(other))
            assert _path_within_workspace(other / "f.py", root) is True

    @pytest.mark.asyncio
    async def test_read_accepts_declared_outside_root(self, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        other = tmp_path / "sandhi"
        other.mkdir()
        target = other / "contract.rs"
        target.write_text("pub struct UsageEventV1;\n")
        monkeypatch.setenv(fs.EXTRA_READ_ROOTS_ENV, str(other))
        paths = MagicMock(project_root=str(project))
        with (
            patch("victor.config.settings.get_project_paths", return_value=paths),
            patch("subprocess.run", side_effect=FileNotFoundError("no git")),
        ):
            result = await read(path=str(target))
        assert "UsageEventV1" in result

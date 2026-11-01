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

"""Tests for find_project_root — git root detection with edge cases.

Covers: regular repos, submodules (.git file), worktrees, nested repos,
non-git projects (pyproject.toml fallback), temp root avoidance, and
existing .victor/project.db adoption.
"""

import tempfile
from pathlib import Path

import pytest

from victor.config.settings import find_project_root


class TestFindProjectRootGitRepo:
    """Regular git repos (.git directory)."""

    def test_finds_git_root_from_subdirectory(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / ".git").mkdir()
        sub = root / "src" / "deep"
        sub.mkdir(parents=True)

        assert find_project_root(sub) == root

    def test_finds_git_root_from_root_itself(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / ".git").mkdir()

        assert find_project_root(root) == root


class TestFindProjectRootSubmodule:
    """Submodules (.git file, not directory)."""

    def test_submodule_git_file_is_boundary(self, tmp_path: Path) -> None:
        root = tmp_path / "parent"
        root.mkdir()
        (root / ".git").mkdir()

        submodule = root / "sub"
        submodule.mkdir()
        # Submodules have .git as a FILE pointing to parent's modules dir
        (submodule / ".git").write_text("gitdir: ../.git/modules/sub")
        deep = submodule / "src"
        deep.mkdir()

        assert find_project_root(deep) == submodule

    def test_submodule_does_not_adopt_parent_victor(self, tmp_path: Path) -> None:
        root = tmp_path / "parent"
        root.mkdir()
        (root / ".git").mkdir()
        (root / ".victor").mkdir()
        (root / ".victor" / "project.db").write_text("")

        submodule = root / "sub"
        submodule.mkdir()
        (submodule / ".git").write_text("gitdir: ../.git/modules/sub")
        deep = submodule / "src"
        deep.mkdir()

        # Should stop at submodule boundary, not adopt parent's .victor
        assert find_project_root(deep) == submodule


class TestFindProjectRootNestedRepos:
    """Nested repos (a .git directory within another repo)."""

    def test_nested_repo_is_separate_project(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / ".git").mkdir()

        inner = outer / "vendor" / "lib"
        inner.mkdir(parents=True)
        (inner / ".git").mkdir()
        deep = inner / "src"
        deep.mkdir()

        assert find_project_root(deep) == inner

    def test_outer_repo_not_adopted_from_nested(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / ".git").mkdir()
        (outer / ".victor").mkdir()
        (outer / ".victor" / "project.db").write_text("")

        inner = outer / "nested"
        inner.mkdir()
        (inner / ".git").mkdir()
        deep = inner / "src"
        deep.mkdir()

        # Should stop at inner .git, not adopt outer's .victor
        assert find_project_root(deep) == inner


class TestFindProjectRootNonGit:
    """Non-git projects — fallback to project markers."""

    def test_finds_pyproject_toml(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]")
        sub = root / "src"
        sub.mkdir()

        assert find_project_root(sub) == root

    def test_finds_package_json(self, tmp_path: Path) -> None:
        root = tmp_path / "nodeproject"
        root.mkdir()
        (root / "package.json").write_text("{}")
        sub = root / "lib"
        sub.mkdir()

        assert find_project_root(sub) == root

    def test_fallback_to_start_when_no_markers(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        assert find_project_root(empty) == empty


class TestFindProjectRootExistingVictor:
    """Existing .victor/project.db adoption."""

    def test_adopts_existing_victor_db(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / ".git").mkdir()
        (root / ".victor").mkdir()
        (root / ".victor" / "project.db").write_text("")
        sub = root / "src"
        sub.mkdir()

        assert find_project_root(sub) == root

    def test_victor_db_stronger_than_git_boundary(self, tmp_path: Path) -> None:
        """An existing .victor/project.db at the git root is found first."""
        root = tmp_path / "repo"
        root.mkdir()
        (root / ".git").mkdir()
        (root / ".victor").mkdir()
        (root / ".victor" / "project.db").write_text("")

        assert find_project_root(root) == root


class TestFindProjectRootTempRootAvoidance:
    """Temp root must not be adopted as project root."""

    def test_does_not_adopt_temp_root_victor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_temp = tmp_path / "faketmp"
        fake_temp.mkdir()
        (fake_temp / ".victor").mkdir()
        (fake_temp / ".victor" / "project.db").write_text("")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_temp))

        case_dir = fake_temp / "pytest-of-user" / "pytest-1" / "case0"
        case_dir.mkdir(parents=True)

        # Should NOT adopt fake_temp's .victor
        result = find_project_root(case_dir)
        assert result != fake_temp
        assert result == case_dir  # fallback to start

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

"""Project database root resolution (walk-up to the owning project).

Regression coverage for the bug where querying a *subdirectory* (e.g. graph
analytics scoped to ``src/network``) resolved to a fresh empty
``src/network/.victor/project.db`` instead of the repository's real
``<root>/.victor/project.db`` — which made broad graph modes report the database
as "unavailable" and littered stray ``.victor/`` directories through the tree.
"""

import tempfile
from pathlib import Path

import pytest

from victor.core.database import (
    _normalize_project_database_paths,
    resolve_project_db_root,
)


def _fresh_dir(tmp_path: Path, name: str) -> Path:
    """A unique directory under tmp_path, free of autouse-fixture artifacts."""
    d = (tmp_path / name).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_indexed_repo(root: Path) -> None:
    (root / ".git").mkdir(exist_ok=True)
    (root / ".victor").mkdir(exist_ok=True)
    (root / ".victor" / "project.db").write_text("")


def test_subdirectory_resolves_to_repo_project_db(tmp_path: Path) -> None:
    root = _fresh_dir(tmp_path, "repo")
    _make_indexed_repo(root)
    sub = root / "src" / "network"
    sub.mkdir(parents=True)

    project_root, project_dir, db_path = _normalize_project_database_paths(sub)

    assert project_root == root
    assert project_dir == root / ".victor"
    assert db_path == root / ".victor" / "project.db"
    assert resolve_project_db_root(sub) == root


def test_subdirectory_lookup_creates_no_stray_victor_dir(tmp_path: Path) -> None:
    root = _fresh_dir(tmp_path, "repo")
    _make_indexed_repo(root)
    sub = root / "src" / "services"
    sub.mkdir(parents=True)

    _normalize_project_database_paths(sub)

    assert not (sub / ".victor").exists()


def test_root_with_project_db_resolves_to_itself(tmp_path: Path) -> None:
    root = _fresh_dir(tmp_path, "repo")
    _make_indexed_repo(root)

    project_root, _project_dir, db_path = _normalize_project_database_paths(root)

    assert project_root == root
    assert db_path == root / ".victor" / "project.db"


def test_fresh_repo_initializes_in_place(tmp_path: Path) -> None:
    """A not-yet-indexed repo still initializes its DB at the requested path."""
    root = _fresh_dir(tmp_path, "repo")
    (root / ".git").mkdir()  # repo boundary, but no project.db yet
    sub = root / "a" / "b"
    sub.mkdir(parents=True)

    project_root, _project_dir, db_path = _normalize_project_database_paths(sub)

    # Bounded by .git: no ancestor project.db, so it stays at the requested path.
    assert project_root == sub
    assert db_path == sub / ".victor" / "project.db"


def test_walk_up_stops_at_git_boundary(tmp_path: Path) -> None:
    """An ancestor project.db above the git root must not be adopted."""
    outer = _fresh_dir(tmp_path, "outer")
    (outer / ".victor").mkdir()
    (outer / ".victor" / "project.db").write_text("")  # unrelated outer project

    repo = outer / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # inner repo boundary, no project.db
    sub = repo / "src"
    sub.mkdir()

    project_root, _project_dir, _db_path = _normalize_project_database_paths(sub)

    # Must not cross the inner .git boundary up into ``outer``.
    assert project_root == sub


def test_walk_up_never_adopts_temp_root_project_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray ``.victor`` at the world-writable system temp root is poison.

    Once any process leaves (or plants) ``<tempdir>/.victor/project.db``, every
    resolution starting under the temp dir — including every pytest ``tmp_path``,
    which has no ``.git`` bound — would adopt that one shared database. The walk
    must stop at the temp root instead.
    """
    fake_temp_root = _fresh_dir(tmp_path, "faketmp")
    (fake_temp_root / ".victor").mkdir()
    (fake_temp_root / ".victor" / "project.db").write_text("")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_temp_root))

    case_dir = fake_temp_root / "pytest-of-user" / "pytest-1" / "case0"
    case_dir.mkdir(parents=True)

    project_root, _project_dir, db_path = _normalize_project_database_paths(case_dir)

    assert project_root == case_dir
    assert db_path == case_dir / ".victor" / "project.db"


def test_temp_root_itself_still_resolves_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving the temp root directly keeps its own DB via the in-place fallback."""
    fake_temp_root = _fresh_dir(tmp_path, "faketmp")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_temp_root))

    project_root, _project_dir, db_path = _normalize_project_database_paths(fake_temp_root)

    assert project_root == fake_temp_root
    assert db_path == fake_temp_root / ".victor" / "project.db"


def test_explicit_db_path_is_respected(tmp_path: Path) -> None:
    db_file = _fresh_dir(tmp_path, "custom") / "project.db"

    project_root, project_dir, db_path = _normalize_project_database_paths(db_file)

    assert db_path == db_file
    assert project_dir == db_file.parent
    assert project_root == db_file.parent.parent

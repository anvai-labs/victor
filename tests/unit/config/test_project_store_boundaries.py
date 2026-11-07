"""Store ownership boundaries: global home state and explicit index roots.

Observed: a source tree under ~/.local/share would have been adopted into the
global ~/.victor/project.db, and a nested source tree was indexed into the
enclosing repository's database with paths relative to the inner tree.
"""


import pytest

from victor.config.settings import find_project_root
from victor.core.database import _find_existing_project_dir, resolve_project_db_root


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home directory whose .victor holds the global store, as on real hosts."""
    home = tmp_path / "home"
    (home / ".victor").mkdir(parents=True)
    (home / ".victor" / "project.db").touch()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_a_tree_under_home_is_not_adopted_into_global_state(home):
    tree = home / ".local" / "share" / "sources" / "iceberg" / "tree"
    tree.mkdir(parents=True)
    assert _find_existing_project_dir(tree.resolve()) is None
    assert find_project_root(tree) != home


def test_starting_at_home_keeps_its_own_store(home):
    assert _find_existing_project_dir(home.resolve()) == (home / ".victor").resolve()
    assert find_project_root(home) == home.resolve()


def test_a_project_under_home_still_adopts_its_own_root_store(home):
    repo = home / "code" / "repo"
    (repo / ".victor").mkdir(parents=True)
    (repo / ".victor" / "project.db").touch()
    (repo / "src" / "pkg").mkdir(parents=True)
    assert (
        _find_existing_project_dir((repo / "src" / "pkg").resolve()) == (repo / ".victor").resolve()
    )


def test_an_explicit_index_root_claims_its_store_inside_an_enclosing_project(tmp_path):
    from victor.ui.commands.graph import _claim_index_store

    outer = tmp_path / "outer"
    (outer / ".victor").mkdir(parents=True)
    (outer / ".victor" / "project.db").touch()
    root = (outer / "build" / "snapshot" / "tree").resolve()
    root.mkdir(parents=True)
    assert resolve_project_db_root(root) == outer.resolve()
    assert _claim_index_store(root) == outer.resolve()
    assert (root / ".victor" / "project.db").is_file()
    assert resolve_project_db_root(root) == root
    assert _claim_index_store(root) is None  # already owns its store


def test_a_fresh_index_root_without_ancestors_initializes_in_place(tmp_path):
    from victor.ui.commands.graph import _claim_index_store

    root = (tmp_path / "standalone").resolve()
    root.mkdir()
    assert _claim_index_store(root) is None
    assert not (root / ".victor").exists()

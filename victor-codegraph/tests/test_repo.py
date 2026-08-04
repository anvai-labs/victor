"""Repo-walk tests — fully offline (tmp tree + stdlib ast for the .py files)."""

from __future__ import annotations

from victor_codegraph import chunk_path, chunk_repo, iter_source_files, parse_path


def _make_tree(root):
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("def a():\n    return b()\ndef b():\n    return 1\n")
    (root / "pkg" / "notes.txt").write_text("not source")  # unknown extension
    (root / "README.md").write_text("# docs")  # markdown is not in the language map
    # noise dirs that must be skipped
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.py").write_text("def vendored():\n    pass\n")
    (root / ".git").mkdir()
    (root / ".git" / "hook.py").write_text("def hook():\n    pass\n")
    return root


def test_iter_source_files_filters_and_skips_noise(tmp_path):
    _make_tree(tmp_path)
    found = {p.name for p in iter_source_files(tmp_path)}
    assert found == {"a.py"}  # .txt/.md skipped; node_modules/.git pruned


def test_iter_source_files_language_filter(tmp_path):
    _make_tree(tmp_path)
    assert {p.name for p in iter_source_files(tmp_path, languages=["python"])} == {"a.py"}
    assert list(iter_source_files(tmp_path, languages=["rust"])) == []


def test_iter_single_file(tmp_path):
    f = tmp_path / "solo.py"
    f.write_text("def f():\n    return 1\n")
    assert [p.name for p in iter_source_files(f)] == ["solo.py"]
    assert list(iter_source_files(f, languages=["rust"])) == []


def test_chunk_path(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f():\n    return 1\n")
    chunks = chunk_path(f)
    assert chunks and any(c.metadata.get("simple_name") == "f" for c in chunks)


def test_chunk_path_unreadable_returns_empty(tmp_path):
    assert chunk_path(tmp_path / "does_not_exist.py") == []


def test_parse_path(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def a():\n    return b()\ndef b():\n    return 1\n")
    parsed = parse_path(f)
    assert parsed is not None
    assert {s.simple_name for s in parsed.symbols} == {"a", "b"}


def test_chunk_repo_streams_all_files(tmp_path):
    _make_tree(tmp_path)
    chunks = list(chunk_repo(tmp_path))
    names = {c.metadata.get("simple_name") for c in chunks}
    assert {"a", "b"} <= names  # symbols from pkg/a.py, vendored dirs excluded

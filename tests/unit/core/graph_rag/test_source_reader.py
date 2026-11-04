"""Descriptor-bound source snapshots reject links, races and special files."""

import os

import pytest

from victor.core.graph_rag.source_reader import read_project_source
from victor.core.graph_rag.structural_clones import scan


@pytest.mark.skipif(os.name != "posix", reason="descriptor traversal requires POSIX")
def test_regular_source_is_bounded(tmp_path):
    (tmp_path / "a.py").write_bytes(b"hello")
    data, info = read_project_source(tmp_path, "a.py", 10)
    assert data == b"hello" and info.st_size == 5
    data, info = read_project_source(tmp_path, "a.py", 2)
    assert data == b"" and info.st_size == 5


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor capability check")
def test_missing_secure_traversal_fails_before_source_open(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(OSError, match="Secure descriptor"):
        read_project_source(tmp_path, "a.py", 10)


@pytest.mark.skipif(os.name != "posix", reason="POSIX links and FIFO")
def test_external_symlink_and_fifo_cannot_become_source_units(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "private.py"
    outside.write_text("def secret(x):\n return x+1\n")
    (project / "linked.py").symlink_to(outside)
    os.mkfifo(project / "pipe.py")
    report = scan(project, min_nodes=5, retain_source=True)
    assert not report.coverage_complete
    assert report.stats["unreadable_files"] == 2
    assert not report.units and not report.source_manifest


@pytest.mark.skipif(os.name != "posix", reason="descriptor traversal race")
def test_ancestor_replacement_cannot_redirect_an_open_directory(tmp_path, monkeypatch):
    folder = tmp_path / "project" / "folder"
    folder.mkdir(parents=True)
    (folder / "a.py").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_bytes(b"outside")
    original = os.open

    def replace(path, flags, *args, **kwargs):
        if path == "a.py":
            folder.rename(folder.with_name("old"))
            folder.symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {replace})
    monkeypatch.setattr(os, "open", replace)
    data, _ = read_project_source(folder.parent, "folder/a.py", 100)
    assert data == b"inside"


@pytest.mark.skipif(os.name != "posix", reason="descriptor traversal race")
def test_leaf_symlink_replacement_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / "a.py"
    target.write_bytes(b"inside")
    outside = tmp_path / "secret"
    outside.write_bytes(b"outside")
    original = os.open

    def replace(path, flags, *args, **kwargs):
        if path == "a.py":
            target.unlink()
            target.symlink_to(outside)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {replace})
    monkeypatch.setattr(os, "open", replace)
    with pytest.raises(OSError):
        read_project_source(tmp_path, "a.py", 100)


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py"])
def test_nonlocal_paths_are_rejected(tmp_path, name):
    with pytest.raises(OSError):
        read_project_source(tmp_path, name, 100)


@pytest.mark.skipif(os.name != "posix", reason="descriptor traversal race")
def test_inventory_root_replacement_cannot_redirect_source_reads(tmp_path, monkeypatch):
    from victor.core.graph_rag import structural_clones

    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def inside(x):\n return x+1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("def secret(x):\n return x-1\n")

    def replace_root(*args, **kwargs):
        project.rename(tmp_path / "original")
        project.symlink_to(outside, target_is_directory=True)
        return ["a.py"]

    monkeypatch.setattr(structural_clones, "_source_files", replace_root)
    with pytest.raises(OSError, match="root changed"):
        scan(project, min_nodes=5, retain_source=True, collect_only=True)

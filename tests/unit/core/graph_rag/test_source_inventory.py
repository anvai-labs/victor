"""Inventory has the same directory authority as source reads and bounded Git I/O."""

import os
from pathlib import Path
import subprocess

import pytest

from victor.core.graph_rag import source_inventory
from victor.core.graph_rag.source_reader import project_source_reader
from victor.core.graph_rag.structural_clones import scan


def test_git_ignore_fifo_is_a_bounded_failure_not_filesystem_fallback(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("def f(x):\n return x+1\n")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    fifo = tmp_path / "ignore-fifo"
    os.mkfifo(fifo)
    subprocess.run(["git", "config", "core.excludesFile", str(fifo)], cwd=tmp_path, check=True)
    monkeypatch.setattr(source_inventory, "INVENTORY_TIMEOUT", 0.2)
    with project_source_reader(tmp_path) as reader:
        with pytest.raises(ValueError, match="time budget"):
            source_inventory.source_files(reader.descriptor, {".py"})


def test_replaced_and_restored_path_cannot_change_owned_inventory(tmp_path, monkeypatch):
    project, outside, original = (tmp_path / name for name in ("project", "outside", "original"))
    project.mkdir()
    outside.mkdir()
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    for name in ("a.py", "b.py"):
        (project / name).write_text(source)
    (outside / "a.py").write_text("def secret(x):\n return x-1\n")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "a.py"], cwd=project, check=True)
    run = source_inventory.OwnedProcess.run_sync

    def replaced(job):
        project.rename(original)
        project.symlink_to(outside, target_is_directory=True)
        try:
            return run(job)
        finally:
            project.unlink()
            original.rename(project)

    monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", replaced)
    result = scan(project, min_nodes=5, retain_source=True)
    assert result.coverage_complete
    assert {u.file_path for u in result.units} == {"a.py", "b.py"}
    assert result.families
    assert all("secret" not in u.source_code for u in result.units)


def test_git_failure_does_not_silently_broaden_selection(tmp_path, monkeypatch):
    def failed(job):
        (job.work / "backend.log").write_bytes(b"")
        (job.work / "backend.stderr").write_bytes(b"fatal: bad config")
        return 128, "test"

    monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", failed)
    with project_source_reader(tmp_path) as reader:
        with pytest.raises(ValueError, match="inventory failed"):
            source_inventory.source_files(reader.descriptor, {".py"})


def test_repository_worktree_override_cannot_hide_owned_source_files(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    for name in ("a.py", "b.py"):
        (project / name).write_text(source)
    (outside / "a.py").write_text(source)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "a.py"], cwd=project, check=True)
    subprocess.run(["git", "config", "core.worktree", str(outside)], cwd=project, check=True)
    result = scan(project, min_nodes=5)
    assert result.coverage_complete
    assert {u.file_path for u in result.units} == {"a.py", "b.py"}
    assert result.families


def test_git_unavailable_uses_owned_filesystem_walk(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def f():\n return 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "hidden.py").write_text("def hidden(): pass\n")
    monkeypatch.setattr(source_inventory.shutil, "which", lambda _: None)
    with project_source_reader(tmp_path) as reader:
        assert source_inventory.source_files(reader.descriptor, {".py"}) == ["a.py"]


def test_relative_git_executable_is_resolved_before_descriptor_chdir(tmp_path, monkeypatch):
    controller = tmp_path / "controller"
    controller.mkdir()
    tools = controller / "tools"
    tools.mkdir()
    (tools / "git").write_text("trusted executable fixture")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(controller)
    monkeypatch.setattr(source_inventory.shutil, "which", lambda _: "tools/git")
    received = []

    def inspect(job):
        received.append(job.arguments[0])
        (job.work / "backend.log").write_bytes(b"a.py\0")
        (job.work / "backend.stderr").write_bytes(b"")
        return 0, "test"

    monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", inspect)
    with project_source_reader(project) as reader:
        assert source_inventory.source_files(reader.descriptor, {".py"}) == ["a.py"]
    assert received == [str(tools / "git")]


def test_zero_source_entries_cannot_bypass_filesystem_work_budget(tmp_path, monkeypatch):
    for i in range(3):
        (tmp_path / f"entry{i}.txt").write_text("not selected source")
    monkeypatch.setattr(source_inventory.shutil, "which", lambda _: None)
    monkeypatch.setattr(source_inventory, "MAX_WALK_ENTRIES", 2)
    with project_source_reader(tmp_path) as reader:
        with pytest.raises(ValueError, match="examined-entry budget"):
            source_inventory.source_files(reader.descriptor, {".py"})


def test_filesystem_directory_and_deadline_budgets_fail_closed(tmp_path, monkeypatch):
    (tmp_path / "child").mkdir()
    monkeypatch.setattr(source_inventory.shutil, "which", lambda _: None)
    monkeypatch.setattr(source_inventory, "MAX_WALK_DIRECTORIES", 1)
    with project_source_reader(tmp_path) as reader:
        with pytest.raises(ValueError, match="directory/depth budget"):
            source_inventory.source_files(reader.descriptor, {".py"})
    monkeypatch.setattr(source_inventory, "MAX_WALK_DIRECTORIES", 100)
    monkeypatch.setattr(source_inventory, "INVENTORY_TIMEOUT", -1)
    with project_source_reader(tmp_path) as reader:
        with pytest.raises(ValueError, match="time budget"):
            source_inventory.source_files(reader.descriptor, {".py"})


@pytest.mark.parametrize("redirect", [False, True])
def test_nested_root_preserves_index_prefix_and_ancestor_ignore(tmp_path, redirect):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    selected = project / "src"
    selected.mkdir()
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    for name in ("a.py", "b.py", "ignored.py"):
        (selected / name).write_text(source)
    (project / "outside.py").write_text(source)
    (project / ".gitignore").write_text("src/ignored.py\n")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "src/a.py", "outside.py"], cwd=project, check=True)
    if redirect:
        subprocess.run(["git", "config", "core.worktree", str(outside)], cwd=project, check=True)
    result = scan(selected, min_nodes=5)
    assert result.coverage_complete
    assert {u.file_path for u in result.units} == {"a.py", "b.py"}
    assert result.families


def test_moved_and_restored_nested_root_cannot_change_git_ancestor(tmp_path, monkeypatch):
    original, foreign = tmp_path / "original", tmp_path / "foreign"
    original.mkdir()
    foreign.mkdir()
    selected = original / "src"
    selected.mkdir()
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    for name in ("a.py", "b.py"):
        (selected / name).write_text(source)
    for root in (original, foreign):
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "src/a.py"], cwd=original, check=True)
    (foreign / ".git" / "info" / "exclude").write_text("b.py\n")
    assert scan(selected, min_nodes=5).coverage_complete
    run = source_inventory.OwnedProcess.run_sync

    def moved(job):
        selected.rename(foreign / "src")
        try:
            return run(job)
        finally:
            (foreign / "src").rename(selected)

    monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", moved)
    with pytest.raises(ValueError, match="inventory failed"):
        scan(selected, min_nodes=5)


@pytest.mark.parametrize("redirect_common", [False, True])
def test_linked_worktree_nested_inventory_retains_its_git_context(
    tmp_path, monkeypatch, redirect_common
):
    repository, linked = tmp_path / "repository", tmp_path / "linked"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "src").mkdir()
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    (repository / "src" / "a.py").write_text(source)
    subprocess.run(["git", "add", "src/a.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "worktree", "add", "-qb", "fixture", str(linked)], cwd=repository, check=True
    )
    (linked / "src" / "b.py").write_text(source)
    result = scan(linked / "src", min_nodes=5)
    assert result.coverage_complete
    assert {u.file_path for u in result.units} == {"a.py", "b.py"}
    assert result.families
    if redirect_common:
        git_directory = linked.joinpath(".git").read_text().removeprefix("gitdir: ").strip()
        selector = Path(git_directory) / "commondir"
        before = selector.read_bytes()
        run = source_inventory.OwnedProcess.run_sync

        def redirected(job):
            selector.write_text(str(repository))
            try:
                return run(job)
            finally:
                selector.write_bytes(before)

        monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", redirected)
        with pytest.raises(ValueError, match="inventory failed"):
            scan(linked / "src", min_nodes=5)


@pytest.mark.parametrize("marker_type", ["file", "directory"])
def test_injected_nearer_git_metadata_cannot_change_owned_inventory(
    tmp_path, monkeypatch, marker_type
):
    project, foreign = tmp_path / "project", tmp_path / "foreign"
    project.mkdir()
    foreign.mkdir()
    selected = project / "src"
    selected.mkdir()
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    for name in ("a.py", "b.py"):
        (selected / name).write_text(source)
    for root in (project, foreign):
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "src/a.py"], cwd=project, check=True)
    (foreign / ".git" / "info" / "exclude").write_text("b.py\n")
    run = source_inventory.OwnedProcess.run_sync

    def injected(job):
        marker = selected / ".git"
        if marker_type == "file":
            marker.write_text(f"gitdir: {foreign / '.git'}\n")
        else:
            marker.symlink_to(foreign / ".git", target_is_directory=True)
        try:
            return run(job)
        finally:
            marker.unlink()

    monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", injected)
    with pytest.raises(ValueError, match="inventory failed"):
        scan(selected, min_nodes=5)


@pytest.mark.parametrize("mutation", ["gitfile", "git_target", "commondir_injection"])
def test_git_selector_indirection_is_bound_at_inventory_boundary(tmp_path, monkeypatch, mutation):
    project, metadata, foreign = tmp_path / "project", tmp_path / "metadata", tmp_path / "foreign"
    foreign.mkdir()
    subprocess.run(
        ["git", "init", "-q", "--separate-git-dir", str(metadata), str(project)], check=True
    )
    subprocess.run(["git", "init", "-q"], cwd=foreign, check=True)
    source = "def f(x):\n value=x+1\n return value*3+4\n"
    for name in ("a.py", "b.py"):
        (project / name).write_text(source)
    subprocess.run(["git", "add", "a.py"], cwd=project, check=True)
    assert scan(project, min_nodes=5).coverage_complete
    run = source_inventory.OwnedProcess.run_sync

    def redirected(job):
        if mutation == "gitfile":
            marker = project / ".git"
            before = marker.read_bytes()
            marker.write_text(f"gitdir: {foreign / '.git'}\n")
        elif mutation == "git_target":
            metadata.rename(tmp_path / "saved-metadata")
            metadata.symlink_to(foreign / ".git", target_is_directory=True)
        else:
            (metadata / "commondir").write_text(str(foreign / ".git"))
        try:
            return run(job)
        finally:
            if mutation == "gitfile":
                marker.write_bytes(before)
            elif mutation == "git_target":
                metadata.unlink()
                (tmp_path / "saved-metadata").rename(metadata)
            else:
                (metadata / "commondir").unlink()

    monkeypatch.setattr(source_inventory.OwnedProcess, "run_sync", redirected)
    with pytest.raises(ValueError, match="inventory failed"):
        scan(project, min_nodes=5)


@pytest.mark.parametrize("contents", [b"gitdir: " + b"x" * 4096, b"invalid", b"gitdir: \n"])
def test_malformed_or_oversized_gitfile_fails_closed(tmp_path, contents):
    (tmp_path / ".git").write_bytes(contents)
    with pytest.raises(ValueError, match="repository selector"):
        scan(tmp_path, min_nodes=5)

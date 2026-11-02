"""Windows source namespace checks and real-OS handle qualification."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from victor.core.graph_rag import windows_source
from victor.core.graph_rag.source_reader import project_source_reader, read_project_source


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "a/../b.py",
        "a//b.py",
        "/a.py",
        "C:/a.py",
        "C:a.py",
        "\\\\server\\share\\a.py",
        "a\\b.py",
        "a.py:secret",
        "NUL.py",
        "COM1.py",
        "LPT¹.txt",
        "a. /b.py",
        "a.py.",
        "a.py ",
        "a\x00.py",
        "a?.py",
    ],
)
def test_namespace_aliases_are_rejected(name):
    with pytest.raises(OSError):
        windows_source.relative_parts(name)


def test_relative_unicode_source_name_is_preserved():
    assert windows_source.relative_parts("src/café.py") == ("src", "café.py")


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, 2**64])
def test_job_budget_is_rejected_before_api_or_handle_allocation(limit, monkeypatch):
    from victor.core.graph_rag import windows_process

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid budget reached Windows API allocation")

    monkeypatch.setattr(windows_process.ct, "WinDLL", unexpected, raising=False)
    with pytest.raises(ValueError, match="memory budget"):
        windows_process._Job(limit)


@pytest.mark.parametrize("configure_failure", [False, True])
def test_job_handle_is_closed_on_success_and_constructor_failure(monkeypatch, configure_failure):
    from types import SimpleNamespace
    from victor.core.graph_rag import windows_process

    created, closed = [], []

    def create(*args):
        created.append(123)
        return 123

    def close(handle):
        closed.append(handle)
        return True

    def configure(*args):
        if configure_failure:
            raise RuntimeError("controlled configuration failure")
        return True

    def success(*args):
        return True

    api = SimpleNamespace(
        CreateJobObjectW=create,
        CloseHandle=close,
        SetInformationJobObject=configure,
        AssignProcessToJobObject=success,
        TerminateJobObject=success,
        QueryInformationJobObject=success,
        OpenProcess=success,
        IsProcessInJob=success,
        GetProcessMemoryInfo=success,
    )
    monkeypatch.setattr(windows_process.ct, "WinDLL", lambda *a, **kw: api, raising=False)
    if configure_failure:
        with pytest.raises(RuntimeError, match="controlled configuration failure"):
            windows_process._Job(1024)
    else:
        windows_process._Job(1024).close()
    assert created == closed == [123]


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows handle APIs")
class TestWindowsHandles:
    def test_bounded_read_and_identity(self, tmp_path):
        target = tmp_path / "src" / "café.py"
        target.parent.mkdir()
        target.write_bytes(b"hello")
        with project_source_reader(tmp_path) as reader:
            data, info = reader("src/café.py", 5)
            assert data == b"hello" and info.st_size == 5
            assert info.st_ino and info.st_mtime_ns
            assert reader("src/café.py", 4)[0] == b""
            reader.verify_root_path()
            with pytest.raises(OSError):
                tmp_path.rename(tmp_path.with_name(tmp_path.name + "-moved"))
        target.write_bytes(b"after")
        assert read_project_source(tmp_path, "src/café.py", 5)[0] == b"after"

    def test_reparse_leaf_and_ancestor_are_rejected(self, tmp_path):
        project, outside = tmp_path / "project", tmp_path / "outside"
        project.mkdir()
        outside.mkdir()
        (outside / "a.py").write_bytes(b"private")
        # Junction creation needs no Developer Mode or elevated symlink privilege.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(project / "linked"), str(outside)],
            check=True,
            capture_output=True,
        )
        try:
            with pytest.raises(OSError):
                read_project_source(project, "linked/a.py", 100)
            with pytest.raises(OSError):
                read_project_source(project / "linked", "a.py", 100)
        finally:
            os.rmdir(project / "linked")
        (project / "linked.py").symlink_to(outside / "a.py")
        with pytest.raises(OSError, match="reparse"):
            read_project_source(project, "linked.py", 100)

    def test_pinned_ancestor_cannot_be_replaced_before_leaf_open(self, tmp_path, monkeypatch):
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "a.py").write_bytes(b"inside")
        original = windows_source._open
        attempts = []

        def replace(api, parent, name, directory):
            if name == "a.py":
                with pytest.raises(OSError):
                    folder.rename(tmp_path / "moved")
                attempts.append(name)
            return original(api, parent, name, directory)

        monkeypatch.setattr(windows_source, "_open", replace)
        assert read_project_source(tmp_path, "folder/a.py", 100)[0] == b"inside"
        assert attempts == ["a.py"]

    def test_subst_drive_alias_is_rejected(self, tmp_path):
        existing = {drive.casefold() for drive in os.listdrives()}
        drive = next(
            f"{letter}:"
            for letter in "ZYXWVUTSRQPONMLKJIHGFED"
            if f"{letter}:\\".casefold() not in existing
        )
        (tmp_path / "a.py").write_bytes(b"source")
        subprocess.run(["subst", drive, str(tmp_path)], check=True, capture_output=True)
        try:
            with pytest.raises(OSError, match="aliased drive roots"):
                read_project_source(Path(drive + "\\"), "a.py", 100)
        finally:
            subprocess.run(["subst", drive, "/D"], check=True, capture_output=True)

    def test_reparse_directory_fails_inventory(self, tmp_path):
        from victor.core.graph_rag.source_inventory import source_files

        project, outside = tmp_path / "project", tmp_path / "outside"
        project.mkdir()
        outside.mkdir()
        (outside / "private.py").write_bytes(b"private")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(project / "linked"), str(outside)],
            check=True,
            capture_output=True,
        )
        try:
            with project_source_reader(project) as reader:
                with pytest.raises(OSError, match="reparse directories"):
                    source_files(reader.descriptor, {".py"})
        finally:
            os.rmdir(project / "linked")

    def test_inventory_bounds_and_git_boundary(self, tmp_path, monkeypatch):
        from victor.core.graph_rag import windows_inventory as source_inventory

        (tmp_path / "a.py").write_bytes(b"source")
        (tmp_path / "ignored.txt").write_bytes(b"other")
        with project_source_reader(tmp_path) as reader:
            assert source_inventory.source_files(reader.descriptor, {".py"}) == ["a.py"]
            monkeypatch.setattr(source_inventory, "MAX_WALK_ENTRIES", 1)
            with pytest.raises(ValueError, match="entry budget"):
                source_inventory.source_files(reader.descriptor, {".py"})
        monkeypatch.setattr(source_inventory, "MAX_WALK_ENTRIES", 200_000)
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "ignored.py").write_bytes(b"ignored")
        (tmp_path / "tracked.py").write_bytes(b"tracked")
        subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
        monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path.parent))
        with project_source_reader(tmp_path) as reader:
            assert source_inventory.source_files(reader.descriptor, {".py"}) == [
                "a.py",
                "tracked.py",
            ]
        child = tmp_path / "child"
        child.mkdir()
        (child / "local.py").write_bytes(b"local")
        with project_source_reader(child) as reader:
            assert source_inventory.source_files(reader.descriptor, {".py"}) == ["local.py"]

    def test_linked_worktree_selector_fails_cleanly(self, tmp_path):
        from victor.core.graph_rag.source_inventory import source_files

        (tmp_path / ".git").write_text("gitdir: ../private\n")
        with project_source_reader(tmp_path) as reader:
            with pytest.raises(OSError, match="linked-worktree selectors"):
                source_files(reader.descriptor, {".py"})

    def test_existing_commondir_selector_fails_cleanly(self, tmp_path):
        from victor.core.graph_rag.source_inventory import source_files

        metadata = tmp_path / ".git"
        metadata.mkdir()
        (metadata / "commondir").write_text("../external\n")
        with project_source_reader(tmp_path) as reader:
            with pytest.raises(OSError, match="common-repository selectors"):
                source_files(reader.descriptor, {".py"})

    def test_structural_scan_retains_actual_source_hash_and_coverage(self, tmp_path):
        import hashlib
        from victor.core.graph_rag.structural_clones import scan

        data = b"def add(value):\n    return value + 1\n"
        (tmp_path / "a.py").write_bytes(data)
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        report = scan(tmp_path, min_nodes=5, retain_source=True, collect_only=True)
        assert report.coverage_complete
        assert report.source_manifest == {"a.py": hashlib.sha256(data).hexdigest()}
        assert len(report.units) == 1

    @pytest.mark.parametrize(
        "budget",
        ["MAX_WALK_DIRECTORIES", "MAX_WALK_DEPTH", "MAX_INVENTORY_BYTES", "INVENTORY_TIMEOUT"],
    )
    def test_inventory_resource_limits_fail_closed(self, tmp_path, monkeypatch, budget):
        from victor.core.graph_rag import windows_inventory

        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "a.py").write_bytes(b"source")
        monkeypatch.setattr(windows_inventory, budget, -1 if budget == "INVENTORY_TIMEOUT" else 0)
        with project_source_reader(tmp_path) as reader:
            with pytest.raises(ValueError, match="budget"):
                windows_inventory.source_files(reader.descriptor, {".py"})

    def test_external_process_has_job_ownership(self, tmp_path):
        from victor.core.graph_rag.bounded_process import OwnedProcess

        code, scope = OwnedProcess(
            [sys.executable, "-I", "-S", "-c", "print('owned')"],
            tmp_path,
            5,
            max_rss=256 * 1024 * 1024,
            artifacts={tmp_path / "backend.log": 1024},
        ).run_sync()
        assert code == 0 and scope == "owned-windows-job-polled-rss-and-hard-commit"
        assert (tmp_path / "backend.log").read_bytes().strip() == b"owned"

    def test_hard_commit_cap_denies_large_allocation(self, tmp_path):
        from victor.core.graph_rag.bounded_process import OwnedProcess

        script = (
            "try:\n data = bytearray(1024*1024*1024)\n"
            "except MemoryError:\n print('hard-commit-cap'); raise SystemExit(73)\n"
        )
        code, _ = OwnedProcess(
            [sys.executable, "-I", "-S", "-c", script],
            tmp_path,
            5,
            max_rss=256 * 1024 * 1024,
            artifacts={tmp_path / "backend.log": 1024},
        ).run_sync()
        assert code == 73
        assert b"hard-commit-cap" in (tmp_path / "backend.log").read_bytes()

    def test_baseline_publication_fails_before_creating_a_destination(self, tmp_path):
        from victor.core.graph_rag.copy_artifacts import _parent_directory

        with pytest.raises(ValueError, match="Secure baseline publication is unavailable"):
            with _parent_directory(tmp_path / "new" / "baseline.json"):
                pytest.fail("Unqualified Windows baseline publication was enabled")
        assert list(tmp_path.iterdir()) == []

    def test_job_assignment_failure_never_starts_tool(self, tmp_path, monkeypatch):
        from victor.core.graph_rag.bounded_process import OwnedProcess
        from victor.core.graph_rag.windows_process import _Job

        def reject(self, process):
            raise OSError("assignment rejected")

        monkeypatch.setattr(_Job, "attach", reject)
        marker = tmp_path / "must-not-exist"
        with pytest.raises(OSError, match="assignment rejected"):
            OwnedProcess(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,sys;pathlib.Path(sys.argv[1]).touch()",
                    str(marker),
                ],
                tmp_path,
                5,
                max_rss=256 * 1024 * 1024,
                artifacts={},
            ).run_sync()
        assert not marker.exists()

    async def test_cancellation_terminates_descendants(self, tmp_path):
        import asyncio
        import psutil
        from victor.core.graph_rag.bounded_process import OwnedProcess

        script = (
            "import subprocess,sys,pathlib,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
        )
        pidfile = tmp_path / "descendant.pid"
        task = asyncio.create_task(
            OwnedProcess(
                [sys.executable, "-I", "-S", "-c", script, str(pidfile)],
                tmp_path,
                10,
                max_rss=256 * 1024 * 1024,
                artifacts={},
            ).run()
        )
        try:
            for _ in range(200):
                if pidfile.exists():
                    break
                await asyncio.sleep(0.02)
            assert pidfile.exists()
            descendant = psutil.Process(int(pidfile.read_text()))
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        descendant.wait(timeout=3)

    @pytest.mark.parametrize("kind", ["time", "artifact"])
    def test_external_tool_budgets(self, tmp_path, kind):
        from victor.core.graph_rag.bounded_process import OwnedProcess

        script = "import time;time.sleep(60)" if kind == "time" else "print('x'*10000)"
        with pytest.raises(ValueError, match=f"{kind} budget"):
            OwnedProcess(
                [sys.executable, "-I", "-S", "-c", script],
                tmp_path,
                0.5 if kind == "time" else 5,
                max_rss=256 * 1024 * 1024,
                artifacts={tmp_path / "backend.log": 100},
            ).run_sync()

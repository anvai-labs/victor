"""External calls retain cancellation, resource bounds and fresh ownership."""

import asyncio
import os
import subprocess
import sys

import pytest

from victor.core.graph_rag.bounded_process import OwnedProcess


async def test_async_cancellation_reaps_the_reserved_leader(tmp_path, monkeypatch):
    created = []
    spawn = subprocess.Popen

    def observe(*args, **kwargs):
        process = spawn(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", observe)
    job = OwnedProcess(
        [sys.executable, "-I", "-S", "-c", "import time;time.sleep(30)"],
        tmp_path,
        3,
        max_rss=256 * 1024 * 1024,
        artifacts={tmp_path / "backend.log": 1024},
        recursive=False,
    )
    task = asyncio.create_task(job.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(created) == 1
    assert created[0].returncode is not None


def test_stale_status_cannot_short_circuit_a_new_call(tmp_path):
    (tmp_path / "backend.status").write_text("0")
    with pytest.raises(ValueError, match="fresh owned"):
        OwnedProcess(
            [sys.executable, "-c", "raise RuntimeError('must not execute')"],
            tmp_path,
            1,
            max_rss=256 * 1024 * 1024,
            artifacts={},
            recursive=False,
        ).run_sync()


def test_direct_child_memory_is_bounded_without_process_enumeration(tmp_path):
    with pytest.raises(ValueError, match="memory budget"):
        OwnedProcess(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import time;x=bytearray(80*1024*1024);time.sleep(30)",
            ],
            tmp_path,
            3,
            max_rss=32 * 1024 * 1024,
            artifacts={tmp_path / "backend.log": 1024},
            recursive=False,
        ).run_sync()


@pytest.mark.parametrize("change_during_tool", [False, True])
def test_metadata_identity_is_checked_before_and_after_tool(tmp_path, change_during_tool):
    project, work = tmp_path / "project", tmp_path / "work"
    project.mkdir()
    work.mkdir()
    marker = project / ".git"
    marker.mkdir()
    original = marker.stat()
    identity = project.stat()
    if not change_during_tool:
        marker.rename(project / "old-git")
        marker.mkdir()
    code = (
        "from pathlib import Path;print('executed');"
        "Path('.git').rename('old-git');Path('.git').mkdir()"
    )
    descriptor = os.open(project, os.O_RDONLY | os.O_DIRECTORY)
    try:
        status, _ = OwnedProcess(
            [sys.executable, "-I", "-S", "-c", code],
            work,
            3,
            max_rss=256 * 1024 * 1024,
            artifacts={work / "backend.log": 4096},
            recursive=False,
            cwd_descriptor=descriptor,
            pass_fds=(descriptor,),
            cwd_ancestry=((identity.st_dev, identity.st_ino),),
            cwd_marker=(".git", original.st_dev, original.st_ino),
        ).run_sync()
    finally:
        os.close(descriptor)
    assert status == 125
    assert ("executed" in (work / "backend.log").read_text()) == change_during_tool

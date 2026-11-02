"""Owned qualification mutations cannot redirect original files or inputs."""

import importlib.util
import hashlib
import os
from pathlib import Path
import sqlite3
import sys

import pytest


def load(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("owned_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name != "posix", reason="owned mutation harness requires POSIX")
def test_descriptor_writes_replace_aliases_and_publish_only_at_completion(tmp_path):
    module = load("body_qualification_io")
    output = tmp_path / "published"
    original = tmp_path / "original.py"
    original.write_bytes(b"preserve")
    with module.OwnedWork(output) as owned:
        owned.write_source("nested/a.py", b"first")
        assert owned.read_source("nested/a.py") == b"first"
        leaf = owned.path / "source/nested/a.py"
        leaf.unlink()
        leaf.symlink_to(original)
        owned.write_source("nested/a.py", b"second")
        assert original.read_bytes() == b"preserve"
        assert owned.read_source("nested/a.py") == b"second"
        owned.move_source("nested/a.py", "nested/b.py")
        owned.unlink_source("nested/b.py")
        assert output.is_dir()
        assert not (output / "QUALIFICATION_COMPLETE").exists()
        owned.write("answer.json", b"{}")
        owned.publish()
    assert (output / "answer.json").read_bytes() == b"{}"
    assert (output / "QUALIFICATION_COMPLETE").is_file()
    assert original.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name != "posix", reason="POSIX link race")
def test_source_parent_alias_and_publication_alias_are_rejected(tmp_path):
    module = load("body_qualification_io")
    original = tmp_path / "original"
    original.mkdir()
    (original / "a.py").write_bytes(b"preserve")
    output = tmp_path / "published"
    with module.OwnedWork(output) as owned:
        (owned.path / "source/linked").symlink_to(original, target_is_directory=True)
        with pytest.raises(OSError):
            owned.write_source("linked/a.py", b"corrupt")
        output.rename(tmp_path / "moved-output")
        output.symlink_to(original, target_is_directory=True)
        with pytest.raises(ValueError, match="changed|link"):
            owned.publish()
    assert (original / "a.py").read_bytes() == b"preserve"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor publication")
def test_fresh_output_reservation_cannot_replace_a_racing_empty_directory(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    output = tmp_path / "publication"
    mkdir = module.os.mkdir
    raced_identity = []

    def reserve(path, *args, **kwargs):
        if path == output.name:
            mkdir(path, *args, **kwargs)
            raced_identity.append(output.stat().st_ino)
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "mkdir", reserve)
    with pytest.raises(ValueError, match="new destination"):
        with module.OwnedWork(output):
            pytest.fail("A racing destination was replaced")
    assert output.stat().st_ino == raced_identity[0]
    assert list(output.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor publication")
def test_completion_marker_is_exclusive_and_parent_alias_during_publication_fails(
    tmp_path, monkeypatch
):
    module = load("body_qualification_io")
    parent = tmp_path / "parent"
    parent.mkdir()
    original = tmp_path / "original"
    original.mkdir()
    (original / "preserve").write_bytes(b"unchanged")
    with module.OwnedWork(parent / "publication") as owned:
        marker = owned.path / "QUALIFICATION_COMPLETE"
        marker.write_bytes(b"preexisting")
        with pytest.raises(ValueError, match="already exists"):
            owned.publish()
        assert marker.read_bytes() == b"preexisting"
        marker.unlink()
        original_open = module.os.open

        def replace_parent(path, *args, **kwargs):
            if path == "QUALIFICATION_COMPLETE":
                parent.rename(tmp_path / "moved-parent")
                parent.symlink_to(original, target_is_directory=True)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(module.os, "open", replace_parent)
        with pytest.raises(ValueError, match="ancestor"):
            owned.publish()
        assert not owned.published
        assert not (tmp_path / "moved-parent/publication/QUALIFICATION_COMPLETE").exists()
        assert list(original.iterdir()) == [original / "preserve"]
        assert (original / "preserve").read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor cwd")
def test_owned_git_ignores_inherited_selectors_and_global_callbacks(tmp_path, monkeypatch):
    module = load("qualify_body_incremental")
    original = tmp_path / "original"
    original.mkdir()
    (original / "a.py").write_bytes(b"preserve")
    marker = tmp_path / "must-not-execute"
    hook = tmp_path / "hook"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o700)
    config = tmp_path / "global-config"
    config.write_text(f"[core]\nfsmonitor = {hook}\nhooksPath = {tmp_path}\n")
    monkeypatch.setenv("GIT_DIR", str(original / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(original))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    with module.OwnedWork(tmp_path / "result") as owned:
        owned.write_source("a.py", b"owned")
        module._git(owned, "init", "--template=", "-q")
        module._git(owned, "add", "--all")
        assert (owned.path / "source/.git/index").is_file()
    assert not (original / ".git").exists()
    assert original.joinpath("a.py").read_bytes() == b"preserve"
    assert not marker.exists()


def test_sqlite_input_is_snapshotted_without_writing_original_or_sidecars(tmp_path):
    module = load("body_qualification_io")
    database = tmp_path / "cache.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE body_vectors_v1(profile,input_hash,vector,spans,checksum)")
    before = database.read_bytes()
    existing = {path.name for path in tmp_path.iterdir()}
    with module.cache_database(database) as db:
        assert db.execute("SELECT COUNT(*) FROM body_vectors_v1").fetchone()[0] == 0
    assert database.read_bytes() == before
    assert {path.name for path in tmp_path.iterdir()} == existing
    (tmp_path / "cache.sqlite-wal").write_bytes(b"active")
    with pytest.raises(ValueError, match="checkpointed"):
        with module.cache_database(database):
            pytest.fail("Live cache was accepted")


def test_sqlite_input_byte_and_row_budgets_fail_closed(tmp_path):
    module = load("body_qualification_io")
    large = tmp_path / "large.sqlite"
    with large.open("wb") as stream:
        stream.truncate(module.MAX_DATABASE_BYTES + 1)
    with pytest.raises(ValueError, match="budget"):
        with module.cache_database(large):
            pytest.fail("Oversized cache was accepted")
    rows = tmp_path / "rows.sqlite"
    with sqlite3.connect(rows) as db:
        db.execute("CREATE TABLE body_vectors_v1(profile,input_hash,vector,spans,checksum)")
        db.executemany(
            "INSERT INTO body_vectors_v1 VALUES(?,?,?,?,?)",
            [("p", str(i), b"x", "[]", "c") for i in range(module.MAX_CACHE_ROWS + 1)],
        )
    with pytest.raises(ValueError, match="row budget"):
        with module.cache_database(rows):
            pytest.fail("Over-budget cache was accepted")


def test_sqlite_sidecar_created_during_capture_is_rejected(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    database = tmp_path / "cache.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE body_vectors_v1(profile,input_hash,vector,spans,checksum)")
    read = module.ProjectSourceReader.__call__

    def capture(reader, relative, limit):
        result = read(reader, relative, limit)
        (tmp_path / "cache.sqlite-journal").write_bytes(b"racing writer")
        return result

    monkeypatch.setattr(module.ProjectSourceReader, "__call__", capture)
    with pytest.raises(ValueError, match="closed checkpointed"):
        with module.cache_database(database):
            pytest.fail("A cache with a racing journal was accepted")


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned writes")
def test_zero_progress_write_fails_once_and_removes_its_owned_temporary(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    with module.OwnedWork(tmp_path / "publication") as owned:
        owned.write_source("a.py", b"preserve")
        calls = []

        def no_progress(*args):
            calls.append(args)
            assert len(calls) == 1, "Zero-progress write was retried"
            return 0

        monkeypatch.setattr(module.os, "write", no_progress)
        with pytest.raises(OSError, match="no progress"):
            owned.write_source("a.py", b"replacement")
        assert len(calls) == 1
        assert owned.read_source("a.py") == b"preserve"
        assert {path.name for path in (owned.path / "source").iterdir()} == {"a.py"}


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned writes")
def test_failed_write_cleanup_preserves_a_replaced_temporary_inode(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    with module.OwnedWork(tmp_path / "publication") as owned:
        original_write = os.write
        replacement = []

        def replace_temporary(*args):
            temporary = next((owned.path / "source").glob(".r1-write-*"))
            temporary.unlink()
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                original_write(descriptor, b"unrelated entry")
            finally:
                os.close(descriptor)
            replacement.append(temporary)
            raise OSError("injected write failure")

        monkeypatch.setattr(module.os, "write", replace_temporary)
        with pytest.raises(OSError, match="injected"):
            owned.write_source("a.py", b"replacement")
        assert replacement[0].read_bytes() == b"unrelated entry"


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned publication")
def test_failed_completion_cleanup_preserves_a_replaced_marker_inode(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    with module.OwnedWork(tmp_path / "publication") as owned:
        marker = owned.path / "QUALIFICATION_COMPLETE"

        def replace_marker(*args):
            marker.unlink()
            marker.write_bytes(b"unrelated marker")
            raise OSError("injected marker fsync failure")

        monkeypatch.setattr(module.os, "fsync", replace_marker)
        with pytest.raises(OSError, match="injected"):
            owned.publish()
        assert marker.read_bytes() == b"unrelated marker"
        assert not owned.published


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned writes")
def test_successful_fsync_cannot_commit_a_replaced_write_temporary(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    with module.OwnedWork(tmp_path / "publication") as owned:
        owned.write_source("a.py", b"preserve")
        replacement = []

        def replace_temporary(*args):
            temporary = next((owned.path / "source").glob(".r1-write-*"))
            temporary.unlink()
            temporary.write_bytes(b"unrelated bytes")
            replacement.append(temporary)

        monkeypatch.setattr(module.os, "fsync", replace_temporary)
        with pytest.raises(ValueError, match="file changed"):
            owned.write_source("a.py", b"replacement")
        assert owned.read_source("a.py") == b"preserve"
        assert replacement[0].read_bytes() == b"unrelated bytes"


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned writes")
def test_replaced_destination_after_leaf_rename_cannot_report_success(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    with module.OwnedWork(tmp_path / "publication") as owned:
        rename = os.rename

        def replace_destination(before, after, *args, **kwargs):
            rename(before, after, *args, **kwargs)
            destination = owned.path / "source" / after
            destination.unlink()
            destination.write_bytes(b"unrelated destination")

        monkeypatch.setattr(module.os, "rename", replace_destination)
        with pytest.raises(ValueError, match="file changed"):
            owned.write_source("a.py", b"replacement")
        assert owned.read_source("a.py") == b"unrelated destination"
        assert not owned.published


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned publication")
def test_successful_fsync_cannot_certify_a_replaced_completion_marker(tmp_path, monkeypatch):
    module = load("body_qualification_io")
    with module.OwnedWork(tmp_path / "publication") as owned:
        marker = owned.path / "QUALIFICATION_COMPLETE"
        replaced = []

        def replace_marker(*args):
            if not replaced:
                marker.unlink()
                marker.write_bytes(b"unrelated marker")
                replaced.append(True)

        monkeypatch.setattr(module.os, "fsync", replace_marker)
        with pytest.raises(ValueError, match="file changed"):
            owned.publish()
        assert marker.read_bytes() == b"unrelated marker"
        assert not owned.published


@pytest.mark.skipif(os.name != "posix", reason="POSIX bounded process owner")
@pytest.mark.parametrize("kind", ["artifact", "time", "memory"])
def test_observer_child_budgets_reap_owned_processes(tmp_path, kind):
    module = load("qualify_body_incremental")
    script = {
        "artifact": "print('x'*100000)",
        "time": "import time;time.sleep(30)",
        "memory": "import time;data=bytearray(64*1024*1024);time.sleep(30)",
    }[kind]
    with module.OwnedWork(tmp_path / "result") as owned:
        with pytest.raises(ValueError, match=kind + " budget"):
            module._owned_call(
                [sys.executable, "-I", "-S", "-c", script],
                owned.fd,
                timeout=0.2 if kind == "time" else 5,
                max_rss=32 * 1024 * 1024 if kind == "memory" else 256 * 1024 * 1024,
                max_output=128,
            )


@pytest.mark.skipif(os.name != "posix", reason="descriptor-owned POSIX cwd")
def test_cache_opens_relative_to_pinned_cwd_after_prefix_replacement(tmp_path, monkeypatch):
    probe = load("benchmark_body_cache")
    io = load("body_qualification_io")
    from victor.core.graph_rag.body_clones import LocalBodyEncoder
    from victor.core.graph_rag.structural_clones import StructuralReport, extract_units

    source = "def f(x):\n return x+1\n\ndef g(x):\n return x-1\n"
    report = StructuralReport(
        [],
        [],
        {"source_files": 1, "selected_regions": 1},
        units=extract_units("a.py", source, None, min_nodes=1, retain_source=True),
        source_manifest={"a.py": hashlib.sha256(source.encode()).hexdigest()},
    )
    monkeypatch.setattr(probe, "scan", lambda *args, **kwargs: report)

    class Encoder(LocalBodyEncoder):
        def windows(self, text):
            return [text], [(0, len(text))]

        def encode(self, texts):
            return [[1] * self.options.dimension for _ in texts]

    monkeypatch.setattr(
        probe, "LocalBodyEncoder", lambda options: Encoder(options, producer_id="tests/io-v1")
    )
    original = tmp_path / "original"
    (original / "source").mkdir(parents=True)
    (original / "source/a.py").write_text(source)
    with io.OwnedWork(tmp_path / "publication") as owned:
        owned.write_source("a.py", source.encode())
        monkeypatch.chdir(owned.path)
        analyze = probe.analyze_bodies
        received = []

        def replace_prefix(*args, **kwargs):
            assert kwargs["cache_path"] == Path("body.sqlite")
            if not received:
                owned.path.rename(tmp_path / "saved-private")
                owned.path.symlink_to(original, target_is_directory=True)
            received.append(kwargs["cache_path"])
            return analyze(*args, **kwargs)

        monkeypatch.setattr(probe, "analyze_bodies", replace_prefix)
        observation = probe.benchmark(
            Path("source"), Path("body.sqlite"), state="cold", details=True, owned_cache=True
        )
        assert len(received) == 2
        assert observation["cache_io_mode"] == "descriptor_cwd_relative_leaf"
        assert (tmp_path / "saved-private/body.sqlite").is_file()
        assert not (original / "body.sqlite").exists()

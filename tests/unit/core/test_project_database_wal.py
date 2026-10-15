import sqlite3
from contextlib import closing
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from victor.core import database
from pathlib import Path

from victor.core.database import (
    ProjectDatabaseManager,
    get_project_database,
    reset_project_database,
)


def _assert_project_connection_pragmas(db: ProjectDatabaseManager) -> None:
    conn = db.get_connection()

    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()
    synchronous = conn.execute("PRAGMA synchronous").fetchone()
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
    wal_autocheckpoint = conn.execute("PRAGMA wal_autocheckpoint").fetchone()
    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()

    assert journal_mode is not None and journal_mode[0] == "wal"
    assert synchronous is not None and synchronous[0] == 1
    assert foreign_keys is not None and foreign_keys[0] == 1
    assert wal_autocheckpoint is not None
    assert wal_autocheckpoint[0] == ProjectDatabaseManager._WAL_AUTOCHECKPOINT_PAGES
    # P6: deliberately 5000ms (NOT the 30s connect timeout) — with the undo-log
    # write made non-fatal, a short bounded wait beats a long stall.
    assert busy_timeout is not None and busy_timeout[0] == 5000


def test_project_database_reapplies_connection_pragmas_after_reopen(
    tmp_path: Path,
) -> None:
    db = get_project_database(tmp_path)

    try:
        _assert_project_connection_pragmas(db)

        db.close()

        _assert_project_connection_pragmas(db)
    finally:
        reset_project_database(tmp_path)


def test_project_database_uses_incremental_auto_vacuum(tmp_path: Path) -> None:
    db = get_project_database(tmp_path)
    try:
        # 0=NONE, 1=FULL, 2=INCREMENTAL — a fresh project DB must reclaim incrementally.
        mode = db.get_connection().execute("PRAGMA auto_vacuum").fetchone()
        assert mode is not None and mode[0] == 2
    finally:
        reset_project_database(tmp_path)


def test_maintain_checkpoints_and_reclaims_space(tmp_path: Path) -> None:
    db = get_project_database(tmp_path)
    try:
        # Use the same raw connection maintain() uses so visibility is unambiguous.
        conn = db._get_raw_connection()
        conn.execute("CREATE TABLE bulk (id INTEGER PRIMARY KEY, blob TEXT)")
        conn.executemany(
            "INSERT INTO bulk (blob) VALUES (?)",
            [("x" * 2000,) for _ in range(5000)],
        )
        conn.commit()
        # Materialize the WAL into the main file so the baseline reflects the real data size.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        full_size = db.db_path.stat().st_size
        assert full_size > 1_000_000  # ~10 MB of data is now in the main file

        conn.execute("DELETE FROM bulk")
        conn.commit()

        stats = db.maintain()

        assert stats["path"] == str(db.db_path)
        # VACUUM reclaimed the freed pages: the file shrank well below the populated size.
        assert stats["size_after"] < full_size
        # WAL is truncated by the checkpoint — no oversized -wal left behind.
        wal = db.db_path.with_name(db.db_path.name + "-wal")
        assert (not wal.exists()) or wal.stat().st_size < full_size
    finally:
        reset_project_database(tmp_path)


def test_legacy_database_gets_one_time_auto_vacuum_migration(tmp_path: Path) -> None:
    """Co-design review item 26a: a project DB created before auto_vacuum was
    set has the pragma silently pending — it only takes effect after a VACUUM.
    The first connection must run that VACUUM once, activating INCREMENTAL
    going forward, without requiring a manual `victor db maintain`."""
    import sqlite3

    # Build a legacy DB: auto_vacuum explicitly NONE (the pre-existing default),
    # with enough content to be a realistic file.
    legacy_path = tmp_path / "legacy" / "project.db"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(legacy_path))
    try:
        conn.execute("PRAGMA auto_vacuum=NONE")
        conn.execute("CREATE TABLE bulk (id INTEGER PRIMARY KEY, blob TEXT)")
        conn.executemany("INSERT INTO bulk (blob) VALUES (?)", [("x" * 100,) for _ in range(200)])
        conn.commit()
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0  # 0 = NONE
    finally:
        conn.close()

    db = get_project_database(legacy_path)
    try:
        mode = db.get_connection().execute("PRAGMA auto_vacuum").fetchone()
        assert mode is not None and mode[0] == 2, (
            "one-time VACUUM migration must activate auto_vacuum=INCREMENTAL "
            "on a legacy database"
        )
        assert db._auto_vacuum_migration_done is True
    finally:
        reset_project_database(legacy_path)


def test_fresh_database_skips_auto_vacuum_migration(tmp_path: Path) -> None:
    """A fresh DB already has INCREMENTAL applied at creation: the migration
    must be a no-op (no VACUUM attempted, flag stays False)."""
    db = get_project_database(tmp_path)
    try:
        assert db._auto_vacuum_migration_done is False
        mode = db.get_connection().execute("PRAGMA auto_vacuum").fetchone()
        assert mode is not None and mode[0] == 2
    finally:
        reset_project_database(tmp_path)


@pytest.fixture
def live_wal_source(tmp_path):
    root = tmp_path / "database ?#% paths"
    root.mkdir()
    source = root / "project.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("INSERT INTO evidence VALUES (1, 'committed in WAL')")
        conn.commit()
        yield root, conn


def test_consolidation_backup_is_independent_and_preserves_wal(live_wal_source, tmp_path):
    import shutil
    import stat

    root, source = live_wal_source
    database.DatabaseConsolidator(root)._backup_legacy_databases()
    backups = list(root.glob("project.db.bak.*"))
    assert len(backups) == 1
    restored = tmp_path / "restore.db"
    shutil.copyfile(backups[0], restored)
    with closing(sqlite3.connect(restored)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]
        assert conn.execute("SELECT * FROM evidence").fetchall() == [(1, "committed in WAL")]
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert source.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1


def test_same_clock_backups_do_not_replace_prior_snapshot(live_wal_source, monkeypatch):
    root, source = live_wal_source
    monkeypatch.setattr(database, "datetime", SimpleNamespace(now=lambda: datetime(2026, 10, 14)))
    consolidator = database.DatabaseConsolidator(root)
    consolidator._backup_legacy_databases()
    first = next(root.glob("project.db.bak.*"))
    original = first.read_bytes()
    source.execute("INSERT INTO evidence VALUES (2, 'later commit')")
    source.commit()
    consolidator._backup_legacy_databases()
    backups = list(root.glob("project.db.bak.*"))
    assert len(backups) == 2
    assert first.read_bytes() == original
    counts = []
    for backup in backups:
        with closing(sqlite3.connect(backup)) as conn:
            counts.append(conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
    assert sorted(counts) == [1, 2]


@pytest.mark.parametrize("failure", ["corrupt", "timeout", "publish"])
def test_backup_failure_blocks_consolidation(live_wal_source, monkeypatch, failure):
    root, source = live_wal_source
    prior = root / "project.db.bak.prior"
    prior.write_bytes(b"preserve previous backup")
    consolidator = database.DatabaseConsolidator(root)
    migrate = Mock()
    monkeypatch.setattr(consolidator, "_migrate_global_database", migrate)
    monkeypatch.setattr(consolidator, "_migrate_project_databases", Mock())
    if failure == "corrupt":
        (root / "conversation.db").write_bytes(b"not a sqlite database")
        expected = sqlite3.DatabaseError
    elif failure == "timeout":
        source.execute("CREATE TABLE payload (data TEXT)")
        source.executemany("INSERT INTO payload VALUES (?)", [("x" * 4000,) for _ in range(600)])
        source.commit()
        real_connect = sqlite3.connect
        progress_calls = []
        clock = [0.0]

        class ExpiringConnection(sqlite3.Connection):
            def backup(self, target, *, progress, **kwargs):
                def expire(status, remaining, total):
                    progress_calls.append((remaining, total))
                    clock[0] = 1000.0
                    progress(status, remaining, total)

                return super().backup(target, progress=expire, **kwargs)

        monkeypatch.setattr(
            database.sqlite3,
            "connect",
            lambda *args, **kwargs: real_connect(*args, **kwargs, factory=ExpiringConnection),
        )
        monkeypatch.setattr(database, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        expected = TimeoutError
    else:
        import os

        monkeypatch.setattr(os, "link", Mock(side_effect=OSError("publication failed")))
        expected = OSError
    with pytest.raises(expected):
        consolidator.consolidate()
    if failure == "timeout":
        assert len(progress_calls) == 1
        remaining, total = progress_calls[0]
        assert 0 < remaining < total  # The deadline aborted a partially copied snapshot.
    migrate.assert_not_called()
    assert not consolidator._consolidated_flag.exists()
    assert prior.read_bytes() == b"preserve previous backup"
    assert list(root.glob("project.db.bak.*")) == [prior]
    assert not list(root.glob(".*.partial"))


@pytest.mark.parametrize("failure", ["integrity", "file_sync", "directory_sync", "collision"])
def test_backup_publication_failures_preserve_evidence(
    live_wal_source, monkeypatch, failure, caplog
):
    import os
    import stat

    if failure == "directory_sync" and os.name != "posix":
        pytest.skip("Directory fsync is a POSIX publication guarantee")
    root, source = live_wal_source
    real_connect = sqlite3.connect
    real_sync, real_link = os.fsync, os.link
    prior = root / "project.db.bak.prior"
    prior.write_bytes(b"prior")

    class CheckedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if failure == "integrity" and sql == "PRAGMA quick_check":
                return SimpleNamespace(fetchall=lambda: [("corrupt index",)])
            return super().execute(sql, parameters)

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=CheckedConnection)

    def sync(fd):
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if (failure == "directory_sync" and is_directory) or (
            failure == "file_sync" and not is_directory
        ):
            raise OSError("injected sync failure")
        real_sync(fd)

    def link(src, dst):
        if failure == "collision":
            Path(dst).write_bytes(b"other backup must survive")
        real_link(src, dst)

    monkeypatch.setattr(database.sqlite3, "connect", connect)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "link", link)
    consolidator = database.DatabaseConsolidator(root)
    migrate = Mock()
    monkeypatch.setattr(consolidator, "_migrate_global_database", migrate)
    with pytest.raises((sqlite3.DatabaseError, OSError)):
        consolidator.consolidate()
    migrate.assert_not_called()
    assert not consolidator._consolidated_flag.exists()
    assert prior.read_bytes() == b"prior"
    backups = [p for p in root.glob("project.db.bak.*") if p != prior]
    if failure == "directory_sync":
        assert len(backups) == 1
        with closing(real_connect(backups[0])) as conn:
            assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1
        assert "durability is unconfirmed" in caplog.text
    elif failure == "collision":
        assert len(backups) == 1 and backups[0].read_bytes() == b"other backup must survive"
    else:
        assert not backups
    assert not list(root.glob(".*.partial"))


def test_online_backup_remains_consistent_when_writer_commits(live_wal_source, monkeypatch):
    root, writer = live_wal_source
    writer.execute("CREATE TABLE payload (data TEXT)")
    writer.executemany("INSERT INTO payload VALUES (?)", [("x" * 4000,) for _ in range(600)])
    writer.commit()
    real_connect = sqlite3.connect
    committed = False

    class InterleavedConnection(sqlite3.Connection):
        def backup(self, target, *, progress, **kwargs):
            def interleave(status, remaining, total):
                nonlocal committed
                if remaining and not committed:
                    writer.execute("INSERT INTO evidence VALUES (2, 'during backup')")
                    writer.commit()
                    committed = True
                progress(status, remaining, total)

            return super().backup(target, progress=interleave, **kwargs)

    monkeypatch.setattr(
        database.sqlite3,
        "connect",
        lambda *args, **kwargs: real_connect(*args, **kwargs, factory=InterleavedConnection),
    )
    backup = database._backup_sqlite_snapshot(root / "project.db")
    assert committed
    with closing(real_connect(backup)) as restored:
        assert restored.execute("SELECT COUNT(*) FROM payload").fetchone()[0] == 600
        assert restored.execute("SELECT id FROM evidence ORDER BY id").fetchall() == [(1,), (2,)]


def test_vanished_backup_source_is_not_recreated(tmp_path):
    before = set(tmp_path.iterdir())
    missing = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        database._backup_sqlite_snapshot(missing)
    assert not missing.exists()
    assert set(tmp_path.iterdir()) == before

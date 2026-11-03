"""POSIX ownership and bounded checkpointed SQLite input for qualification only."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
import errno
import os
from pathlib import Path
import sqlite3
import stat
import time
from types import TracebackType
import uuid

from victor.core.graph_rag.source_reader import ProjectSourceReader, project_source_reader

MAX_DATABASE_BYTES = 16 * 1024 * 1024
MAX_CACHE_ROWS = 2000
MAX_MANIFEST_BYTES = 512 * 1024


def source_parts(relative: str) -> tuple[str, ...]:
    parts = tuple(relative.split("/"))
    if (
        not relative
        or any(part in {"", ".", ".."} for part in parts)
        or Path(relative).is_absolute()
        or "\\" in relative
        or "\0" in relative
    ):
        raise ValueError("Invalid qualification source manifest path")
    if any(
        part.casefold() in {".git", ".gitattributes", ".gitignore", ".gitmodules", ".lfsconfig"}
        for part in parts
    ):
        raise ValueError("Git control files cannot be qualification sources")
    return parts


def bounded_read(path: Path, limit: int) -> bytes:
    path = path.absolute()
    with project_source_reader(path.parent) as reader:
        value = _read_from(reader, path.name, limit)
        reader.verify_root_path()
        return value


def _read_from(reader: ProjectSourceReader, name: str, limit: int) -> bytes:
    before = os.stat(name, dir_fd=reader.descriptor, follow_symlinks=False)
    value, after = reader(name, limit)

    def identity(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    if after.st_size > limit or len(value) > limit or identity(before) != identity(after):
        raise ValueError("Qualification input changed or exceeded its byte budget")
    return value


def _write_all(descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:
            raise OSError(errno.EIO, "Qualification write made no progress")
        remaining = remaining[count:]


def _unlink_owned(directory: int, name: str, created: os.stat_result | None) -> None:
    try:
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if created is not None and (current.st_dev, current.st_ino) == (
        created.st_dev,
        created.st_ino,
    ):
        os.unlink(name, dir_fd=directory)


def _verify_owned(directory: int, name: str, created: os.stat_result) -> None:
    current = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        created.st_dev,
        created.st_ino,
    ):
        raise ValueError("Owned qualification file changed or became a link")


@contextmanager
def cache_database(path: Path) -> Iterator[sqlite3.Connection]:
    """Never give SQLite the original pathname; reject live/WAL input snapshots."""
    path = path.absolute()
    with project_source_reader(path.parent) as reader:

        def check_sidecars() -> None:
            for suffix in ("-wal", "-shm", "-journal"):
                try:
                    os.stat(path.name + suffix, dir_fd=reader.descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise ValueError("Qualification requires a closed checkpointed cache")

        check_sidecars()
        raw = _read_from(reader, path.name, MAX_DATABASE_BYTES)
        check_sidecars()
        reader.verify_root_path()
    if raw[:16] != b"SQLite format 3\0" or raw[18:20] != b"\x01\x01":
        raise ValueError("Qualification requires a checkpointed DELETE-journal SQLite cache")
    db = sqlite3.connect(":memory:")
    try:
        db.deserialize(raw)
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 128 * 1024)
        db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 32 * 1024)
        deadline = time.monotonic() + 5
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        objects = db.execute(
            "SELECT type,name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*' LIMIT 3"
        ).fetchmany(3)
        if objects != [("table", "body_vectors_v1")]:
            raise ValueError("Unexpected qualification cache schema")
        if db.execute("SELECT COUNT(*) FROM body_vectors_v1").fetchone()[0] > MAX_CACHE_ROWS:
            raise ValueError("Qualification cache exceeds its row budget")
        yield db
    except sqlite3.Error as exc:
        raise ValueError("Invalid or over-budget qualification cache") from exc
    finally:
        db.close()


class OwnedWork:
    """Reserve a new tree exclusively; a final exclusive marker certifies completion."""

    def __init__(self, output: Path) -> None:
        self.output = output.absolute()
        self.path = self.output
        self.fd = self.source_fd = self.parent_fd = -1
        self.name = self.output.name
        self._ancestor_fds: list[int] = []
        self._ancestor_links: list[tuple[int, str, int]] = []
        self.published = False

    def __enter__(self) -> OwnedWork:
        if not self.output.name:
            raise ValueError("Qualification publication requires a new destination")
        if os.name != "posix" or not all(
            hasattr(os, key) for key in ("O_DIRECTORY", "O_NOFOLLOW", "fchdir")
        ):
            raise OSError(errno.ENOTSUP, "Owned qualification mutations require POSIX descriptors")
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._ancestor_fds.append(current)
        try:
            for part in self.output.parent.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                self._ancestor_links.append((current, part, child))
                self._ancestor_fds.append(child)
                current = child
            self.parent_fd = current
            try:
                os.mkdir(self.name, mode=0o700, dir_fd=current)
            except FileExistsError as exc:
                raise ValueError("Qualification publication requires a new destination") from exc
            self.fd = os.open(
                self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
            )
            self.verify_ancestors()
            os.mkdir("source", mode=0o700, dir_fd=self.fd)
            self.source_fd = os.open(
                "source", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd
            )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        for name in ("source_fd", "fd"):
            descriptor = getattr(self, name)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, name, -1)
        for descriptor in reversed(self._ancestor_fds):
            os.close(descriptor)
        self._ancestor_fds.clear()
        self._ancestor_links.clear()
        self.parent_fd = -1

    def verify_ancestors(self) -> None:
        for parent, name, descriptor in self._ancestor_links:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            pinned = os.fstat(descriptor)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
                pinned.st_dev,
                pinned.st_ino,
            ):
                raise ValueError("Owned qualification ancestor changed or became a link")

    def verify(self) -> None:
        self.verify_ancestors()
        for parent, name, descriptor in (
            (self.parent_fd, self.name, self.fd),
            (self.fd, "source", self.source_fd),
        ):
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            pinned = os.fstat(descriptor)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
                pinned.st_dev,
                pinned.st_ino,
            ):
                raise ValueError("Owned qualification directory changed or became a link")

    @contextmanager
    def _parent(
        self, relative: str, *, source: bool, create: bool = False
    ) -> Iterator[tuple[int, str]]:
        parts = source_parts(relative)
        self.verify()
        directory = self.source_fd if source else self.fd
        with ExitStack() as stack:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=directory)
                    except FileExistsError:
                        pass
                directory = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                stack.callback(os.close, directory)
            yield directory, parts[-1]

    def write(self, relative: str, value: bytes, *, source: bool = False) -> None:
        with self._parent(relative, source=source, create=True) as (directory, name):
            temporary = ".r1-write-" + uuid.uuid4().hex
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            created: os.stat_result | None = None
            try:
                created = os.fstat(descriptor)
                _write_all(descriptor, value)
                os.fsync(descriptor)
                _verify_owned(directory, temporary, created)
                os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                _verify_owned(directory, name, created)
            finally:
                try:
                    _unlink_owned(directory, temporary, created)
                finally:
                    os.close(descriptor)
        self.verify()

    def write_source(self, relative: str, value: bytes) -> None:
        self.write(relative, value, source=True)

    def read_source(self, relative: str) -> bytes:
        self.verify()
        info = os.fstat(self.source_fd)
        value, metadata = ProjectSourceReader(
            self.path / "source", self.source_fd, (info.st_dev, info.st_ino)
        )(relative, 4 * 1024 * 1024)
        if metadata.st_size > 4 * 1024 * 1024:
            raise ValueError("Source exceeds qualification byte budget")
        self.verify()
        return value

    def unlink_source(self, relative: str) -> None:
        with self._parent(relative, source=True) as (directory, name):
            os.unlink(name, dir_fd=directory)
        self.verify()

    def move_source(self, old: str, new: str) -> None:
        with (
            self._parent(old, source=True) as (left, before),
            self._parent(new, source=True, create=True) as (right, after),
        ):
            try:
                os.stat(after, dir_fd=right, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("Qualification rename destination already exists")
            os.rename(before, after, src_dir_fd=left, dst_dir_fd=right)
        self.verify()

    def seed_cache(self, seed: Path) -> None:
        with cache_database(seed) as original, closing(sqlite3.connect(":memory:")) as snapshot:
            deadline = time.monotonic() + 5

            def progress(status: int, remaining: int, total: int) -> None:
                if time.monotonic() > deadline:
                    raise ValueError("Cache backup exceeded its time budget")

            original.backup(snapshot, pages=64, progress=progress)
            raw = snapshot.serialize()
        if len(raw) > MAX_DATABASE_BYTES:
            raise ValueError("Cache backup exceeded its byte budget")
        self.write("body.sqlite", raw)

    def publish(self) -> None:
        self.verify()
        marker = "QUALIFICATION_COMPLETE"
        try:
            descriptor = os.open(
                marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd
            )
        except FileExistsError as exc:
            raise ValueError("Qualification completion marker already exists") from exc
        created: os.stat_result | None = None
        try:
            created = os.fstat(descriptor)
            _write_all(descriptor, b"victor-incremental-body-qualification-v2\n")
            os.fsync(descriptor)
            os.fsync(self.fd)
            os.fsync(self.parent_fd)
            self.verify()
            _verify_owned(self.fd, marker, created)
        except BaseException:
            _unlink_owned(self.fd, marker, created)
            raise
        finally:
            os.close(descriptor)
        self.published = True

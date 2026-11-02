"""Bounded, lazy projection of SCIP document occurrence inputs.

The whole protobuf arena and unconsumed symbol/diagnostic messages are never
materialized. This is an occurrence view, not validation of unused index fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
import errno
import io
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.graph_rag.source_reader import ProjectSourceReader

MAX_INDEX_BYTES = 512 * 1024**2
MAX_DOCUMENT_BYTES = 16 * 1024**2
MAX_OCCURRENCES = 100_000
MAX_DOCUMENTS = 100_000
MAX_WORK = 1_000_000


def _varint(stream: BinaryIO, end: int) -> int:
    value = 0
    for shift in range(0, 70, 7):
        if stream.tell() >= end:
            raise ValueError("Truncated SCIP varint")
        raw = stream.read(1)
        if not raw or (shift == 63 and raw[0] > 1):
            raise ValueError("Invalid SCIP varint")
        value |= (raw[0] & 127) << shift
        if raw[0] < 128:
            return value
    raise ValueError("Oversized SCIP varint")


def _next_field(stream: BinaryIO, end: int) -> tuple[int, int, int, int]:
    tag = _varint(stream, end)
    number, wire = tag >> 3, tag & 7
    if not 1 <= number < 2**29:
        raise ValueError("Invalid SCIP field number")
    start = stream.tell()
    if wire == 0:
        _varint(stream, end)
        stop = stream.tell()
    elif wire in {1, 5}:
        stop = start + (8 if wire == 1 else 4)
    elif wire == 2:
        size = _varint(stream, end)
        start, stop = stream.tell(), stream.tell() + size
    else:
        raise ValueError("Unsupported SCIP wire type")
    if stop > end:
        raise ValueError("SCIP field exceeds its message")
    return number, wire, start, stop


def _encoded(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _path(value: bytes) -> str:
    if not 1 <= len(value) <= 16_384:
        raise ValueError("Invalid SCIP document path size")
    name = value.decode("utf-8")
    if (
        name.startswith("/")
        or "\\" in name
        or "\0" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("SCIP document paths must be canonical relative POSIX paths")
    return name


def _project_occurrence(raw: bytes, step: Callable[[], None]) -> bytes:
    stream = io.BytesIO(raw)
    projected = bytearray()
    coordinates = 0
    while stream.tell() < len(raw):
        step()
        field_start = stream.tell()
        number, wire, start, stop = _next_field(stream, len(raw))
        if number == 1:
            if wire not in {0, 2} or stop - start > 40:
                raise ValueError("SCIP occurrence range exceeds its coordinate budget")
            if wire == 0:
                coordinates += 1
            else:
                stream.seek(start)
                while stream.tell() < stop:
                    _varint(stream, stop)
                    coordinates += 1
                    if coordinates > 4:
                        raise ValueError("SCIP occurrence range exceeds four coordinates")
            if coordinates > 4:
                raise ValueError("SCIP occurrence range exceeds four coordinates")
            projected.extend(raw[field_start:stop])
        elif number == 2:
            if wire != 2 or stop - start > 64 * 1024:
                raise ValueError("SCIP symbol exceeds its string budget")
            projected.extend(raw[field_start:stop])
        stream.seek(stop)
    return bytes(projected)


class ScipDocumentReader(Mapping[str, Any]):
    """Own a bounded artifact and decode only requested occurrence projections.

    Default project artifacts must borrow the scan's pinned source root. An
    explicitly supplied external artifact gets its own pinned parent. Windows
    uses the existing secure immutable snapshot seam; POSIX seeks regular fds.
    """

    def __init__(
        self,
        path: Path,
        *,
        source_root: ProjectSourceReader | None = None,
        max_bytes: int = MAX_INDEX_BYTES,
        max_document_bytes: int = MAX_DOCUMENT_BYTES,
        max_occurrences: int = MAX_OCCURRENCES,
    ) -> None:
        self._closed = True
        from victor.core.graph_rag.source_reader import project_source_reader

        for value, cap in (
            (max_bytes, MAX_INDEX_BYTES),
            (max_document_bytes, MAX_DOCUMENT_BYTES),
            (max_occurrences, MAX_OCCURRENCES),
        ):
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError("Invalid SCIP input budget")
        self._stack = ExitStack()
        self._closed = False
        self._snapshot = os.name == "nt"
        self._max_document_bytes, self._max_occurrences = max_document_bytes, max_occurrences
        self._offsets: dict[str, tuple[int, int]] = {}
        self.decoded_documents = 0
        self.skipped_headers = 0
        self._work = 0
        try:
            path = path.absolute()
            self._root = source_root or self._stack.enter_context(
                project_source_reader(path.parent)
            )
            relative = path.relative_to(self._root.root)
            if self._snapshot:
                data, self.stat = self._root(relative.as_posix(), max_bytes)
                self._stream: BinaryIO = self._stack.enter_context(io.BytesIO(data))
            else:
                directory = os.dup(self._root.descriptor)
                try:
                    for component in relative.parts[:-1]:
                        if component in {"", ".", ".."}:
                            raise ValueError("Invalid SCIP artifact path")
                        child = os.open(
                            component,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory,
                        )
                        os.close(directory)
                        directory = child
                    descriptor = os.open(
                        relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
                    )
                finally:
                    os.close(directory)
                self._stack.callback(os.close, descriptor)
                self.stat = os.fstat(descriptor)
                if not stat.S_ISREG(self.stat.st_mode) or not 0 <= self.stat.st_size <= max_bytes:
                    raise ValueError("SCIP artifact must be a bounded regular file")
                self._stream = self._stack.enter_context(os.fdopen(descriptor, "rb", closefd=False))
            if not stat.S_ISREG(self.stat.st_mode) or not 0 <= self.stat.st_size <= max_bytes:
                raise ValueError("SCIP artifact must be a bounded regular file")
            self._identity = self._stat_key(self.stat)
            self._index()
            self._verify()
        except BaseException:
            self._closed = True
            self._stack.close()
            raise

    @staticmethod
    def _stat_key(info: os.stat_result) -> tuple[int, ...]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _verify(self) -> None:
        if self._closed:
            raise ValueError("SCIP reader is closed")
        self._root.verify_root_path()
        if not self._snapshot and self._stat_key(os.fstat(self._stream.fileno())) != self._identity:
            raise OSError(errno.ESTALE, "SCIP artifact changed during analysis")

    def _step(self) -> None:
        self._work += 1
        if self._work > MAX_WORK:
            raise ValueError("SCIP index work budget exceeded")

    def _index(self) -> None:
        total_names = 0
        while self._stream.tell() < self.stat.st_size:
            self._step()
            number, wire, start, stop = _next_field(self._stream, self.stat.st_size)
            if number == 2:
                if wire != 2:
                    raise ValueError("Invalid SCIP document wire encoding")
                name = self._prefix_path(start, stop)
                if name is not None:
                    if name in self._offsets:
                        raise ValueError("Duplicate SCIP document path")
                    total_names += len(name.encode())
                    if len(self._offsets) >= MAX_DOCUMENTS or total_names > 16 * 1024**2:
                        raise ValueError("SCIP document inventory budget exceeded")
                    self._offsets[name] = (start, stop - start)
                else:
                    self.skipped_headers += 1
            self._stream.seek(stop)

    def _prefix_path(self, start: int, stop: int) -> str | None:
        self._stream.seek(start)
        while self._stream.tell() < stop:
            self._step()
            number, wire, value_start, value_stop = _next_field(self._stream, stop)
            if number == 1:
                if wire != 2 or not 1 <= value_stop - value_start <= 16_384:
                    raise ValueError("Invalid SCIP path wire encoding")
                self._stream.seek(value_start)
                return _path(self._stream.read(value_stop - value_start))
            self._stream.seek(value_stop)
        return None

    def __iter__(self) -> Iterator[str]:
        return iter(self._offsets)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, name: str) -> Any:
        from victor.core.graph_rag._scip import scip_pb2

        self._verify()
        offset, size = self._offsets[name]
        if size > self._max_document_bytes:
            raise ValueError("SCIP document exceeds its byte budget")
        self._stream.seek(offset)
        raw = self._stream.read(size)
        if len(raw) != size:
            raise OSError(errno.ESTALE, "SCIP artifact changed during document read")
        stream, projected, occurrences = io.BytesIO(raw), bytearray(), 0
        work = 0

        def step() -> None:
            nonlocal work
            work += 1
            if work > MAX_WORK:
                raise ValueError("SCIP document projection work budget exceeded")

        while stream.tell() < len(raw):
            step()
            field_start = stream.tell()
            number, wire, start, stop = _next_field(stream, len(raw))
            if number in {1, 4, 5, 6}:
                if number in {1, 4, 5} and wire != 2 or number == 6 and wire != 0:
                    raise ValueError("Invalid SCIP document scalar wire encoding")
                projected.extend(raw[field_start:stop])
            elif number == 2:
                occurrences += 1
                if wire != 2 or occurrences > self._max_occurrences:
                    raise ValueError("SCIP occurrence budget exceeded")
                occurrence = _project_occurrence(raw[start:stop], step)
                projected.extend(b"\x12" + _encoded(len(occurrence)) + occurrence)
            stream.seek(stop)
        try:
            bindings: Any = scip_pb2  # Generated descriptors expose message classes dynamically.
            document = bindings.Document.FromString(bytes(projected))
        except Exception as exc:
            raise ValueError("Invalid SCIP occurrence projection") from exc
        if document.relative_path != name:
            raise ValueError("SCIP document path changed after prefix selection")
        self._verify()
        self.decoded_documents += 1
        return document

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stack.close()

    def __enter__(self) -> ScipDocumentReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

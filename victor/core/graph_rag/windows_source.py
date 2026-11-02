"""Read-only Windows source handles; no path-based traversal fallback.

Only local drive paths and disk files are accepted. NtOpenFile binds each single
component to its already-owned parent. All reparse tags are rejected, including
junctions and cloud placeholders. Handles deny write/delete sharing while owned.
The ctypes ABI uses fixed-width Windows scalars, including on non-Windows hosts
which import the namespace validator. API loading happens only on Windows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
import ctypes as ct
import errno
import os
from pathlib import Path, PureWindowsPath
import stat
import struct
from typing import Any

_DWORD = ct.c_uint32
_HANDLE = ct.c_void_p
_REPARSE = 0x400
_DIRECTORY = 0x10
_INVALID = ct.c_void_p(-1).value
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "123456789¹²³"
}


def relative_parts(relative: str) -> tuple[str, ...]:
    """Reject alternate namespaces, streams, device aliases and normalization."""
    parts = tuple(relative.split("/"))
    if not relative or any(
        not part
        or part in {".", ".."}
        or part[-1:] in {".", " "}
        or part.split(".", 1)[0].upper() in _RESERVED
        or any(ord(char) < 32 or char in '\\:<>"|?*' for char in part)
        for part in parts
    ):
        raise OSError(errno.EINVAL, "Invalid Windows project-relative source path")
    return parts


class _UnicodeString(ct.Structure):
    _fields_ = [("length", ct.c_uint16), ("maximum", ct.c_uint16), ("buffer", _HANDLE)]


class _ObjectAttributes(ct.Structure):
    _fields_ = [
        ("length", _DWORD),
        ("root", _HANDLE),
        ("name", ct.POINTER(_UnicodeString)),
        ("attributes", _DWORD),
        ("security", _HANDLE),
        ("quality", _HANDLE),
    ]


class _IoStatus(ct.Structure):
    _fields_ = [("status", _HANDLE), ("information", ct.c_size_t)]


class _RootHandle(int):
    """An integer OS handle retaining its context-owned namespace ancestors."""

    ancestors: tuple[int, ...]

    def __new__(cls, value: int, ancestors: tuple[int, ...]) -> _RootHandle:
        instance = super().__new__(cls, value)
        instance.ancestors = ancestors
        return instance


class _FileInformation(ct.Structure):
    # FILETIME is two DWORDs: it has four-byte rather than uint64 alignment.
    _pack_ = 4
    _fields_ = [
        ("attributes", _DWORD),
        ("creation", ct.c_uint64),
        ("access", ct.c_uint64),
        ("write", ct.c_uint64),
        ("volume", _DWORD),
        ("size_high", _DWORD),
        ("size_low", _DWORD),
        ("links", _DWORD),
        ("index_high", _DWORD),
        ("index_low", _DWORD),
    ]


class _Api:
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError(errno.ENOTSUP, "Windows source handles require Windows")
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        native = ct.WinDLL("ntdll", use_last_error=True)

        def bind(dll: Any, name: str, result: Any, arguments: list[Any]) -> Any:
            function = getattr(dll, name)
            function.restype, function.argtypes = result, arguments
            return function

        self.close = bind(kernel, "CloseHandle", ct.c_int, [_HANDLE])
        self.create = bind(
            kernel,
            "CreateFileW",
            _HANDLE,
            [ct.c_wchar_p, _DWORD, _DWORD, _HANDLE, _DWORD, _DWORD, _HANDLE],
        )
        self.open = bind(
            native,
            "NtOpenFile",
            ct.c_int32,
            [
                ct.POINTER(_HANDLE),
                _DWORD,
                ct.POINTER(_ObjectAttributes),
                ct.POINTER(_IoStatus),
                _DWORD,
                _DWORD,
            ],
        )
        self.error = bind(native, "RtlNtStatusToDosError", _DWORD, [ct.c_int32])
        self.info = bind(
            kernel, "GetFileInformationByHandle", ct.c_int, [_HANDLE, ct.POINTER(_FileInformation)]
        )
        self.kind = bind(kernel, "GetFileType", _DWORD, [_HANDLE])
        self.drive = bind(kernel, "GetDriveTypeW", _DWORD, [ct.c_wchar_p])
        self.enumerate = bind(
            kernel, "GetFileInformationByHandleEx", ct.c_int, [_HANDLE, ct.c_int, _HANDLE, _DWORD]
        )
        self.read = bind(
            kernel, "ReadFile", ct.c_int, [_HANDLE, _HANDLE, _DWORD, ct.POINTER(_DWORD), _HANDLE]
        )
        self.path = bind(
            kernel, "GetFinalPathNameByHandleW", _DWORD, [_HANDLE, ct.c_wchar_p, _DWORD, _DWORD]
        )
        self.volume = bind(
            kernel,
            "GetVolumeInformationByHandleW",
            ct.c_int,
            [
                _HANDLE,
                ct.c_wchar_p,
                _DWORD,
                ct.POINTER(_DWORD),
                ct.POINTER(_DWORD),
                ct.POINTER(_DWORD),
                ct.c_wchar_p,
                _DWORD,
            ],
        )


def _failure() -> OSError:
    return ct.WinError(ct.get_last_error())


def _information(api: _Api, handle: int, directory: bool) -> os.stat_result:
    raw = _FileInformation()
    if not api.info(handle, ct.byref(raw)):
        raise _failure()
    if (
        api.kind(handle) != 1
        or raw.attributes & (0x40 | _REPARSE)
        or bool(raw.attributes & _DIRECTORY) != directory
    ):
        raise OSError(errno.EINVAL, "Windows source is a reparse point or unsupported file type")
    mode = stat.S_IFDIR if directory else stat.S_IFREG
    times = [(value - 116444736000000000) * 100 for value in (raw.access, raw.write, raw.creation)]
    return os.stat_result(
        (
            mode | 0o400,
            (raw.index_high << 32) | raw.index_low,
            raw.volume,
            raw.links,
            0,
            0,
            (raw.size_high << 32) | raw.size_low,
            *(value / 1e9 for value in times),
        ),
        dict(zip(("st_atime_ns", "st_mtime_ns", "st_ctime_ns"), times)),
    )


def _open(api: _Api, parent: int, name: str, directory: bool) -> int:
    if len(relative_parts(name)) != 1:
        raise OSError(errno.EINVAL, "Expected one Windows source path component")
    encoded = name.encode("utf-16-le", errors="strict")
    if len(encoded) > 65532:
        raise OSError(errno.ENAMETOOLONG, "Windows source name exceeds UnicodeString bounds")
    buffer = ct.create_unicode_buffer(name)
    string = _UnicodeString(len(encoded), len(encoded) + 2, ct.cast(buffer, _HANDLE))
    attributes = _ObjectAttributes(
        ct.sizeof(_ObjectAttributes), parent, ct.pointer(string), 0x40, None, None
    )
    handle, status = _HANDLE(), _IoStatus()
    result = api.open(
        ct.byref(handle),
        0x100081,
        ct.byref(attributes),
        ct.byref(status),
        1,
        0x200020 | (1 if directory else 0x40),
    )
    if result < 0:
        raise ct.WinError(api.error(result))
    selected = int(handle.value or 0)
    try:
        _information(api, selected, directory)
    except BaseException:
        api.close(selected)
        raise
    return selected


@contextmanager
def root_handle(root: Path) -> Iterator[tuple[int, os.stat_result]]:
    """Pin every local-drive ancestor; reject UNC, device paths and reparses."""
    api = _Api()
    path = PureWindowsPath(str(root))
    if (
        len(path.drive) != 2
        or path.drive[1] != ":"
        or not path.root
        or str(root).startswith(("\\\\", "//"))
    ):
        raise OSError(errno.ENOTSUP, "Windows source requires an absolute local drive path")
    parts = path.parts[1:]
    # PureWindowsPath normalizes dot and duplicate separators; validate raw first.
    relative_parts(str(root)[3:].replace("\\", "/")) if parts else None
    if api.drive(path.anchor) != 3:
        raise OSError(errno.ENOTSUP, "Windows source requires a local fixed disk")
    with ExitStack() as stack:
        handle = api.create(path.anchor, 0x100081, 1, None, 3, 0x02200000, None)
        if handle == _INVALID:
            raise _failure()
        stack.callback(api.close, handle)
        ancestors = [handle]
        _information(api, handle, True)
        if handle_path(api, handle).casefold() != path.anchor.casefold():
            raise OSError(errno.ENOTSUP, "Windows source rejects aliased drive roots")
        filesystem = ct.create_unicode_buffer(32)
        if not api.volume(handle, None, 0, None, None, None, filesystem, len(filesystem)):
            raise _failure()
        if filesystem.value != "NTFS":
            raise OSError(errno.ENOTSUP, "Windows source requires a local NTFS volume")
        for part in parts:
            handle = _open(api, handle, part, True)
            stack.callback(api.close, handle)
            ancestors.append(handle)
        yield _RootHandle(handle, tuple(ancestors)), _information(api, handle, True)


def read_source(root: int, relative: str, limit: int) -> tuple[bytes, os.stat_result]:
    """Acquire only the bounded bytes of one regular handle, without replay."""
    parts = relative_parts(relative)
    if limit < 1:
        raise OSError(errno.EINVAL, "Invalid Windows source byte budget")
    api = _Api()
    with ExitStack() as stack:
        directory = root
        for part in parts[:-1]:
            directory = _open(api, directory, part, True)
            stack.callback(api.close, directory)
        handle = _open(api, directory, parts[-1], False)
        stack.callback(api.close, handle)
        info = _information(api, handle, False)
        if info.st_size > limit:
            return b"", info
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            buffer = ct.create_string_buffer(min(remaining, 64 * 1024))
            count = _DWORD()
            if not api.read(handle, buffer, len(buffer), ct.byref(count), None):
                raise _failure()
            if not count.value:
                break
            chunks.append(buffer.raw[: count.value])
            remaining -= count.value
        return b"".join(chunks), _information(api, handle, False)


def entries(api: _Api, handle: int) -> Iterator[tuple[str, int]]:
    """Enumerate FILE_FULL_DIR_INFO batches on the owned directory handle."""
    buffer = ct.create_string_buffer(64 * 1024)
    information_class = 15  # FileFullDirectoryRestartInfo, then resumable class 14.
    while True:
        if not api.enumerate(handle, information_class, buffer, len(buffer)):
            if ct.get_last_error() == 18:  # ERROR_NO_MORE_FILES
                return
            raise _failure()
        information_class = 14
        offset = 0
        while True:
            if offset + 68 > len(buffer):
                raise ValueError("Malformed Windows source directory record")
            following, _, *fields = struct.unpack_from("<II6qIII", buffer, offset)
            attributes, size = fields[6:8]
            end = offset + 68 + size
            if size % 2 or end > len(buffer) or (following and following < 68 + size):
                raise ValueError("Malformed Windows source directory filename")
            name = buffer.raw[offset + 68 : end].decode("utf-16-le", errors="strict")
            if name not in {".", ".."}:
                relative_parts(name)
                yield name, attributes
            if not following:
                break
            offset += following


def handle_path(api: _Api, handle: int) -> str:
    """Recover a process cwd only while all namespace ancestors stay pinned."""
    buffer = ct.create_unicode_buffer(32768)
    length = api.path(handle, buffer, len(buffer), 0)
    if not length:
        raise _failure()
    if length >= len(buffer) or not buffer.value.startswith("\\\\?\\"):
        raise OSError(errno.ENOTSUP, "Unsupported Windows source handle namespace")
    path = buffer.value[4:]
    if len(path) < 3 or path[1:3] != ":\\":
        raise OSError(errno.ENOTSUP, "Windows source requires a local drive handle")
    return path


@contextmanager
def git_context(
    api: _Api, root: int, check: Callable[[], None]
) -> Iterator[tuple[str, str, str] | None]:
    """Pin ordinary Git metadata and verify nearest-repository boundaries.

    Explicit Git directories and GIT_COMMON_DIR bypass mutable discovery and
    commondir selectors. Linked-worktree .git files remain unsupported.
    """
    if not isinstance(root, _RootHandle):
        raise OSError(errno.EINVAL, "Windows inventory requires a context-owned root handle")
    ancestry: list[tuple[int, int | None]] = []
    with ExitStack() as stack:
        context = None
        if len(root.ancestors) > 1024:
            raise ValueError("Windows source inventory exceeded its ancestor budget")
        for current in reversed(root.ancestors):
            check()
            marker = None
            for name, attributes in entries(api, current):
                check()
                if name.casefold() == ".git":
                    if not attributes & _DIRECTORY:
                        raise OSError(
                            errno.ENOTSUP, "Windows linked-worktree selectors are unavailable"
                        )
                    marker = _open(api, current, name, True)
                    stack.callback(api.close, marker)
                    for selector, _selector_attributes in entries(api, marker):
                        check()
                        if selector.casefold() == "commondir":
                            raise OSError(
                                errno.ENOTSUP, "Windows common-repository selectors are unavailable"
                            )
                    context = (
                        handle_path(api, root),
                        handle_path(api, current),
                        handle_path(api, marker),
                    )
                    break
            ancestry.append((current, marker))
            if marker is not None:
                break

        def verify() -> None:
            for directory, expected in ancestry:
                check()
                markers = []
                for name, _attributes in entries(api, directory):
                    check()
                    if name.casefold() == ".git":
                        markers.append(name)
                if bool(markers) != (expected is not None) or len(markers) > 1:
                    raise OSError(errno.ESTALE, "Windows source repository context changed")
                if expected is not None:
                    selected = _open(api, directory, markers[0], True)
                    try:
                        left, right = _information(api, expected, True), _information(
                            api, selected, True
                        )
                        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                            raise OSError(
                                errno.ESTALE, "Windows source repository identity changed"
                            )
                    finally:
                        api.close(selected)

        verify()
        yield context
        verify()

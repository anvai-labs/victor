# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only ``.npz`` loader with no numpy dependency (FEP-0012 runtime path).

The shipped edge-classifier artifact is an ``np.savez`` archive. Runtime
inference must work in a bare ``pip install victor-ai`` (numpy-optional), so
this module parses the few dtypes the artifact actually uses straight from the
zip container:

- ``.npy`` v1.0 headers (6-byte magic + 2-byte version + uint16 header length;
  ``np.savez`` never emits v2.0 headers for headers < 64 KB)
- little-endian ``<i1/i2/i4/i8`` and ``<f4/f8`` payload via ``struct``
- ``<U<n`` fixed-width UTF-32-LE strings with null padding

The dev-only trainer (:mod:`victor.ml.trainer`) keeps using numpy to WRITE
artifacts; this module is deliberately read-only, so the artifact format is
owned by numpy and only consumed here.

C-order is assumed (``np.savez`` default); Fortran-order files are rejected
loudly rather than mis-read. Object arrays are rejected (they would require
pickle — the reader is untrusted-input safe).
"""

from __future__ import annotations

import ast
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_NPY_MAGIC = b"\x93NUMPY"
_SUPPORTED_DTYPES = {"<i1", "<i2", "<i4", "<i8", "<f4", "<f8"}


@dataclass(frozen=True)
class NpyArray:
    """A parsed ``.npy`` array: shape, dtype descr, and raw little-endian data."""

    shape: Tuple[int, ...]
    descr: str
    fortran_order: bool
    data: bytes

    def size(self) -> int:
        out = 1
        for dim in self.shape:
            out *= dim
        return out


def _parse_npy_header(raw: bytes, source: str) -> Tuple[Tuple[int, ...], str, bool, bytes]:
    if raw[:6] != _NPY_MAGIC:
        raise ValueError(f"{source}: not a numpy .npy stream (bad magic)")
    major, minor = raw[6], raw[7]
    if (major, minor) != (1, 0):
        raise ValueError(
            f"{source}: unsupported .npy version {major}.{minor} "
            "(reader accepts v1.0 only — re-save with numpy<2 defaults)"
        )
    (header_len,) = struct.unpack("<H", raw[8:10])
    header_text = raw[10 : 10 + header_len].decode("utf-8")
    try:
        header = ast.literal_eval(header_text.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{source}: malformed .npy header: {exc}") from exc
    descr = header["descr"]
    is_unicode = descr.startswith("<U") and descr[2:].isdigit()
    if descr not in _SUPPORTED_DTYPES and not is_unicode:
        if descr.startswith("|O"):
            raise ValueError(f"{source}: object arrays are not supported (pickle)")
        raise ValueError(f"{source}: unsupported dtype {descr!r}")
    if header["fortran_order"]:
        raise ValueError(f"{source}: Fortran-order arrays are not supported")
    shape = header["shape"]
    if isinstance(shape, int):  # 0-d arrays serialize as a bare int
        shape = (shape,)
    payload = raw[10 + header_len :]
    expected = 1
    for dim in shape:
        expected *= dim
    expected *= _dtype_size(descr)
    if len(payload) < expected:
        raise ValueError(f"{source}: truncated payload ({len(payload)} < {expected} bytes)")
    return tuple(shape), descr, False, payload[:expected]


def _dtype_size(descr: str) -> int:
    if descr.startswith("<U"):
        return 4 * int(descr[2:])
    return int(descr[2:])


def read_npz(path: "str | Path") -> Dict[str, NpyArray]:
    """Read an ``np.savez`` archive into raw arrays, numpy-free.

    Raises:
        ValueError: On non-npz input, unsupported versions/dtypes, Fortran
            order, or truncated payloads.
    """
    out: Dict[str, NpyArray] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            name = info.filename[: -len(".npy")]
            out[name] = NpyArray(*_parse_npy_header(archive.read(info), info.filename))
    if not out:
        raise ValueError(f"{path}: no .npy members found in archive")
    return out


# ---------------------------------------------------------------- converters


def as_float_vector(a: NpyArray) -> List[float]:
    """1-D ``<f4``/``<f8`` array → ``List[float]``."""
    if a.descr not in ("<f4", "<f8"):
        raise ValueError(f"expected float array, got {a.descr!r}")
    if len(a.shape) != 1:
        raise ValueError(f"expected 1-D array, got shape {a.shape}")
    fmt = "<%d%s" % (a.size(), "f" if a.descr == "<f4" else "d")
    return list(struct.unpack(fmt, a.data))


def as_float_rows(a: NpyArray) -> List[List[float]]:
    """2-D ``<f4``/``<f8`` array → ``List[List[float]]`` (C-order rows)."""
    if a.descr not in ("<f4", "<f8"):
        raise ValueError(f"expected float array, got {a.descr!r}")
    if len(a.shape) != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    rows, cols = a.shape
    fmt = "<%d%s" % (cols, "f" if a.descr == "<f4" else "d")
    step = _dtype_size(a.descr) * cols
    return [list(struct.unpack(fmt, a.data[r * step : (r + 1) * step])) for r in range(rows)]


def as_int_vector(a: NpyArray) -> List[int]:
    """1-D signed-int array → ``List[int]``."""
    if not a.descr.startswith("<i"):
        raise ValueError(f"expected int array, got {a.descr!r}")
    if len(a.shape) != 1:
        raise ValueError(f"expected 1-D array, got shape {a.shape}")
    code = {"1": "b", "2": "h", "4": "i", "8": "q"}[a.descr[2]]
    return list(struct.unpack("<%d%s" % (a.size(), code), a.data))


def _decode_u(a: NpyArray) -> List[str]:
    width = int(a.descr[2:])
    total = a.size()
    codepoints = struct.unpack("<%dI" % (total * width), a.data)
    strings: List[str] = []
    for i in range(total):
        chunk = codepoints[i * width : (i + 1) * width]
        strings.append("".join(chr(cp) for cp in chunk).rstrip("\x00"))
    return strings


def as_str_list(a: NpyArray) -> List[str]:
    """1-D ``<U<n`` array → ``List[str]``."""
    if not a.descr.startswith("<U"):
        raise ValueError(f"expected unicode array, got {a.descr!r}")
    if len(a.shape) != 1:
        raise ValueError(f"expected 1-D array, got shape {a.shape}")
    return _decode_u(a)


def as_scalar_str(a: NpyArray) -> str:
    """0-d/1-element ``<U<n`` array → ``str``."""
    values = _decode_u(a) if a.descr.startswith("<U") else None
    if values is None or len(values) != 1:
        raise ValueError(f"expected single-element unicode array, got shape {a.shape}")
    return values[0]


def as_scalar_float(a: NpyArray) -> float:
    """0-d/1-element ``<f4``/``<f8`` array → ``float``."""
    if a.descr not in ("<f4", "<f8"):
        raise ValueError(f"expected float array, got {a.descr!r}")
    if a.shape == ():
        (value,) = struct.unpack("<%s" % ("f" if a.descr == "<f4" else "d"), a.data)
        return value
    if a.descr not in ("<f4", "<f8") or a.size() != 1 or len(a.shape) > 1:
        raise ValueError(f"expected single-element float array, got shape {a.shape}")
    (value,) = struct.unpack("<%s" % ("f" if a.descr == "<f4" else "d"), a.data)
    return value

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

"""The numpy-free .npz reader: format handling and shipped-artifact drift.

The reader must parse exactly what ``np.savez`` writes for the FEP-0012
artifact's dtypes and reject everything else loudly. The artifact-drift test
pins the shipped ``edge_classifier_v1.npz`` so a silent format change breaks
HERE, not in production inference.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from victor.ml.npz_reader import (
    NpyArray,
    as_float_rows,
    as_float_vector,
    as_int_vector,
    as_scalar_float,
    as_scalar_str,
    as_str_list,
    read_npz,
)

ARTIFACT = Path(__file__).resolve().parents[3] / "victor" / "models" / "edge_classifier_v1.npz"


def _write_npy(fh, descr: str, shape: tuple, flat: list) -> None:
    """Write a minimal .npy v1.0 stream (the dtypes the reader supports)."""
    header = repr({"descr": descr, "fortran_order": False, "shape": shape})
    header += " " * max(0, 64 - len(header) - 10) + "\n"
    if descr.startswith("<U"):
        width = int(descr[2:])
        payload = b"".join(
            struct.pack(f"<{width}I", *[ord(c) for c in s.ljust(width, "\x00")]) for s in flat
        )
    else:
        code = {"<i1": "b", "<i2": "h", "<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]
        payload = struct.pack(f"<{len(flat)}{code}", *flat)
    fh.write(
        b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header)) + header.encode() + payload
    )


def _write_npz(path: Path, arrays: dict) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, (descr, shape, flat) in arrays.items():
            with zf.open(f"{name}.npy", "w", force_zip64=True) as fh:
                _write_npy(fh, descr, shape, flat)


def test_reader_roundtrip_selfwritten_npz(tmp_path: Path):
    arrays = {
        "spec": ("<U1", (), ["1"]),
        "alpha": ("<f8", (), [0.3]),
        "hashes": ("<i8", (3,), [10, -20, 30]),
        "coefs": ("<f8", (2, 3), [0.5, -1.0, 2.0, 0.25, -0.5, 1.75]),
        "bias": ("<f8", (3,), [0.1, 0.2, 0.3]),
        "labels": ("<U6", (2,), ["pass", "fail"]),
    }
    path = tmp_path / "model.npz"
    _write_npz(path, arrays)

    data = read_npz(path)
    assert set(data) == set(arrays)
    assert as_scalar_str(data["spec"]) == "1"
    assert as_scalar_float(data["alpha"]) == 0.3
    assert as_int_vector(data["hashes"]) == [10, -20, 30]
    assert as_float_rows(data["coefs"]) == [[0.5, -1.0, 2.0], [0.25, -0.5, 1.75]]
    assert as_float_vector(data["bias"]) == [0.1, 0.2, 0.3]
    assert as_str_list(data["labels"]) == ["pass", "fail"]


def test_reader_rejects_bad_version(tmp_path: Path):
    path = tmp_path / "v2.npz"
    with zipfile.ZipFile(path, "w") as zf:
        with zf.open("x.npy", "w") as fh:
            header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1, )}"
            fh.write(b"\x93NUMPY" + bytes([2, 0]) + struct.pack("<I", len(header)) + header)
    with pytest.raises(ValueError, match="unsupported .npy version"):
        read_npz(path)


def test_reader_rejects_object_and_fortran(tmp_path: Path):
    path = tmp_path / "obj.npz"
    with zipfile.ZipFile(path, "w") as zf:
        with zf.open("x.npy", "w") as fh:
            header = "{'descr': '|O', 'fortran_order': False, 'shape': (1, )}"
            header = header + " " * max(0, 64 - len(header) - 10) + "\n"
            fh.write(
                b"\x93NUMPY"
                + bytes([1, 0])
                + struct.pack("<H", len(header))
                + header.encode()
                + b"\x00"
            )
    with pytest.raises(ValueError, match="object arrays"):
        read_npz(path)

    path2 = tmp_path / "fortran.npz"
    with zipfile.ZipFile(path2, "w") as zf:
        with zf.open("x.npy", "w") as fh:
            header = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 2)}"
            header = header + " " * max(0, 64 - len(header) - 10) + "\n"
            fh.write(
                b"\x93NUMPY"
                + bytes([1, 0])
                + struct.pack("<H", len(header))
                + header.encode()
                + b"\x00" * 32
            )
    with pytest.raises(ValueError, match="Fortran"):
        read_npz(path2)


def test_reader_rejects_empty_archive(tmp_path: Path):
    path = tmp_path / "empty.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "not arrays")
    with pytest.raises(ValueError, match="no .npy members"):
        read_npz(path)


def test_shipped_artifact_loads_and_predicts():
    """Artifact-drift guard: the shipped npz must stay parseable by the
    numpy-free reader, with the exact head structure inference depends on."""
    data = read_npz(ARTIFACT)
    assert len(data) == 14
    assert as_scalar_str(data["feature_spec_version"]) == "1"
    assert as_scalar_str(data["model_version"]) == "outcomes-1"
    assert as_scalar_float(data["alpha"]) == pytest.approx(0.3)

    names = as_str_list(data["head_names"])
    assert sorted(names) == ["stage_detection", "task_completion"]
    expected_labels = {
        "stage_detection": ["fail", "pass"],
        "task_completion": ["fail", "partial", "pass"],
    }
    for name in names:
        head_labels = as_str_list(data[f"{name}__labels"])
        assert head_labels == expected_labels[name]
        hashes = as_int_vector(data[f"{name}__hashes"])
        coefs = as_float_rows(data[f"{name}__coefs"])
        bias = as_float_vector(data[f"{name}__bias"])
        assert len(hashes) == len(coefs) and len(bias) == len(head_labels)
        for row in coefs:
            assert len(row) == len(head_labels)
        assert 0.0 < as_scalar_float(data[f"{name}__threshold"]) < 1.0


def test_reader_matches_np_load_on_shipped_artifact():
    """Where numpy exists, the reader must agree with np.load byte-for-byte."""
    np = pytest.importorskip("numpy")
    data = read_npz(ARTIFACT)
    reference = np.load(ARTIFACT, allow_pickle=False)
    for name in reference.files:
        arr = reference[name]
        got = data[name]
        assert list(got.shape) == list(arr.shape)
        assert got.descr == arr.dtype.str
        if arr.dtype.kind == "U":
            if arr.ndim == 0:
                assert as_scalar_str(got) == str(arr)
            else:
                assert as_str_list(got) == arr.tolist()
        elif arr.dtype.kind == "f":
            if arr.ndim == 0:
                assert as_scalar_float(got) == float(arr)
            elif arr.ndim == 1:
                assert as_float_vector(got) == arr.tolist()
            else:
                assert as_float_rows(got) == arr.tolist()
        elif arr.dtype.kind == "i":
            assert as_int_vector(got) == arr.tolist()

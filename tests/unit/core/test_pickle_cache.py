# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unit tests for the shared pickle load-validate-save helper."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import pytest

from victor.core.pickle_cache import (
    invalid,
    load_validated_pickle,
    save_pickle_with_metadata,
    valid,
)

LOGGER = logging.getLogger("test_pickle_cache")


def _version_validator(expected: str):
    def _check(cache_data: Dict[str, Any]):
        if cache_data.get("version") != expected:
            return invalid(delete=True, reason="version mismatch")
        return valid()

    return _check


def _hash_validator(expected: str):
    def _check(cache_data: Dict[str, Any]):
        if cache_data.get("hash") != expected:
            # Hash mismatch keeps the file (no delete), mirroring the call sites.
            return invalid(delete=False)
        return valid()

    return _check


def _write_pickle(path: Path, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_valid_load_returns_dict(tmp_path: Path) -> None:
    """A cache passing every validator is returned intact."""
    cache_file = tmp_path / "cache.pkl"
    payload = {"version": "1", "hash": "abc", "data": [1, 2, 3]}
    _write_pickle(cache_file, payload)

    result = load_validated_pickle(
        cache_file,
        validators=[_version_validator("1"), _hash_validator("abc")],
        logger=LOGGER,
        label="Test",
    )

    assert result == payload
    assert cache_file.exists()


def test_missing_file_returns_none(tmp_path: Path) -> None:
    """A nonexistent cache file yields None without error."""
    cache_file = tmp_path / "absent.pkl"

    result = load_validated_pickle(
        cache_file,
        validators=[_version_validator("1")],
        logger=LOGGER,
        label="Test",
    )

    assert result is None


def test_version_mismatch_returns_none_and_deletes(tmp_path: Path) -> None:
    """A version mismatch returns None and deletes the stale file."""
    cache_file = tmp_path / "cache.pkl"
    _write_pickle(cache_file, {"version": "0", "hash": "abc"})

    result = load_validated_pickle(
        cache_file,
        validators=[_version_validator("1"), _hash_validator("abc")],
        logger=LOGGER,
        label="Test",
    )

    assert result is None
    assert not cache_file.exists()  # deleted on version mismatch


def test_hash_mismatch_returns_none_keeps_file(tmp_path: Path) -> None:
    """A hash mismatch returns None but intentionally keeps the file."""
    cache_file = tmp_path / "cache.pkl"
    _write_pickle(cache_file, {"version": "1", "hash": "stale"})

    result = load_validated_pickle(
        cache_file,
        validators=[_version_validator("1"), _hash_validator("abc")],
        logger=LOGGER,
        label="Test",
    )

    assert result is None
    assert cache_file.exists()  # kept because hash validator requested delete=False


def test_corrupt_file_returns_none_and_deletes(tmp_path: Path) -> None:
    """A corrupt (non-pickle) file returns None and is deleted."""
    cache_file = tmp_path / "cache.pkl"
    cache_file.write_bytes(b"this is not a pickle stream")

    result = load_validated_pickle(
        cache_file,
        validators=[_version_validator("1")],
        logger=LOGGER,
        label="Test",
    )

    assert result is None
    assert not cache_file.exists()


def test_validators_run_in_order(tmp_path: Path) -> None:
    """The first failing validator short-circuits later ones."""
    cache_file = tmp_path / "cache.pkl"
    _write_pickle(cache_file, {"version": "0", "hash": "stale"})

    calls: list[str] = []

    def _v1(cache_data: Dict[str, Any]):
        calls.append("v1")
        return invalid(delete=True, reason="v1 fail")

    def _v2(cache_data: Dict[str, Any]):
        calls.append("v2")
        return valid()

    result = load_validated_pickle(
        cache_file,
        validators=[_v1, _v2],
        logger=LOGGER,
        label="Test",
    )

    assert result is None
    assert calls == ["v1"]  # v2 never ran


def test_save_round_trip(tmp_path: Path) -> None:
    """save_pickle_with_metadata writes data that load returns unchanged."""
    cache_file = tmp_path / "cache.pkl"
    payload = {"version": "1", "hash": "abc", "data": {"k": "v"}}

    saved = save_pickle_with_metadata(cache_file, payload, logger=LOGGER, label="Test")
    assert saved is True
    assert cache_file.exists()

    result = load_validated_pickle(
        cache_file,
        validators=[_version_validator("1"), _hash_validator("abc")],
        logger=LOGGER,
        label="Test",
    )
    assert result == payload


def test_save_failure_returns_false(tmp_path: Path) -> None:
    """A write to an invalid path is swallowed and returns False."""
    # A directory path cannot be opened for writing as a file.
    bad_path = tmp_path / "subdir"
    bad_path.mkdir()

    saved = save_pickle_with_metadata(bad_path, {"x": 1}, logger=LOGGER, label="Test")
    assert saved is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

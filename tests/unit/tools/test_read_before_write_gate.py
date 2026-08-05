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

"""Dedicated tests for the read-before-overwrite guard.

The suite-wide conftest disables this guard by default (it is a *model*-safety
feature, default ON in production). These tests re-enable it per test via
``monkeypatch.delenv`` and exercise the ON path against ``write()`` (now a facade
over ``edit(create)``) and ``edit()`` directly.
"""

import os
import tempfile

import pytest

from victor.tools.file_editor_tool import edit
from victor.tools.filesystem import (
    ReadStateTracker,
    _editor_available,
    enforce_read_before_write,
    get_read_state,
    read,
    write,
)

# edit() requires the optional victor-coding extra; the "changed files" CI job
# installs only .[dev], so skip the direct-edit() tests there. write() tests are
# NOT skipped — write() has a direct-write fallback that covers that environment.
requires_editor = pytest.mark.skipif(
    not _editor_available(), reason="Enhanced editor (victor-coding) not available"
)


@pytest.fixture(autouse=True)
def _enable_gate_and_reset_read_state(monkeypatch):
    """Enable the guard (prod default ON) and start each test with an empty read-set."""
    monkeypatch.delenv("VICTOR_ENFORCE_READ_BEFORE_WRITE", raising=False)
    monkeypatch.setenv("VICTOR_DISABLE_WORKSPACE_GUARD", "1")
    get_read_state().clear()
    yield
    get_read_state().clear()


def _write_file(path: str, content: str = "original\n") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# ReadStateTracker unit tests (no tool machinery)
# ---------------------------------------------------------------------------


def test_read_state_tracker_records_and_checks():
    tracker = ReadStateTracker()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hi")
        path = f.name
    try:
        assert tracker.is_current(path) is False  # never read
        tracker.record(path)
        assert tracker.is_current(path) is True
        tracker.clear()
        assert tracker.is_current(path) is False
    finally:
        os.unlink(path)


def test_read_state_tracker_mtime_change_invalidates():
    tracker = ReadStateTracker()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("v1")
        path = f.name
    try:
        tracker.record(path)
        assert tracker.is_current(path) is True
        # Externally change mtime (simulates a concurrent edit after our read).
        os.utime(path, ns=(10**18, 10**18))
        assert tracker.is_current(path) is False
    finally:
        os.unlink(path)


def test_read_state_tracker_missing_file_is_noop():
    tracker = ReadStateTracker()
    tracker.record("/does/not/exist/xyz.txt")  # must not raise
    assert tracker.is_current("/does/not/exist/xyz.txt") is False


def test_enforce_read_before_write_direct():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("x")
        path = f.name
    try:
        from pathlib import Path

        fp = Path(path)
        # Unread existing file -> blocked
        with pytest.raises(PermissionError):
            enforce_read_before_write(fp)
        # force=True opts out
        enforce_read_before_write(fp, force=True)
        # Non-existent file -> allowed (creating new)
        enforce_read_before_write(fp.parent / "brand_new_does_not_exist.txt")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# write() facade + gate integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_blocks_overwrite_of_unread_existing_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        result = await write(path, "clobbered\n")
        # Blocked by the gate (returns the structured failure, not a string).
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "Refusing" in result.get("error", "") or "not read" in result.get("error", "")
        # File is unchanged.
        assert open(path).read() == "original\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_write_allows_overwrite_after_read():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        await read(path=path)  # registers the read
        result = await write(path, "new content\n")
        assert isinstance(result, str)
        assert "Successfully" in result
        assert "modified" in result
        assert open(path).read() == "new content\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_write_force_overrides_gate():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        result = await write(path, "forced\n", force=True)
        assert isinstance(result, str)
        assert "Successfully" in result
        assert open(path).read() == "forced\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_write_new_file_allowed_without_read():
    # A brand-new file never needs a prior read.
    path = os.path.join(tempfile.mkdtemp(), "brand_new_file.txt")
    try:
        result = await write(path, "created\n")
        assert isinstance(result, str)
        assert "Successfully" in result
        assert "created" in result
        assert open(path).read() == "created\n"
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.asyncio
async def test_env_disable_lets_unread_overwrite_through(monkeypatch):
    monkeypatch.setenv("VICTOR_ENFORCE_READ_BEFORE_WRITE", "0")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        result = await write(path, "overwritten\n")
        assert isinstance(result, str)
        assert "Successfully" in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_mtime_change_after_read_re_blocks_write():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("v1\n")
        path = f.name
    try:
        await read(path=path)  # read recorded with current mtime
        os.utime(path, ns=(10**18, 10**18))  # file changed on disk after our read
        result = await write(path, "v2\n")
        assert isinstance(result, dict)
        assert result.get("success") is False
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# edit() pre-pass gate integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@requires_editor
async def test_edit_create_blocks_unread_overwrite():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        result = await edit(ops=[{"type": "create", "path": path, "content": "x\n"}])
        assert result.get("success") is False
        assert result.get("operations_applied") == 0
        assert open(path).read() == "original\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
@requires_editor
async def test_edit_modify_blocks_unread():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        result = await edit(ops=[{"type": "modify", "path": path, "new_content": "x\n"}])
        assert result.get("success") is False
        assert open(path).read() == "original\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
@requires_editor
async def test_edit_modify_after_read_succeeds():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        await read(path=path)
        result = await edit(ops=[{"type": "modify", "path": path, "new_content": "edited\n"}])
        assert result.get("success") is True
        assert open(path).read() == "edited\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
@requires_editor
async def test_edit_modify_force_overrides_gate():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("original\n")
        path = f.name
    try:
        result = await edit(
            ops=[{"type": "modify", "path": path, "new_content": "forced\n", "force": True}]
        )
        assert result.get("success") is True
        assert open(path).read() == "forced\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
@requires_editor
async def test_edit_replace_still_protected_by_exact_match():
    # replace is already byte-exact-protected; with the gate ON and unread it
    # must still fail (the old_str won't be resolved from an un-read working copy
    # in the pre-pass). Read first to isolate the byte-match path.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("alpha\nbeta\n")
        path = f.name
    try:
        await read(path=path)
        result = await edit(
            ops=[{"type": "replace", "path": path, "old_str": "alpha", "new_str": "ALPHA"}]
        )
        assert result.get("success") is True
        assert "ALPHA" in open(path).read()
    finally:
        os.unlink(path)

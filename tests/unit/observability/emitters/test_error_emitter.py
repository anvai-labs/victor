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

"""Characterisation tests for ErrorEventEmitter, including warning de-duplication."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import pytest

import victor.observability.emitters.error_emitter as error_emitter_mod
from victor.observability.emitters import ErrorEventEmitter, IErrorEventEmitter

from .conftest import FakeBus, SyncRecorder


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(error_emitter_mod, "time", SimpleNamespace(time=c.time))
    return c


def _warnings(recorder: SyncRecorder) -> List[Any]:
    return [c["data"] for c in recorder.calls if c["topic"] == "warning.raised"]


def test_implements_protocol_and_dedup_defaults() -> None:
    e = ErrorEventEmitter()
    assert isinstance(e, IErrorEventEmitter)
    assert e._dedup_enabled is True
    assert e._dedup_window == 300
    assert e._max_suppressed == 10
    assert e._warning_dedup == {}


async def test_error_async_payload(fake_bus: FakeBus) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    try:
        raise ValueError("bad")
    except ValueError as exc:
        assert await e.error_async(exc, True, context={"c": 1}, extra="y") is True
    topic, data = fake_bus.emitted[0]
    assert topic == "error.raised"
    assert data["error"] == "bad"
    assert data["error_type"] == "ValueError"
    assert data["recoverable"] is True
    assert data["context"] == {"c": 1}
    assert data["extra"] == "y"
    assert data["category"] == "error"
    assert "ValueError: bad" in data["traceback"]


def test_error_sync_payload(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    e.error(KeyError("k"), False)
    call = sync_recorder.calls[0]
    assert call["topic"] == "error.raised"
    assert call["source"] == "ErrorEventEmitter"
    data = call["data"]
    assert data["error"] == "'k'"
    assert data["error_type"] == "KeyError"
    assert data["recoverable"] is False
    assert data["context"] == {}
    assert "category" not in data


def test_traceback_truncated_to_last_2000_chars(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_tb = "a" * 1000 + "b" * 2000
    monkeypatch.setattr(error_emitter_mod.traceback, "format_exc", lambda: long_tb)
    ErrorEventEmitter(bus=fake_bus).error(RuntimeError("x"), True)
    assert sync_recorder.calls[0]["data"]["traceback"] == "b" * 2000


def test_duplicate_warnings_suppressed_then_forced(
    fake_bus: FakeBus,
    sync_recorder: SyncRecorder,
    clock: Clock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    payload = {"warning": "disk low", "level": "w"}
    with caplog.at_level(logging.DEBUG):
        e.emit("warning.raised", payload)  # first: emitted unchanged
        for _ in range(8):  # counts 2..9: suppressed
            clock.now += 1
            e.emit("warning.raised", payload)
    assert _warnings(sync_recorder) == [payload]
    assert "Suppressed duplicate warning: disk low" in caplog.text

    clock.now += 1  # count 10 -> forced emission with suffix, entry reset
    e.emit("warning.raised", payload)
    forced = _warnings(sync_recorder)[-1]
    assert forced == {
        "warning": "disk low (repeated 10 times in last 9s)",
        "level": "w",
        # entry is popped before the lookup, so the count falls back to 1
        "suppressed_count": 1,
    }
    assert payload == {"warning": "disk low", "level": "w"}  # caller's dict untouched
    assert e._warning_dedup == {}

    e.emit("warning.raised", payload)  # fresh cycle: emitted again
    assert len(_warnings(sync_recorder)) == 3


def test_dedup_window_expiry_allows_reemission(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, clock: Clock
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    e.emit("warning.raised", {"warning": "w"})
    e.emit("warning.raised", {"warning": "w"})  # suppressed
    clock.now += 301
    e.emit("warning.raised", {"warning": "w"})  # window expired
    assert len(_warnings(sync_recorder)) == 2


def test_distinct_warnings_not_suppressed(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, clock: Clock
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    e.emit("warning.raised", {"warning": "a"})
    e.emit("warning.raised", {"warning": "b"})
    assert len(_warnings(sync_recorder)) == 2


def test_dedup_disabled_emits_every_warning(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, clock: Clock
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    e._dedup_enabled = False
    for _ in range(3):
        e.emit("warning.raised", {"warning": "w"})
    assert len(_warnings(sync_recorder)) == 3
    assert e._warning_dedup == {}


def test_dedup_only_applies_to_warning_topic(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, clock: Clock
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    for _ in range(3):
        e.emit("error.raised", {"warning": "w"})
    assert len(sync_recorder.calls) == 3
    assert e._warning_dedup == {}


async def test_async_path_does_not_dedup(fake_bus: FakeBus, clock: Clock) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    for _ in range(3):
        await e.emit_async("warning.raised", {"warning": "w"})
    assert len(fake_bus.emitted) == 3


def test_missing_warning_field_deduplicates_on_empty_string(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, clock: Clock
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    e.emit("warning.raised", {})
    e.emit("warning.raised", {})
    assert len(_warnings(sync_recorder)) == 1


def test_cleanup_expired_warnings(
    fake_bus: FakeBus, sync_recorder: SyncRecorder, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    e = ErrorEventEmitter(bus=fake_bus)
    e.emit("warning.raised", {"warning": "old"})
    clock.now += 200
    e.emit("warning.raised", {"warning": "new"})
    clock.now += 150  # "old" last seen 350s ago, "new" 150s ago
    with caplog.at_level(logging.DEBUG):
        e.cleanup_expired_warnings()
    assert len(e._warning_dedup) == 1
    assert "Cleaned up 1 expired warning dedup entries" in caplog.text

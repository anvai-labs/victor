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

"""Characterisation tests for LifecycleEventEmitter typed helpers."""

from __future__ import annotations

import pytest

from victor.observability.emitters import ILifecycleEventEmitter, LifecycleEventEmitter

from .conftest import FakeBus, SyncRecorder


def test_implements_protocol() -> None:
    assert isinstance(LifecycleEventEmitter(), ILifecycleEventEmitter)


async def test_async_helpers(fake_bus: FakeBus) -> None:
    e = LifecycleEventEmitter(bus=fake_bus)
    assert await e.session_start_async("s1", agent_id="a") is True
    await e.session_end_async("s1", 42.0)
    assert fake_bus.emitted == [
        ("lifecycle.session.start", {"session_id": "s1", "agent_id": "a", "category": "lifecycle"}),
        (
            "lifecycle.session.end",
            {"session_id": "s1", "duration_ms": 42.0, "category": "lifecycle"},
        ),
    ]


def test_sync_helpers(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    e = LifecycleEventEmitter(bus=fake_bus)
    e.session_start("s1", agent_id="a")
    e.session_end("s1", 42.0)
    assert [(c["topic"], c["data"], c["source"]) for c in sync_recorder.calls] == [
        ("lifecycle.session.start", {"session_id": "s1", "agent_id": "a"}, "LifecycleEventEmitter"),
        (
            "lifecycle.session.end",
            {"session_id": "s1", "duration_ms": 42.0},
            "LifecycleEventEmitter",
        ),
    ]


def test_track_session_emits_end_even_on_error(
    fake_bus: FakeBus, sync_recorder: SyncRecorder
) -> None:
    e = LifecycleEventEmitter(bus=fake_bus)
    with pytest.raises(ValueError):
        with e.track_session("s2", agent_id="a"):
            raise ValueError("x")
    assert [c["topic"] for c in sync_recorder.calls] == [
        "lifecycle.session.start",
        "lifecycle.session.end",
    ]
    end = sync_recorder.calls[1]["data"]
    assert end["session_id"] == "s2" and end["agent_id"] == "a" and end["duration_ms"] >= 0

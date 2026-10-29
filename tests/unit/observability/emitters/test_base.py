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

"""Characterisation tests for the shared emit template of every event emitter.

Each emitter tags async payloads with its category, routes sync emission through
emit_event_sync with its class name as source, honours the disabled flag on
the async path only, and degrades to a no-op when no bus is available.
"""

from __future__ import annotations

import logging
from typing import Any, Type

import pytest

import victor.core.events as core_events
from victor.observability.emitters import (
    ErrorEventEmitter,
    IEventEmitter,
    LifecycleEventEmitter,
    ModelEventEmitter,
    StateEventEmitter,
    ToolEventEmitter,
)

from .conftest import FakeBus, SyncRecorder

EMITTERS = [
    pytest.param(ErrorEventEmitter, "error", id="error"),
    pytest.param(LifecycleEventEmitter, "lifecycle", id="lifecycle"),
    pytest.param(ModelEventEmitter, "model", id="model"),
    pytest.param(StateEventEmitter, "state", id="state"),
    pytest.param(ToolEventEmitter, "tool", id="tool"),
]


def _no_global_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> Any:
        raise RuntimeError("no bus")

    monkeypatch.setattr(core_events, "get_observability_bus", _raise)


@pytest.mark.parametrize("cls,category", EMITTERS)
def test_satisfies_event_emitter_protocol_and_starts_enabled(cls: Type[Any], category: str) -> None:
    emitter = cls()
    assert isinstance(emitter, IEventEmitter)
    assert emitter.is_enabled() is True
    emitter.disable()
    assert emitter.is_enabled() is False
    emitter.enable()
    assert emitter.is_enabled() is True


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_emit_async_adds_category_without_mutating_input(
    cls: Type[Any], category: str, fake_bus: FakeBus
) -> None:
    emitter = cls(bus=fake_bus)
    payload = {"k": 1}
    assert await emitter.emit_async("some.topic", payload) is True
    assert fake_bus.emitted == [("some.topic", {"k": 1, "category": category})]
    assert payload == {"k": 1}


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_emit_async_category_overrides_payload_category(
    cls: Type[Any], category: str, fake_bus: FakeBus
) -> None:
    emitter = cls(bus=fake_bus)
    await emitter.emit_async("t", {"category": "other"})
    assert fake_bus.emitted[0][1]["category"] == category


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_emit_async_returns_bus_result(cls: Type[Any], category: str) -> None:
    emitter = cls(bus=FakeBus(result=False))
    assert await emitter.emit_async("t", {}) is False


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_emit_async_disabled_skips_bus(
    cls: Type[Any], category: str, fake_bus: FakeBus
) -> None:
    emitter = cls(bus=fake_bus)
    emitter.disable()
    assert await emitter.emit_async("t", {"k": 1}) is False
    assert fake_bus.emitted == []


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_emit_async_bus_failure_returns_false_and_logs(
    cls: Type[Any], category: str, caplog: pytest.LogCaptureFixture
) -> None:
    emitter = cls(bus=FakeBus(raises=ValueError("boom")))
    with caplog.at_level(logging.DEBUG):
        assert await emitter.emit_async("t", {}) is False
    assert f"Failed to emit {category} event: boom" in caplog.text
    # Logged on the concrete emitter's module logger.
    assert {r.name for r in caplog.records if "Failed to emit" in r.message} == {cls.__module__}


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_emit_async_without_bus_returns_false(
    cls: Type[Any], category: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_global_bus(monkeypatch)
    assert await cls().emit_async("t", {}) is False


@pytest.mark.parametrize("cls,category", EMITTERS)
async def test_falls_back_to_global_bus(
    cls: Type[Any], category: str, fake_bus: FakeBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core_events, "get_observability_bus", lambda: fake_bus)
    assert await cls().emit_async("t", {}) is True
    assert fake_bus.emitted == [("t", {"category": category})]


@pytest.mark.parametrize("cls,category", EMITTERS)
def test_emit_sync_routes_through_emit_event_sync(
    cls: Type[Any], category: str, fake_bus: FakeBus, sync_recorder: SyncRecorder
) -> None:
    emitter = cls(bus=fake_bus)
    payload = {"k": 1}
    emitter.emit("some.topic", payload)
    assert sync_recorder.calls == [
        {"bus": fake_bus, "topic": "some.topic", "data": {"k": 1}, "source": cls.__name__}
    ]
    # The sync path does not add the category tag.
    assert "category" not in sync_recorder.calls[0]["data"]


@pytest.mark.parametrize("cls,category", EMITTERS)
def test_emit_sync_ignores_disabled_flag(
    cls: Type[Any], category: str, fake_bus: FakeBus, sync_recorder: SyncRecorder
) -> None:
    emitter = cls(bus=fake_bus)
    emitter.disable()
    emitter.emit("t", {})
    assert len(sync_recorder.calls) == 1


@pytest.mark.parametrize("cls,category", EMITTERS)
def test_emit_sync_without_bus_is_noop(
    cls: Type[Any],
    category: str,
    sync_recorder: SyncRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_global_bus(monkeypatch)
    cls().emit("t", {})
    assert sync_recorder.calls == []


@pytest.mark.parametrize("cls,category", EMITTERS)
def test_emit_sync_swallows_and_logs_failures(
    cls: Type[Any],
    category: str,
    fake_bus: FakeBus,
    sync_recorder: SyncRecorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sync_recorder.raises = RuntimeError("kaput")
    with caplog.at_level(logging.DEBUG):
        cls(bus=fake_bus).emit("t", {})
    assert f"Failed to emit {category} event: kaput" in caplog.text
    assert {r.name for r in caplog.records if "Failed to emit" in r.message} == {cls.__module__}

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

"""Characterisation tests for StateEventEmitter typed helpers."""

from __future__ import annotations

from victor.observability.emitters import IStateEventEmitter, StateEventEmitter

from .conftest import FakeBus, SyncRecorder

EXPECTED = {"old_stage": "A", "new_stage": "B", "confidence": 0.5, "agent_id": "x"}


def test_implements_protocol() -> None:
    assert isinstance(StateEventEmitter(), IStateEventEmitter)


async def test_state_transition_async(fake_bus: FakeBus) -> None:
    e = StateEventEmitter(bus=fake_bus)
    assert await e.state_transition_async("A", "B", 0.5, agent_id="x") is True
    assert fake_bus.emitted == [("state.transition", {**EXPECTED, "category": "state"})]


def test_state_transition_sync(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    StateEventEmitter(bus=fake_bus).state_transition("A", "B", 0.5, agent_id="x")
    assert sync_recorder.calls == [
        {
            "bus": fake_bus,
            "topic": "state.transition",
            "data": EXPECTED,
            "source": "StateEventEmitter",
        }
    ]

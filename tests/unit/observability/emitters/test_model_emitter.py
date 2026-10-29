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

"""Characterisation tests for ModelEventEmitter typed helpers."""

from __future__ import annotations

from victor.observability.emitters import IModelEventEmitter, ModelEventEmitter

from .conftest import FakeBus, SyncRecorder


def test_implements_protocol() -> None:
    assert isinstance(ModelEventEmitter(), IModelEventEmitter)


def _expected() -> list:
    return [
        ("model.request", {"provider": "p", "model": "m", "prompt_tokens": 10, "x": 1}),
        (
            "model.response",
            {
                "provider": "p",
                "model": "m",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "latency_ms": 9.0,
            },
        ),
        (
            "model.streaming_delta",
            {"provider": "p", "model": "m", "delta": "d" * 500, "delta_length": 600},
        ),
        (
            "model.error",
            {"provider": "p", "model": "m", "error": "oops", "error_type": "TimeoutError"},
        ),
    ]


async def test_async_helpers(fake_bus: FakeBus) -> None:
    e = ModelEventEmitter(bus=fake_bus)
    assert await e.model_request_async("p", "m", 10, x=1) is True
    await e.model_response_async("p", "m", 10, 5, 9.0)
    await e.model_streaming_delta_async("p", "m", "d" * 600)
    await e.model_error_async("p", "m", TimeoutError("oops"))
    assert fake_bus.emitted == [(t, {**d, "category": "model"}) for t, d in _expected()]


def test_sync_helpers(fake_bus: FakeBus, sync_recorder: SyncRecorder) -> None:
    e = ModelEventEmitter(bus=fake_bus)
    e.model_request("p", "m", 10, x=1)
    e.model_response("p", "m", 10, 5, 9.0)
    e.model_streaming_delta("p", "m", "d" * 600)
    e.model_error("p", "m", TimeoutError("oops"))
    assert [(c["topic"], c["data"]) for c in sync_recorder.calls] == _expected()
    assert {c["source"] for c in sync_recorder.calls} == {"ModelEventEmitter"}

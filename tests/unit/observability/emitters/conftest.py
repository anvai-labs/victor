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

"""Shared fakes for the observability event emitter tests."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

import victor.core.events.emit_helper as emit_helper


class FakeBus:
    """Records async emissions; can be told to fail or return False."""

    def __init__(self, result: bool = True, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.emitted: List[Tuple[str, Dict[str, Any]]] = []

    async def emit(self, topic: str, data: Dict[str, Any]) -> bool:
        if self.raises is not None:
            raise self.raises
        self.emitted.append((topic, data))
        return self.result


class SyncRecorder:
    """Stands in for emit_event_sync and records every call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.raises: Exception | None = None

    def __call__(self, bus: Any, *, topic: str, data: Dict[str, Any], source: str) -> None:
        if self.raises is not None:
            raise self.raises
        self.calls.append({"bus": bus, "topic": topic, "data": data, "source": source})


@pytest.fixture
def fake_bus() -> FakeBus:
    return FakeBus()


@pytest.fixture
def sync_recorder(monkeypatch: pytest.MonkeyPatch) -> SyncRecorder:
    recorder = SyncRecorder()
    monkeypatch.setattr(emit_helper, "emit_event_sync", recorder)
    return recorder

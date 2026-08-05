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

"""W3d/G7 D4: internal orchestration kwargs must not reach provider.chat()."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    _INTERNAL_KWARG_KEYS,
    OrchestratorProtocolAdapter,
)


async def test_internal_kwargs_stripped_before_provider_chat():
    provider = SimpleNamespace(chat=AsyncMock(return_value="ok"))
    orch = SimpleNamespace(provider=provider)
    adapter = OrchestratorProtocolAdapter.__new__(OrchestratorProtocolAdapter)
    adapter._orchestrator = orch

    await adapter.execute_turn(
        messages=[],
        model="m",
        temperature=0.7,
        max_tokens=100,
        execution_mode="fast",
        topology_action="escalate",
        provider_hint="x",
        reasoning_effort="high",  # a real param — must survive
    )

    _, kwargs = provider.chat.call_args
    for internal in _INTERNAL_KWARG_KEYS:
        assert internal not in kwargs
    assert kwargs["reasoning_effort"] == "high"

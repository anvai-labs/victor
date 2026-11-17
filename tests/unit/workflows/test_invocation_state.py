# Copyright 2025 Vijaykumar Singh <anvaiops.com>
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

"""Tests for workflow invocation input-state translation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.workflows.invocation_state import create_invocation_state


@pytest.mark.asyncio
async def test_existing_checkpoint_translates_omitted_context_to_resume_input():
    workflow = SimpleNamespace(start_node="start")
    checkpointer = MagicMock()
    checkpointer.load = AsyncMock(return_value=MagicMock())

    result = await create_invocation_state(workflow, None, "thread", checkpointer)

    assert result is None
    checkpointer.load.assert_awaited_once_with("thread")


@pytest.mark.asyncio
async def test_missing_checkpoint_translates_omitted_context_to_fresh_state():
    workflow = SimpleNamespace(start_node="start")
    checkpointer = MagicMock()
    checkpointer.load = AsyncMock(return_value=None)

    result = await create_invocation_state(workflow, None, "thread", checkpointer)

    assert result is not None
    assert dict(result.items())["_current_node"] == "start"
    checkpointer.load.assert_awaited_once_with("thread")

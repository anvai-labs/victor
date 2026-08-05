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

"""Tests for shared agentic-graph state helpers."""

import pytest

from victor.framework.agentic_graph._state_utils import (
    normalize_state_argument,
    unwrap_state,
)
from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.framework.graph import CopyOnWriteState


class TestUnwrapState:
    """Tests for unwrap_state."""

    def test_returns_model_instance_unchanged(self):
        """Model input should be returned as-is."""
        state = AgenticLoopStateModel(query="Test")

        result = unwrap_state(state)

        assert result is state

    def test_converts_dict_to_model(self):
        """Dict input should be validated into the canonical model."""
        result = unwrap_state({"query": "Test", "iteration": 1})

        assert isinstance(result, AgenticLoopStateModel)
        assert result.query == "Test"
        assert result.iteration == 1

    def test_unwraps_copy_on_write_state(self):
        """Copy-on-write input should be unwrapped to the canonical model."""
        wrapped = CopyOnWriteState(AgenticLoopStateModel(query="Test"))

        result = unwrap_state(wrapped)

        assert isinstance(result, AgenticLoopStateModel)
        assert result.query == "Test"

    def test_rejects_unsupported_state_type(self):
        """Unsupported state types should fail fast."""
        with pytest.raises(TypeError, match="Unsupported agentic graph state type"):
            unwrap_state(object())

    def test_rejects_unsupported_copy_on_write_payload(self):
        """Wrapped unsupported payloads should fail fast."""
        with pytest.raises(TypeError, match="Unsupported CopyOnWriteState payload"):
            unwrap_state(CopyOnWriteState(["not", "a", "state"]))


class TestNormalizeStateArgument:
    """Tests for normalize_state_argument."""

    @pytest.mark.asyncio
    async def test_normalizes_async_positional_state(self):
        """Async callables should receive normalized positional state."""

        @normalize_state_argument()
        async def sample_node(state: AgenticLoopStateModel) -> str:
            return state.query

        result = await sample_node({"query": "Async dict"})

        assert result == "Async dict"

    def test_normalizes_sync_keyword_state(self):
        """Sync callables should receive normalized keyword state."""

        @normalize_state_argument()
        def sample_edge(*, state: AgenticLoopStateModel) -> str:
            return state.query

        result = sample_edge(state=CopyOnWriteState(AgenticLoopStateModel(query="Sync wrapped")))

        assert result == "Sync wrapped"

    @pytest.mark.asyncio
    async def test_normalizes_method_state_with_custom_position(self):
        """Methods should support state normalization after self."""

        class SampleAdapter:
            @normalize_state_argument(position=1)
            async def call(self, state: AgenticLoopStateModel) -> str:
                return state.query

        result = await SampleAdapter().call({"query": "Method dict"})

        assert result == "Method dict"

    @pytest.mark.asyncio
    async def test_raises_type_error_for_unsupported_async_state(self):
        """Decorator should surface state-type errors before node logic runs."""

        @normalize_state_argument()
        async def sample_node(state: AgenticLoopStateModel) -> str:
            return state.query

        with pytest.raises(TypeError, match="Unsupported agentic graph state type"):
            await sample_node(object())

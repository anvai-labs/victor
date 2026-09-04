# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Tool-selection history projection equivalence.

_select_tools_for_turn used to pydantic-serialize the ENTIRE conversation
history on every model turn (``[msg.model_dump() ...]``) while its consumers
read only role, full content, and tool_calls names (co-design review U1-1).
The replacement projection must be equivalent for those consumers: same
cache key via the real cache-key builder, same keys the selector accesses,
and no aliasing back into the live history.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from victor.agent.conversation.types import ConversationMessage
from victor.agent.tool_selection.history_projection import _selector_history_projection
from victor.storage.cache.generic_result_cache import _create_tool_selection_cache_key


class _PydanticMessage(BaseModel):
    """Mirrors the pydantic message shape the chat context exposes."""

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


def _fixture() -> list:
    return [
        _PydanticMessage(role="user", content="search for the pending action"),
        _PydanticMessage(
            role="assistant",
            content="running it now",
            tool_calls=[{"function": {"name": "web_search", "arguments": "{}"}}],
        ),
        _PydanticMessage(role="user", content="thanks"),
    ]


class TestSelectorHistoryProjection:
    def test_projection_matches_model_dump_on_consumed_keys(self):
        msgs = _fixture()
        projection = _selector_history_projection(msgs)
        dump = [m.model_dump() for m in msgs]

        assert len(projection) == len(dump)
        for projected, dumped in zip(projection, dump):
            assert set(projected) <= set(dumped)
            assert projected["role"] == dumped["role"]
            assert projected["content"] == dumped["content"]
            assert projected.get("tool_calls") == dumped.get("tool_calls")

    def test_cache_key_equivalent_to_model_dump(self):
        msgs = _fixture()
        projection = _selector_history_projection(msgs)
        dump = [m.model_dump() for m in msgs]

        key_projection = _create_tool_selection_cache_key(
            "run the search", conversation_history=projection, conversation_depth=7
        )
        key_dump = _create_tool_selection_cache_key(
            "run the search", conversation_history=dump, conversation_depth=7
        )
        assert key_projection == key_dump

    def test_projection_does_not_alias_live_history(self):
        msgs = _fixture()
        projection = _selector_history_projection(msgs)

        # Mutating the projection's tool_calls list must not touch the message.
        projection[1]["tool_calls"].append({"function": {"name": "injected"}})
        assert len(msgs[1].tool_calls) == 1

    def test_dataclass_messages_project_via_getattr(self):
        msgs = [
            ConversationMessage(role="user", content="hello"),
            ConversationMessage(
                role="assistant",
                content="hi",
                tool_calls=[{"name": "shell"}],
            ),
        ]
        projection = _selector_history_projection(msgs)
        assert projection[0] == {"role": "user", "content": "hello"}
        assert projection[1]["tool_calls"] == [{"name": "shell"}]

    def test_empty_and_missing_fields(self):
        assert _selector_history_projection([]) is None

        class _Bare:
            pass

        assert _selector_history_projection([_Bare()]) == [{"role": None, "content": ""}]


class TestAdversarialProjectionGuarantees:
    """Negatives from adversarial review of this PR."""

    def test_inner_tool_call_dicts_not_aliased(self):
        """Mutating a projected tool_call dict must not touch the live
        message (the outer list was copied; inner dicts were not)."""
        msgs = [
            _PydanticMessage(
                role="assistant",
                content="running",
                tool_calls=[{"function": {"name": "web_search", "arguments": "{}"}}],
            )
        ]
        projection = _selector_history_projection(msgs)
        projection[0]["tool_calls"][0]["function"]["name"] = "injected"
        assert msgs[0].tool_calls[0]["function"]["name"] == "web_search"

    def test_cache_key_changes_when_history_changes(self):
        """Anti-collapse: different history must produce a different cache
        key (guards against a projection that collapses all inputs)."""
        key_a = _create_tool_selection_cache_key(
            "q", conversation_history=_selector_history_projection(_fixture()), conversation_depth=5
        )
        key_b = _create_tool_selection_cache_key(
            "q",
            conversation_history=_selector_history_projection(
                _fixture()[:-1] + [_PydanticMessage(role="user", content="different")]
            ),
            conversation_depth=5,
        )
        assert key_a != key_b

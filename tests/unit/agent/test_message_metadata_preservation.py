# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Message metadata must survive the paths that rebuild messages.

``MessageSource`` exists so compaction can tell permanent ground truth from
transient agent scaffolding — ``conversation/scoring.py`` scores AGENT_GUIDANCE
at 0.2 and USER_TYPED at 1.0 precisely so throwaway guidance is evicted first.

Two paths silently discard it, which collapses that distinction: every message
becomes ``UNKNOWN``, falls through to ``_ROLE_SCORES``, and agent guidance —
injected with ``role="user"`` — scores 0.8, the same as something the user
actually typed. Scaffolding then competes with real user intent for retention,
and can evict it.

That is the same theme as the defect class this work came from: agent-authored
text being indistinguishable from the user's.
"""

from __future__ import annotations

import logging

from victor.agent.conversation.scoring import _ROLE_SCORES, _SOURCE_ROLE_OVERRIDES
from victor.agent.conversation.types import (
    MESSAGE_SOURCE_METADATA_KEY,
    ConversationMessage,
    MessageSource,
)
from victor.providers.base import Message


def _guidance_message() -> Message:
    return Message(
        role="user",
        content="[status] 3 of 20 tool calls used",
        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_GUIDANCE.value},
    )


class TestBridgePreservesSource:
    """ConversationMessage.from_provider_message must carry metadata across."""

    def test_source_survives_the_bridge(self):
        bridged = ConversationMessage.from_provider_message(_guidance_message())

        assert bridged.source is MessageSource.AGENT_GUIDANCE

    def test_explicit_metadata_wins_over_the_message_s_own(self):
        """Callers passing metadata explicitly must still override."""
        bridged = ConversationMessage.from_provider_message(
            _guidance_message(),
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.USER_TYPED.value},
        )

        assert bridged.source is MessageSource.USER_TYPED

    def test_message_without_metadata_still_bridges(self):
        bridged = ConversationMessage.from_provider_message(Message(role="user", content="hello"))

        assert bridged.source is MessageSource.UNKNOWN
        assert bridged.metadata == {}


class TestScaffoldingDoesNotOutrankUserIntent:
    """The consequence the bridge fix exists to restore."""

    def test_guidance_scores_below_typed_user_content(self):
        guidance = ConversationMessage.from_provider_message(_guidance_message())
        typed = ConversationMessage.from_provider_message(
            Message(
                role="user",
                content="implement the metrics registry",
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.USER_TYPED.value},
            )
        )

        guidance_weight = _SOURCE_ROLE_OVERRIDES.get(guidance.source)
        typed_weight = _SOURCE_ROLE_OVERRIDES.get(typed.source)

        assert guidance_weight is not None, (
            "guidance resolved to UNKNOWN, so it falls through to "
            f"_ROLE_SCORES['user']={_ROLE_SCORES['user']} — the same as real user "
            "intent, which is exactly what MessageSource exists to prevent"
        )
        assert typed_weight is not None
        assert guidance_weight < typed_weight


class TestTurnPrefixPreservesTheMessageItRewrites:
    """Prepending the turn prefix must not strip the rest of the message."""

    def _apply_prefix(self, messages, prefix):
        from victor.agent.prompt_prefix import apply_turn_prefix

        return apply_turn_prefix(messages, prefix)

    def test_metadata_survives_prefixing(self):
        original = Message(
            role="user",
            content="do the thing",
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.USER_TYPED.value},
        )

        result = self._apply_prefix([original], "<guidance>\n")

        assert result[0].content.startswith("<guidance>\n")
        assert result[0].metadata == original.metadata

    def test_name_and_tool_fields_survive_prefixing(self):
        original = Message(role="user", content="do the thing", name="operator")

        result = self._apply_prefix([original], "<guidance>\n")

        assert result[0].name == "operator"

    def test_prefix_targets_the_last_user_message(self):
        messages = [
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="second"),
        ]

        result = self._apply_prefix(messages, "P:")

        assert result[0].content == "first"
        assert result[2].content == "P:second"

    def test_empty_prefix_is_a_no_op(self):
        messages = [Message(role="user", content="unchanged")]

        assert self._apply_prefix(messages, "")[0].content == "unchanged"


class TestDroppedNudgesAreNotSilent:
    """A discarded message must leave a trace."""

    def test_bracket_nudge_drop_is_logged(self, caplog):
        """Under cache optimisation a bracketed system nudge is dropped.

        Dropping it is correct — appending a second system message would break
        the byte-stable cached prefix — but doing it via a bare `return` meant a
        caller lost the message with no trace anywhere.
        """
        from victor.agent.system_nudges import log_dropped_system_nudge

        with caplog.at_level(logging.WARNING, logger="victor.agent.system_nudges"):
            dropped = log_dropped_system_nudge("[FILES: a.py]")

        assert dropped is True
        assert "[FILES: a.py]" in caplog.text
        assert "reminder manager" in caplog.text.lower()

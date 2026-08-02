# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for MessageSource origin tracking on ConversationMessage."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from victor.agent.conversation.types import (
    ConversationMessage,
    MESSAGE_SOURCE_METADATA_KEY,
    MessagePriority,
    MessageSource,
)
from victor.agent.conversation.scoring import (
    CONTROLLER_WEIGHTS,
    ScoringWeights,
    score_messages,
)

# ---------------------------------------------------------------------------
# Enum round-trip
# ---------------------------------------------------------------------------


def test_message_source_values_are_short():
    """All enum string values must be short codes (≤4 chars) to keep SQLite compact."""
    for member in MessageSource:
        assert len(member.value) <= 4, f"{member.name} value '{member.value}' exceeds 4 chars"


def test_message_source_round_trip():
    for member in MessageSource:
        assert MessageSource(member.value) is member


# ---------------------------------------------------------------------------
# source property on ConversationMessage
# ---------------------------------------------------------------------------


def test_source_defaults_to_unknown():
    msg = ConversationMessage(role="user", content="hello")
    assert msg.source is MessageSource.UNKNOWN


def test_source_setter_round_trip():
    msg = ConversationMessage(role="user", content="hello")
    msg.source = MessageSource.USER_TYPED
    assert msg.source is MessageSource.USER_TYPED
    assert msg.metadata[MESSAGE_SOURCE_METADATA_KEY] == MessageSource.USER_TYPED.value


def test_source_getter_from_preloaded_metadata():
    msg = ConversationMessage(
        role="user",
        content="hello",
        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_NUDGE.value},
    )
    assert msg.source is MessageSource.AGENT_NUDGE


def test_source_getter_bad_string_returns_unknown():
    msg = ConversationMessage(
        role="user",
        content="hello",
        metadata={MESSAGE_SOURCE_METADATA_KEY: "not_a_valid_code"},
    )
    assert msg.source is MessageSource.UNKNOWN


# ---------------------------------------------------------------------------
# Wire format: to_provider_format() must not expose metadata or source
# ---------------------------------------------------------------------------


def test_provider_format_has_no_source_key():
    msg = ConversationMessage(role="user", content="hello")
    msg.source = MessageSource.USER_TYPED
    pf = msg.to_provider_format()
    assert "source" not in pf
    assert "metadata" not in pf
    assert pf["role"] == "user"
    assert pf["content"] == "hello"


# ---------------------------------------------------------------------------
# Scoring: source overrides role score
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=timezone.utc)


def _make_msg(role: str, source: MessageSource, content: str = "x") -> ConversationMessage:
    msg = ConversationMessage(role=role, content=content, timestamp=_NOW)
    msg.source = source
    return msg


def _role_only_scores(messages: list) -> list[float]:
    weights = ScoringWeights(priority=0.0, recency=0.0, role=1.0, length=0.0, semantic=0.0)
    pairs = score_messages(messages, weights=weights)
    msg_to_score = {id(m): s for m, s in pairs}
    return [msg_to_score[id(m)] for m in messages]


def test_user_typed_scores_higher_than_unknown_user():
    """USER_TYPED (override=1.0) should outscore UNKNOWN (falls through to role score 0.8)."""
    typed = _make_msg("user", MessageSource.USER_TYPED)
    unknown = _make_msg("user", MessageSource.UNKNOWN)
    scores = _role_only_scores([typed, unknown])
    assert scores[0] > scores[1]


def test_agent_nudge_scores_lower_than_user_typed():
    typed = _make_msg("user", MessageSource.USER_TYPED)
    nudge = _make_msg("user", MessageSource.AGENT_NUDGE)
    scores = _role_only_scores([typed, nudge])
    assert scores[0] > scores[1]


def test_agent_continuation_scores_low():
    cont = _make_msg("user", MessageSource.AGENT_CONTINUATION)
    scores = _role_only_scores([cont])
    assert scores[0] == pytest.approx(0.1, abs=1e-6)


def test_unknown_source_uses_role_fallback():
    """UNKNOWN source falls through to _ROLE_SCORES['user'] = 0.8."""
    unknown_user = _make_msg("user", MessageSource.UNKNOWN)
    scores = _role_only_scores([unknown_user])
    assert scores[0] == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# Priority from source in ConversationStore._determine_priority
# ---------------------------------------------------------------------------


def test_determine_priority_user_typed_is_critical():
    from victor.agent.conversation.store import ConversationStore

    store = ConversationStore.__new__(ConversationStore)
    priority = store._determine_priority("user", None, source=MessageSource.USER_TYPED)
    assert priority is MessagePriority.CRITICAL


def test_determine_priority_nudge_is_ephemeral():
    from victor.agent.conversation.store import ConversationStore

    store = ConversationStore.__new__(ConversationStore)
    priority = store._determine_priority("user", None, source=MessageSource.AGENT_NUDGE)
    assert priority is MessagePriority.EPHEMERAL


def test_determine_priority_continuation_is_ephemeral():
    from victor.agent.conversation.store import ConversationStore

    store = ConversationStore.__new__(ConversationStore)
    priority = store._determine_priority("user", None, source=MessageSource.AGENT_CONTINUATION)
    assert priority is MessagePriority.EPHEMERAL


def test_determine_priority_guidance_is_low():
    from victor.agent.conversation.store import ConversationStore

    store = ConversationStore.__new__(ConversationStore)
    priority = store._determine_priority("user", None, source=MessageSource.AGENT_GUIDANCE)
    assert priority is MessagePriority.LOW


def test_determine_priority_unknown_falls_through_to_role_logic():
    from victor.agent.conversation.store import ConversationStore

    store = ConversationStore.__new__(ConversationStore)
    # UNKNOWN source → existing role-based logic (role="user" → HIGH)
    priority = store._determine_priority("user", None, source=MessageSource.UNKNOWN)
    assert priority in (
        MessagePriority.HIGH,
        MessagePriority.MEDIUM,
        MessagePriority.CRITICAL,
    )


# ---------------------------------------------------------------------------
# Compaction: Message.metadata carries source for controller's USER_TYPED guard
# ---------------------------------------------------------------------------


def test_message_metadata_round_trips_through_message_history():
    """Verify that metadata passed to MessageHistory.add_message() is stored on
    the Message object — this is what the controller's USER_TYPED guard reads."""
    from victor.agent.message_history import MessageHistory

    hist = MessageHistory()
    hist.add_message(
        "user",
        "hello",
        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.USER_TYPED.value},
    )
    msg = hist._messages[-1]
    assert msg.metadata is not None
    assert msg.metadata.get(MESSAGE_SOURCE_METADATA_KEY) == MessageSource.USER_TYPED.value


def test_nudge_metadata_round_trips_through_message_history():
    """AGENT_NUDGE metadata is preserved so the controller can mark nudges EPHEMERAL."""
    from victor.agent.message_history import MessageHistory

    hist = MessageHistory()
    hist.add_message(
        "user",
        "nudge text",
        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_NUDGE.value},
    )
    msg = hist._messages[-1]
    assert msg.metadata is not None
    assert msg.metadata.get(MESSAGE_SOURCE_METADATA_KEY) == MessageSource.AGENT_NUDGE.value


def test_build_internal_history_metadata_stores_source():
    from victor.agent.conversation.history_metadata import (
        build_internal_history_metadata,
    )

    meta = build_internal_history_metadata("nudge", source=MessageSource.AGENT_NUDGE)
    assert meta[MESSAGE_SOURCE_METADATA_KEY] == MessageSource.AGENT_NUDGE.value


def test_build_internal_history_metadata_no_source_omits_key():
    from victor.agent.conversation.history_metadata import (
        build_internal_history_metadata,
    )

    meta = build_internal_history_metadata("some_kind")
    assert MESSAGE_SOURCE_METADATA_KEY not in meta

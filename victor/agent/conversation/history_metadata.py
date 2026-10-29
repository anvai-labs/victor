"""Metadata helpers for interactive history visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

INTERACTIVE_HISTORY_KEY = "interactive_history"
INTERNAL_PROMPT_KIND_KEY = "internal_prompt_kind"

if TYPE_CHECKING:
    from victor.agent.conversation.types import MessageSource


def build_internal_history_metadata(
    kind: str,
    source: Optional["MessageSource"] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Mark a message as internal so it stays out of interactive history.

    Args:
        kind: Internal prompt kind (e.g. "nudge", "continuation")
        source: MessageSource origin — stored as metadata["source"] for compaction scoring
        **extra: Additional metadata fields
    """
    from victor.agent.conversation.types import (
        MESSAGE_SOURCE_METADATA_KEY,
    )

    metadata: dict[str, Any] = {
        INTERACTIVE_HISTORY_KEY: False,
        INTERNAL_PROMPT_KIND_KEY: kind,
    }
    if source is not None:
        metadata[MESSAGE_SOURCE_METADATA_KEY] = source.value
    metadata.update(extra)
    return metadata


def is_hidden_from_interactive_history(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether metadata marks a message as hidden from input history."""
    if not metadata:
        return False
    return metadata.get(INTERACTIVE_HISTORY_KEY) is False


def is_user_turn(role: str, metadata: Mapping[str, Any] | None) -> bool:
    """Identify a genuine user turn, retaining compatibility with untagged history.

    Runtime guidance can use the user wire role without becoming a new task.
    History retention and provider assembly must use the same provenance rule.
    """
    from victor.agent.conversation.types import MESSAGE_SOURCE_METADATA_KEY, MessageSource

    metadata = metadata or {}
    internal_sources = {
        source.value
        for source in MessageSource
        if source not in (MessageSource.USER_TYPED, MessageSource.UNKNOWN)
    }
    return (
        role == "user"
        and metadata.get(MESSAGE_SOURCE_METADATA_KEY) not in internal_sources
        and metadata.get(INTERACTIVE_HISTORY_KEY) is not False
        and not metadata.get(INTERNAL_PROMPT_KIND_KEY)
    )


def mid_user_turn(messages) -> bool:
    """True when non-anchor activity exists after the last genuine user turn.

    KV prefix stability: compaction and other history rewrites are sanctioned
    only at user-turn boundaries; intra-loop rewrites mutate already-sent
    provider bytes. Hidden runtime messages (failure hints, verification
    feedback) do not count as anchors, so appending them does not end a turn.
    """
    from typing import Sequence

    sequence: Sequence = messages
    for message in reversed(sequence):
        if is_user_turn(message.role, message.metadata):
            return message is not sequence[-1]
    return False

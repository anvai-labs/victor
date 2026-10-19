"""Tests for apply_turn_prefix placement (genuine-anchor targeting)."""

from __future__ import annotations

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.prompt_prefix import apply_turn_prefix
from victor.providers.base import Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


async def test_targets_genuine_anchor_not_hidden_tail():
    """A hidden user-role hint appended after the anchor must not steal the prefix."""
    from victor.agent.prompt_prefix import apply_turn_prefix

"""Tests for conversation settings (mirrors victor/config/conversation_settings.py)."""

from __future__ import annotations

from victor.config.conversation_settings import (
    DEFAULT_MAX_CONVERSATION_HISTORY,
    ConversationSettings,
)
from victor.config.settings import Settings


def test_default_max_conversation_history_is_single_source_of_truth():
    # The count trim is a rare safety net (KV prefix stability): a mid-loop
    # trim mutates already-sent provider bytes, so the cap is deliberately
    # generous while char-budget assembly and boundary compaction bound context.
    assert DEFAULT_MAX_CONVERSATION_HISTORY == 2000


def test_settings_and_nested_model_agree_with_the_constant():
    assert Settings().max_conversation_history == DEFAULT_MAX_CONVERSATION_HISTORY
    assert ConversationSettings().max_conversation_history == DEFAULT_MAX_CONVERSATION_HISTORY

# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Conversation management package.

Canonical modules:
- types: ConversationMessage, MessageRole, MessagePriority
- scoring: score_messages(), ScoringWeights, weight presets
- assembler: TurnBoundaryContextAssembler (context budget)
- message_store: MessageStore (SOLID extraction)
- context_handler: ContextOverflowHandler
- session_manager: SessionManager
- embedding_manager: EmbeddingManager
"""

from victor.agent.conversation.types import (
    ConversationMessage,
    MESSAGE_SOURCE_METADATA_KEY,
    MessagePriority,
    MessageRole,
    MessageSource,
)
from victor.agent.conversation.scoring import (
    CONTROLLER_WEIGHTS,
    DEFAULT_WEIGHTS,
    STORE_WEIGHTS,
    ScoringWeights,
    score_messages,
)
from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.conversation.message_store import MessageStore
from victor.agent.conversation.context_handler import ContextOverflowHandler
from victor.agent.conversation.session_manager import SessionManager
from victor.agent.conversation.embedding_manager import EmbeddingManager

__all__ = [
    # Canonical types
    "ConversationMessage",
    "MESSAGE_SOURCE_METADATA_KEY",
    "MessagePriority",
    "MessageRole",
    "MessageSource",
    # Scoring
    "CONTROLLER_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "STORE_WEIGHTS",
    "ScoringWeights",
    "score_messages",
    # Assembler
    "TurnBoundaryContextAssembler",
    # SOLID extraction modules
    "MessageStore",
    "ContextOverflowHandler",
    "SessionManager",
    "EmbeddingManager",
]

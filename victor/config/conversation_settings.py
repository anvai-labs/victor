"""Conversation persistence and history limits."""

from __future__ import annotations

from pydantic import BaseModel

#: Single source of truth for the message-count history cap. Referenced by the
#: Settings field, MessageHistory's constructor/serialization defaults and both
#: factory call sites so the cap cannot drift again (it previously had three
#: conflicting defaults: 100000 / 100 / 100). 2000 keeps the count trim a rare
#: safety net: char-budget assembly and turn-boundary compaction are the real
#: context controls, and a mid-loop trim mutates already-sent provider bytes
#: (KV prefix stability), so when it does fire it logs a sanctioned boundary.
DEFAULT_MAX_CONVERSATION_HISTORY = 2000

#: Budget-aware compaction: cumulative session prefill volume (real
#: prompt_tokens accumulated from provider responses) that forces compaction
#: at the next user-turn boundary even under window utilization. Un-compacted
#: history resend amplification is what burns token budgets (measured 2026-10:
#: 813k tokens in 24 calls of one turn). KV prefix caching keeps this trigger
#: quiet for cache-stable sessions — only genuinely resent prefill counts.
#: ``None`` disables the spend trigger.
DEFAULT_MAX_SESSION_PREFILL_TOKENS = 400_000


class ConversationSettings(BaseModel):
    """Conversation persistence and history limits."""

    conversation_memory_enabled: bool = True
    conversation_embeddings_enabled: bool = True
    max_conversation_history: int = DEFAULT_MAX_CONVERSATION_HISTORY
    session_idle_timeout: int = 180

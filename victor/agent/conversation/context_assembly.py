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

"""Context assembly extracted from AgentOrchestrator (self-analysis round 2, W3).

The orchestrator remains the facade (service-layer guard): its method
names and mock.patch targets stay; the implementation lives here as
module functions taking the orchestrator, mirroring the
``ToolStrategyRuntime(self)`` extraction idiom. Every function is the
VERBATIM body of the orchestrator method it replaces — no behavior
change, only a new home.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from victor.core.provider_messages import Message
from victor.agent.prompt_prefix import apply_turn_prefix

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


def conversation_message_count(orchestrator: "AgentOrchestrator") -> int:
    """Return the current conversation length across history implementations."""
    conversation = getattr(orchestrator, "conversation", None)
    if conversation is None:
        return 0

    messages = getattr(conversation, "messages", None)
    if messages is not None:
        try:
            return len(messages)
        except TypeError:
            pass

    message_count = getattr(conversation, "message_count", None)
    if callable(message_count):
        try:
            return int(message_count())
        except TypeError:
            pass

    return 0


def session_tools(orchestrator: "AgentOrchestrator") -> Optional[list]:
    """Session-locked tools for cache-friendly API calls.

    Returns the currently enabled tool set (frozen at session start) so the
    tool prefix remains byte-identical across API calls in the same session.
    When cache optimization is disabled, returns None (use per-turn).
    """
    if not orchestrator._cache_optimization_enabled:
        return None
    locked = getattr(orchestrator, "_session_tools", None)
    if locked is None:
        try:
            enabled_tools = orchestrator.get_enabled_tools()
            locked = [
                tool
                for tool in orchestrator.tools.list_tools(only_enabled=True)
                if getattr(tool, "name", None) in enabled_tools
            ]
            if locked:
                orchestrator._session_tools = locked
                logger.debug(
                    "[cache] Session tools locked: %d tools (prefix-stable)",
                    len(locked),
                )
                return locked
        except Exception:
            return None
    return locked


def assembled_messages_for_call(
    orchestrator: "AgentOrchestrator",
    current_query: Optional[str] = None,
    *,
    selected_tools: Optional[list[Any]] = None,
    planned_tools: Optional[list[Any]] = None,
    goals: Optional[list[str]] = None,
) -> List[Message]:
    """Context-assembled messages for provider calls.

    When context assembler is available: keeps system prompt + ledger +
    last N full turns + score-selected older messages within budget.
    Falls back to raw orchestrator.messages when assembler unavailable.

    When the canonical prompt pipeline is available, dynamic prompt guidance
    (skills, reminders, GEPA/MiPRO/CoT optimizations, credit guidance) is
    prepended to the last user message instead of being injected as system
    messages. For KV-capable providers this also keeps the system prompt
    byte-identical across turns for provider prefix caching.
    """
    orchestrator._get_prompt_builder_runtime().ensure_system_prompt_current()

    assembler = getattr(orchestrator, "_context_assembler", None)
    if assembler is None:
        messages = list(orchestrator.messages)
    else:
        max_chars = orchestrator._get_max_context_chars()
        messages = assembler.assemble(orchestrator.messages, max_chars, current_query=current_query)

        if len(messages) < len(orchestrator.messages):
            total_original = sum(len(m.content) for m in orchestrator.messages)
            total_assembled = sum(len(m.content) for m in messages)
            logger.debug(
                "[context] Assembled %d/%d messages (%dK/%dK chars, budget=%dK)",
                len(messages),
                len(orchestrator.messages),
                total_assembled // 1024,
                total_original // 1024,
                max_chars // 1024,
            )

    # Boundary guard: ensure all items are Message objects before sending
    # to the provider. Converts dicts/strings that may have leaked in from
    # session restore, compaction, or serialization round-trips.
    normalized = []
    for m in messages:
        if isinstance(m, Message):
            normalized.append(m)
        elif isinstance(m, dict) and "role" in m:
            normalized.append(Message(**{k: v for k, v in m.items() if k in Message.model_fields}))
        elif isinstance(m, str):
            normalized.append(Message(role="assistant", content=m))
        else:
            logger.debug("Dropping non-Message item from assembled messages: %s", type(m))
    if len(normalized) != len(messages):
        logger.debug(
            "[context] Normalized %d/%d non-Message items in assembled messages",
            len(messages) - len(normalized),
            len(messages),
        )
    messages = normalized

    # Log KV prefix fingerprint for observability (system-prompt head only;
    # the full-payload append-only classification is recorded below).
    if orchestrator._kv_optimization_enabled:
        logger.debug(
            "[kv-cache] prefix=%s frozen=%s tools_cached=%s",
            orchestrator._kv_prefix_fingerprint(),
            getattr(orchestrator, "_system_prompt_frozen", False),
            getattr(orchestrator, "_last_sorted_tool_names", None) is not None,
        )

    # Canonical per-turn prompt optimizations live on the prompt pipeline.
    # KV-capable providers additionally benefit from stable system-prompt
    # caching, but Tier C providers still need the same dynamic guidance.
    # KV prefix stability: the turn prefix is frozen per user turn (anchor
    # identity) by the turn-prefix runtime; mid-loop failure signals are
    # appended as hidden messages rather than re-rendering sent bytes.
    prefix = ""
    pipeline = getattr(orchestrator, "_prompt_pipeline", None)
    if messages and pipeline:
        from victor.agent.turn_prefix_runtime import compose_turn_prefix_for_call

        prefix = compose_turn_prefix_for_call(
            orchestrator,
            messages,
            current_query=current_query,
            goals=goals,
            selected_tools=selected_tools,
            planned_tools=planned_tools,
        )

    final_messages = apply_turn_prefix(messages, prefix)

    # Full-payload prefix-stability classification: within a user turn the
    # serialized payload must extend the previous call's bytes so provider
    # prefix caching holds; sanctioned boundaries are announced via the
    # tracker and logged once. Observability must never break the call.
    # (Inline lazy wiring: the facade ratchet forbids new probe methods.)
    if orchestrator._kv_optimization_enabled:
        try:
            tracker = getattr(orchestrator, "_prefix_call_tracker", None)
            if tracker is None:
                from victor.agent.kv_prefix_observability import PrefixCallTracker

                tracker = PrefixCallTracker()
                orchestrator._prefix_call_tracker = tracker
            tracker.record(
                final_messages,
                tools=getattr(orchestrator, "_last_sorted_tool_names", None),
            )
        except Exception:  # noqa: BLE001 - diagnostics only
            logger.debug("[kv-cache] prefix tracker skipped (record failed)", exc_info=True)

    return final_messages

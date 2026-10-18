"""Turn-prefix composition runtime (KV prefix stability).

Owns the per-user-turn lifecycle of the composed turn prefix so the
orchestrator facade stays thin:

- The anchor is the identity of the last genuine user message. Intra-loop
  provider calls (same anchor) REUSE the frozen prefix bytes — re-composing
  per call mutated the bytes of the already-sent anchor message and defeated
  provider prefix caching (measured: cache hits frozen at the tools+system
  head while input grew 9x).
- Mid-loop failure signals are delivered as appended hidden user messages
  (the proven verification-feedback channel) instead of re-rendering the
  prefix; the next call carries them in its append-only tail.
- A new anchor recomposes (sanctioned single-miss boundary), announced to the
  KV prefix tracker so the divergence is classified, not mysterious.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from victor.agent.conversation.history_metadata import is_user_turn

if TYPE_CHECKING:
    from victor.providers.base import Message

logger = logging.getLogger(__name__)


def turn_anchor(messages: List["Message"]) -> Optional[int]:
    """Identity of the last genuine user message (intra-turn stability key).

    Hidden runtime messages (``interactive_history=False``) are not genuine
    user turns, so appended guidance never moves the anchor.
    """
    for message in reversed(messages):
        if is_user_turn(message.role, message.metadata):
            return id(message)
    return None


def _append_midturn_failure_hint(orch: Any, injector: Any) -> None:
    """Deliver a raised failure signal as an appended hidden message."""
    hint = None
    try:
        hint = injector.get_failure_hint(
            injector._last_failure_category,
            injector._last_failure_error,
        )
    except Exception:  # noqa: BLE001 - hint is best-effort
        hint = None
    if hint:
        try:
            from victor.agent.conversation.history_metadata import (
                build_internal_history_metadata,
            )

            # Appended to history (hidden from interactive view); the NEXT
            # provider call carries it in its payload tail — append-only, no
            # already-sent bytes change.
            orch.conversation.add_message(
                "user",
                hint,
                metadata=build_internal_history_metadata(kind="failure_hint"),
            )
        except Exception:  # noqa: BLE001 - never break assembly
            logger.debug("mid-turn failure hint append failed", exc_info=True)
    injector._last_failure_category = None
    injector._last_failure_error = None


def compose_turn_prefix_for_call(
    orch: Any,
    messages: List["Message"],
    *,
    current_query: Optional[str],
    goals: Optional[list] = None,
    selected_tools: Optional[list] = None,
    planned_tools: Optional[list] = None,
) -> str:
    """Compose (or reuse) the turn prefix for one provider call."""
    anchor = turn_anchor(messages)
    anchor_unchanged = (
        anchor is not None
        and anchor == getattr(orch, "_frozen_turn_prefix_anchor", None)
        and getattr(orch, "_frozen_turn_prefix", None) is not None
    )

    pipeline = getattr(orch, "_prompt_pipeline", None)
    if pipeline is None:
        return ""

    if anchor_unchanged:
        injector = getattr(orch, "_optimization_injector", None)
        if injector is not None and injector._last_failure_category is not None:
            _append_midturn_failure_hint(orch, injector)
        return orch._frozen_turn_prefix

    from victor.agent.prompt_pipeline import TurnContext, coerce_prompt_overlays

    injector = getattr(orch, "_optimization_injector", None)
    turn_ctx = TurnContext(
        provider_name=orch.provider_name or "",
        model=getattr(orch, "_model", "") or "",
        task_type=getattr(orch, "_current_task_type", "default"),
        active_skill_prompt=(
            orch.get_skill_user_prefix() if hasattr(orch, "get_skill_user_prefix") else None
        ),
        last_turn_failed=(injector._last_failure_category is not None if injector else False),
        last_failure_category=(injector._last_failure_category if injector else None),
        last_failure_error=(injector._last_failure_error if injector else None),
        task_guidance_text=getattr(orch, "_dynamic_task_guidance", None),
    )
    runtime_context_overrides = getattr(orch, "_runtime_tool_context_overrides", None)
    if isinstance(runtime_context_overrides, dict):
        turn_ctx.prompt_overlays = coerce_prompt_overlays(
            runtime_context_overrides.get("prompt_overlays")
        )

    reminder_mgr = getattr(orch, "reminder_manager", None)
    if reminder_mgr:
        turn_ctx.reminder_text = reminder_mgr.get_consolidated_reminder()

    # Last genuine user message text for KNN few-shot matching
    last_user_msg = ""
    for message in reversed(messages):
        if is_user_turn(message.role, message.metadata):
            last_user_msg = message.content[:200]
            break

    prompt_runtime = orch._get_prompt_builder_runtime()
    turn_ctx.dynamic_tool_guidance = prompt_runtime.build_dynamic_tool_guidance(
        current_query or last_user_msg,
        goals=goals,
        planned_tools=planned_tools,
        selected_tools=selected_tools,
    )

    prefix = pipeline.compose_turn_prefix(last_user_msg, turn_ctx)
    orch._record_prompt_optimization_metadata(turn_ctx)

    # Clear failure state after injecting hint
    if injector and turn_ctx.last_turn_failed:
        injector._last_failure_category = None
        injector._last_failure_error = None

    orch._frozen_turn_prefix = prefix
    orch._frozen_turn_prefix_anchor = anchor
    _note_user_turn_boundary(orch, anchor)
    return prefix


def _note_user_turn_boundary(orch: Any, anchor: Optional[int]) -> None:
    """Classify anchor changes as user-turn boundaries for the KV tracker."""
    if anchor is None:
        return
    if anchor != getattr(orch, "_last_recorded_turn_anchor", None):
        tracker = getattr(orch, "_prefix_call_tracker", None)
        if tracker is not None:
            tracker.note_boundary("user_turn")
        orch._last_recorded_turn_anchor = anchor

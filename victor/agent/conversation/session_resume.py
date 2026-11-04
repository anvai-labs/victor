# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Cross-visit session resume: hydrate the live conversation from storage.

The framework keeps two message stores that ``reset_conversation`` clears
separately — the ``ContextService`` buffer and the ``ConversationController``
history (the streaming turn loop reads the latter's captured reference). A
correct resume must repopulate BOTH, or the agent "looks resumed" but does not
recall the prior turns through the executor. This pure helper does exactly
that, kept out of any hotspot module so it stays independently testable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def hydrate_session(
    context_service: Any,
    conversation_controller: Any,
    session_data: Dict[str, Any],
    orchestrator: Any = None,
) -> Dict[str, Any]:
    """Repopulate both live message stores from a stored session.

    Symmetric to ``ChatService.reset_conversation`` (which clears both): the
    context buffer is cleared and re-filled, and the controller's history is
    repointed at the stored session, so every read path recalls the prior
    turns.

    Args:
        context_service: The live ``ContextService`` (needs ``clear_messages``
            / ``add_messages``).
        conversation_controller: The live ``ConversationController`` (needs
            ``load_history``); tolerated as ``None`` / missing the method.
        session_data: A stored-session dict from ``load_session`` — carries
            ``conversation`` (a ``MessageHistory`` dict) and ``metadata``.
        orchestrator: Optional orchestrator instance for refreshing system prompt
            with current project context. When provided, the system prompt is
            rebuilt to include the current workspace's AGENTS.md/CLAUDE.md/etc.

    Returns:
        The session ``metadata`` (title, message_count, …) for UI display.
    """
    from victor.agent.message_history import MessageHistory

    conversation = session_data.get("conversation", {}) or {}
    history = MessageHistory.from_dict(conversation)
    non_system = [m for m in history.messages if getattr(m, "role", None) != "system"]

    # Context buffer: clear (retaining system) then re-add the stored turns.
    if context_service is not None:
        context_service.clear_messages(retain_system=True)
        if non_system:
            context_service.add_messages(non_system)

    # Conversation controller: repoint its history at the stored session.
    if conversation_controller is not None and hasattr(conversation_controller, "load_history"):
        conversation_controller.load_history(history)

        # Re-anchor token tracking after resume. The controller's token accounting
        # state (_last_actual_prompt_tokens, _session_prompt_tokens, etc.) is reset
        # on init, so after restoring a large conversation we need to estimate the
        # current token count to ensure compaction triggers correctly.
        # Without this, a 125k/128k session appears as 0 utilization and compaction
        # never fires.
        try:
            total_chars = sum(len(m.content) for m in history.messages)
            if total_chars > 0:
                # Estimate tokens using the controller's chars_per_token_estimate
                # (default 3.5 chars/token). This is rough but sufficient to get
                # utilization into the right ballpark for compaction decisions.
                chars_per_token = getattr(
                    conversation_controller.config, "chars_per_token_estimate", 3.5
                )
                estimated_tokens = total_chars // max(1, int(chars_per_token))
                # Use record_actual_usage to anchor the token tracking. We pass
                # the estimated tokens as if they were real prompt_tokens from
                # the API. This sets _last_actual_prompt_tokens and calibrates
                # the chars-per-token ratio for future projections.
                conversation_controller.record_actual_usage(
                    prompt_tokens=estimated_tokens, total_chars=total_chars
                )
                # Also seed _session_prompt_tokens so the budget-aware spend
                # trigger in _should_compact() sees the cumulative volume.
                conversation_controller._session_prompt_tokens = estimated_tokens
                logger.info(
                    "Re-anchored token tracking on resume: ~%d tokens from %d chars",
                    estimated_tokens,
                    total_chars,
                )
        except Exception as e:
            logger.warning(f"Failed to re-anchor token tracking on resume: {e}")

    # Refresh system prompt with current project context (AGENTS.md, CLAUDE.md, etc.)
    # The stored session may have been created in a different workspace, so we need
    # to rebuild the system prompt with the current project's instructions.
    if orchestrator is not None:
        try:
            prompt_builder_runtime = getattr(orchestrator, "_prompt_builder_runtime", None)
            if prompt_builder_runtime is None:
                prompt_builder_runtime = getattr(orchestrator, "prompt_builder_runtime", None)

            if prompt_builder_runtime and hasattr(
                prompt_builder_runtime, "ensure_system_prompt_current"
            ):
                prompt_builder_runtime.ensure_system_prompt_current()
                logger.info("Refreshed system prompt with current project context on resume")
            elif hasattr(orchestrator, "build_system_prompt"):
                # Fallback: rebuild and set the system prompt directly
                new_system_prompt = orchestrator.build_system_prompt()
                if conversation_controller and hasattr(
                    conversation_controller, "set_system_prompt"
                ):
                    conversation_controller.set_system_prompt(new_system_prompt)
                    conversation_controller.ensure_system_message()
                    logger.info("Rebuilt system prompt with current project context on resume")
        except Exception as e:
            logger.warning(f"Failed to refresh system prompt on resume: {e}")

    logger.debug("Resumed session with %d prior messages", len(non_system))
    return session_data.get("metadata", {}) or {}

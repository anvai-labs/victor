# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

    logger.debug("Resumed session with %d prior messages", len(non_system))
    return session_data.get("metadata", {}) or {}

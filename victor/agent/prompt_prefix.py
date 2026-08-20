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

"""Placement of the composed turn prefix onto an assembled message list.

``UnifiedPromptPipeline.compose_turn_prefix`` builds the per-turn guidance block;
this puts it where the provider will see it. Keeping placement next to
composition means the two rules that matter live in one place: it rides the
*last user* message (so it stays outside the cached system prefix), and it must
not disturb anything else about that message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from victor.providers.base import Message

__all__ = ["apply_turn_prefix"]


def apply_turn_prefix(messages: List["Message"], prefix: str) -> List["Message"]:
    """Prepend ``prefix`` to the last user message, in place.

    Copies rather than reconstructs. Building a fresh ``Message`` here discarded
    everything except role and content — metadata (and with it ``MessageSource``,
    which compaction scoring keys on), ``name``, ``tool_calls`` and
    ``tool_call_id`` — from the message it replaced.

    Args:
        messages: Assembled provider messages. Returned unchanged when there is
            no prefix, or no user message to carry it.
        prefix: Composed turn prefix, already framed by the caller.

    Returns:
        The same list, with the last user message's content prefixed.
    """
    if not prefix:
        return messages

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            messages[i] = messages[i].model_copy(update={"content": prefix + messages[i].content})
            break

    return messages

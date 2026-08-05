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

"""Unit tests for ConversationLog assistant-reply capture (copy support).

The app's ``ctrl+c`` action copies the last assistant reply when nothing is
selected; these tests pin the plain-text accumulation that feeds it. They use an
unmounted widget (no running app) — ``begin_turn("")`` writes nothing and TOKEN
actions only buffer text, so no Textual render is triggered.
"""

from __future__ import annotations

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind
from victor.ui.tui.conversation import ConversationLog


def _token(text: str) -> RenderAction:
    return RenderAction(kind=RenderKind.TOKEN, text=text)


def test_last_response_text_accumulates_tokens() -> None:
    log = ConversationLog()
    log.begin_turn("")
    log.feed_action(_token("Hello, "))
    log.feed_action(_token("world."))
    assert log.last_response_text() == "Hello, world."


def test_last_response_text_ignores_non_token_actions() -> None:
    log = ConversationLog()
    log.begin_turn("")
    log.feed_action(_token("answer"))
    # Reasoning/thinking must not leak into the copied reply.
    log.feed_action(RenderAction(kind=RenderKind.THINKING, text="secret chain of thought"))
    assert log.last_response_text() == "answer"


def test_begin_turn_resets_response_text() -> None:
    log = ConversationLog()
    log.begin_turn("")
    log.feed_action(_token("stale"))
    log.begin_turn("")
    assert log.last_response_text() == ""


def test_last_response_text_empty_before_any_tokens() -> None:
    log = ConversationLog()
    log.begin_turn("")
    assert log.last_response_text() == ""

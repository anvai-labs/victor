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

"""Natural per-iteration tool-flow segmentation contract (no chainlit dependency)."""

from __future__ import annotations

from victor.ui.chat_app.event_mapping import RenderKind, segment_turn

K = RenderKind


def test_interleaved_text_tools_text():
    # Two iterations: text → (parallel tools) → text → (parallel tools) → final text.
    kinds = [
        K.TOKEN,
        K.TOKEN,
        K.TOOL_START,
        K.TOOL_END,
        K.TOKEN,
        K.TOOL_START,
        K.TOOL_START,
        K.TOOL_END,
        K.TOOL_END,
        K.TOKEN,
    ]
    assert segment_turn(kinds) == ["text", "tools", "text", "tools", "text"]


def test_thinking_and_ignore_do_not_open_segments():
    kinds = [K.THINKING, K.IGNORE, K.TOKEN, K.IGNORE, K.TOOL_START, K.TOOL_END]
    assert segment_turn(kinds) == ["text", "tools"]


def test_contiguous_text_is_one_segment():
    assert segment_turn([K.TOKEN, K.TOKEN, K.TOKEN]) == ["text"]


def test_tools_only():
    assert segment_turn([K.TOOL_START, K.TOOL_END]) == ["tools"]


def test_empty_turn():
    assert segment_turn([]) == []


def test_error_starts_a_text_segment_after_tools():
    assert segment_turn([K.TOOL_END, K.ERROR]) == ["tools", "text"]

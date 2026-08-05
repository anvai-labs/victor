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

"""P1 trust/transparency: call_id correlation + informed approval prompts (no chainlit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from victor.ui.chat_app.approval import _argument_preview, _format_prompt
from victor.ui.chat_app.event_mapping import RenderKind, map_event
from victor.framework.hitl import ApprovalRequest


@dataclass
class FakeEvent:
    event_type: Any
    content: Optional[str] = None
    tool_name: Optional[str] = None
    result: Any = None
    metadata: Optional[Dict[str, Any]] = None


# --- call_id correlation (per-call tool steps) ---


def test_tool_call_carries_call_id_from_metadata():
    action = map_event(
        FakeEvent(
            "tool_call",
            tool_name="bash",
            metadata={"tool_call_id": "call_42", "arguments": {}},
        )
    )
    assert action.kind is RenderKind.TOOL_START
    assert action.call_id == "call_42"


def test_tool_result_carries_call_id_from_result_payload():
    action = map_event(
        FakeEvent("tool_result", tool_name="bash", result={"id": "call_42", "result": "ok"})
    )
    assert action.kind is RenderKind.TOOL_END
    assert action.call_id == "call_42"


def test_call_id_absent_is_none():
    assert map_event(FakeEvent("tool_call", tool_name="bash")).call_id is None


# --- informed approval prompts ---


def _req(tool: str, arguments: dict) -> ApprovalRequest:
    return ApprovalRequest(
        id="r1",
        title=f"Approve tool: {tool}",
        description="Policy requests approval.",
        context={"tool_name": tool, "arguments": arguments},
    )


def test_approval_shows_bash_command():
    out = _format_prompt(_req("bash", {"command": "rm -rf build"}))
    assert "```bash" in out and "rm -rf build" in out
    assert "`bash`" in out


def test_approval_shows_diff_for_edit():
    out = _format_prompt(_req("edit_file", {"diff": "- old\n+ new"}))
    assert "```diff" in out and "+ new" in out


def test_approval_shows_content_for_write():
    out = _format_prompt(_req("write_file", {"path": "x.py", "content": "print('hi')"}))
    assert "print('hi')" in out


def test_approval_generic_tool_shows_arg_summary():
    out = _argument_preview("some_tool", {"q": "hello"})
    assert "q" in out and "hello" in out


def test_approval_no_args_is_safe():
    out = _format_prompt(_req("bash", {}))
    assert "`bash`" in out  # still shows the tool, no crash

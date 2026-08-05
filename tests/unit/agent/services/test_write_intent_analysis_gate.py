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

"""WRITE_ALLOWED must not force mutation tools onto read-oriented analysis tasks."""

from __future__ import annotations

from types import SimpleNamespace

from victor.agent.action_authorizer import ActionIntent
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _runtime(task_type: str) -> ToolSelectionRuntime:
    return ToolSelectionRuntime(SimpleNamespace(_current_task_type=task_type))


def test_analysis_task_does_not_force_mutation_tools():
    rt = _runtime("analyze")
    tools = [_tool("read"), _tool("ls")]
    out = rt._ensure_write_tools_for_write_intent(tools, ActionIntent.WRITE_ALLOWED)
    assert out == tools  # unchanged — no edit/write/shell injected


def test_search_and_research_tasks_also_skipped():
    for task_type in ("search", "research"):
        rt = _runtime(task_type)
        tools = [_tool("read")]
        out = rt._ensure_write_tools_for_write_intent(tools, ActionIntent.WRITE_ALLOWED)
        assert out == tools


def test_non_write_intent_returns_unchanged():
    rt = _runtime("general")
    tools = [_tool("read")]
    assert rt._ensure_write_tools_for_write_intent(tools, ActionIntent.READ_ONLY) == tools


def test_action_task_still_force_adds_mutation_tools(monkeypatch):
    rt = _runtime("general")  # not a read-oriented type
    monkeypatch.setattr(
        rt,
        "_available_tool_defs_by_name",
        lambda: {
            "edit": _tool("edit"),
            "write": _tool("write"),
            "shell": _tool("shell"),
        },
    )
    out = rt._ensure_write_tools_for_write_intent([_tool("read")], ActionIntent.WRITE_ALLOWED)
    names = {ToolSelectionRuntime._tool_name(t) for t in out}
    assert {"edit", "write", "shell"} <= names

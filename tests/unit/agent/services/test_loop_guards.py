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

"""Regression tests for agentic-loop guards.

Covers two defects that let a pure-analysis run spin for 15 iterations:
  - write-intent forced edit/write/shell onto read-oriented turns because the guard read
    a never-assigned attribute (``_current_task_type``);
  - the streaming spin detector was never fed tool-call signatures, so a model re-issuing
    the same tool call every turn was not detected.
"""

from types import SimpleNamespace

from victor.agent.action_authorizer import ActionIntent
from victor.agent.services.chat_stream_executor import (
    _count_productive_tools,
    _tool_call_signatures,
)
from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime
from victor.agent.streaming.tool_execution import ToolExecutionResult
from victor.agent.turn_policy import SpinDetector, SpinState

# --- plateau accounting reads the streaming ToolExecutionResult shape ------------------


def test_count_productive_tools_from_streaming_result():
    """Regression: the streaming ToolExecutionResult exposes tool_results, NOT a
    successful_tool_count property — counting must not raise AttributeError."""
    result = ToolExecutionResult()
    result.tool_results = [{"success": True}, {"success": False}, {"success": True}]
    result.tool_calls_executed = 3
    assert _count_productive_tools(result) == 2  # only the successful results


def test_count_productive_tools_empty_and_none():
    assert _count_productive_tools(None) == 0
    assert _count_productive_tools(ToolExecutionResult()) == 0  # no tools ran


def test_count_productive_tools_falls_back_to_executed_count():
    """When no per-result detail is available, fall back to whether any tool executed."""
    obj = SimpleNamespace(tool_calls_executed=2)  # no tool_results attribute
    assert _count_productive_tools(obj) == 2


def _runtime(**host_attrs) -> ToolSelectionRuntime:
    return ToolSelectionRuntime(OrchestratorProtocolAdapter(SimpleNamespace(**host_attrs)))


# --- #3 write-intent guard reads a populated task type ---------------------------------


def test_current_task_type_reads_unified_tracker():
    rt = _runtime(unified_tracker=SimpleNamespace(task_type="analyze"))
    assert rt._current_task_type() == "analyze"


def test_current_task_type_reads_enum_value():
    rt = _runtime(unified_tracker=SimpleNamespace(task_type=SimpleNamespace(value="research")))
    assert rt._current_task_type() == "research"


def test_current_task_type_empty_when_unset():
    # No unified_tracker and no legacy attribute -> empty string, never raises.
    rt = _runtime(unified_tracker=None)
    assert rt._current_task_type() == ""


def test_write_tools_not_forced_on_analysis_turn():
    """An analysis turn must not get edit/write/shell injected even when WRITE_ALLOWED."""
    rt = _runtime(unified_tracker=SimpleNamespace(task_type="analyze"))
    tools = [SimpleNamespace(name="read"), SimpleNamespace(name="code_search")]
    result = rt._ensure_write_tools_for_write_intent(tools, ActionIntent.WRITE_ALLOWED)
    assert result is tools  # unchanged: no mutation tools appended


def test_write_tools_not_forced_on_search_turn():
    rt = _runtime(unified_tracker=SimpleNamespace(task_type="search"))
    tools = [SimpleNamespace(name="read")]
    assert rt._ensure_write_tools_for_write_intent(tools, ActionIntent.WRITE_ALLOWED) is tools


# --- #4 streaming spin detector is fed tool-call signatures ----------------------------


def test_tool_call_signatures_stable_and_repeatable():
    calls = [{"name": "code_search", "arguments": {"query": "foo", "mode": "semantic"}}]
    # Argument order must not change the signature.
    calls_reordered = [{"name": "code_search", "arguments": {"mode": "semantic", "query": "foo"}}]
    assert _tool_call_signatures(calls) == _tool_call_signatures(calls_reordered)
    # Different args -> different signature.
    other = [{"name": "code_search", "arguments": {"query": "bar"}}]
    assert _tool_call_signatures(calls) != _tool_call_signatures(other)
    assert _tool_call_signatures(None) == set()


def test_repeated_signatures_terminate_spin_detector():
    """The signatures the executor now feeds drive the SpinDetector to TERMINATED."""
    detector = SpinDetector()
    sig = _tool_call_signatures(
        [{"name": "code_search", "arguments": {"query": "Jensen Huang co-design"}}]
    )
    detector.record_turn(has_tool_calls=True, tool_count=1, tool_signatures=sig)
    assert detector.state != SpinState.TERMINATED
    detector.record_turn(has_tool_calls=True, tool_count=1, tool_signatures=sig)
    detector.record_turn(has_tool_calls=True, tool_count=1, tool_signatures=sig)
    # repetition_threshold=3 -> terminate once the same signature repeats across 3 turns.
    assert detector.state == SpinState.TERMINATED

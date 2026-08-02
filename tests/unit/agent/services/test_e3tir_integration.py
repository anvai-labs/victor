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

"""E3-TIR exploration reranker integration at the tool-selection seam."""

from types import SimpleNamespace

import pytest

import victor.agent.services.tool_selection_runtime as tsr
from victor.agent.services.tool_selection_runtime import (
    ToolSelectionRuntime,
    _get_e3tir_shared_store,
    _reset_e3tir_state_for_tests,
)
from victor.tools.e3_tir_selector import E3TIRToolSelector


def _tool(name):
    return SimpleNamespace(name=name)


def _tools():
    return [_tool("read"), _tool("write"), _tool("shell"), _tool("grep")]


@pytest.fixture(autouse=True)
def _reset():
    _reset_e3tir_state_for_tests()
    yield
    _reset_e3tir_state_for_tests()


def test_disabled_by_default_is_noop():
    """With the flag off (default), the seam is a pass-through and caches a None reranker."""
    runtime = SimpleNamespace(_kv_optimization_enabled=False, _current_task_type="general")
    seam = ToolSelectionRuntime(runtime)
    out = seam._apply_e3tir_exploration(_tools(), "do the thing")
    assert [t.name for t in out] == ["read", "write", "shell", "grep"]
    assert runtime._e3tir_reranker is None  # cached "disabled" result


def test_kv_provider_skips_exploration():
    """KV-prefix-caching providers must keep a stable tool order — E3-TIR is skipped."""
    runtime = SimpleNamespace(_kv_optimization_enabled=True, _current_task_type="general")
    # Even with a live reranker present, KV stability wins.
    runtime._e3tir_reranker = E3TIRToolSelector()
    out = ToolSelectionRuntime(runtime)._apply_e3tir_exploration(_tools(), "x")
    assert [t.name for t in out] == ["read", "write", "shell", "grep"]


def test_reranker_reorders_without_losing_tools():
    """When enabled (non-KV), E3-TIR reranks the set but never drops a tool."""
    runtime = SimpleNamespace(
        _kv_optimization_enabled=False,
        _current_task_type="general",
        _e3tir_reranker=E3TIRToolSelector(),
    )
    out = ToolSelectionRuntime(runtime)._apply_e3tir_exploration(_tools(), "x")
    assert sorted(t.name for t in out) == ["grep", "read", "shell", "write"]


def test_empty_tools_is_safe():
    runtime = SimpleNamespace(_kv_optimization_enabled=False, _current_task_type="general")
    assert ToolSelectionRuntime(runtime)._apply_e3tir_exploration([], "x") == []
    assert ToolSelectionRuntime(runtime)._apply_e3tir_exploration(None, "x") is None


def test_outcome_hook_feeds_shared_store():
    """A TOOL_EXECUTED RL event must flow into the shared experience store, so the
    reranker learns from real tool outcomes."""
    from victor.framework.rl.hooks import RLEvent, RLEventType, get_rl_hooks

    store = _get_e3tir_shared_store()  # subscribes the outcome hook
    assert len(store) == 0

    # The hook registry only dispatches to custom handlers once an RL coordinator is set
    # (in production this is bootstrapped by use_learning_from_execution, which E3-TIR
    # requires). Provide a minimal stub so emit() reaches the custom handler.
    get_rl_hooks().set_coordinator(SimpleNamespace(is_closed=False))

    get_rl_hooks().emit(
        RLEvent(
            type=RLEventType.TOOL_EXECUTED,
            tool_name="code_search",
            success=True,
            quality_score=0.9,
            task_type="coding",
        )
    )

    stats = store.get_stats("code_search")
    assert stats.total_uses >= 1
    assert stats.successes >= 1


def test_shared_store_is_singleton_within_process():
    a = _get_e3tir_shared_store()
    b = _get_e3tir_shared_store()
    assert a is b

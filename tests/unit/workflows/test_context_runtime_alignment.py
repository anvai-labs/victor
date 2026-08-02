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

"""Runtime-alignment checks for legacy workflow context compatibility."""

from victor.workflows import context as context_module


def test_create_execution_context_uses_shared_runtime_defaults() -> None:
    ctx = context_module.create_execution_context(
        {"input": "value"},
        workflow_id="wf-123",
        workflow_name="example_workflow",
    )

    # Pydantic models use attribute access, not subscript
    assert ctx.data == {"input": "value"}
    assert ctx.workflow_id == "wf-123"
    assert ctx.workflow_name == "example_workflow"
    assert ctx.current_node == ""
    assert ctx.node_results == {}
    assert ctx.parallel_results == {}
    assert ctx.hitl_pending is False
    assert ctx.hitl_response is None


def test_from_compiler_workflow_state_preserves_workflow_name() -> None:
    ctx = context_module.from_compiler_workflow_state(
        {
            "_workflow_id": "wf-123",
            "_workflow_name": "compiler_workflow",
            "_current_node": "step1",
            "_node_results": {"step1": {"success": True}},
            "_parallel_results": {"branch": {"done": True}},
            "_hitl_pending": True,
            "_hitl_response": {"approved": True},
            "input": "value",
        }
    )

    assert ctx.data == {"input": "value"}
    assert ctx.workflow_id == "wf-123"
    assert ctx.workflow_name == "compiler_workflow"
    assert ctx.current_node == "step1"
    assert ctx.node_results == {"step1": {"success": True}}
    assert ctx.parallel_results == {"branch": {"done": True}}
    assert ctx.hitl_pending is True
    assert ctx.hitl_response == {"approved": True}
    assert ctx.visited_nodes == ["step1"]


def test_to_compiler_workflow_state_includes_workflow_name() -> None:
    ctx = context_module.create_execution_context(
        {"input": "value"},
        workflow_id="wf-123",
        workflow_name="context_workflow",
    )
    ctx.current_node = "step1"

    state = context_module.to_compiler_workflow_state(ctx)

    assert state["input"] == "value"
    assert state["_workflow_id"] == "wf-123"
    assert state["_workflow_name"] == "context_workflow"
    assert state["_current_node"] == "step1"

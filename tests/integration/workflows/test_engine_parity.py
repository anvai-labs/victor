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

"""Engine-parity battery — ADR-030 (single graph execution engine).

Runs the same ``WorkflowDefinition`` through both engines — the BFS
``CompiledWorkflowExecutor`` and the ``StateGraphExecutor``/CompiledGraph
path — and pins result equivalence on the scenarios where the engines are
required to agree:

- linear transform chains,
- condition branches (both arms),
- parallel fan-out with all children succeeding (join=all/any/merge),
- failure propagation (a raising transform fails the run),
- ``join=all`` with a failing child (both engines must fail the run).

Known, intended divergence (NOT pinned as parity): under ``join=any`` /
``join=merge`` / ``join=first`` with a *partial* child failure, the two
engines currently disagree — the BFS walker fails the whole run via
``context.has_failures()`` even though its own join policy completed the
group, while the compiled path lets the join policy decide (U6-F5's
"let join policy decide" remedy). ADR-030 deletes the BFS walker, so the
battery pins the *desired* survivor semantics for those cases instead of
enforcing equivalence with soon-to-be-deleted behavior.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
)
from victor.workflows.unified_executor import (
    CompiledWorkflowExecutor,
    StateGraphExecutor,
)

pytestmark = [pytest.mark.integration, pytest.mark.workflows]


def _orchestrator():
    orchestrator = MagicMock()
    orchestrator.settings = MagicMock()
    return orchestrator


def _normalized_bfs(result) -> Dict[str, Any]:
    """Normalize a BFS ``WorkflowResult`` to the common parity shape."""
    return {
        "success": result.success,
        "state": dict(result.context.data),
        "error": result.error,
        "nodes": set(result.context.node_results),
    }


def _normalized_sg(result) -> Dict[str, Any]:
    """Normalize a compiled-path ``ExecutorResult`` to the common parity shape."""
    return {
        "success": result.success,
        "state": dict(result.state),
        "error": result.error,
        "nodes": set(result.nodes_executed),
    }


async def _run_both(workflow: WorkflowDefinition, initial_context: Dict[str, Any]):
    orchestrator = _orchestrator()
    bfs = _normalized_bfs(
        await CompiledWorkflowExecutor(orchestrator).execute(workflow, dict(initial_context))
    )
    sg = _normalized_sg(
        await StateGraphExecutor(orchestrator).execute(workflow, dict(initial_context))
    )
    return bfs, sg


def _linear() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="parity_linear",
        start_node="double",
        nodes={
            "double": TransformNode(
                id="double",
                name="Double",
                transform=lambda ctx: {**ctx, "v": ctx["v"] * 2},
                next_nodes=["add"],
            ),
            "add": TransformNode(
                id="add",
                name="Add",
                transform=lambda ctx: {**ctx, "w": ctx.get("w", 0) + 5},
            ),
        },
    )


def _condition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="parity_condition",
        start_node="check",
        nodes={
            "check": ConditionNode(
                id="check",
                name="Check",
                condition=lambda ctx: "yes" if ctx["flag"] else "no",
                branches={"yes": "yes_node", "no": "no_node"},
            ),
            "yes_node": TransformNode(
                id="yes_node",
                name="Yes",
                transform=lambda ctx: {**ctx, "path": "yes"},
            ),
            "no_node": TransformNode(
                id="no_node",
                name="No",
                transform=lambda ctx: {**ctx, "path": "no"},
            ),
        },
    )


def _parallel(strategy: str, b_raises: bool) -> WorkflowDefinition:
    def ok_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {**ctx, "b": 2}

    def bad_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("child b failed")

    return WorkflowDefinition(
        name=f"parity_parallel_{strategy}",
        start_node="fan",
        nodes={
            "fan": ParallelNode(
                id="fan",
                name="Fan",
                parallel_nodes=["a", "b"],
                join_strategy=strategy,
            ),
            "a": TransformNode(id="a", name="A", transform=lambda ctx: {**ctx, "a": 1}),
            "b": TransformNode(id="b", name="B", transform=bad_b if b_raises else ok_b),
        },
    )


def _parallel_all_raise(strategy: str) -> WorkflowDefinition:
    def raise_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("child a failed")

    def raise_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("child b failed")

    return WorkflowDefinition(
        name=f"parity_parallel_all_raise_{strategy}",
        start_node="fan",
        nodes={
            "fan": ParallelNode(
                id="fan",
                name="Fan",
                parallel_nodes=["a", "b"],
                join_strategy=strategy,
            ),
            "a": TransformNode(id="a", name="A", transform=raise_a),
            "b": TransformNode(id="b", name="B", transform=raise_b),
        },
    )


def _raising() -> WorkflowDefinition:
    def boom(ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("node exploded")

    return WorkflowDefinition(
        name="parity_raising",
        start_node="boom",
        nodes={"boom": TransformNode(id="boom", name="Boom", transform=boom)},
    )


class TestEngineParityAgreement:
    """Scenarios where both engines must produce equivalent outcomes."""

    @pytest.mark.asyncio
    async def test_linear_chain(self):
        bfs, sg = await _run_both(_linear(), {"v": 3})
        assert bfs["success"] is sg["success"] is True
        assert bfs["state"] == sg["state"] == {"v": 6, "w": 5}
        assert bfs["nodes"] == sg["nodes"] == {"double", "add"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag,expected_path", [(True, "yes"), (False, "no")])
    async def test_condition_branches(self, flag: bool, expected_path: str):
        bfs, sg = await _run_both(_condition(), {"flag": flag})
        for outcome in (bfs, sg):
            assert outcome["success"] is True
            assert outcome["state"]["path"] == expected_path
        assert bfs["nodes"] == sg["nodes"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["all", "any", "merge"])
    async def test_parallel_all_children_succeed(self, strategy: str):
        bfs, sg = await _run_both(_parallel(strategy, b_raises=False), {})
        for outcome in (bfs, sg):
            assert outcome["success"] is True
            assert outcome["state"] == {"a": 1, "b": 2}
        # Engine shape difference (accepted): the BFS walker records each child
        # as its own node result; the compiled path executes the group as ONE
        # node, so children never enter node_history.
        assert "fan" in bfs["nodes"] and "fan" in sg["nodes"]
        assert sg["nodes"] == {"fan"}
        assert {"a", "b"} <= bfs["nodes"]

    @pytest.mark.asyncio
    async def test_raising_transform_fails_the_run(self):
        """Failure propagation parity: a raising transform node must fail the
        run on BOTH engines with the node's error surfaced."""
        bfs, sg = await _run_both(_raising(), {})
        for outcome in (bfs, sg):
            assert outcome["success"] is False
            assert "node exploded" in (outcome["error"] or "")

    @pytest.mark.asyncio
    async def test_parallel_join_all_with_failing_child_fails_the_run(self):
        """join=all with a failing child must fail the run on BOTH engines
        (the child's successful partial state may differ per engine's COW
        behavior, so only success and error content are pinned here)."""
        bfs, sg = await _run_both(_parallel("all", b_raises=True), {})
        for outcome in (bfs, sg):
            assert outcome["success"] is False
            assert "child b failed" in (outcome["error"] or "")


class TestJoinPolicyDecidesSurvivorSemantics:
    """The desired survivor semantics for partial parallel failures.

    Per U6-F5's remedy ("merge successful branch states, attach per-child
    errors, let join policy decide"), the compiled path lets the join policy
    decide the group outcome instead of failing the whole run. The BFS walker
    currently disagrees (it fails the run via ``context.has_failures()`` even
    when the join policy completed the group) — that behavior is deleted with
    the walker in ADR-030 step 3, so these assertions pin the survivor only.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["any", "merge", "first"])
    async def test_partial_failure_with_join_tolerance_succeeds(self, strategy: str):
        orchestrator = _orchestrator()
        result = await StateGraphExecutor(orchestrator).execute(
            _parallel(strategy, b_raises=True), {}
        )
        assert result.success is True, (
            f"join={strategy} tolerates a partial child failure; the run must "
            f"succeed and attach the per-child error (got error={result.error!r})"
        )
        assert result.state.get("a") == 1, "successful child's writes must survive"
        assert "b" not in result.state, "failed child's writes must not leak"

    @pytest.mark.asyncio
    async def test_join_first_all_children_fail(self):
        """join=first with ZERO successful children fails the group."""
        orchestrator = _orchestrator()
        result = await StateGraphExecutor(orchestrator).execute(_parallel_all_raise("first"), {})
        assert result.success is False
        assert "failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_join_all_partial_failure_still_fails_the_run(self):
        orchestrator = _orchestrator()
        result = await StateGraphExecutor(orchestrator).execute(_parallel("all", b_raises=True), {})
        assert result.success is False
        assert "child b failed" in (result.error or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
async def test_agent_node_compiled_execution_propagates_result(outcome: str):
    """Agent outcomes survive compilation, including failed returned results."""
    spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=outcome == "success",
            summary="review complete",
            error="agent rejected task" if outcome == "failure" else None,
            tool_calls_used=3,
        )
    )
    if outcome == "exception":
        spawn.side_effect = RuntimeError("agent spawn crashed")
    workflow = WorkflowDefinition(
        name="agent_result",
        start_node="agent",
        nodes={
            "agent": AgentNode(
                id="agent",
                name="Agent",
                role="researcher",
                goal="Review {{task}}",
                output_key="summary",
                next_nodes=["after"],
            ),
            "after": TransformNode(
                id="after", name="After", transform=lambda ctx: {"continued": True}
            ),
        },
    )
    with patch(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        return_value=SimpleNamespace(spawn=spawn),
    ):
        result = await StateGraphExecutor(_orchestrator()).execute(workflow, {"task": "repo"})
    spawn.assert_awaited_once()
    assert spawn.await_args.kwargs["task"] == "Review repo"
    assert result.success is (outcome == "success")
    assert result.node_results["agent"].success is (outcome == "success")
    if outcome == "success":
        assert result.state["summary"] == "review complete"
        assert result.state["continued"] is True
        assert result.node_results["agent"].tool_calls_used == 3
    else:
        assert "continued" not in result.state
        assert "summary" not in result.state
        expected = "agent rejected task" if outcome == "failure" else "agent spawn crashed"
        assert expected in result.error
        assert expected in result.node_results["agent"].error


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["all", "any", "first", "merge"])
async def test_parallel_result_metadata_respects_join_policy(strategy: str):
    result = await StateGraphExecutor(_orchestrator()).execute(
        _parallel(strategy, b_raises=True), {}
    )
    assert result.success is (strategy != "all")
    assert result.state["a"] == 1
    assert result.node_results["a"].success is True
    assert result.node_results["b"].success is False
    assert "child b failed" in result.node_results["b"].error
    assert result.node_results["fan"].success is (strategy != "all")


@pytest.mark.asyncio
async def test_parallel_join_only_considers_current_group():
    workflow = _parallel("any", b_raises=True)
    workflow.nodes["fan"].next_nodes = ["second"]
    workflow.nodes["second"] = ParallelNode(
        id="second", name="Second", parallel_nodes=["c"], join_strategy="all"
    )
    workflow.nodes["c"] = TransformNode(id="c", name="C", transform=lambda ctx: {"c": 3})
    result = await StateGraphExecutor(_orchestrator()).execute(workflow, {})
    assert result.success is True
    assert result.state["a"] == 1
    assert result.state["c"] == 3
    assert result.node_results["b"].success is False
    assert result.node_results["second"].success is True
    assert set(result.node_results["second"].output) == {"c"}


@pytest.mark.asyncio
async def test_parallel_merge_preserves_writes_and_deletions_from_successful_children():
    workflow = _parallel("all", b_raises=False)

    def change_a(ctx):
        ctx["nested"]["value"] = 2
        ctx.pop("removed")
        return {"counter": 1, "conflict": "a"}

    workflow.nodes["a"].transform = change_a
    workflow.nodes["b"].transform = lambda ctx: {"conflict": "b"}
    initial = {"counter": 0, "nested": {"value": 0}, "removed": True, "conflict": "old"}
    result = await StateGraphExecutor(_orchestrator()).execute(workflow, initial)
    assert result.success is True
    assert result.state == {"counter": 1, "nested": {"value": 2}, "conflict": "b"}
    assert initial["nested"] == {"value": 0}
    assert initial["removed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_values", [[1, 2], [float("nan"), 2]])
async def test_parallel_merge_accepts_array_state_without_ambiguous_equality(initial_values):
    np = pytest.importorskip("numpy")
    workflow = _parallel("all", b_raises=False)
    workflow.nodes["a"].transform = lambda ctx: {"array": np.array([3, 4])}
    workflow.nodes["b"].transform = lambda ctx: {"b": 2}
    result = await StateGraphExecutor(_orchestrator()).execute(
        workflow, {"array": np.array(initial_values)}
    )
    assert result.success is True
    np.testing.assert_array_equal(result.state["array"], [3, 4])


@pytest.mark.asyncio
@pytest.mark.parametrize("placeholder", ["{files}", "{{files}}"])
async def test_agent_node_interpolates_mapped_goal_without_formatting_literal_braces(placeholder):
    spawn = AsyncMock(
        return_value=SimpleNamespace(success=True, summary="done", error=None, tool_calls_used=0)
    )
    workflow = WorkflowDefinition(
        name="mapped_agent_goal",
        start_node="agent",
        nodes={
            "agent": AgentNode(
                id="agent",
                name="Agent",
                role="researcher",
                goal="Analyze " + placeholder + ' with {"literal": true} and {unknown}',
                input_mapping={"files": "paths"},
            )
        },
    )
    with patch(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        return_value=SimpleNamespace(spawn=spawn),
    ):
        result = await StateGraphExecutor(_orchestrator()).execute(workflow, {"paths": "foo.py"})
    assert result.success is True
    assert spawn.await_args.kwargs["task"] == (
        'Analyze foo.py with {"literal": true} and {unknown}'
    )


@pytest.mark.asyncio
async def test_transform_result_contains_returned_user_output():
    workflow = WorkflowDefinition(
        name="transform_output",
        start_node="transform",
        nodes={
            "transform": TransformNode(
                id="transform",
                name="Transform",
                transform=lambda ctx: {"answer": ctx["value"] * 2, "_internal": "hidden"},
            )
        },
    )
    result = await StateGraphExecutor(_orchestrator()).execute(workflow, {"value": 4})
    assert result.success is True
    assert result.node_results["transform"].output == {"answer": 8}


@pytest.mark.asyncio
async def test_agent_node_fails_without_an_orchestrator():
    workflow = WorkflowDefinition(
        name="missing_agent_runtime",
        start_node="agent",
        nodes={"agent": AgentNode(id="agent", name="Agent", role="researcher", goal="Review")},
    )
    result = await StateGraphExecutor().execute(workflow, {"input": 1})
    assert result.success is False
    assert "No orchestrator available" in result.error
    assert result.node_results["agent"].success is False
    assert "agent" not in result.state


async def test_definition_ordinary_successors_preserve_bfs_order_and_shared_descendant():
    definition = WorkflowDefinition(
        name="ordinary_successors",
        start_node="a",
        nodes={
            name: TransformNode(
                id=name,
                name=name,
                transform=lambda state, name=name: {"order": state.get("order", []) + [name]},
                next_nodes=successors,
            )
            for name, successors in {
                "a": ["b", "c"],
                "b": ["d"],
                "c": ["d"],
                "d": [],
            }.items()
        },
    )
    bfs, compiled = await _run_both(definition, {})
    assert bfs["success"] is compiled["success"] is True
    assert bfs["state"] == compiled["state"] == {"order": ["a", "b", "c", "d"]}
    assert bfs["nodes"] == compiled["nodes"] == {"a", "b", "c", "d"}

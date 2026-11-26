# Copyright 2026 Vijaykumar Singh <vijay@anaiops.com>
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

"""Async pipeline TDD for the planning runtime (team-adapter execution).

Drives the REAL execution loop — batch dispatch, evidence contracts,
retry, produces-flow into plan_state, blocked-dependency handling, and
the final summary — against a scripted team adapter, plus the plan-
generation override resolution and the chat_with_planning decision
paths.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from victor.agent.planning.base import PlanResult, StepResult
from victor.agent.planning.readable_schema import ReadableTaskPlan
from victor.agent.services.planning_runtime import (
    PlanningCoordinator,
    PlanningMode,
)


def _plan(steps=None) -> ReadableTaskPlan:
    return ReadableTaskPlan(
        name="pipeline plan",
        complexity="moderate",
        desc="drive the team pipeline",
        steps=steps
        or [
            [1, "research", "analyze the module", "read,grep", []],
            [2, "feature", "apply the change", "write", [1]],
        ],
    )


def _coordinator(planning_cfg=None):
    orchestrator = MagicMock()
    orchestrator.provider = MagicMock()
    orchestrator.model = "base-model"
    orchestrator.profile = None
    orchestrator.planning_model_override = None
    orchestrator.active_session_id = "session-root"
    renderer = MagicMock()
    renderer.console = MagicMock()
    coordinator = PlanningCoordinator(orchestrator, renderer=renderer)
    if planning_cfg is not None:
        coordinator.config = planning_cfg
    return coordinator, orchestrator


class _ScriptedAdapter:
    """Team adapter whose execute_step follows a per-step script."""

    def __init__(self, script):
        # script: {step_id: [StepResult, StepResult, ...]} popped per call
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def should_use_team(self, plan):
        return True

    async def execute_step(self, *, plan, execution_plan, step, root_session_id, plan_state):
        self.calls.append((str(step.id), dict(plan_state)))
        outcomes = self.script.get(str(step.id))
        if not outcomes:
            return StepResult(success=True, output=f"default-{step.id}", tool_calls_used=1)
        return outcomes.pop(0)


def _install_adapter(monkeypatch, adapter):
    import victor.agent.planning.team_execution as team_execution_module

    monkeypatch.setattr(team_execution_module, "PlanningTeamExecutionAdapter", lambda orch: adapter)


class TestTeamPipelineExecution:
    async def test_happy_path_flows_outputs_and_produces(self, monkeypatch):
        coordinator, _ = _coordinator()
        adapter = _ScriptedAdapter(
            {
                "1": [
                    StepResult(
                        success=True,
                        output="auth.py: verify() at line 12 handles expiry",
                        tool_calls_used=2,
                    )
                ],
                2: [
                    StepResult(
                        success=True,
                        output="patched verify() to refresh tokens",
                        tool_calls_used=3,
                    )
                ],
            }
        )
        _install_adapter(monkeypatch, adapter)

        result = await coordinator._execute_plan(_plan(), user_approved=True)

        assert result.success is True
        assert result.steps_completed == 2
        assert result.steps_failed == 0
        # tool_calls accumulate from the recorded per-step results
        assert result.total_tool_calls == sum(
            r.tool_calls_used for r in result.step_results.values()
        )
        # Dependency ordering: step 2 dispatched after step 1 completed.
        assert [c[0] for c in adapter.calls] == ["1", "2"]

    async def test_produces_key_populates_plan_state_for_downstream(self, monkeypatch):
        seen_state = {}

        class CapturingAdapter(_ScriptedAdapter):
            async def execute_step(self, **kwargs):
                seen_state[str(kwargs["step"].id)] = dict(kwargs["plan_state"])
                return await super().execute_step(**kwargs)

        adapter = CapturingAdapter(
            {
                "1": [
                    StepResult(
                        success=True,
                        output=(
                            "alpha.py (module loader)\n"
                            "beta.py (config parser)\n"
                            "gamma.py (type definitions)"
                        ),
                        tool_calls_used=1,
                    )
                ],
                "2": [StepResult(success=True, output="consumed the list", tool_calls_used=1)],
            }
        )
        coordinator, _ = _coordinator()
        _install_adapter(monkeypatch, adapter)
        plan = _plan(
            [
                {
                    "id": 1,
                    "type": "research",
                    "desc": "list modules",
                    "tools": "read",
                    "deps": [],
                    "produces": "modules",
                },
                {
                    "id": 2,
                    "type": "feature",
                    "desc": "use the list",
                    "tools": "write",
                    "deps": [1],
                    "loop_over": "modules",
                },
            ]
        )
        result = await coordinator._execute_plan(plan, user_approved=True)

        assert result.success is True
        # The producing step's output is visible to the consumer by key.
        assert seen_state["2"].get("modules") == [
            "alpha.py (module loader)",
            "beta.py (config parser)",
            "gamma.py (type definitions)",
        ]

    async def test_failing_step_marks_plan_failed(self, monkeypatch):
        coordinator, _ = _coordinator()
        adapter = _ScriptedAdapter(
            {
                "1": [
                    StepResult(
                        success=False,
                        output="",
                        error="provider timeout",
                        tool_calls_used=0,
                    )
                ],
                2: [StepResult(success=True, output="still ran", tool_calls_used=1)],
            }
        )
        _install_adapter(monkeypatch, adapter)
        result = await coordinator._execute_plan(_plan(), user_approved=True)

        assert result.success is False
        assert result.steps_failed >= 1
        assert result.step_results["1"].success is False

    async def test_adapter_exception_becomes_failed_step_result(self, monkeypatch):
        class ExplodingAdapter(_ScriptedAdapter):
            async def execute_step(self, **kwargs):
                raise RuntimeError("worker crashed")

        coordinator, _ = _coordinator()
        _install_adapter(monkeypatch, ExplodingAdapter({}))
        result = await coordinator._execute_plan(_plan(), user_approved=True)

        assert result.success is False
        assert "worker crashed" in result.step_results["1"].error

    async def test_empty_required_produces_gets_one_agentic_retry(self, monkeypatch):
        coordinator, _ = _coordinator()
        adapter = _ScriptedAdapter(
            {
                "1": [
                    StepResult(
                        success=True,
                        output="I will look at the modules now.",
                        tool_calls_used=1,
                        metadata={
                            "evidence_validation": {"passed": False},
                            "empty_required_produces": {"modules": 0},
                        },
                    ),
                    StepResult(
                        success=True,
                        output=(
                            "src/alpha.py defines load(); src/beta.py parses " "the runtime config"
                        ),
                        tool_calls_used=2,
                    ),
                ],
                2: [StepResult(success=True, output="done", tool_calls_used=1)],
            }
        )
        _install_adapter(monkeypatch, adapter)
        plan = _plan(
            [
                {
                    "id": 1,
                    "type": "research",
                    "desc": "list modules",
                    "tools": "read",
                    "deps": [],
                    "produces": "modules",
                },
                {"id": 2, "type": "feature", "desc": "use the list", "tools": "write", "deps": [1]},
            ]
        )
        result = await coordinator._execute_plan(plan, user_approved=True)

        # The hollow first attempt was retried and the second succeeded.
        step1_calls = [c for c in adapter.calls if c[0] == "1"]
        assert len(step1_calls) == 2
        assert result.step_results["1"].success is True


class TestPlanEvidenceContractRecompute:
    def test_recompute_flips_statuses_and_joins_output(self):
        coordinator, _ = _coordinator()
        plan = _plan()
        execution_plan = plan.to_execution_plan()
        steps = execution_plan.steps

        result = PlanResult(plan_id=execution_plan.id, success=True, total_steps=2)
        result.step_results[steps[0].id] = StepResult(
            success=True,
            output="src/auth.py line 12: verify() handles expiry",
            tool_calls_used=2,
        )
        result.step_results[steps[1].id] = StepResult(
            success=False, output="second output", error="hollow", tool_calls_used=0
        )

        coordinator._apply_plan_evidence_contracts(execution_plan, result)

        assert steps[0].status.value == "completed"
        assert steps[1].status.value == "failed"
        assert result.steps_completed == 1
        assert result.steps_failed == 1
        assert result.success is False
        assert "src/auth.py line 12" in result.final_output
        assert "second output" in result.final_output

    def test_missing_step_result_preserves_blocked_and_skipped(self):
        coordinator, _ = _coordinator()
        plan = _plan()
        execution_plan = plan.to_execution_plan()
        from victor.agent.planning.base import StepStatus

        execution_plan.steps[0].status = StepStatus.BLOCKED
        execution_plan.steps[1].status = StepStatus.SKIPPED
        result = PlanResult(plan_id=execution_plan.id, success=True, total_steps=2)

        coordinator._apply_plan_evidence_contracts(execution_plan, result)

        # No step results to validate: statuses untouched, success False
        # because nothing completed.
        assert result.steps_completed == 0
        assert result.success is False


class TestGeneratePlanOverrides:
    async def _run(self, monkeypatch, orchestrator_attrs):
        coordinator, orchestrator = _coordinator()
        for attr, value in orchestrator_attrs.items():
            setattr(orchestrator, attr, value)
        captured = {}

        async def fake_generate(**kwargs):
            captured.update(kwargs)
            return _plan()

        with patch(
            "victor.agent.services.planning_runtime.generate_task_plan",
            fake_generate,
        ):
            plan = await coordinator._generate_plan("do a complex thing")
        return captured, plan

    async def test_cli_provider_colon_model_override(self, monkeypatch):
        fake_provider = MagicMock(name="zai-provider")
        with patch(
            "victor.providers.registry.create_provider_or_none",
            return_value=fake_provider,
        ):
            captured, plan = await self._run(
                monkeypatch, {"planning_model_override": "zai:glm-5.3"}
            )
        assert captured["provider"] is fake_provider
        assert captured["model"] == "glm-5.3"

    async def test_cli_override_falls_back_when_registry_misses(self, monkeypatch):
        coordinator, orchestrator = _coordinator()
        orchestrator.planning_model_override = "zai:glm-5.3"
        captured = {}

        async def fake_generate(**kwargs):
            captured.update(kwargs)
            return _plan()

        with (
            patch(
                "victor.providers.registry.create_provider_or_none",
                return_value=None,
            ),
            patch(
                "victor.agent.services.planning_runtime.generate_task_plan",
                fake_generate,
            ),
        ):
            await coordinator._generate_plan("anything")
        # Registry miss keeps the parent provider rather than crashing.
        assert captured["provider"] is orchestrator.provider

    async def test_cli_bare_model_override(self, monkeypatch):
        captured, _ = await self._run(monkeypatch, {"planning_model_override": "qwen3-max"})
        assert captured["model"] == "qwen3-max"

    async def test_profile_overrides_apply_without_cli(self, monkeypatch):
        fake_provider = MagicMock(name="profile-provider")
        profile = SimpleNamespace(planning_provider="zai", planning_model="glm-5.3", name="prof")
        with patch(
            "victor.providers.registry.create_provider_or_none",
            return_value=fake_provider,
        ):
            captured, _ = await self._run(monkeypatch, {"profile": profile})
        assert captured["provider"] is fake_provider
        assert captured["model"] == "glm-5.3"

    async def test_defaults_pass_parent_provider_and_model(self, monkeypatch):
        coordinator, orchestrator = _coordinator()
        captured = {}

        async def fake_generate(**kwargs):
            captured.update(kwargs)
            return _plan()

        with patch(
            "victor.agent.services.planning_runtime.generate_task_plan",
            fake_generate,
        ):
            await coordinator._generate_plan("anything")
        assert captured["provider"] is orchestrator.provider
        assert captured["model"] == "base-model"


class TestChatWithPlanningPaths:
    async def test_never_mode_takes_direct_chat(self):
        coordinator, _ = _coordinator()
        coordinator._direct_chat = AsyncMock(return_value=MagicMock(name="resp"))
        await coordinator.chat_with_planning("anything", mode=PlanningMode.NEVER)
        coordinator._direct_chat.assert_awaited_once_with("anything")

    async def test_generation_failure_falls_back_to_direct_chat(self, monkeypatch):
        coordinator, _ = _coordinator()
        assert coordinator.config.fallback_on_planning_failure is True
        coordinator._direct_chat = AsyncMock(return_value=MagicMock(name="fallback"))

        async def boom(**kwargs):
            raise RuntimeError("planner down")

        with patch(
            "victor.agent.services.planning_runtime.generate_task_plan",
            boom,
        ):
            response = await coordinator.chat_with_planning(
                "first analyze, then refactor, then test each part"
            )
        assert response is coordinator._direct_chat.return_value

    async def test_rejected_plan_returns_rejection_response(self, monkeypatch):
        coordinator, _ = _coordinator()
        coordinator.config.show_plan_before_execution = True
        rejection = MagicMock(name="rejection")
        coordinator._generate_plan_rejected_response = AsyncMock(return_value=rejection)
        coordinator._show_plan_to_user = AsyncMock(
            return_value=SimpleNamespace(proceed=False, user_approved_execution=False)
        )
        coordinator._execute_plan = AsyncMock()

        async def fake_generate(**kwargs):
            return _plan()

        with patch(
            "victor.agent.services.planning_runtime.generate_task_plan",
            fake_generate,
        ):
            response = await coordinator.chat_with_planning(
                "first analyze, then refactor, then test each part"
            )
        assert response is rejection
        coordinator._execute_plan.assert_not_called()

    async def test_happy_path_executes_and_summarizes(self, monkeypatch):
        coordinator, _ = _coordinator()
        coordinator.config.show_plan_before_execution = False
        adapter = _ScriptedAdapter({})
        _install_adapter(monkeypatch, adapter)
        final = MagicMock(name="final")
        coordinator._generate_final_response = AsyncMock(return_value=final)
        coordinator._attach_planning_response_metadata = MagicMock()

        async def fake_generate(**kwargs):
            return _plan()

        with patch(
            "victor.agent.services.planning_runtime.generate_task_plan",
            fake_generate,
        ):
            response = await coordinator.chat_with_planning(
                "first analyze, then refactor, then test each part"
            )
        assert response is final
        coordinator._generate_final_response.assert_awaited_once()
        coordinator._attach_planning_response_metadata.assert_called_once()


class TestSummaryChain:
    """Final-response generation, summary prompt, artifacts, diagnostics."""

    def _result(self, success=True):
        plan = _plan()
        execution_plan = plan.to_execution_plan()
        result = PlanResult(plan_id=execution_plan.id, success=success, total_steps=2)
        result.step_results["1"] = StepResult(
            success=True,
            output="src/auth.py line 12: verify() handles expiry",
            tool_calls_used=2,
            artifacts=["reports/auth.md"],
            metadata={
                "evidence_validation": {
                    "passed": True,
                    "reason": "durable artifacts or sources recorded",
                    "tool_calls_used": 2,
                    "has_file_reference": True,
                },
                "provider_retry_diagnostics": [
                    {
                        "provider": "zai",
                        "model": "glm-5.3",
                        "retry_count": 2,
                        "last_error": "429 rate limited",
                    }
                ],
            },
        )
        result.step_results["2"] = StepResult(
            success=False,
            output="",
            error="hollow output",
            tool_calls_used=0,
            metadata={
                "evidence_validation": {
                    "passed": False,
                    "reason": "no tool-backed execution was recorded",
                }
            },
        )
        result.final_output = "aggregated findings text"
        return plan, result

    def test_build_summary_prompt_carries_every_section(self):
        coordinator, _ = _coordinator()
        plan, result = self._result()
        prompt = coordinator._build_summary_prompt(plan, result)

        assert "Summarize the planned task execution" in prompt
        assert "Steps completed:" in prompt
        assert "src/auth.py line 12" in prompt
        assert "reports/auth.md" in prompt
        assert "hollow output" in prompt
        assert "passed" in prompt and "failed" in prompt
        assert "Provider retry diagnostics" in prompt
        assert "429 rate limited" in prompt
        assert "Aggregated step output" in prompt
        assert "do not invent repository findings" in prompt

    async def test_failed_plan_response_needs_no_model_call(self):
        coordinator, orchestrator = _coordinator()
        plan, result = self._result(success=False)
        result.metadata["plan_execution_state"] = {
            "failed_step_ids": ["2"],
            "blocked_step_ids": [],
        }
        response = await coordinator._generate_final_response(plan, result)
        assert "Plan did not complete" in response.content
        assert "Failed steps: 2." in response.content
        # Artifact paths are guaranteed present even on failure.
        assert "reports/auth.md" in response.content
        orchestrator._metrics_coordinator.complete.assert_not_called()

    async def test_successful_plan_response_uses_summary_prompt_and_appends_artifacts(self):
        from victor.providers.base import CompletionResponse

        coordinator, orchestrator = _coordinator()
        plan, result = self._result(success=True)
        result.steps_failed = 0

        async def fake_complete(provider, *, messages, **kwargs):
            assert "Summarize the planned task execution" in messages[0]["content"]
            return CompletionResponse(content="summary without artifact", role="assistant")

        orchestrator._metrics_coordinator.complete = fake_complete
        response = await coordinator._generate_final_response(plan, result)
        assert response.content.startswith("summary without artifact")
        # Missing artifact paths are appended durably.
        assert "reports/auth.md" in response.content

    def test_artifact_collection_dedupes_across_steps(self):
        plan, result = self._result()
        result.step_results["2"].artifacts = ["reports/auth.md", "notes/other.md"]
        artifacts = PlanningCoordinator._collect_step_artifacts(result)
        assert artifacts == ["reports/auth.md", "notes/other.md"]


class TestProducesCoercionAndConsumption:
    def test_prose_markdown_report_preserved_as_single_item(self):
        coordinator, _ = _coordinator()
        plan = _plan(
            [
                {
                    "id": 1,
                    "type": "research",
                    "desc": "write the findings report",
                    "tools": "read,write",
                    "deps": [],
                    "produces": "findings",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        report = (
            "# Findings\n\n- The alpha module is stale and duplicates the loader.\n"
            "- The beta module lacks tests for config parsing.\n"
            "- The gamma types leak internals across the crate boundary.\n"
            "- Retry policy is computed independently in three call sites.\n"
            "- None of the call sites honor the shared circuit breaker.\n"
            "- The migration notes are out of date with the current schema.\n"
        ) * 3
        step_result = StepResult(
            success=True,
            output=report,
            tool_calls_used=4,
            metadata={"evidence_validation": {"passed": True}},
        )
        coerced = PlanningCoordinator._coerce_required_produces_items(
            step, step_result, "findings", []
        )
        # A substantive markdown report is preserved as one synthesis input.
        assert len(coerced) == 1
        assert "Findings" in coerced[0]

    def test_extracted_lists_pass_through_unchanged(self):
        plan = _plan([{"id": 1, "type": "research", "desc": "list", "tools": "read", "deps": []}])
        step = plan.to_execution_plan().steps[0]
        step_result = StepResult(success=True, output="x", tool_calls_used=1)
        coerced = PlanningCoordinator._coerce_required_produces_items(
            step, step_result, "modules", ["a.py", "b.py"]
        )
        assert coerced == ["a.py", "b.py"]

    def test_downstream_consumption_detection(self):
        plan = _plan(
            [
                {
                    "id": 1,
                    "type": "research",
                    "desc": "produce",
                    "tools": "read",
                    "deps": [],
                    "produces": "modules",
                },
                {
                    "id": 2,
                    "type": "feature",
                    "desc": "consume via loop",
                    "tools": "write",
                    "deps": [1],
                    "loop_over": "modules",
                },
                {
                    "id": 3,
                    "type": "feature",
                    "desc": "consume via inputs",
                    "tools": "write",
                    "deps": [1],
                    "inputs": ["modules"],
                },
            ]
        )
        execution_plan = plan.to_execution_plan()
        assert PlanningCoordinator._produces_consumed_downstream(execution_plan, "1", "modules")
        assert not PlanningCoordinator._produces_consumed_downstream(
            execution_plan, "1", "nonexistent_key"
        )


class TestContextCompaction:
    async def test_context_service_compaction_short_circuits(self):
        coordinator, _ = _coordinator()
        coordinator._compact_with_context_service = AsyncMock(return_value=True)
        await coordinator._compact_context_if_needed()
        coordinator._compact_with_context_service.assert_awaited_once()

    async def test_capability_compactor_runs_when_present(self):
        coordinator, orchestrator = _coordinator()
        coordinator._compact_with_context_service = AsyncMock(return_value=False)
        compactor = MagicMock()
        compactor.check_and_compact.return_value = MagicMock(compacted=True)
        orchestrator.has_capability = MagicMock(return_value=True)
        orchestrator.get_capability_value = MagicMock(return_value=compactor)
        orchestrator.conversation = MagicMock()
        orchestrator.conversation.get_latest_user_message.return_value = "query"
        await coordinator._compact_context_if_needed()
        compactor.check_and_compact.assert_called_once()

    async def test_no_capability_is_a_noop(self):
        coordinator, orchestrator = _coordinator()
        coordinator._compact_with_context_service = AsyncMock(return_value=False)
        orchestrator.has_capability = MagicMock(return_value=False)
        await coordinator._compact_context_if_needed()  # must not raise


class TestParallelBatchDispatch:
    async def test_independent_steps_share_a_consistent_state_snapshot(self, monkeypatch):
        seen_batches = []

        class ParallelAdapter(_ScriptedAdapter):
            async def execute_step(self, **kwargs):
                seen_batches.append((str(kwargs["step"].id), dict(kwargs["plan_state"])))
                import asyncio

                await asyncio.sleep(0.01)
                return StepResult(
                    success=True,
                    output=f"src/mod{kwargs['step'].id}.py analyzed",
                    tool_calls_used=1,
                )

        coordinator, _ = _coordinator()
        _install_adapter(monkeypatch, ParallelAdapter({}))
        plan = _plan(
            [
                {"id": 1, "type": "research", "desc": "first", "tools": "read", "deps": []},
                {"id": 2, "type": "research", "desc": "second", "tools": "read", "deps": []},
            ]
        )
        result = await coordinator._execute_plan(plan, user_approved=True)

        assert result.success is True
        assert result.steps_completed == 2
        # Both independent steps dispatched concurrently in one batch.
        assert len(seen_batches) == 2
        # Neither saw the other's writes: same consistent snapshot for the batch.
        assert seen_batches[0][1] == seen_batches[1][1] == {}


class TestFailedDependencySkipping:
    async def test_failure_blocks_transitive_dependents_but_independent_branch_runs(
        self, monkeypatch
    ):
        class BranchAdapter(_ScriptedAdapter):
            async def execute_step(self, **kwargs):
                sid = str(kwargs["step"].id)
                if sid == "1":
                    return StepResult(
                        success=False, output="", error="provider timeout", tool_calls_used=0
                    )
                return StepResult(
                    success=True,
                    output=f"src/branch{sid}.py analyzed independently",
                    tool_calls_used=2,
                )

        coordinator, _ = _coordinator()
        _install_adapter(monkeypatch, BranchAdapter({}))
        plan = _plan(
            [
                {"id": 1, "type": "research", "desc": "fails", "tools": "read", "deps": []},
                {"id": 2, "type": "feature", "desc": "depends on 1", "tools": "write", "deps": [1]},
                {"id": 3, "type": "feature", "desc": "depends on 2", "tools": "write", "deps": [2]},
                {"id": 4, "type": "research", "desc": "independent", "tools": "read", "deps": []},
            ]
        )
        result = await coordinator._execute_plan(plan, user_approved=True)

        assert result.success is False
        # Transitive dependents blocked with the dependency named.
        assert "failed required dependencies" in result.step_results["2"].error
        assert (
            "2" in result.step_results["3"].metadata.get("blocked_by_failed_dependencies", [])
            or "failed required dependencies" in result.step_results["3"].error
        )
        # The independent branch still executed and completed.
        assert result.step_results["4"].success is True
        assert result.steps_completed == 1


class TestPlanApprovalPrompt:
    async def test_non_interactive_stdin_auto_approves(self, monkeypatch):
        import io

        coordinator, _ = _coordinator()
        plan = _plan()
        console = MagicMock()
        monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"")))

        approved = await coordinator._request_plan_approval(plan, console)

        assert approved is True
        console.print.assert_called()

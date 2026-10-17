import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.planning.base import PlanResult, StepResult
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.services.planning_runtime import (
    PlanningConfig,
    PlanningMode,
    PlanningRuntimeService,
    _PlanProgressDisplay,
)
from victor.framework.execution_checkpoint import ApprovalState
from victor.providers.base import CompletionResponse


@pytest.mark.asyncio
async def test_execute_plan_routes_complex_plan_through_team_adapter():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Arc Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust Arc usage by workspace",
        steps=[
            ["1", "research", "Map Rust workspaces", "ls,read"],
            ["2", "review", "Review Arc usage", "grep,read", ["1"]],
        ],
    )
    adapter = MagicMock()
    adapter.should_use_team.return_value = True
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(
                success=True,
                output="Read Cargo.toml and clients/rust/Cargo.toml; found 2 Rust workspaces.",
                tool_calls_used=2,
            ),
            StepResult(
                success=True,
                output="Reviewed src/lib.rs:10 and clients/rust/src/lib.rs:4 for Arc usage.",
                tool_calls_used=3,
            ),
        ]
    )

    with (
        patch(
            "victor.agent.planning.team_execution.PlanningTeamExecutionAdapter",
            return_value=adapter,
        ),
        patch("victor.agent.planning.autonomous.AutonomousPlanner") as planner_cls,
    ):
        result = await service._execute_plan(plan, user_approved=True)

    planner_cls.assert_not_called()
    assert adapter.execute_step.await_count == 2
    first_call = adapter.execute_step.await_args_list[0].kwargs
    assert first_call["root_session_id"] == "session_root"
    assert first_call["step"].id == "1"
    assert result.success is True
    assert result.steps_completed == 2
    assert result.total_tool_calls == 5
    assert "Cargo.toml" in result.final_output
    execution_state = result.metadata["plan_execution_state"]
    assert execution_state["plan_id"] == result.plan_id
    assert execution_state["execution_mode"] == "team_adapter"
    assert execution_state["success"] is True
    assert execution_state["step_statuses"] == {"1": "completed", "2": "completed"}
    assert execution_state["completed_step_ids"] == ["1", "2"]
    assert execution_state["failed_step_ids"] == []


@pytest.mark.asyncio
async def test_planning_compaction_prefers_context_service_before_legacy_compactor():
    context_service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": True}),
        compact_context=AsyncMock(return_value=2),
    )
    legacy_compactor = MagicMock()
    orchestrator = SimpleNamespace(
        _context_service=context_service,
        settings=SimpleNamespace(context_compaction_strategy="semantic"),
        tool_calls_used=0,
        conversation=SimpleNamespace(get_latest_user_message=MagicMock(return_value="review plan")),
        has_capability=MagicMock(return_value=True),
        get_capability_value=MagicMock(return_value=legacy_compactor),
    )
    service = PlanningRuntimeService(orchestrator)

    await service._compact_context_if_needed()

    context_service.get_compaction_recommendation.assert_called_once()
    context_service.compact_context.assert_awaited_once_with(
        strategy="semantic",
        min_messages=6,
    )
    legacy_compactor.check_and_compact.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False, "false", None, 1])
async def test_chat_with_planning_attaches_execution_state_to_response_metadata(success):
    service = PlanningRuntimeService(
        SimpleNamespace(),
        config=PlanningConfig(show_plan_before_execution=False),
    )
    plan = ReadableTaskPlan(
        name="Graph Native Plan",
        complexity=TaskComplexity.COMPLEX,
        desc="Exercise plan execution state metadata",
        steps=[["1", "research", "Map files", "read"]],
    )
    execution_state = {
        "plan_id": "plan-1",
        "execution_mode": "team_adapter",
        "success": success,
        "step_statuses": {"1": "completed"},
    }
    result = PlanResult(
        plan_id="plan-1",
        success=success,
        total_steps=1,
        steps_completed=1,
        metadata={"plan_execution_state": execution_state},
    )
    response = CompletionResponse(
        content="summary",
        role="assistant",
        metadata={"provider": "test-provider", "agentic_loop_success": True},
    )
    service._compact_context_if_needed = AsyncMock()
    service._generate_plan = AsyncMock(return_value=plan)
    service._execute_plan = AsyncMock(return_value=result)
    service._generate_final_response = AsyncMock(return_value=response)

    if type(success) is not bool:
        with pytest.raises(ValueError, match="Plan success must be a boolean"):
            await service.chat_with_planning("plan this", mode=PlanningMode.ALWAYS)
        service._generate_final_response.assert_not_awaited()
        assert service._compact_context_if_needed.await_count == 1
        return

    returned = await service.chat_with_planning("plan this", mode=PlanningMode.ALWAYS)

    assert returned is response
    assert returned.metadata["provider"] == "test-provider"
    assert returned.metadata["plan_execution_state"] == execution_state
    assert returned.metadata["planning"]["mode"] == "planned"
    assert returned.metadata["planning"]["plan_name"] == "Graph Native Plan"
    assert returned.metadata["planning"]["steps_completed"] == 1
    assert returned.metadata["planning"]["steps_total"] == 1

    assert returned.metadata["agentic_loop_success"] is success
    assert returned.metadata["planning"]["success"] is success
    if not success:
        assert returned.metadata["agentic_loop_error"]


@pytest.mark.asyncio
async def test_team_plan_execution_bounds_independent_step_concurrency():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(
        orchestrator,
        config=PlanningConfig(max_parallel_steps=2),
    )
    plan = ReadableTaskPlan(
        name="Large Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review many independent areas",
        steps=[
            ["1", "research", "Map workspace A", "read"],
            ["2", "research", "Map workspace B", "read"],
            ["3", "research", "Map workspace C", "read"],
            ["4", "research", "Map workspace D", "read"],
        ],
    )
    adapter = MagicMock()
    active = 0
    max_active = 0

    async def execute_step(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return StepResult(
            success=True,
            output=f"Read workspace_{kwargs['step'].id}/Cargo.toml and found 3 files.",
            tool_calls_used=1,
        )

    adapter.execute_step = AsyncMock(side_effect=execute_step)

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 4
    assert adapter.execute_step.await_count == 4
    assert max_active == 2


@pytest.mark.asyncio
async def test_team_plan_blocks_read_only_step_when_required_inventory_fails():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source after inventory",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Enumerate all Rust source files",
                "tools": ["shell", "read"],
            },
            {
                "id": "2",
                "type": "review",
                "desc": "Review Arc usage in inventoried files",
                "tools": ["grep", "read"],
                "deps": ["1"],
            },
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(success=True, output="inventory complete", tool_calls_used=1),
            StepResult(
                success=True,
                output="Reviewed src/lib.rs:12 and src/storage/mod.rs:44 for Arc usage.",
                tool_calls_used=2,
            ),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is False
    assert result.steps_completed == 0
    assert result.steps_failed == 1
    assert adapter.execute_step.await_count == 1
    assert result.step_results["2"].tool_calls_used == 0
    assert "Insufficient execution evidence" in result.error_message
    execution_state = result.metadata["plan_execution_state"]
    assert execution_state["execution_mode"] == "team_adapter"
    assert execution_state["success"] is False
    assert execution_state["step_statuses"] == {"1": "failed", "2": "blocked"}
    assert execution_state["failed_step_ids"] == ["1"]
    assert execution_state["skipped_step_ids"] == []
    assert execution_state["blocked_step_ids"] == ["2"]


@pytest.mark.asyncio
async def test_team_plan_accepts_concrete_read_heavy_evidence():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source after inventory",
        steps=[
            ["1", "analyze", "Enumerate all Rust source files", "shell,read"],
            ["2", "review", "Review Arc usage in inventoried files", "grep,read", [1]],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(
                success=True,
                output="Enumerated 192 Rust files, including src/lib.rs and src/storage/mod.rs.",
                tool_calls_used=1,
            ),
            StepResult(
                success=True,
                output="Reviewed src/lib.rs:12 and src/storage/mod.rs:44 for Arc usage.",
                tool_calls_used=2,
            ),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 2
    assert result.steps_failed == 0
    assert adapter.execute_step.await_count == 2
    assert "192 Rust files" in result.final_output


@pytest.mark.asyncio
async def test_team_plan_accepts_concrete_file_target_from_step_description():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust workspace",
        steps=[
            ["1", "analyze", "Read rust/Cargo.toml to map workspace members", "read"],
            ["2", "review", "Review Arc usage in mapped crates", "grep,read", [1]],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(success=True, output="Workspace manifest inspected.", tool_calls_used=1),
            StepResult(
                success=True,
                output="Reviewed rust/crates/core/src/lib.rs:12 for Arc usage.",
                tool_calls_used=2,
            ),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 2
    validation = result.step_results["1"].metadata["evidence_validation"]
    assert validation["passed"] is True
    assert validation["has_file_reference"] is True


@pytest.mark.asyncio
async def test_team_plan_accepts_directory_tree_mapping_scope():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust workspace",
        steps=[
            [
                "1",
                "research",
                "Map the full directory tree of rust/ to identify src/, benches/, tests/",
                "shell,read",
            ],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        return_value=StepResult(
            success=True,
            output="Directory tree mapped.",
            tool_calls_used=1,
        )
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    validation = result.step_results["1"].metadata["evidence_validation"]
    assert validation["passed"] is True
    assert validation["has_directory_scope"] is True


def test_read_only_plan_does_not_require_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Read-only Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust code without changing files",
        steps=[
            ["1", "analyze", "Read Cargo.toml", "read"],
            ["2", "review", "Search Arc usages", "grep,code_search,overview"],
            ["3", "doc", "Summarize findings in chat", "read"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is False


def test_checklist_planning_step_does_not_require_read_evidence_contract():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Checklist",
        complexity=TaskComplexity.COMPLEX,
        desc="Create Rust review checklist",
        steps=[
            [
                "1",
                "doc",
                "Build master Rust best practices checklist for Arc and immutability",
                "write",
            ],
            ["2", "review", "Present checklist to user for approval", "read", [1]],
        ],
    )
    execution_plan = plan.to_execution_plan()

    assert service._step_requires_evidence_contract(execution_plan.steps[0]) is False
    assert service._step_requires_evidence_contract(execution_plan.steps[1]) is False


def test_unpack_step_handles_list_format():
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = ["1", "analyze", "Discover workspace", "read,grep"]
    sid, stype, sdesc, stools = svc._unpack_step(step)
    assert sid == "1"
    assert stype == "analyze"
    assert sdesc == "Discover workspace"
    assert stools == "read,grep"


def test_unpack_step_handles_dict_format():
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = {
        "id": "2",
        "type": "feature",
        "desc": "Create checklist",
        "tools": ["write", "read"],
    }
    sid, stype, sdesc, stools = svc._unpack_step(step)
    assert sid == "2"
    assert stype == "feature"
    assert sdesc == "Create checklist"
    assert stools == ["write", "read"]


def test_unpack_step_dict_uses_description_fallback():
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = {"id": "3", "type": "compute", "description": "Run check"}
    _, _, sdesc, _ = svc._unpack_step(step)
    assert sdesc == "Run check"


def test_serializable_plan_step_preserves_runtime_fields():
    step = {
        "id": "2",
        "type": "analyze",
        "desc": "Inventory review targets",
        "tools": ["read", "ls"],
        "deps": ["1"],
        "inputs": ["manifest_summary"],
        "exec": "agent",
        "produces": "review_targets",
        "tool_calls": 20,
        "exit": ["return one target per line"],
    }

    persisted = PlanningRuntimeService._serializable_plan_step(step)

    assert persisted["description"] == "Inventory review targets"
    assert persisted["deps"] == ["1"]
    assert persisted["inputs"] == ["manifest_summary"]
    assert persisted["exec"] == "agent"
    assert persisted["produces"] == "review_targets"
    assert persisted["tool_calls"] == 20
    assert persisted["exit"] == ["return one target per line"]


def test_plan_requires_execution_approval_works_with_dict_steps():
    """Rich dict steps must not crash _plan_requires_execution_approval."""
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    plan = ReadableTaskPlan(
        name="Mixed plan",
        complexity=TaskComplexity.COMPLEX,
        desc="Test dict step approval check",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Read workspace", "tools": ["read"]},
            {"id": "2", "type": "deployment", "desc": "Deploy", "tools": ["shell"]},
        ],
    )
    result = svc._plan_requires_execution_approval(plan)
    assert result is True


@pytest.mark.asyncio
async def test_failed_plan_reports_recorded_status_without_summary_provider(tmp_path):
    artifact = tmp_path / "inventory.md"
    artifact.write_text("Verified repository inventory", encoding="utf-8")
    provider = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("summary unavailable")))
    service = PlanningRuntimeService(
        SimpleNamespace(provider=provider, model="test", max_tokens=100)
    )
    plan = ReadableTaskPlan(
        name="Audit",
        desc="Inspect code",
        complexity=TaskComplexity.COMPLEX,
        steps=[["1", "research", "Inventory", "read"], ["2", "review", "Review", "read"]],
    )
    result = PlanResult(
        plan_id="audit",
        success=False,
        total_steps=3,
        steps_completed=1,
        step_results={
            "1": StepResult(success=True, output="Inventory recorded", artifacts=[str(artifact)]),
            "2": StepResult(success=False, output="", error="Provider failed"),
            "3": StepResult(
                success=False, output="", metadata={"blocked_by_failed_dependencies": ["2"]}
            ),
        },
        metadata={
            "plan_execution_state": {
                "failed_step_ids": ["2"],
                "blocked_step_ids": ["3"],
                "step_statuses": {"1": "completed", "2": "failed", "3": "blocked"},
            }
        },
    )
    response = await service._generate_final_response(plan, result)
    provider.chat.assert_not_awaited()
    assert "1/3" in response.content
    assert "Failed steps: 2" in response.content
    assert "Blocked steps: 3" in response.content
    assert str(artifact) in response.content
    assert response.usage is None  # No invented provider accounting.


def test_summary_prompt_surfaces_evidence_validation_failure():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Evidence Audit",
        complexity=TaskComplexity.COMPLEX,
        desc="Audit Rust evidence quality",
        steps=[
            ["1", "analyze", "Enumerate Rust files", "shell,read"],
            ["2", "review", "Review Arc usage", "grep,read", [1]],
        ],
    )
    result = SimpleNamespace(
        steps_completed=0,
        total_steps=2,
        success=False,
        final_output="",
        error_message="Insufficient execution evidence for step 1",
        step_results={
            "1": StepResult(
                success=False,
                output="inventory complete",
                error=(
                    "Insufficient execution evidence for step 1: "
                    "step output is only a generic completion marker"
                ),
                tool_calls_used=1,
                metadata={
                    "evidence_validation": {
                        "passed": False,
                        "reason": "step output is only a generic completion marker",
                        "tool_calls_used": 1,
                        "has_file_reference": False,
                        "has_counted_scope": False,
                        "has_artifacts": False,
                        "source_count": 0,
                    }
                },
            )
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert "Evidence validation: failed" in prompt
    assert "step output is only a generic completion marker" in prompt
    assert "tool_calls=1" in prompt
    assert "file_ref=False" in prompt
    assert "Do not report failed evidence-validation steps as completed" in prompt


def test_summary_prompt_reports_aggregate_evidence_validation_counts():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Evidence Coverage",
        complexity=TaskComplexity.COMPLEX,
        desc="Audit which completed steps have evidence backing",
        steps=[
            ["1", "analyze", "Map Rust workspaces", "read"],
            ["2", "review", "Review Arc usage", "grep,read", [1]],
            ["3", "doc", "Summarize findings", "read", [2]],
            ["4", "review", "Review allocation patterns", "grep,read", [1]],
        ],
    )
    result = SimpleNamespace(
        steps_completed=3,
        total_steps=4,
        success=False,
        final_output="",
        error_message="",
        step_results={
            "1": StepResult(
                success=True,
                output="Read Cargo.toml and clients/rust/Cargo.toml.",
                tool_calls_used=2,
                metadata={
                    "evidence_validation": {
                        "passed": True,
                        "reason": "concrete file or scope evidence found",
                    }
                },
            ),
            "2": StepResult(
                success=True,
                output="Reviewed Arc usage.",
                tool_calls_used=2,
            ),
            "3": StepResult(
                success=False,
                output="report complete",
                error="Insufficient execution evidence",
                tool_calls_used=1,
                metadata={
                    "evidence_validation": {
                        "passed": False,
                        "reason": "step output is only a generic completion marker",
                    }
                },
            ),
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert (
        "Evidence validation summary: passed=1; failed=1; exempt=0; missing=1; not_run=1"
    ) in prompt
    assert "Treat steps missing evidence validation as unverified" in prompt


def test_summary_prompt_surfaces_provider_retry_diagnostics():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Retry Audit",
        complexity=TaskComplexity.COMPLEX,
        desc="Audit Rust execution evidence and provider stability",
        steps=[
            ["1", "analyze", "Map Rust workspaces", "read"],
            ["2", "review", "Review Arc usage", "grep,read", [1]],
        ],
    )
    result = SimpleNamespace(
        steps_completed=1,
        total_steps=2,
        success=False,
        final_output="Read root Cargo.toml and clients/rust/Cargo.toml.",
        error_message="Provider recovered after transient disconnects",
        metadata={
            "provider_retry_diagnostics": [
                {
                    "provider": "zai",
                    "model": "glm-5.1",
                    "retry_count": 3,
                    "last_error": "Server disconnected without sending a response.",
                }
            ]
        },
        step_results={
            "1": StepResult(
                success=True,
                output="Read root Cargo.toml and clients/rust/Cargo.toml.",
                tool_calls_used=2,
                metadata={
                    "provider_retry_diagnostics": {
                        "provider": "zai",
                        "model": "glm-5.1",
                        "retry_count": 1,
                        "last_error": "Server disconnected without sending a response.",
                    }
                },
            )
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert "Provider retry diagnostics:" in prompt
    assert "provider=zai" in prompt
    assert "model=glm-5.1" in prompt
    assert "retry_count=3" in prompt
    assert "retry_count=1" in prompt
    assert "Server disconnected without sending a response" in prompt
    assert "Mention provider retry diagnostics separately from repository findings" in prompt


def test_shell_plan_requires_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust files using a command",
        steps=[
            ["1", "analyze", "Enumerate Rust files", "shell"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is True


def test_write_plan_requires_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Write Report",
        complexity=TaskComplexity.MODERATE,
        desc="Write findings to disk",
        steps=[
            ["1", "doc", "Write findings report", "write"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is True


@pytest.mark.asyncio
async def test_read_only_plan_decision_uses_not_required_approval_state():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Read-only Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review without changing files",
        steps=[["1", "review", "Read docs", "read"]],
    )

    with patch.object(service, "_save_plan_to_disk"):
        decision = await service._show_plan_with_console(plan)

    assert decision.proceed is True
    assert decision.user_approved_execution is False
    assert decision.approval_state is ApprovalState.NOT_REQUIRED


@pytest.mark.asyncio
async def test_effectful_plan_decision_uses_approved_approval_state():
    service = PlanningRuntimeService(SimpleNamespace(), config=PlanningConfig(auto_approve=False))
    plan = ReadableTaskPlan(
        name="Effectful Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )

    with (
        patch.object(service, "_save_plan_to_disk"),
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=True)),
    ):
        decision = await service._show_plan_with_console(plan)

    assert decision.proceed is True
    assert decision.user_approved_execution is True
    assert decision.approval_state is ApprovalState.APPROVED


@pytest.mark.asyncio
async def test_effectful_plan_decision_uses_rejected_approval_state():
    service = PlanningRuntimeService(SimpleNamespace(), config=PlanningConfig(auto_approve=False))
    plan = ReadableTaskPlan(
        name="Rejected Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )

    with (
        patch.object(service, "_save_plan_to_disk"),
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=False)),
    ):
        decision = await service._show_plan_with_console(plan)

    assert decision.proceed is False
    assert decision.user_approved_execution is False
    assert decision.approval_state is ApprovalState.REJECTED


@pytest.mark.asyncio
async def test_plan_approval_prompt_explains_enter_and_reject_options():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Effectful Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )
    console = MagicMock()

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=True)),
    ):
        approved = await service._request_plan_approval(plan, console)

    assert approved is True
    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    assert "Press Enter to execute" in rendered
    assert "type y then Enter" in rendered
    assert "Type n then Enter to reject" in rendered


# ---------------------------------------------------------------------------
# Plan-state: _extract_list_from_output
# ---------------------------------------------------------------------------


class TestExtractListFromOutput:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    def test_bullet_list(self) -> None:
        out = "- core\n- util\n- cli"
        items = self.svc._extract_list_from_output(out)
        assert items == ["core", "util", "cli"]

    def test_numbered_list(self) -> None:
        out = "1. core\n2. util\n3. cli"
        items = self.svc._extract_list_from_output(out)
        assert items == ["core", "util", "cli"]

    def test_plain_newlines(self) -> None:
        out = "core\nutil\ncli"
        items = self.svc._extract_list_from_output(out)
        assert items == ["core", "util", "cli"]

    def test_empty_string_returns_empty(self) -> None:
        assert self.svc._extract_list_from_output("") == []

    def test_single_line_falls_back_to_whole_string(self) -> None:
        out = "Just one item without newlines"
        items = self.svc._extract_list_from_output(out)
        assert items == ["Just one item without newlines"]

    def test_filters_long_lines(self) -> None:
        long_line = "x" * 201
        out = f"core\n{long_line}\nutil"
        items = self.svc._extract_list_from_output(out)
        assert "core" in items
        assert "util" in items
        assert long_line not in items

    def test_asterisk_bullets(self) -> None:
        out = "* alpha\n* beta"
        items = self.svc._extract_list_from_output(out)
        assert items == ["alpha", "beta"]

    def test_mixed_bullet_styles(self) -> None:
        out = "• one\n- two\n* three"
        items = self.svc._extract_list_from_output(out)
        assert items == ["one", "two", "three"]

    def test_none_sentinel_returns_empty(self) -> None:
        """Regression: '(none)' sentinel must yield [] not ['(none)'].

        Bug: agents instructed to output '(none)' when they found nothing would
        produce a one-item list ['(none)'], which the prose-guard sometimes let
        through, causing conditional nodes to see a non-empty list and take the
        wrong branch.
        Fix: _extract_list_from_output now checks for the exact sentinel before
        any other extraction so it always returns [].
        """
        assert self.svc._extract_list_from_output("(none)") == []

    def test_none_sentinel_with_surrounding_whitespace_returns_empty(self) -> None:
        """(none) sentinel with leading/trailing whitespace is still recognised."""
        assert self.svc._extract_list_from_output("  (none)  ") == []

    def test_prose_with_embedded_paths_returns_tokens(self) -> None:
        """Prose narration that embeds slash-path tokens should yield those paths.

        GLM-5.x models sometimes output 'Now let me examine crates/state and
        crates/tools...' instead of a plain list. The secondary embedded-token scan
        should rescue the actual crate paths.
        """
        out = "Now let me examine crates/state and crates/tools to review the workspace."
        items = self.svc._extract_list_from_output(out)
        assert "crates/state" in items
        assert "crates/tools" in items

    def test_prose_without_embedded_paths_returns_empty(self) -> None:
        """Pure narration prose with no path-like tokens returns []."""
        out = "Now let me begin reading the files in the workspace to understand the structure."
        items = self.svc._extract_list_from_output(out)
        assert items == []

    def test_prose_with_embedded_snake_case_returns_tokens(self) -> None:
        """Snake_case identifiers embedded in prose are extracted as fallback items."""
        out = "I will now examine workspace_members and edge_runtime to understand the setup."
        items = self.svc._extract_list_from_output(out)
        assert "workspace_members" in items
        assert "edge_runtime" in items

    def test_xml_thinking_blocks_are_stripped(self) -> None:
        """<thinking>...</thinking> blocks from reasoning models must not appear as items.

        GLM-5.x emits <thinking> XML blocks in content.  Without stripping,
        '</thinking>' and '<tool_call >' surface as list items.
        """
        out = (
            "<thinking>\nLet me read the source files.\n</thinking>\n"
            "rust/crates/state\n"
            "rust/crates/tools\n"
            "<tool_call >\n<tool_name >read</tool_name >\n</tool_call >"
        )
        items = self.svc._extract_list_from_output(out)
        assert "rust/crates/state" in items
        assert "rust/crates/tools" in items
        # XML tags must not appear as items
        assert not any("<" in item or ">" in item for item in items)
        assert not any("thinking" in item for item in items)

    def test_malformed_tool_call_blocks_are_stripped(self) -> None:
        """Malformed provider tool-call markup must not become produced findings."""
        out = (
            "<tool_call questionable>\n"
            '{"name": "read_file", "arguments": {"path": "rust/crates/state/src/lib.rs"}}\n'
            "</tool_call]\n\n"
            "<tool_call questionable>\n"
            '{"name": "read_file", "arguments": {"path": "rust/crates/python-bindings/src/thinking.rs"}}\n'
            "</tool_call"
        )

        assert self.svc._extract_list_from_output(out) == []

    def test_deterministic_summary_prose_does_not_include_hyphenated_noise(
        self,
    ) -> None:
        """'Deterministic read-only execution...' prose must not yield 'read-only' as a token.

        When a step's only surviving line is the deterministic execution summary,
        the secondary path-token scan should return slash-paths extracted from the
        full output — not hyphenated adjectives like 'read-only'.
        """
        # Simulate: the sole line after extraction is the execution summary;
        # the full output also contains actual crate paths in thinking content.
        out = (
            "Deterministic read-only execution completed for shell: 1 succeeded, 0 failed.\n"
            "rust/crates/edge-runtime/src/agent\n"
            "rust/crates/edge-runtime/src/lib"
        )
        items = self.svc._extract_list_from_output(out)
        # Path tokens must be present
        assert any("/" in item for item in items)
        # 'read-only' must not appear (no "/" so filtered out)
        assert "read-only" not in items

    def test_prose_fallback_extraction_does_not_emit_warning(self, caplog) -> None:
        """Benign prose fallback extraction should not disrupt the progress display."""
        import logging

        out = (
            "Deterministic read-only execution completed for shell: 1 succeeded, 0 failed.\n"
            "rust/crates/edge-runtime/src/agent\n"
            "rust/crates/edge-runtime/src/lib"
        )

        with caplog.at_level(logging.WARNING, logger="victor.agent.services.planning_runtime"):
            items = self.svc._extract_list_from_output(out)

        assert items
        assert not caplog.records

    def test_xml_only_output_returns_empty(self) -> None:
        """Output consisting entirely of XML tags after stripping returns []."""
        out = "<thinking>\nsome internal reasoning\n</thinking>"
        items = self.svc._extract_list_from_output(out)
        assert items == []

    def test_truncation_message_filtered_from_list(self) -> None:
        """Lines containing file-truncation narration must not appear as list items.

        When sub-agents read large files, the tool emits 'The file was truncated.'
        and the model echoes 'Let me read the middle portion...'.  These lines
        contaminated per_crate_findings with prose instead of crate paths.
        """
        out = (
            "[rust/crates/protocol]\n"
            "The file was truncated. Let me read the middle portion...\n"
            "[rust/crates/state]\n"
            "Let me read the next section to get more context.\n"
            "[rust/crates/tools]\n"
        )
        items = self.svc._extract_list_from_output(out)
        # Real crate references must be present
        assert "rust/crates/protocol" in items or any("protocol" in i for i in items)
        assert "rust/crates/state" in items or any("state" in i for i in items)
        assert "rust/crates/tools" in items or any("tools" in i for i in items)
        # Truncation narration must be absent
        assert not any(
            "truncated" in i.lower() for i in items
        ), f"truncation phrase in items: {items}"
        assert not any(
            "let me read" in i.lower() for i in items
        ), f"continuation phrase in items: {items}"

    def test_firstresponder_halted_line_filtered(self) -> None:
        """'FirstResponderTool halted, entering general response mode.' must be filtered."""
        out = "FirstResponderTool halted, entering general response mode.\ncore\nutil"
        items = self.svc._extract_list_from_output(out)
        assert "core" in items
        assert "util" in items
        assert not any("halted" in i.lower() for i in items), f"halted phrase in items: {items}"

    def test_let_me_examine_filtered(self) -> None:
        """'Let me examine...' continuation lines must not appear as list items."""
        out = "rust/crates/protocol\nLet me examine the remaining files.\nrust/crates/state"
        items = self.svc._extract_list_from_output(out)
        assert not any("let me examine" in i.lower() for i in items)

    def test_clean_list_unaffected_by_filter(self) -> None:
        """A clean crate list without any continuation lines must pass through unchanged."""
        out = "rust/crates/protocol\nrust/crates/state\nrust/crates/tools"
        items = self.svc._extract_list_from_output(out)
        assert items == [
            "rust/crates/protocol",
            "rust/crates/state",
            "rust/crates/tools",
        ]

    def test_delta_symbol_items_stripped_from_output(self) -> None:
        """Items containing the ∂ character (U+2202) must be stripped from the extracted list.

        Regression: ZAI/GLM-5.1 uses ∂ as tool-call field separators in its output.
        Lines like 'read∂', '/path∂rust/crates/edge-runtime/src/agent.rs∂',
        '<_Tool_Dependency>∂' were leaking into per_crate_findings and cross_crate_findings,
        polluting plan_state with model-internal formatting artifacts.
        """
        # Simulate the output format GLM-5.1 produces when it interleaves tool calls
        out = (
            "rust/crates/protocol\n"
            "rust/crates/state\n"
            "Let me explore the edge-runtime submodules. ∂\n"  # ∂ suffix
            "∂\n"  # bare ∂ line
            "<_Tool_Dependency>∂\n"
            "read∂\n"
            "/path∂rust/crates/edge-runtime/src/agent.rs∂\n"
            "rust/crates/tools\n"
        )
        items = self.svc._extract_list_from_output(out)
        # Real crate paths must survive
        assert "rust/crates/protocol" in items
        assert "rust/crates/state" in items
        assert "rust/crates/tools" in items
        # ∂-bearing items must be removed
        assert not any(
            "∂" in i for i in items
        ), f"Items containing ∂ must be filtered; got: {items}"

    def test_actually_let_me_reread_continuation_filtered(self) -> None:
        """'Actually, let me re-read...' continuation lines must not appear as list items.

        Regression: the per_crate_findings list for step 7a contained
        'Actually, let me re-read specifying lines to get the truncated portion:'
        because _CONTINUATION_RE matched 'let me read' but not 'let me re-read'
        or sentences starting with 'Actually, let me'.
        """
        out = (
            "rust/crates/protocol\n"
            "Actually, let me re-read specifying lines to get the truncated portion:\n"
            "rust/crates/state\n"
            "Actually, let me look at this more carefully.\n"
            "rust/crates/tools\n"
        )
        items = self.svc._extract_list_from_output(out)
        # Real crate paths must survive
        assert "rust/crates/protocol" in items
        assert "rust/crates/state" in items
        assert "rust/crates/tools" in items
        # Continuation lines must be removed
        assert not any(
            "actually" in i.lower() for i in items
        ), f"'Actually...' continuation lines must be filtered; got: {items}"
        assert not any(
            "re-read" in i.lower() for i in items
        ), f"'let me re-read' lines must be filtered; got: {items}"

    def test_now_i_have_complete_picture_continuation_filtered(self) -> None:
        """'Now I have the complete picture...' must not appear as a list item.

        Regression: step 7a per_crate_findings list contained
        'Now I have the complete picture. This crate is a single-file crate...'
        between real crate bracket refs.
        """
        out = (
            "[rust/crates/protocol]\n"
            "Now I have the complete picture. This crate is a single-file crate (`src/lib.rs`)."
            " Let me review the truncated portion more carefully\n"
            "[rust/crates/state]\n"
            "Now I understand the codebase structure fully.\n"
            "[rust/crates/tools]\n"
        )
        items = self.svc._extract_list_from_output(out)
        assert "[rust/crates/protocol]" in items
        assert "[rust/crates/state]" in items
        assert "[rust/crates/tools]" in items
        assert not any(
            "now i have" in i.lower() for i in items
        ), f"'Now I have...' lines must be filtered; got: {items}"
        assert not any(
            "now i understand" in i.lower() for i in items
        ), f"'Now I understand...' lines must be filtered; got: {items}"

    def test_i_need_to_see_truncated_continuation_filtered(self) -> None:
        """'I need to see the truncated middle portion...' must not appear as a list item.

        Regression: step 7a per_crate_findings list contained
        'I need to see the truncated middle portion of lib.rs:'
        which is a planning statement, not a finding.
        """
        out = (
            "[rust/crates/state]\n"
            "I need to see the truncated middle portion of lib.rs:\n"
            "[rust/crates/tools]\n"
            "I need to read more of the source file to complete the analysis.\n"
        )
        items = self.svc._extract_list_from_output(out)
        assert "[rust/crates/state]" in items
        assert "[rust/crates/tools]" in items
        assert not any(
            "i need to" in i.lower() for i in items
        ), f"'I need to...' continuation lines must be filtered; got: {items}"


def test_produces_consumed_downstream_requires_nonempty_output():
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepType

    producer = PlanStep(
        id="2",
        description="Inventory review targets",
        step_type=StepType.RESEARCH,
        context={"produces": "review_targets"},
    )
    consumer = PlanStep(
        id="3",
        description="Analyze targets",
        step_type=StepType.RESEARCH,
        depends_on=["2"],
        inputs=["review_targets"],
    )
    plan = ExecutionPlan(id="p", goal="g", steps=[producer, consumer])

    assert PlanningRuntimeService._produces_consumed_downstream(plan, "2", "review_targets") is True


# ---------------------------------------------------------------------------
# Evidence contract: _is_directory_listing_only + _assess_step_evidence
# ---------------------------------------------------------------------------


class TestIsDirectoryListingOnly:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    def test_pure_ls_output_is_listing(self) -> None:
        ls_out = "src/main.rs\nsrc/lib.rs\nCargo.toml\nREADME.md"
        assert self.svc._is_directory_listing_only(ls_out) is True

    def test_source_file_content_is_not_listing(self) -> None:
        code = 'fn main() {\n    println!("hello");\n}\n'
        assert self.svc._is_directory_listing_only(code) is False

    def test_empty_string_is_not_listing(self) -> None:
        assert self.svc._is_directory_listing_only("") is False

    def test_mixed_paths_and_code_is_not_listing(self) -> None:
        # 30% code lines → below 70% threshold
        text = "src/main.rs\n" * 3 + "fn foo() {\n    return 1;\n}\n"
        assert self.svc._is_directory_listing_only(text) is False

    def test_paths_without_extension_below_threshold(self) -> None:
        # Lines without '/' and without extension → below 60% → not a listing
        text = "hello world\nno path here\njust text"
        assert self.svc._is_directory_listing_only(text) is False

    def test_bracket_header_listing_is_detected(self) -> None:
        # Subagent format that previously bypassed the detector:
        # alternating [path] header lines and bare path lines.  This was the
        # exact output that escaped plan step 7a / step 10 in plan_e53a5ce4.
        text = (
            "[rust/Cargo.toml]\n"
            "rust/Cargo.toml\n"
            "\n"
            "[rust/crates/edge-runtime/Cargo.toml]\n"
            "rust/crates/edge-runtime/Cargo.toml\n"
            "\n"
            "[rust/crates/protocol/Cargo.toml]\n"
            "rust/crates/protocol/Cargo.toml\n"
        )
        assert self.svc._is_directory_listing_only(text) is True

    def test_non_rust_path_listing_is_detected(self) -> None:
        # Generic path listing across languages — ensures the extension regex
        # is multi-language, not rust-specific.
        text = (
            "src/auth/login.py\n"
            "src/auth/oauth.py\n"
            "frontend/components/Button.tsx\n"
            "frontend/hooks/useAuth.ts\n"
            "build.gradle.kts\n"
            "pom.xml\n"
        )
        assert self.svc._is_directory_listing_only(text) is True


class TestIsDegenerateOutput:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    def test_single_unbroken_token_loop_is_degenerate(self) -> None:
        # Reproduces plan step 8 of plan_e53a5ce4 (EventHandlerExecutorAction loop).
        output = "EventHandlerExecutorAction" * 200
        is_deg, reason = self.svc._is_degenerate_output(output)
        assert is_deg is True
        assert "single unbroken" in reason

    def test_few_unique_tokens_with_many_repeats_is_degenerate(self) -> None:
        output = "foo bar baz " * 100
        is_deg, reason = self.svc._is_degenerate_output(output)
        assert is_deg is True

    def test_dominant_short_token_is_degenerate(self) -> None:
        # 50 repetitions of "noop" plus a small amount of filler — noop dominates
        output = ("noop " * 50) + "x" * 10
        is_deg, reason = self.svc._is_degenerate_output(output)
        assert is_deg is True

    def test_normal_long_output_is_not_degenerate(self) -> None:
        output = (
            "This is a detailed analysis of the authentication module. "
            "Several files were inspected including login handlers, session "
            "management, and middleware components. Three issues were found "
            "ranging from configuration drift to potential session fixation."
        ) * 5
        is_deg, reason = self.svc._is_degenerate_output(output)
        assert is_deg is False
        assert reason == ""

    def test_short_output_is_not_degenerate(self) -> None:
        # Outputs under 200 chars never trigger the detector — too noisy on
        # legitimately short statuses like "Done." or "Found 3 matches."
        is_deg, _ = self.svc._is_degenerate_output("a" * 199)
        assert is_deg is False

    def test_consecutive_bigram_repetition_is_degenerate(self) -> None:
        # 60 consecutive repetitions of the same token mid-output
        output = "Starting analysis. " + ("STUCK " * 60) + " then more content."
        is_deg, reason = self.svc._is_degenerate_output(output)
        assert is_deg is True


class TestAssessStepEvidence:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    @staticmethod
    def _make_step(desc: str = "Analyze code") -> SimpleNamespace:
        from victor.agent.planning.base import StepType

        return SimpleNamespace(
            id="step1",
            description=desc,
            step_type=StepType.RESEARCH,
            artifacts=[],
            context={},
        )

    @staticmethod
    def _make_result(output: str, tool_calls: int = 1, artifacts=None) -> StepResult:
        return StepResult(
            success=True,
            output=output,
            tool_calls_used=tool_calls,
            artifacts=artifacts or [],
        )

    def test_ls_only_output_fails_evidence_contract(self) -> None:
        step = self._make_step()
        # ls output: extension matches regex but no content
        result = self._make_result("src/main.rs\nsrc/lib.rs\nCargo.toml", tool_calls=1)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert evidence["is_directory_listing_only"] is True
        assert evidence["has_file_reference"] is False

    def test_file_read_output_passes_evidence_contract(self) -> None:
        step = self._make_step()
        code = (
            "fn parse_args() -> Args {\n"
            '    let matches = App::new("tool").get_matches();\n'
            '    Args { verbose: matches.is_present("verbose") }\n'
            "}"
        )
        result = self._make_result(code, tool_calls=2)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed
        assert evidence["has_content_tool"] is True

    def test_explicit_content_tool_in_metadata_passes(self) -> None:
        step = self._make_step()
        result = self._make_result("Some output without obvious code patterns", tool_calls=1)
        metadata = {"tool_names_used": ["read", "grep"]}
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, metadata)
        assert evidence["has_content_tool"] is True

    def test_zero_tool_calls_always_fails(self) -> None:
        step = self._make_step()
        result = self._make_result("fn foo() { return 42; }", tool_calls=0)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert "no tool-backed" in reason

    def test_write_step_passes_with_short_output(self) -> None:
        """Regression: synthesis/write steps produce short summaries, not file content.

        Bug: step 9 (type=doc, exec=agent, tools=[write]) used all 10 tool budget
        re-reading prior results and produced only 95 chars output — too short for the
        multi-tool analysis threshold (240 chars). Evidence contract failed despite the
        step having completed the write.
        Fix: write-tool steps pass evidence when tool_calls >= 1 and output >= 20 chars.
        """
        from victor.agent.planning.base import StepType

        step = SimpleNamespace(
            id="9",
            description="Synthesize all findings into a prioritized report",
            step_type=StepType.RESEARCH,  # doc maps to RESEARCH
            artifacts=[],
            context={},
            allowed_tools=["write"],
        )
        # 95 chars — realistic short confirmation from a write step
        result = self._make_result(
            "Report written to rust_best_practices_report.md (2847 words).",
            tool_calls=10,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed, f"write step should pass evidence contract; reason={reason}"
        assert evidence["is_write_step"] is True
        assert "write tool" in reason

    def test_non_write_step_still_requires_substantive_output(self) -> None:
        """Non-write steps still need 240+ chars or concrete file/scope evidence."""
        step = self._make_step("Review Arc usage in codebase")
        # 95 chars, no file refs, no code patterns
        result = self._make_result("Analysis complete. Found some patterns.", tool_calls=10)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert evidence["is_write_step"] is False

    def test_intent_phrase_with_5_plus_tool_calls_passes_evidence(self) -> None:
        """Regression: ZAI/GLM sub-agent final message is an intent phrase after running
        8 tool calls. The evidence contract must not reject the step in this case
        — 5+ actual tool executions are sufficient evidence of substantive work even when
        the spawn summary is accidentally captured as a planning statement.

        Production failure: step 8 (cross-crate analysis), tools=8, chars=62,
        output="Now let me read the source files to find cross-crate patterns:"
        """
        step = self._make_step("Cross-crate analysis: shared Arc patterns, redundant clones")
        result = self._make_result(
            "Now let me read the source files to find cross-crate patterns:",
            tool_calls=8,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed, (
            f"Step with 8 tool calls must pass even when spawn summary is an intent phrase; "
            f"reason={reason}"
        )

    def test_required_produces_intent_phrase_with_5_plus_tool_calls_fails_evidence(
        self,
    ) -> None:
        """Required plan-state artifacts must return content, not a spawn summary.

        Cross-crate findings can run several reads/searches and still end with a
        short intent phrase.  Tool count alone is not enough when downstream steps
        need the produced artifact text in plan_state.
        """
        from victor.agent.planning.base import StepType

        step = SimpleNamespace(
            id="8",
            description="Cross-crate analysis: shared Arc patterns, redundant clones",
            step_type=StepType.RESEARCH,
            artifacts=[],
            context={"produces": "cross_crate_findings"},
            allowed_tools=["read", "code_search"],
        )
        result = self._make_result(
            "Now let me read the source files to find cross-crate patterns:",
            tool_calls=8,
        )

        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})

        assert not passed
        assert "required produced artifact" in reason
        assert evidence["tool_calls_used"] == 8

    def test_required_produces_path_listing_artifact_fails_evidence(self) -> None:
        """Persisted artifacts are not enough when the produced value is only paths.

        Regression: cross-crate/final-report steps persisted a 180-char artifact
        containing only Cargo.toml paths, then passed because artifacts were present.
        """
        from victor.agent.planning.base import StepType

        step = SimpleNamespace(
            id="8",
            description="Cross-crate analysis: shared Arc patterns across crate boundaries",
            step_type=StepType.RESEARCH,
            artifacts=[],
            context={"produces": "cross_crate_findings"},
            allowed_tools=["read", "grep", "code_search"],
        )
        output = "\n".join(
            [
                "rust/Cargo.toml",
                "rust/crates/edge-runtime/Cargo.toml",
                "rust/crates/protocol/Cargo.toml",
                "rust/crates/python-bindings/Cargo.toml",
                "rust/crates/state/Cargo.toml",
                "rust/crates/tools/Cargo.toml",
            ]
        )
        result = self._make_result(
            output,
            tool_calls=6,
            artifacts=[".victor/plans/artifacts/p/step_8_cross_crate_findings.md"],
        )

        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})

        assert not passed
        assert "only a file/path listing" in reason
        assert evidence["has_artifacts"] is True
        assert evidence["requires_produced_artifact"] is True

    def test_required_produces_unresolved_tool_markup_artifact_fails_evidence(
        self,
    ) -> None:
        """Malformed reasoning/tool-call transcripts are not substantive findings."""
        from victor.agent.planning.base import StepType

        step = SimpleNamespace(
            id="7a",
            description="Deep review each workspace crate one-by-one",
            step_type=StepType.RESEARCH,
            artifacts=[],
            context={"produces": "per_crate_findings"},
            allowed_tools=["read", "grep", "code_search"],
        )
        output = (
            "[rust/crates/python-bindings/Cargo.toml]\n"
            "Let me read the source files.\n\n"
            '<tool_call name="read">\n'
            '<parameter name="path">rust/crates/python-bindings/src/lib.rs</parameter>\n'
            "</tool_call>\n"
        )
        result = self._make_result(
            output,
            tool_calls=10,
            artifacts=[".victor/plans/artifacts/p/step_7a_per_crate_findings.md"],
        )

        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})

        assert not passed
        assert "unresolved tool-call markup" in reason
        assert evidence["has_artifacts"] is True
        assert evidence["has_unresolved_tool_markup"] is True

    def test_intent_phrase_with_4_tool_calls_still_fails(self) -> None:
        """With fewer than 5 tool calls, the intent-phrase check applies normally.
        4 tools is below the 'substantive work' threshold.
        """
        step = self._make_step("Cross-crate analysis: shared Arc patterns")
        result = self._make_result(
            "Now let me read the source files to find cross-crate patterns:",
            tool_calls=4,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed, "4 tool calls with an intent-phrase output should still fail"
        assert "intent" in reason

    def test_intent_phrase_with_few_tools_still_rejected(self) -> None:
        """The existing intent-phrase rejection must fire for 1-2 tool calls
        — this prevents hollow steps from slipping through."""
        step = self._make_step("Analyze Arc usage")
        result = self._make_result("Let me use grep to search for Arc patterns.", tool_calls=2)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert "intent" in reason

    def test_multi_tool_substantive_output_passes_evidence_contract(self) -> None:
        """Cross-crate analysis with ≥5 tools and ≥100 chars must pass the evidence
        contract even without explicit file:line references in the text.

        Production failure (run ed439249): step 8 had tools=6, chars=180, no file refs.
        Evidence contract correctly rejected it, but then the rescue path in
        _execute_plan_via_team_adapter fired via 'Insufficient execution evidence' in
        _AGENTIC_LOOP_FP_PATTERNS, storing junk Cargo.toml paths as cross_crate_findings.

        Fix: add a general pass for ≥5 tool calls with ≥100-char output. This makes the
        evidence contract pass directly, removing the need for a rescue that stores bad data.
        """
        step = self._make_step("Cross-crate analysis: shared Arc patterns, redundant clones")
        result = self._make_result(
            "Cross-crate analysis complete. Edge-runtime depends on protocol and state. "
            "Python-bindings wraps all crates. No redundant Arc clones found at interfaces. "
            "Error types are consistent across crate boundaries.",
            tool_calls=6,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed, f"6 tools with 180+ chars must pass; reason={reason}"
        assert "substantive" in reason or "tool" in reason.lower()

    def test_multi_tool_pass_requires_100_chars_minimum(self) -> None:
        """The general multi-tool pass must NOT fire for thin outputs under 100 chars.
        A 39-char output with 10 tool calls should still fail the evidence contract.
        """
        step = self._make_step("Review Arc usage in codebase")
        result = self._make_result("Analysis complete. Found some patterns.", tool_calls=10)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert evidence["is_write_step"] is False

    def test_multi_tool_pass_requires_5_tool_calls_minimum(self) -> None:
        """The general multi-tool pass must NOT fire for fewer than 5 tool calls,
        even if the output is long enough."""
        step = self._make_step("Cross-crate analysis: shared Arc patterns, redundant clones")
        result = self._make_result(
            "Cross-crate analysis complete. Edge-runtime depends on protocol and state. "
            "Python-bindings wraps all crates. No redundant Arc clones found at interfaces. "
            "Error types are consistent across crate boundaries.",
            tool_calls=4,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed


# ---------------------------------------------------------------------------
# Plan-state: _skip_specific_steps
# ---------------------------------------------------------------------------


class TestSkipSpecificSteps:
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType

    @staticmethod
    def _make_exec_plan(*step_ids: str) -> "ExecutionPlan":
        from victor.agent.planning.base import (
            ExecutionPlan,
            PlanStep,
            StepStatus,
            StepType,
        )

        steps = [
            PlanStep(
                id=sid,
                description=f"step {sid}",
                step_type=StepType.RESEARCH,
                status=StepStatus.PENDING,
            )
            for sid in step_ids
        ]
        return ExecutionPlan(id="plan-1", goal="test", steps=steps)

    def test_marks_pending_steps_as_skipped(self) -> None:
        from victor.agent.planning.base import StepStatus

        plan = self._make_exec_plan("1", "2", "3")
        PlanningRuntimeService._skip_specific_steps(plan, ["2", "3"])
        statuses = {s.id: s.status for s in plan.steps}
        assert statuses["1"] == StepStatus.PENDING
        assert statuses["2"] == StepStatus.SKIPPED
        assert statuses["3"] == StepStatus.SKIPPED

    def test_does_not_skip_non_pending_steps(self) -> None:
        from victor.agent.planning.base import StepStatus

        plan = self._make_exec_plan("1", "2")
        plan.steps[1].status = StepStatus.COMPLETED
        PlanningRuntimeService._skip_specific_steps(plan, ["2"])
        assert plan.steps[1].status == StepStatus.COMPLETED

    def test_unknown_step_id_is_ignored(self) -> None:
        plan = self._make_exec_plan("1", "2")
        PlanningRuntimeService._skip_specific_steps(plan, ["99"])
        from victor.agent.planning.base import StepStatus

        assert all(s.status == StepStatus.PENDING for s in plan.steps)

    def test_empty_skip_list_is_noop(self) -> None:
        plan = self._make_exec_plan("1", "2")
        PlanningRuntimeService._skip_specific_steps(plan, [])
        from victor.agent.planning.base import StepStatus

        assert all(s.status == StepStatus.PENDING for s in plan.steps)

    def test_step_ids_coerced_from_strings(self) -> None:
        from victor.agent.planning.base import StepStatus

        plan = self._make_exec_plan("3", "4")
        PlanningRuntimeService._skip_specific_steps(plan, ["3"])
        assert plan.steps[0].status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# Plan-state integration: produces + skip_step_ids in execution loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_state_produces_key_is_stored_after_step():
    """tool/compute steps with 'produces' write their output list to plan_state."""
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    captured_plan_state: dict = {}

    async def _fake_execute_step(**kwargs) -> StepResult:
        # Capture the plan_state as received by step 2 (after step 1 produced it)
        if kwargs.get("plan_state"):
            captured_plan_state.update(kwargs["plan_state"])
        step = kwargs["step"]
        if step.id == "1":
            return StepResult(
                success=True,
                output="- core\n- util\n- cli",
                tool_calls_used=1,
                metadata={},
            )
        return StepResult(success=True, output="done", tool_calls_used=0, metadata={})

    plan = ReadableTaskPlan(
        name="State test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test produces propagation",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Discover members",
                "exec": "tool",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "feature",
                "desc": "Loop over members",
                "exec": "loop",
                "loop_over": "workspace_members",
                "deps": ["1"],
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute_step)

    # Bypass evidence contract so the test focuses on plan_state flow only.
    with patch.object(
        svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r
    ):
        await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # Step 2 should have received workspace_members extracted from step 1's output
    assert "workspace_members" in captured_plan_state
    assert captured_plan_state["workspace_members"] == ["core", "util", "cli"]


@pytest.mark.asyncio
async def test_plan_state_skip_step_ids_marks_branch_skipped():
    """When a conditional step returns skip_step_ids, downstream steps are marked SKIPPED."""
    from victor.agent.planning.base import StepResult, StepStatus
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    async def _fake_execute_step(**kwargs) -> StepResult:
        step = kwargs["step"]
        if step.id == "cond":
            return StepResult(
                success=True,
                output="condition evaluated",
                tool_calls_used=0,
                metadata={"skip_step_ids": ["3b"]},
            )
        return StepResult(success=True, output="done", tool_calls_used=0, metadata={})

    plan = ReadableTaskPlan(
        name="Conditional test",
        complexity=TaskComplexity.COMPLEX,
        desc="Branch routing",
        steps=[
            {
                "id": "cond",
                "type": "analyze",
                "desc": "Check workspace",
                "exec": "conditional",
            },
            {
                "id": "3a",
                "type": "feature",
                "desc": "Loop path",
                "exec": "loop",
                "deps": ["cond"],
            },
            {
                "id": "3b",
                "type": "feature",
                "desc": "Single path",
                "exec": "agent",
                "deps": ["cond"],
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute_step)

    # Bypass evidence contract to focus on branch routing logic.
    with patch.object(
        svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r
    ):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # cond and 3a completed; 3b was SKIPPED (not failed) so steps_completed == 2.
    assert result.steps_completed == 2
    assert result.steps_failed == 0


def test_toml_content_passes_evidence_contract():
    """Cargo.toml / TOML manifest reads must pass evidence even without code keywords."""
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = PlanStep(
        id="dep_audit",
        step_type=StepType.RESEARCH,
        description="Audit Cargo.toml dependencies for version pinning and feature-flag bloat",
        allowed_tools=["read", "grep"],
    )
    # Output that a real sub-agent would produce from reading Cargo.toml files:
    toml_output = (
        "rust/crates/python-bindings/Cargo.toml\n"
        "[dependencies]\n"
        'serde = { version = "1.0", features = ["derive"] }\n'
        'once_cell = "1.19"  # redundant with std::sync::LazyLock (Rust 1.80+)\n'
        'serde_yaml = "0.9"  # deprecated crate — migrate to serde_yml\n'
        "[dev-dependencies]\n"
        'criterion = "0.5"\n'
        'edition = "2021"\n'
    )
    step_result = StepResult(success=True, output=toml_output, tool_calls_used=2)
    result = svc._apply_step_evidence_contract(step, step_result)
    assert result.success, f"TOML dependency audit should pass evidence: {result.error}"


def test_report_cannot_waive_required_input_by_description():
    """A report is still blocked when its declared input failed."""
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    analysis = PlanStep(
        id="dep_audit",
        step_type=StepType.RESEARCH,
        description="Audit dependencies",
        allowed_tools=["read", "grep"],
    )
    analysis.status = StepStatus.FAILED

    synthesis = PlanStep(
        id="report",
        step_type=StepType.IMPLEMENTATION,
        description="Synthesize all findings into a report",
        allowed_tools=["write"],
        depends_on=["dep_audit"],
    )
    synthesis.status = StepStatus.PENDING

    plan = ExecutionPlan(id="test-plan", goal="audit rust", steps=[analysis, synthesis])
    svc._skip_team_plan_dependents(plan, ["dep_audit"])

    assert synthesis.status == StepStatus.BLOCKED
    assert synthesis.depends_on == ["dep_audit"]
    assert synthesis.result.metadata["blocked_by_failed_dependencies"] == ["dep_audit"]


@pytest.mark.asyncio
async def test_failed_branch_blocks_report_but_independent_work_continues():
    """Keep independent work and evidence without executing a blocked report."""
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    executed: list[str] = []

    async def _fake_execute(**kw) -> StepResult:
        step = kw["step"]
        executed.append(step.id)
        if step.id == "2":
            return StepResult(
                success=False,
                output="Now let me read the remaining files.",
                error="Insufficient execution evidence",
                tool_calls_used=3,
            )
        assert step.id != "3", "Required failed input must prevent dispatch"
        return StepResult(success=True, output="src/lib.rs:1 inspected", tool_calls_used=1)

    plan = ReadableTaskPlan(
        name="Independent work after failure",
        complexity=TaskComplexity.COMPLEX,
        desc="Verify independent work survives failed analysis",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Collect baseline findings"},
            {
                "id": "2",
                "type": "analyze",
                "desc": "Cross-crate analysis",
                "deps": ["1"],
                "tools": ["read", "grep"],
            },
            {
                "id": "3",
                "type": "doc",
                "desc": "Synthesize all findings into a prioritized report",
                "deps": ["1", "2"],
                "tools": ["write"],
            },
            {
                "id": "4",
                "type": "research",
                "desc": "Inspect independent README",
                "deps": [],
                "tools": ["read"],
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute)

    with (
        patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda s, r, *a, **kw: r),
        patch.object(svc, "_effective_team_plan_concurrency", return_value=1),
    ):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert set(executed) == {"1", "2", "4"}
    assert result.step_results["3"].success is False
    assert result.step_results["3"].tool_calls_used == 0
    assert result.metadata["plan_execution_state"]["step_statuses"]["3"] == "blocked"
    assert result.success is False


def test_failed_dependency_blocks_non_synthesis_dependents():
    """A failed predecessor still blocks effectful dependent steps.

    Implementation/testing work must not proceed on missing required inputs.
    """
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    failed_scan = PlanStep(
        id="dependency_profile",
        step_type=StepType.RESEARCH,
        description="Scan dependency profiles",
    )
    failed_scan.status = StepStatus.FAILED

    checklist = PlanStep(
        id="checklist",
        step_type=StepType.IMPLEMENTATION,
        description="Apply dependency profile fixes to source files",
        depends_on=["dependency_profile"],
    )
    downstream = PlanStep(
        id="review",
        step_type=StepType.TESTING,
        description="Run tests for dependency profile fixes",
        depends_on=["checklist"],
    )

    plan = ExecutionPlan(
        id="test-plan",
        goal="audit rust",
        steps=[failed_scan, checklist, downstream],
    )
    svc._skip_team_plan_dependents(plan, ["dependency_profile"])

    assert checklist.status == StepStatus.BLOCKED
    assert downstream.status == StepStatus.BLOCKED
    assert "dependency_profile" in (checklist.result.error or "")
    assert "checklist" in (downstream.result.error or "")
    assert plan.get_ready_steps() == []


def test_read_only_dependents_remain_blocked_transitively_without_rewriting_edges():
    """Analysis and report labels cannot override required dependencies."""
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    failed_scan = PlanStep(
        id="per_crate_review",
        step_type=StepType.RESEARCH,
        description="Deep per-crate Rust review",
    )
    failed_scan.status = StepStatus.FAILED

    cross_crate = PlanStep(
        id="cross_crate",
        step_type=StepType.RESEARCH,
        description="Cross-crate analysis of shared Arc patterns",
        allowed_tools=["read", "grep", "code_search"],
        depends_on=["per_crate_review"],
    )
    report = PlanStep(
        id="report",
        step_type=StepType.REVIEW,
        description="Synthesize all findings into a prioritized report",
        allowed_tools=["write"],
        depends_on=["cross_crate"],
    )

    plan = ExecutionPlan(
        id="test-plan",
        goal="audit rust",
        steps=[failed_scan, cross_crate, report],
    )
    svc._skip_team_plan_dependents(plan, ["per_crate_review"])

    assert cross_crate.status == StepStatus.BLOCKED
    assert cross_crate.depends_on == ["per_crate_review"]
    assert report.status == StepStatus.BLOCKED
    assert report.depends_on == ["cross_crate"]
    assert plan.get_ready_steps() == []


@pytest.mark.asyncio
async def test_clarification_fp_rescue_upgrades_failed_step_to_completed():
    """A step FAILED with 'Clarification required' but that produced valid output must be rescued.

    Regression: victor chat -p zai-coding step 5 (Create comprehensive Rust best practices
    checklist) — the sub-agent generated 2830 chars of valid checklist content, but the
    agentic loop decided EvaluationDecision.FAIL because PerceptionIntegration's
    underspecified_target check fired.  The planning engine then skipped all downstream
    steps even though the output WAS produced and the produces key had 4 items.

    Fix: when produces key has items AND output >= 100 chars AND error contains
    'Clarification required', rescue the step as COMPLETED.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepStatus, StepType
    from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    checklist_output = (
        "Arc vs Rc: Use Rc for single-threaded scenarios; use Arc only when data is shared across threads\n"
        "Immutable bindings: Use let instead of let mut wherever possible\n"
        "Concurrency safety: Ensure shared state is protected by Mutex/RwLock or is Arc-wrapped\n"
        "Cow usage: Use Cow<str> or Cow<[T]> when a function may return either borrowed or owned data\n"
    )

    plan = ReadableTaskPlan(
        name="Rescue test",
        complexity=TaskComplexity.COMPLEX,
        desc="Rust best practices rescue",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Map workspace",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "doc",
                "desc": "Create comprehensive Rust best practices checklist",
                "deps": ["1"],
                "produces": "best_practices_checklist",
                "exec": "compute",
            },
            {
                "id": "3",
                "type": "analyze",
                "desc": "Review workspace using checklist",
                "deps": ["2"],
                "inputs": ["workspace_members", "best_practices_checklist"],
                "produces": "review_findings",
                "tools": ["read", "grep"],
            },
        ],
    )

    call_counts: dict[str, int] = {"step1": 0, "step2": 0, "step3": 0}

    async def fake_execute(step, **kwargs):
        if step.id == "1":
            call_counts["step1"] += 1
            return StepResult(
                success=True,
                output="rust/crates/protocol\nrust/crates/state\nrust/crates/tools",
                tool_calls_used=1,
            )
        if step.id == "2":
            call_counts["step2"] += 1
            # Simulate what the real sub-agent did: agentic loop FAILED but content was produced.
            return StepResult(
                success=False,
                output=checklist_output,
                tool_calls_used=0,
                error="Clarification required: target artifact or scope is underspecified",
            )
        if step.id == "3":
            call_counts["step3"] += 1
            return StepResult(success=True, output="review done", tool_calls_used=5)
        return StepResult(success=True, output="ok", tool_calls_used=0)

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=fake_execute)

    with patch.object(
        svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r
    ):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # Step 2 should be rescued and step 3 should run (not skipped).
    assert call_counts["step2"] == 1, "Step 2 must have been attempted"
    assert (
        call_counts["step3"] == 1
    ), "Step 3 must run — step 2 should have been rescued, not treated as failed"
    assert result.steps_completed == 3, f"All 3 steps should complete; got {result.steps_completed}"
    assert result.steps_failed == 0, f"No failures expected after rescue; got {result.steps_failed}"


@pytest.mark.asyncio
async def test_synthesis_step_multi_tool_passes_evidence_contract_directly():
    """A cross-crate analysis step with ≥5 tools and ≥100-char output must pass the
    evidence contract directly — no rescue needed.

    Regression (run ed439249): step 8 (Cross-crate analysis, 6 tools, 180 chars) was
    correctly rejected by the evidence contract (no file refs), then *incorrectly* rescued
    via 'Insufficient execution evidence' in _AGENTIC_LOOP_FP_PATTERNS.  The rescue stored
    junk Cargo.toml paths as cross_crate_findings, making the downstream synthesis step
    report those paths as findings.

    Fix: _assess_step_evidence now passes ≥5 tools + ≥100 chars directly, so the evidence
    contract returns success=True without needing a rescue.  'Insufficient execution evidence'
    was removed from _AGENTIC_LOOP_FP_PATTERNS to prevent false rescues for legitimately
    thin steps.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepStatus, StepType
    from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    synthesis_output = (
        "Cross-crate analysis complete.\n"
        "Shared Arc patterns found across protocol and state crates.\n"
        "Redundant clone() calls identified at crate boundaries.\n"
        "Dependency coupling issues noted between edge-runtime and python-bindings.\n"
        "See per_crate_findings for crate-level detail.\n"
    )

    plan = ReadableTaskPlan(
        name="Synthesis rescue test",
        complexity=TaskComplexity.COMPLEX,
        desc="Rust cross-crate synthesis rescue",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Per-crate review",
                "produces": "per_crate_findings",
            },
            {
                "id": "2",
                "type": "analyze",
                "desc": "Cross-crate analysis: identify shared Arc patterns and redundant clones",
                "deps": ["1"],
                "inputs": ["per_crate_findings"],
                "produces": "cross_crate_findings",
                "tools": ["read", "grep"],
            },
            {
                "id": "3",
                "type": "doc",
                "desc": "Write consolidated report",
                "deps": ["2"],
                "inputs": ["cross_crate_findings"],
                "tools": ["write"],
            },
        ],
    )

    call_counts: dict[str, int] = {"step1": 0, "step2": 0, "step3": 0}

    async def fake_execute(step, **kwargs):
        if step.id == "1":
            call_counts["step1"] += 1
            # Realistic per-crate output with .rs file refs → passes evidence contract.
            return StepResult(
                success=True,
                output=(
                    "rust/crates/protocol/src/types.rs: 4 Arc<Config> usages, 2 shared patterns.\n"
                    "rust/crates/state/src/lib.rs: 3 Arc<SharedState> usages, shared Config.\n"
                    "rust/crates/tools/src/registry.rs: 2 Arc<Registry> singleton usages.\n"
                ),
                tool_calls_used=6,
            )
        if step.id == "2":
            call_counts["step2"] += 1
            # Agent ran 6 tools, produced 183-char substantive output.
            # Evidence contract must pass directly via ≥5 tools + ≥100 chars rule.
            return StepResult(
                success=True,
                output=synthesis_output,
                tool_calls_used=6,
            )
        if step.id == "3":
            call_counts["step3"] += 1
            return StepResult(success=True, output="report.md written", tool_calls_used=1)
        return StepResult(success=True, output="ok", tool_calls_used=0)

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=fake_execute)

    result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert call_counts["step2"] == 1, "Step 2 must have been attempted"
    assert (
        call_counts["step3"] == 1
    ), "Step 3 must run — step 2 passed the evidence contract directly"
    assert result.steps_completed == 3, f"All 3 steps should complete; got {result.steps_completed}"
    assert result.steps_failed == 0, f"No failures expected; got {result.steps_failed}"


@pytest.mark.asyncio
async def test_broad_rust_review_retries_agentically_after_hollow_output():
    """Broad review is qualitative-first and retries with stricter evidence criteria."""
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    plan = ReadableTaskPlan(
        name="Rust retry review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust workspace",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Parse workspace members and crate directory list from Cargo.toml",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "analyze",
                "desc": (
                    "Deep per-crate review: iterate over each workspace member analyzing "
                    "Arc usage and immutable patterns"
                ),
                "deps": ["1"],
                "produces": "per_crate_findings",
                "tools": ["read", "grep", "code_search"],
            },
        ],
    )
    calls: dict[str, int] = {"step1": 0, "step2": 0}
    retry_budgets: list[int] = []

    async def fake_execute(step, **kwargs):
        if step.id == "1":
            calls["step1"] += 1
            return StepResult(
                success=True,
                output="rust/crates/state",
                tool_calls_used=1,
                metadata={
                    "execution_mode": "builtin_compute",
                    "compute_node": "_workspace_members",
                },
            )
        if step.id == "2":
            calls["step2"] += 1
            retry_budgets.append(step.estimated_tool_calls)
            if calls["step2"] == 1:
                return StepResult(
                    success=True,
                    output="rust/Cargo.toml\nrust/crates/state/Cargo.toml",
                    tool_calls_used=6,
                )
            return StepResult(
                success=True,
                output=(
                    "`rust/crates/state/src/lib.rs:2`: `Arc<RwLock<String>>` is shared "
                    "runtime state; verify thread/task sharing is required before keeping Arc."
                ),
                tool_calls_used=12,
            )
        raise AssertionError(f"unexpected step {step.id}")

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=fake_execute)

    result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert calls == {"step1": 1, "step2": 2}
    assert result.success is True
    step2 = result.step_results["2"]
    assert step2.metadata["agentic_retry"]["retry_tool_budget"] >= 25
    assert retry_budgets[1] >= retry_budgets[0]
    assert "Arc<RwLock<String>>" in step2.output


def test_evidence_contract_exempts_synthesis_step_when_inputs_in_plan_state():
    """Evidence contract must be skipped for synthesis steps whose inputs are all in plan_state.

    Regression: step 8 cross-crate analysis declares inputs=['per_crate_findings'] which
    is already populated by step 7.  The evidence contract should not fire for such steps —
    they synthesize from collected data, and requiring file-ref patterns in output text is
    a false positive.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    step = PlanStep(
        id="8",
        step_type=StepType.RESEARCH,
        description="Cross-crate analysis: identify shared Arc patterns",
        inputs=["per_crate_findings"],
        context={"produces": "cross_crate_findings"},
    )

    step_result = StepResult(
        success=True,
        output="Cross-crate analysis: protocol and state share Arc<Config> pattern.",
        tool_calls_used=6,
    )

    # plan_state has per_crate_findings → synthesis step → contract exempt
    plan_state = {"per_crate_findings": ["protocol findings", "state findings"]}
    result = svc._apply_step_evidence_contract(step, step_result, plan_state)
    assert result.success is True, (
        f"Synthesis step with inputs in plan_state should be exempt from evidence contract; "
        f"got success={result.success}, error={getattr(result, 'error', None)}"
    )

    # Without plan_state the contract fires normally (not exempt)
    result_no_state = svc._apply_step_evidence_contract(step, step_result, plan_state=None)
    # With only 6 tools and 57 chars, evidence contract should fail (no file refs, low chars)
    # This verifies the contract IS active when plan_state is absent.
    assert result_no_state.success is False, (
        f"Without plan_state, evidence contract should reject thin 57-char research output; "
        f"got success={result_no_state.success}"
    )


def test_evidence_contract_not_exempt_for_research_step_with_gathering_tools():
    """Research step with data-gathering tools (read/grep/code_search) must NOT be exempt
    from evidence contract even when its declared inputs are present in plan_state.

    Regression: step 9 (cross-crate analysis, allowed_tools=['read','grep','code_search'])
    was being exempted because per_crate_findings was in plan_state.  This allowed the step
    to pass despite producing only an intent statement ("Let me use grep...") with 2 tools
    and 65 chars — yielding cross_crate_findings=[] and a hollow synthesis step downstream.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    step = PlanStep(
        id="9",
        step_type=StepType.RESEARCH,
        description="Cross-crate analysis: identify shared Arc patterns",
        inputs=["per_crate_findings"],
        allowed_tools=["read", "grep", "code_search"],
        context={"produces": "cross_crate_findings"},
    )
    step_result = StepResult(
        success=True,
        output="Let me use grep and file reading to analyze cross-crate patterns.",
        tool_calls_used=2,
        metadata={},
    )

    # plan_state has per_crate_findings — but step has gathering tools → NOT exempt
    plan_state = {"per_crate_findings": ["protocol findings", "state findings"]}
    result = svc._apply_step_evidence_contract(step, step_result, plan_state)
    assert result.success is False, (
        "Research step with data-gathering tools must not be exempt from evidence contract "
        "even when inputs are in plan_state; "
        f"got success={result.success}, error={getattr(result, 'error', None)}"
    )
    assert "Insufficient execution evidence" in (
        result.error or ""
    ), f"Expected evidence contract failure message; got error={result.error}"


@pytest.mark.asyncio
async def test_team_plan_retries_required_produces_when_extraction_is_empty():
    """A synthesis/report step retries when it produces only narration.

    Regression: the live run logged "Now let me examine..." for a final_report step.
    _extract_list_from_output correctly returned [], but the synthesis step was evidence
    exempt, so the plan still reported success with final_report=[].
    """
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source",
        steps=[
            {
                "id": "1",
                "type": "research",
                "desc": "Collect per-crate findings",
                "tools": "read",
                "produces": "per_crate_findings",
            },
            {
                "id": "2",
                "type": "doc",
                "desc": "Synthesize all findings into a prioritized report",
                "tools": "write",
                "deps": ["1"],
                "inputs": ["per_crate_findings"],
                "produces": "final_report",
            },
            {
                "id": "3",
                "type": "review",
                "desc": "Present consolidated report",
                "deps": ["2"],
            },
        ],
    )

    step_2_calls = 0

    async def _fake_execute(**kw) -> StepResult:
        nonlocal step_2_calls
        step = kw["step"]
        if step.id == "1":
            return StepResult(
                success=True,
                output=(
                    "- rust/crates/protocol/src/lib.rs: no Arc usage after full read\n"
                    "- rust/crates/state/src/lib.rs: Arc is used only for shared runtime state\n"
                    "- rust/crates/tools/src/lib.rs: clone usage is localized to tool metadata"
                ),
                tool_calls_used=6,
            )
        if step.id == "2":
            step_2_calls += 1
            if step_2_calls == 2:
                return StepResult(
                    success=True,
                    output=(
                        "## Prioritized report\n"
                        "- rust/crates/protocol/src/lib.rs: no Arc issue found.\n"
                        "- rust/crates/state/src/lib.rs: Arc usage is justified by shared state.\n"
                        "- rust/crates/tools/src/lib.rs: clone usage is localized."
                    ),
                    tool_calls_used=8,
                )
            return StepResult(
                success=True,
                output="Now let me examine the edge-runtime agent and provider modules for deeper analysis",
                tool_calls_used=4,
            )
        if step.id == "3":
            return StepResult(
                success=True, output="Presented consolidated report", tool_calls_used=0
            )
        raise AssertionError(f"unexpected execution of step {step.id}")

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute)

    result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert result.success is True
    assert result.steps_failed == 0
    assert result.step_results["2"].success is True
    assert result.step_results["2"].metadata["agentic_retry"]["original_error"]
    assert step_2_calls == 2


@pytest.mark.asyncio
async def test_team_plan_preserves_substantive_prose_for_required_produces():
    """Substantive prose analysis should feed downstream synthesis even without bullets."""
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    captured_plan_state: dict = {}
    prose = (
        "Cross-crate analysis found that protocol remains the leaf definition crate while "
        "state and tools depend on protocol types for shared request and tool metadata. "
        "edge-runtime combines protocol, state, and tools at the runtime boundary, which "
        "is the primary place where Arc-backed shared ownership should be expected. "
        "python-bindings bridges the Rust workspace into Python and should avoid cloning "
        "large protocol payloads across FFI boundaries unless conversion requires owned "
        "values. The dependency shape is acyclic and there is no evidence that root "
        "framework internals are imported back into lower-level definition crates. "
        "Optimization priorities are to keep protocol structs plain and cheaply movable, "
        "audit edge-runtime shared state for Arc<Mutex> hot paths, and keep cross-crate "
        "interfaces borrowing slices or references where lifetime boundaries allow it."
    )
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source",
        steps=[
            {
                "id": "1",
                "type": "research",
                "desc": "Collect cross-crate findings",
                "tools": "read",
                "produces": "cross_crate_findings",
            },
            {
                "id": "2",
                "type": "doc",
                "desc": "Synthesize all findings into a prioritized report",
                "deps": ["1"],
                "inputs": ["cross_crate_findings"],
            },
        ],
    )

    async def _fake_execute(**kw) -> StepResult:
        step = kw["step"]
        if step.id == "1":
            return StepResult(success=True, output=prose, tool_calls_used=5)
        captured_plan_state.update(kw["plan_state"])
        return StepResult(success=True, output="report complete", tool_calls_used=0)

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute)

    result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert result.success is True
    assert captured_plan_state["cross_crate_findings"] == [prose]
    assert mock_adapter.execute_step.await_count == 2


def test_required_checklist_produces_accepts_substantive_zero_tool_artifact():
    """A generated checklist can be a valid plan artifact without tool calls."""
    from victor.agent.planning.base import StepResult

    step = SimpleNamespace(
        step_type=SimpleNamespace(value="research"),
        execution="compute",
        context={"execution": "compute"},
        description="Create Rust best practices checklist covering Arc and performance",
        inputs=[],
    )
    output = (
        "Rust best practices checklist\n\n"
        "1. Prefer immutable let bindings unless mutation is required.\n"
        "2. Use Arc only for cross-thread shared ownership and Rc for single-thread sharing.\n"
        "3. Avoid cloning large payloads at crate boundaries; borrow slices where possible.\n"
        "4. Prefer Cow when data may be borrowed or owned depending on caller needs.\n"
        "5. Pre-size Vec and HashMap when expected sizes are known.\n"
        "6. Avoid unwrap in library code; propagate typed errors.\n"
        "7. Use RwLock only when read-heavy access patterns justify it.\n"
        "8. Audit async spawn boundaries for Send and lifetime correctness.\n"
        "9. Avoid boxing small uniform values unless trait objects are needed.\n"
        "10. Keep public APIs explicit about ownership transfer.\n"
    ) * 3
    result = StepResult(
        success=True,
        output=output,
        tool_calls_used=0,
        metadata={"evidence_validation": {"passed": True, "reason": "knowledge artifact"}},
    )

    coerced = PlanningRuntimeService._coerce_required_produces_items(
        step,
        result,
        "best_practices_checklist",
        [],
    )

    assert coerced == [output.strip()]


@pytest.mark.asyncio
async def test_team_plan_updates_progress_display_for_step_status_changes():
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    plan = ReadableTaskPlan(
        name="Short Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review one area",
        steps=[["1", "research", "Read source files", "read"]],
    )
    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(
        return_value=StepResult(
            success=True,
            output="Reviewed rust/crates/protocol/src/lib.rs:1 for Arc usage.",
            tool_calls_used=1,
        )
    )
    progress = MagicMock()
    with patch.object(svc, "_create_plan_progress_display", return_value=progress):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert result.success is True
    progress.start.assert_called_once()
    assert progress.update.call_count >= 2
    progress.stop.assert_called_once()


def test_plan_progress_display_hides_transitive_edges():
    steps = [
        SimpleNamespace(id="1", depends_on=[]),
        SimpleNamespace(id="2", depends_on=["1"]),
        SimpleNamespace(id="4", depends_on=["2"]),
        SimpleNamespace(id="5", depends_on=["2", "4"]),
    ]
    by_id = {step.id: step for step in steps}

    successors = _PlanProgressDisplay._build_reduced_successors(steps, by_id)

    assert successors["1"] == ["2"]
    assert successors["2"] == ["4"]
    assert successors["4"] == ["5"]


def test_plan_progress_display_renders_simple_outline_without_graph_jargon():
    steps = [
        SimpleNamespace(
            id="1",
            status=SimpleNamespace(value="completed"),
            result=SimpleNamespace(tool_calls_used=1),
            depends_on=[],
            description="Start",
        ),
        SimpleNamespace(
            id="2",
            status=SimpleNamespace(value="completed"),
            result=SimpleNamespace(tool_calls_used=0),
            depends_on=["1"],
            description="Workspace",
        ),
        SimpleNamespace(
            id="3",
            status=SimpleNamespace(value="pending"),
            result=None,
            depends_on=["2"],
            description="Inventory",
        ),
        SimpleNamespace(
            id="4",
            status=SimpleNamespace(value="pending"),
            result=None,
            depends_on=["2"],
            description="Checklist",
        ),
        SimpleNamespace(
            id="5",
            status=SimpleNamespace(value="pending"),
            result=None,
            depends_on=["2", "4"],
            description="Review checklist",
        ),
        SimpleNamespace(
            id="8",
            status=SimpleNamespace(value="blocked"),
            result=None,
            depends_on=["3", "5"],
            description="Cross-crate analysis",
        ),
    ]
    display = _PlanProgressDisplay(
        ReadableTaskPlan(
            name="DAG",
            complexity=TaskComplexity.COMPLEX,
            desc="dag",
            steps=[],
        ),
        SimpleNamespace(steps=steps),
    )

    panel = display._render_graph(title="Execution Graph")
    text = "\n".join(line.plain for line in panel.renderable.renderables)

    assert "2/6 done" in text
    assert "1 blocked" in text
    assert "parallel group (2 steps)" in text
    assert "edges:" not in text
    assert "L2" not in text
    assert "transitive edges hidden" not in text
    assert text.count("2 done") == 1
    assert text.count("5 pending") == 1
    assert "after 4" in text
    assert "after 2,4" not in text
    assert "(shown above)" not in text


def test_plan_progress_display_counts_skipped_steps_as_terminal_progress():
    steps = [
        SimpleNamespace(
            id="7a",
            status=SimpleNamespace(value="completed"),
            result=SimpleNamespace(tool_calls_used=0),
            depends_on=[],
            description="Multi-crate path",
        ),
        SimpleNamespace(
            id="7b",
            status=SimpleNamespace(value="skipped"),
            result=None,
            depends_on=[],
            description="Single-crate path",
        ),
    ]
    display = _PlanProgressDisplay(
        ReadableTaskPlan(
            name="Branch",
            complexity=TaskComplexity.COMPLEX,
            desc="branch",
            steps=[],
        ),
        SimpleNamespace(steps=steps),
    )

    panel = display._render_graph(title="Execution Graph")
    text = "\n".join(line.plain for line in panel.renderable.renderables)

    assert "2/2 terminal (1 done, 1 skipped)" in text
    assert "1/2 done" not in text


@pytest.mark.asyncio
async def test_failed_step_does_not_store_produces_key_in_plan_state():
    """A step that fails (success=False) must not write its produces_key to plan_state.

    Regression: even when a step's result has success=False (e.g. after evidence contract
    rejects it), the produces storage code used `if produces_key and step_result.output`
    which is truthy for any non-empty string output.  This silently stored empty findings
    (cross_crate_findings=[]) in plan_state, allowing downstream synthesis steps to proceed
    with no real data rather than being BLOCKED.
    """
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    captured_plan_state_step2: dict = {}

    async def _fake_execute(**kw) -> StepResult:
        step = kw["step"]
        ps = dict(kw.get("plan_state") or {})
        if step.id == "1":
            # step 1 fails with a truthy (non-empty) output — the classic regression case
            return StepResult(
                success=False,
                output="Let me analyze the crates.",
                error="Insufficient execution evidence",
                tool_calls_used=2,
                metadata={},
            )
        # step 2 — capture what's in plan_state after step 1 ran
        captured_plan_state_step2.update(ps)
        return StepResult(success=True, output="done", tool_calls_used=0, metadata={})

    plan = ReadableTaskPlan(
        name="Produces fail test",
        complexity=TaskComplexity.COMPLEX,
        desc="Failed step produces should not pollute plan_state",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Gather crate findings",
                "exec": "agent",
                "produces": "crate_findings",
            },
            {
                "id": "2",
                "type": "doc",
                "desc": "Step that runs after step 1",
                "exec": "agent",
                # no dep on step 1 — so it runs and we can observe plan_state
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute)

    # Bypass evidence contract and force sequential execution (1 at a time) so step 2
    # runs after step 1 and receives the live (post-step-1) plan_state.
    with (
        patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda s, r, *a, **kw: r),
        patch.object(svc, "_effective_team_plan_concurrency", return_value=1),
    ):
        await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert "crate_findings" not in captured_plan_state_step2, (
        "Failed step must not store its produces_key in plan_state; "
        f"plan_state seen by step 2: {list(captured_plan_state_step2.keys())}"
    )


def test_exempt_step_has_evidence_validation_metadata():
    """Steps exempt from evidence contract must have evidence_validation set in metadata
    (with exempt=True) so they appear as 'exempt' rather than 'missing' in the summary.

    Regression: execution-type-exempt steps (conditional, approval, compute) returned
    the original step_result unchanged — evidence_validation was never set in metadata,
    causing _format_evidence_validation_summary_for_summary to count them as 'missing'.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    step = PlanStep(
        id="7",
        step_type=StepType.RESEARCH,
        description="Route: multi-crate vs single crate",
        context={"execution": "conditional"},
    )
    step_result = StepResult(success=True, output="multi-crate", tool_calls_used=0, metadata={})

    result = svc._apply_step_evidence_contract(step, step_result)

    ev = (result.metadata or {}).get("evidence_validation")
    assert ev is not None, (
        "Exempt step must have evidence_validation in metadata; " f"metadata={result.metadata}"
    )
    assert (
        ev.get("exempt") is True
    ), f"Exempt step must have exempt=True in evidence_validation; got: {ev}"
    assert (
        ev.get("passed") is True
    ), f"Exempt step must be marked passed=True in evidence_validation; got: {ev}"


def test_builtin_compute_result_is_exempt_from_tool_backed_evidence_contract():
    """Deterministic compute artifacts can satisfy analyze steps without tool calls."""
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = PlanStep(
        id="8",
        step_type=StepType.RESEARCH,
        description="Cross-crate analysis: identify shared Arc patterns",
        allowed_tools=["read", "grep", "code_search"],
        context={"produces": "cross_crate_findings"},
    )
    step_result = StepResult(
        success=True,
        output="# Cross-Crate Rust Findings\n\n- `state` depends on `protocol`.",
        tool_calls_used=0,
        metadata={
            "execution_mode": "builtin_compute",
            "compute_node": "_cross_crate_findings",
        },
    )

    result = svc._apply_step_evidence_contract(step, step_result, {})

    assert result.success is True
    ev = result.metadata.get("evidence_validation")
    assert ev["passed"] is True
    assert ev["exempt"] is True
    assert ev["reason"] == "result-exec=builtin_compute"


def test_final_presentation_review_step_is_exempt_from_evidence_contract():
    """Presenting a finished report to the user is a checkpoint, not code analysis."""
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = PlanStep(
        id="11",
        step_type=StepType.REVIEW,
        description="Present consolidated report to user for feedback and discuss next steps for remediation",
        allowed_tools=[],
    )
    step_result = StepResult(success=True, output="Presented report", tool_calls_used=0)

    result = svc._apply_step_evidence_contract(step, step_result)

    assert result.success is True
    ev = result.metadata.get("evidence_validation")
    assert ev["passed"] is True
    assert ev["exempt"] is True
    assert ev["reason"] == "not-required:step-type"


def test_final_report_output_is_persisted_as_plan_artifact(tmp_path, monkeypatch):
    """Long report outputs should be durable even when terminal summaries truncate."""
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepResult, StepType

    monkeypatch.chdir(tmp_path)
    step = PlanStep(
        id="9",
        step_type=StepType.RESEARCH,
        description="Synthesize all findings into a prioritized report",
        context={"produces": "final_report"},
    )
    execution_plan = ExecutionPlan(id="Rust Arc Review 123", goal="Review Rust", steps=[step])
    output = "# Rust Best Practices Report\n\n" + "\n".join(
        f"- finding {index}: `rust/crates/state/src/lib.rs:{index}`" for index in range(80)
    )
    step_result = StepResult(
        success=True,
        output=output,
        tool_calls_used=0,
        metadata={"execution_mode": "builtin_compute"},
    )

    persisted = PlanningRuntimeService._persist_step_artifact_if_needed(
        execution_plan,
        step,
        step_result,
    )

    assert persisted.artifacts
    artifact_path = tmp_path / persisted.artifacts[0]
    assert artifact_path.exists()
    assert artifact_path.name == "step_9_final_report.md"
    assert "Rust Best Practices Report" in artifact_path.read_text()
    assert persisted.metadata["plan_artifact_path"] == persisted.artifacts[0]
    assert persisted.metadata["plan_artifact_bytes"] == len(output.encode("utf-8"))


def test_summary_prompt_includes_step_artifact_paths():
    """Final summaries should point users to durable report artifacts."""
    from victor.agent.planning.base import PlanResult

    service = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    plan = ReadableTaskPlan(
        name="Artifact summary",
        complexity=TaskComplexity.COMPLEX,
        desc="Write report",
        steps=[
            {
                "id": "9",
                "type": "doc",
                "desc": "Synthesize findings into report",
                "produces": "final_report",
            }
        ],
    )
    result = PlanResult(
        plan_id="p1",
        success=True,
        total_steps=1,
        steps_completed=1,
        step_results={
            "9": StepResult(
                success=True,
                output="# Report\n\ntruncated preview",
                artifacts=[".victor/plans/artifacts/p1/step_9_final_report.md"],
            )
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert "Artifacts:" in prompt
    assert ".victor/plans/artifacts/p1/step_9_final_report.md" in prompt


def test_final_response_appends_missing_artifact_paths():
    """Artifact paths must be user-visible even if provider summary omits them."""
    result = PlanResult(
        plan_id="p1",
        success=True,
        total_steps=1,
        steps_completed=1,
        step_results={
            "9": StepResult(
                success=True,
                output="# Report",
                artifacts=[".victor/plans/artifacts/p1/step_9_final_report.md"],
            )
        },
    )
    response = CompletionResponse(content="Summary without paths", role="assistant")

    PlanningRuntimeService._append_artifact_paths_to_response(response, result)

    assert "Full artifacts:" in response.content
    assert ".victor/plans/artifacts/p1/step_9_final_report.md" in response.content


def test_final_response_does_not_duplicate_existing_artifact_paths():
    result = PlanResult(
        plan_id="p1",
        success=True,
        total_steps=1,
        steps_completed=1,
        step_results={
            "9": StepResult(
                success=True,
                output="# Report",
                artifacts=[".victor/plans/artifacts/p1/step_9_final_report.md"],
            )
        },
    )
    response = CompletionResponse(
        content="Full report: .victor/plans/artifacts/p1/step_9_final_report.md",
        role="assistant",
    )

    PlanningRuntimeService._append_artifact_paths_to_response(response, result)

    assert "Full artifacts:" not in response.content
    assert response.content.count(".victor/plans/artifacts/p1/step_9_final_report.md") == 1


def test_evidence_summary_counts_exempt_separately_from_missing():
    """Evidence validation summary must show 'exempt=N' for legitimately bypassed steps
    rather than lumping them into 'missing=N'.

    'missing' should only reflect steps where evidence_validation was never set due to a
    code bug — not steps where the evidence contract was intentionally skipped.
    """
    from victor.agent.planning.base import PlanResult, PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    plan = ReadableTaskPlan(
        name="Summary test",
        complexity=TaskComplexity.COMPLEX,
        desc="test",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Conditional route",
                "exec": "conditional",
            },
        ],
    )

    execution_plan = plan.to_execution_plan()
    step = execution_plan.steps[0]
    step_result = StepResult(success=True, output="multi-crate", tool_calls_used=0, metadata={})
    result = svc._apply_step_evidence_contract(step, step_result)

    plan_result = PlanResult(
        plan_id="test",
        success=True,
        total_steps=1,
        step_results={"1": result},
    )

    summary = svc._format_evidence_validation_summary_for_summary(plan, plan_result)
    assert summary, "Summary should not be empty"
    assert "missing=0" in summary, f"Exempt steps must not count as missing; summary: {summary}"
    assert (
        "exempt=1" in summary
    ), f"Exempt steps must appear as exempt=1 in summary; summary: {summary}"


def test_artifact_produces_preserves_markdown_as_single_plan_state_item():
    service = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = SimpleNamespace(
        id="8a",
        description="Deep review each workspace member crate",
        context={"produces": "per_crate_findings"},
    )
    output = (
        "# Rust Crate Review: workspace\n\n"
        "## state\n\n"
        "- `rust/crates/state/src/lib.rs:2`: `Arc<` shared ownership\n"
        "- `rust/crates/state/src/lib.rs:3`: `.clone()` owned clone\n"
    )
    step_result = StepResult(
        success=True,
        output=output,
        tool_calls_used=0,
        metadata={"evidence_validation": {"passed": True, "reason": "compute_node"}},
    )
    extracted = service._extract_list_from_output(output)

    coerced = service._coerce_required_produces_items(
        step,
        step_result,
        "per_crate_findings",
        extracted,
    )

    assert coerced == [output.strip()]


async def test_standalone_direct_chat_acquires_public_turn_boundary():
    response = CompletionResponse(content="standalone", role="assistant")
    owner = SimpleNamespace(chat=AsyncMock(return_value=response))
    service = PlanningRuntimeService(owner)
    assert await service._direct_chat("hello") is response
    owner.chat.assert_awaited_once_with("hello")


def test_evidence_validation_preserves_blocked_dispatch_status():
    from victor.agent.planning.base import ExecutionPlan, PlanStep, PlanResult, StepStatus, StepType

    service = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    blocked = PlanStep(id="2", description="Review", step_type=StepType.REVIEW, depends_on=["1"])
    blocked.status = StepStatus.BLOCKED
    blocked.result = StepResult(success=False, output="", error="Required dependency 1 failed")
    plan = ExecutionPlan(id="plan", goal="review", steps=[blocked])
    result = PlanResult(
        plan_id="plan", success=False, total_steps=1, step_results={"2": blocked.result}
    )
    service._apply_plan_evidence_contracts(plan, result)
    assert blocked.status == StepStatus.BLOCKED
    assert result.steps_failed == 0
    assert result.steps_completed == 0
    assert not result.success

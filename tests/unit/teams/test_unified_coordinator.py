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

"""Tests for UnifiedTeamCoordinator."""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional, TypedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.teams import (
    AgentMessage,
    ITeamCoordinator,
    MemberResult,
    MessageType,
    TeamConfig,
    TeamFormation,
    TeamMember,
    TeamResult,
    UnifiedTeamCoordinator,
    create_coordinator,
)
from victor.protocols.team import IDelegateFollowUpCoordinator, ITeamMember
from victor.teams.worktree_runtime import WorktreeRuntimeError


class MockTeamMember:
    """Mock team member for testing."""

    def __init__(self, member_id: str, output: str = "Done"):
        self._id = member_id
        self._output = output
        self._role = MagicMock()
        self._messages: list = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> Any:
        return self._role

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        return self._output

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        self._messages.append(message)
        return None


class StructuredMember(MockTeamMember):
    """Mock member that returns structured execution metadata."""

    def __init__(
        self,
        member_id: str,
        output: str = "Done",
        *,
        changed_files: Optional[list[str]] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(member_id, output=output)
        self._changed_files = list(changed_files or [])
        self._success = success
        self._metadata = dict(metadata or {})
        self.seen_contexts: list[Dict[str, Any]] = []

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.seen_contexts.append(dict(context))
        return {
            "success": self._success,
            "output": self._output,
            "changed_files": list(self._changed_files),
            "metadata": dict(self._metadata),
            "tool_calls_used": 2,
            "discoveries": [f"handled:{self._id}"],
        }


class FailingMember(MockTeamMember):
    """Mock member that always fails."""

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        raise RuntimeError("Task failed")


class TestProtocolCompliance:
    """Tests for ITeamCoordinator protocol compliance."""

    def test_implements_iteamcoordinator(self):
        """UnifiedTeamCoordinator must implement ITeamCoordinator."""
        coordinator = UnifiedTeamCoordinator()
        # Check protocol methods exist
        assert hasattr(coordinator, "add_member")
        assert hasattr(coordinator, "set_formation")
        assert hasattr(coordinator, "execute_task")
        assert hasattr(coordinator, "broadcast")

    def test_implements_delegate_follow_up_capability(self):
        """UnifiedTeamCoordinator should expose delegate follow-up execution capability."""
        coordinator = UnifiedTeamCoordinator()

        assert isinstance(coordinator, IDelegateFollowUpCoordinator)
        assert hasattr(coordinator, "execute_follow_up_request")
        assert hasattr(coordinator, "execute_follow_up_contract")

    def test_fluent_interface(self):
        """Methods should return self for chaining."""
        coordinator = UnifiedTeamCoordinator()
        member = MockTeamMember("m1")

        result = coordinator.add_member(member).set_formation(TeamFormation.PARALLEL)
        assert result is coordinator

    def test_factory_returns_protocol(self):
        """Factory should return ITeamCoordinator implementation."""
        mock_orch = MagicMock()
        coordinator = create_coordinator(orchestrator=mock_orch)
        assert hasattr(coordinator, "add_member")
        assert hasattr(coordinator, "execute_task")

    def test_merge_review_contract_delegates_to_workspace_isolation_service(self):
        """Merge/review contract ownership should stay in workspace isolation service."""
        coordinator = UnifiedTeamCoordinator(lightweight_mode=True)
        workspace = MagicMock()
        workspace.build_merge_review_contract.return_value = {"next_action": "review"}
        coordinator._workspace_isolation = workspace

        result = coordinator._build_merge_review_contract(
            {"worker": {"member_id": "worker"}},
            merge_analysis={"risk_level": "medium"},
            merge_orchestration={"merge_execution_eligible": False},
        )

        assert result == {"next_action": "review"}
        workspace.build_merge_review_contract.assert_called_once_with(
            {"worker": {"member_id": "worker"}},
            merge_analysis={"risk_level": "medium"},
            merge_orchestration={"merge_execution_eligible": False},
        )


class TestExecuteTeamCompatibility:
    """Tests for legacy execute_team entry point."""

    @pytest.mark.asyncio
    async def test_execute_team_invokes_callback_for_each_member(self):
        """Compatibility execute_team should forward per-member completion callbacks."""
        coordinator = UnifiedTeamCoordinator(lightweight_mode=True)

        config = TeamConfig(
            name="Legacy test",
            goal="Run legacy compatibility path",
            members=[
                TeamMember(
                    id="member-1",
                    role="researcher",
                    name="Researcher",
                    goal="Run test",
                ),
                TeamMember(
                    id="member-2",
                    role="executor",
                    name="Executor",
                    goal="Fix test",
                ),
            ],
            formation=TeamFormation.SEQUENTIAL,
        )

        expected_result = TeamResult(
            success=True,
            final_output="done",
            member_results={
                "member-1": MemberResult(
                    member_id="member-1",
                    success=True,
                    output="research complete",
                ),
                "member-2": MemberResult(
                    member_id="member-2",
                    success=False,
                    output="execution failed",
                ),
            },
            formation=TeamFormation.SEQUENTIAL,
            total_tool_calls=4,
            total_duration=0.45,
        )

        # Execute_team should use execute_team_config as compatibility path.
        coordinator.execute_team_config = AsyncMock(return_value=expected_result)  # type: ignore[method-assign]
        callback = MagicMock()

        result = await coordinator.execute_team(config, callback)

        assert result is expected_result
        assert result is coordinator.execute_team_config.return_value
        # callback receives one call per member.
        assert callback.call_count == 2
        callback_calls = [c.args for c in callback.mock_calls if c.args]
        assert len(callback_calls) == 2
        assert any(
            call_args[0] == "member-1"
            and call_args[1]
            == MemberResult(
                member_id="member-1",
                success=True,
                output="research complete",
            )
            for call_args in callback_calls
        )
        assert any(
            call_args[0] == "member-2"
            and call_args[1]
            == MemberResult(
                member_id="member-2",
                success=False,
                output="execution failed",
            )
            for call_args in callback_calls
        )


class TestFormations:
    """Tests for different formation patterns."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Sequential formation should execute members in order."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Result1"))
        coordinator.add_member(MockTeamMember("m2", "Result2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert "m1" in result["member_results"]
        assert "m2" in result["member_results"]
        assert result["formation"] == "sequential"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Parallel formation should execute all members concurrently."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Result1"))
        coordinator.add_member(MockTeamMember("m2", "Result2"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert len(result["member_results"]) == 2
        assert result["formation"] == "parallel"

    @pytest.mark.asyncio
    async def test_parallel_execution_preserves_parent_handoff_metadata(self):
        """Parallel member results should preserve lifecycle handoff metadata."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(
            StructuredMember(
                "m1",
                "Result1",
                metadata={
                    "agent_id": "agent_m1",
                    "team_id": "team_1",
                    "parent_handoff": {"summary": "bounded m1"},
                },
            )
        )
        coordinator.add_member(
            StructuredMember(
                "m2",
                "Result2",
                metadata={
                    "agent_id": "agent_m2",
                    "team_id": "team_1",
                    "parent_handoff": {"summary": "bounded m2"},
                },
            )
        )
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {"request_team_id": "team_1"})

        assert result["formation"] == "parallel"
        member_1 = result["member_results"]["m1"]
        member_2 = result["member_results"]["m2"]
        member_1_metadata = (
            member_1.metadata if hasattr(member_1, "metadata") else member_1["metadata"]
        )
        member_2_metadata = (
            member_2.metadata if hasattr(member_2, "metadata") else member_2["metadata"]
        )
        assert member_1_metadata["agent_id"] == "agent_m1"
        assert member_2_metadata["parent_handoff"]["summary"] == "bounded m2"

    @pytest.mark.asyncio
    async def test_context_reserved_team_keys_do_not_conflict_with_team_context(self):
        """Planning context may already include TeamContext constructor keys."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        member = StructuredMember("m1", "Result1")
        coordinator.add_member(member)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task(
            "Test task",
            {
                "team_id": "team_1",
                "formation": "hierarchical",
                "shared_state": {"caller": "planning"},
            },
        )

        assert result["success"] is True
        assert result["formation"] == "sequential"
        assert result["shared_context"]["caller"] == "planning"
        assert result["shared_context"]["team_id"] == "team_1"
        assert member.seen_contexts[0]["team_id"] == "team_1"

    @pytest.mark.asyncio
    async def test_formation_hint_overrides_default_for_single_execution(self):
        """Per-call formation hints should override the coordinator default."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Result1"))
        coordinator.add_member(MockTeamMember("m2", "Result2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test task", {"formation_hint": "parallel"})

        assert result["success"] is True
        assert result["formation"] == "parallel"
        assert coordinator.formation == TeamFormation.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_hierarchical_execution(self):
        """Hierarchical formation should have supervisor plan and synthesize."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        manager = MockTeamMember("manager", "Plan: do X")
        worker = MockTeamMember("worker", "Did X")
        coordinator.add_member(manager)
        coordinator.add_member(worker)
        coordinator.set_supervisor(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert coordinator.supervisor is manager
        assert coordinator.manager is manager
        assert "manager" in result["member_results"]
        assert "worker" in result["member_results"]
        assert result["formation"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Pipeline formation should chain outputs."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("stage1", "Stage1Output"))
        coordinator.add_member(MockTeamMember("stage2", "Stage2Output"))
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Initial input", {})

        assert result["success"] is True
        assert result["final_output"] == "Stage2Output"  # Last stage output
        assert result["formation"] == "pipeline"

    @pytest.mark.asyncio
    async def test_max_workers_limits_execution_members(self):
        """Per-call max_workers should limit the participating members."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Result1"))
        coordinator.add_member(MockTeamMember("m2", "Result2"))
        coordinator.add_member(MockTeamMember("m3", "Result3"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {"max_workers": 2})

        assert result["success"] is True
        assert result["formation"] == "parallel"
        assert len(result["member_results"]) == 2

    @pytest.mark.asyncio
    async def test_consensus_execution(self):
        """Consensus formation should require agreement."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Agreed"))
        coordinator.add_member(MockTeamMember("m2", "Agreed"))
        coordinator.set_formation(TeamFormation.CONSENSUS)

        result = await coordinator.execute_task("Test task", {"max_consensus_rounds": 1})

        assert result["success"] is True
        assert result.get("consensus_achieved") is True
        assert result["formation"] == "consensus"

    @pytest.mark.asyncio
    async def test_consensus_execution_preserves_member_identity_metadata(self):
        """Consensus should retain each member's identity metadata for attribution."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(
            StructuredMember(
                "reviewer_1",
                "Agreed",
                metadata={"agent_id": "agent_reviewer_1", "team_id": "team_consensus"},
            )
        )
        coordinator.add_member(
            StructuredMember(
                "reviewer_2",
                "Agreed",
                metadata={"agent_id": "agent_reviewer_2", "team_id": "team_consensus"},
            )
        )
        coordinator.set_formation(TeamFormation.CONSENSUS)

        result = await coordinator.execute_task("Review proposal", {"max_consensus_rounds": 1})

        assert result["formation"] == "consensus"
        assert result.get("consensus_achieved") is True
        reviewer_1 = result["member_results"]["reviewer_1"]
        reviewer_2 = result["member_results"]["reviewer_2"]
        reviewer_1_metadata = (
            reviewer_1.metadata if hasattr(reviewer_1, "metadata") else reviewer_1["metadata"]
        )
        reviewer_2_metadata = (
            reviewer_2.metadata if hasattr(reviewer_2, "metadata") else reviewer_2["metadata"]
        )
        assert reviewer_1_metadata["agent_id"] == "agent_reviewer_1"
        assert reviewer_2_metadata["team_id"] == "team_consensus"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_no_members_error(self):
        """Should fail gracefully with no members."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert "No team members" in result["error"]

    @pytest.mark.asyncio
    async def test_member_failure_sequential(self):
        """Sequential should handle member failures."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "OK"))
        coordinator.add_member(FailingMember("m2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert result["member_results"]["m2"].success is False
        assert result["member_results"]["m2"].error is not None

    @pytest.mark.asyncio
    async def test_member_failure_parallel(self):
        """Parallel should handle member failures."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "OK"))
        coordinator.add_member(FailingMember("m2"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert result["member_results"]["m1"].success is True
        assert result["member_results"]["m2"].success is False

    @pytest.mark.asyncio
    async def test_structured_member_outputs_feed_worktree_plan_and_merge_analysis(
        self,
    ):
        """Structured member outputs should drive isolation metadata and merge analysis."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        planner = StructuredMember("planner", "Planned", changed_files=["src/auth/service.py"])
        tester = StructuredMember("tester", "Tested", changed_files=["tests/auth/test_service.py"])
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task(
            "Implement auth flow",
            {
                "team_name": "feature_team",
                "worktree_isolation": True,
                "repo_root": "/repo/project",
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
                "shared_readonly_paths": ["docs"],
            },
        )

        assert result["success"] is True
        assert result["merge_risk_level"] == "low"
        assert result["worktree_plan"]["team_name"] == "feature_team"
        assert planner.seen_contexts[0]["isolation_mode"] == "worktree"
        assert planner.seen_contexts[0]["workspace_root"].endswith("feature_team-planner")
        assert result["member_results"]["planner"].metadata["worktree_assignment"]["member_id"] == (
            "planner"
        )
        assert result["merge_analysis"]["member_changed_files"]["tester"] == [
            "tests/auth/test_service.py"
        ]

    @pytest.mark.asyncio
    async def test_structured_failure_payload_marks_member_failed(self):
        """Structured failure payloads should be preserved without raising."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(
            StructuredMember(
                "m1",
                output="",
                success=False,
                metadata={"failure_stage": "merge"},
            )
        )

        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert result["member_results"]["m1"].success is False
        assert result["member_results"]["m1"].metadata["failure_stage"] == "merge"

    @pytest.mark.asyncio
    async def test_structured_outputs_emit_worker_return_contracts(self):
        """Delegated executions should emit concise per-worker return contracts."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        planner = StructuredMember(
            "planner",
            "Updated auth service and documented the patch.",
            changed_files=["src/auth/service.py"],
            metadata={
                "task_summary": "Patched auth service error handling",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated the flow against auth tests.",
            changed_files=["src/auth/helpers.py"],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_helpers.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task(
            "Implement auth flow",
            {
                "team_name": "feature_team",
                "worktree_isolation": True,
                "repo_root": "/repo/project",
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        contracts = result["worker_return_contracts"]

        assert contracts["planner"]["task_summary"] == "Patched auth service error handling"
        assert contracts["planner"]["changed_files"] == ["src/auth/service.py"]
        assert contracts["planner"]["validation_run"]["status"] == "passed"
        assert contracts["planner"]["merge_risk"]["level"] == "low"
        assert contracts["tester"]["task_summary"] == "Validated the flow against auth tests."
        assert contracts["tester"]["validation_run"]["summary"] == "1 failed"
        assert contracts["tester"]["merge_risk"]["level"] == "medium"
        assert contracts["tester"]["merge_risk"]["reasons"] == ["out_of_scope_writes"]
        assert contracts["tester"]["merge_risk"]["out_of_scope_writes"] == ["src/auth/helpers.py"]
        review_contract = result["merge_review_contract"]
        assert review_contract["merge_ready"] is False
        assert review_contract["review_required"] is True
        assert review_contract["recommended_merge_order"] == ["planner", "tester"]
        assert review_contract["review_required_members"] == ["tester"]
        assert review_contract["validation_failed_members"] == ["tester"]
        assert review_contract["next_action"] == "fix_validation"
        assert review_contract["blocking_issues"][0]["type"] == "validation_failed"
        assert review_contract["blocking_issues"][1]["type"] == "merge_risk_medium"

    @pytest.mark.asyncio
    async def test_materialized_worktree_runtime_enriches_results_and_cleans_up(self):
        """Explicit worktree materialization should add session and merge orchestration metadata."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("m1",))
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_path": "/tmp/feature-m1",
                            "materialized": True,
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={"recommended_merge_order": ["m1"], "materialized": True}
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        result = await coordinator.execute_task(
            "Implement feature",
            {
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "materialize_worktrees": True,
            },
        )

        assert result["success"] is True
        fake_runtime.materialize.assert_called_once()
        fake_runtime.collect_changed_files.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert result["worktree_session"]["materialized"] is True
        assert result["merge_orchestration"]["materialized"] is True
        assert result["merge_review_contract"]["next_action"] == "merge"
        assert result["member_results"]["m1"].metadata["changed_files"] == ["src/auth/service.py"]
        assert result["worktree_cleanup"]["removed"] == ["/tmp/feature-m1"]

    @pytest.mark.asyncio
    async def test_delegate_mode_auto_materializes_worktrees(self):
        """Delegate mode should make worktree isolation turnkey without extra flags."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("m1",))
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_path": "/tmp/feature-m1",
                            "materialized": True,
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={"recommended_merge_order": ["m1"], "materialized": True}
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        result = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
            },
        )

        fake_runtime.materialize.assert_called_once()
        assert result["worktree_session"]["materialized"] is True
        assert result["merge_review_contract"]["next_action"] == "merge"

    @pytest.mark.asyncio
    async def test_delegate_mode_surfaces_workspace_diagnostics_in_follow_up_contract(
        self,
    ):
        """Materialization failures should be visible without bypassing delegate contracts."""
        fake_runtime = SimpleNamespace(
            materialize=MagicMock(
                side_effect=WorktreeRuntimeError(
                    "branch already exists",
                    reason="branch_exists",
                    details={
                        "branch_name": "victor/team/tester-1",
                        "member_id": "tester",
                    },
                )
            )
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        tester = StructuredMember(
            "tester",
            "Validation failed.",
            changed_files=["tests/auth/test_service.py"],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(tester)

        result = await coordinator.execute_task(
            "Validate auth service",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
            },
        )

        fake_runtime.materialize.assert_called_once()
        diagnostic = result["workspace_isolation_diagnostics"][0]
        follow_up = result["delegate_follow_up_contract"]
        approval = follow_up["approval_contract"]

        assert "worktree_session" not in result
        assert result["shared_context"]["workspace_isolation_diagnostics"] == [diagnostic]
        assert diagnostic["operation"] == "materialize"
        assert diagnostic["reason"] == "branch_exists"
        assert diagnostic["details"]["branch_name"] == "victor/team/tester-1"
        assert follow_up["workspace_isolation_diagnostics"] == [diagnostic]
        assert approval["workspace_isolation_diagnostics"] == [diagnostic]
        assert approval["reason"] == "validation_failed"
        assert approval["resume_ready"] is False

    @pytest.mark.asyncio
    async def test_auto_merge_worktrees_executes_guarded_merge_plan(self):
        """Auto-merge should invoke the runtime merge executor when explicitly enabled."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("m1",))
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_path": "/tmp/feature-m1",
                            "materialized": True,
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={"recommended_merge_order": ["m1"], "materialized": True}
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        result = await coordinator.execute_task(
            "Implement feature",
            {
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "materialize_worktrees": True,
                "auto_merge_worktrees": True,
            },
        )

        fake_runtime.execute_merge_orchestration.assert_called_once()
        assert result["merge_execution"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_delegate_mode_auto_merges_execution_eligible_worktrees(self):
        """Delegate mode should auto-merge when the runtime marks the plan safe."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("m1",))
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_path": "/tmp/feature-m1",
                            "materialized": True,
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        result = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
            },
        )

        fake_runtime.execute_merge_orchestration.assert_called_once()
        assert result["merge_execution"]["status"] == "success"
        assert result["merge_review_contract"]["next_action"] == "merge"
        assert result["delegate_follow_up_contract"]["approval_contract"] == {
            "required": False,
            "reason": "merge_executed",
            "recommended_action": "merged",
            "recommended_mode": "auto_apply_safe",
            "resume_ready": False,
            "auto_retry_eligible": False,
            "merge_executed": True,
            "target_member_ids": ["m1"],
            "summary": "Merge orchestration already executed for: m1.",
            "next_steps": [
                {
                    "step": "status_merged",
                    "step_id": "status_merged",
                    "instruction": "Merge orchestration already executed for: m1.",
                    "target_member_ids": ["m1"],
                    "requires_approval": False,
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_delegate_mode_preserves_merge_approval_follow_up_when_auto_merge_disabled(
        self,
    ):
        """Delegate mode should preserve a runnable merge contract when safe auto-merge is disabled."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        result = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        fake_runtime.execute_merge_orchestration.assert_not_called()
        follow_up = result["delegate_follow_up_contract"]
        assert follow_up["next_action"] == "merge"
        assert follow_up["preserve_worktrees"] is True
        assert follow_up["merge_execution_contract"] == {
            "mode": "delegate",
            "next_action": "merge",
            "worktree_session": result["worktree_session"],
            "merge_analysis": result["merge_analysis"],
            "merge_orchestration": result["merge_orchestration"],
            "merge_review_contract": result["merge_review_contract"],
            "worker_return_contracts": result["worker_return_contracts"],
            "context_overrides": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": False,
                "cleanup_worktrees": True,
            },
        }
        approval = follow_up["approval_contract"]
        assert approval["required"] is True
        assert approval["reason"] == "merge_ready"
        assert approval["recommended_action"] == "approve_merge"
        assert approval["target_member_ids"] == ["m1"]
        assert approval["next_steps"] == [
            {
                "step": "approve_merge_execution",
                "step_id": "approve_merge_execution",
                "instruction": "Review and approve merge execution for: m1.",
                "target_member_ids": ["m1"],
                "requires_approval": True,
                "execution_context": {
                    "mode": "delegate",
                    "delegate_merge_contract": follow_up["merge_execution_contract"],
                },
            }
        ]
        assert follow_up["next_steps"] == approval["next_steps"]
        assert follow_up["primary_step_id"] == "approve_merge_execution"
        assert follow_up["step_requests"] == {
            "approve_merge_execution": {
                "mode": "delegate",
                "delegate_follow_up_contract": {
                    "next_steps": follow_up["next_steps"],
                    "primary_step_id": "approve_merge_execution",
                },
                "delegate_next_step_id": "approve_merge_execution",
            }
        }
        assert (
            follow_up["primary_step_request"]
            == follow_up["step_requests"]["approve_merge_execution"]
        )
        assert follow_up["step_execution_requests"] == {
            "approve_merge_execution": {
                "task": "Review and approve merge execution for: m1.",
                "context": follow_up["step_requests"]["approve_merge_execution"],
            }
        }
        assert (
            follow_up["primary_step_execution_request"]
            == follow_up["step_execution_requests"]["approve_merge_execution"]
        )
        assert result["worktree_cleanup"] == {
            "removed": [],
            "skipped": ["/tmp/feature-m1"],
            "errors": [],
            "reason": "preserved_for_follow_up",
        }

    @pytest.mark.asyncio
    async def test_delegate_mode_does_not_auto_merge_ineligible_worktrees(self):
        """Delegate mode should not auto-merge when the runtime flags review risk."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("m1",))
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_path": "/tmp/feature-m1",
                            "materialized": True,
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "high",
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        result = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
            },
        )

        fake_runtime.execute_merge_orchestration.assert_not_called()
        assert "merge_execution" not in result
        assert result["merge_review_contract"]["merge_execution_eligible"] is False
        approval = result["delegate_follow_up_contract"]["approval_contract"]
        assert approval["required"] is True
        assert approval["reason"] == "review_required"
        assert approval["recommended_action"] == "review_then_retry"
        assert approval["recommended_mode"] == "manual_review"
        assert approval["resume_ready"] is True
        assert approval["auto_retry_eligible"] is False
        assert approval["merge_executed"] is False
        assert approval["target_member_ids"] == ["m1"]
        assert (
            approval["summary"] == "Review merge risks before retrying preserved worktrees for: m1."
        )
        assert approval["resume_context"] == {
            "mode": "delegate",
            "delegate_reentry_contract": result["delegate_follow_up_contract"]["reentry_contract"],
        }
        assert approval["task_briefs_by_member"] == {
            "m1": "Review the pending merge risk for m1 (low). Inspect: src/auth/service.py. Prior output: Done."
        }
        assert approval["next_steps"] == [
            {
                "step": "review_worktrees",
                "step_id": "review_worktrees",
                "instruction": "Review merge risks before retrying preserved worktrees for: m1.",
                "target_member_ids": ["m1"],
                "requires_approval": True,
            },
            {
                "step": "resume_delegate_retry",
                "step_id": "resume_delegate_retry",
                "instruction": "Resume preserved worktrees after review for: m1.",
                "target_member_ids": ["m1"],
                "requires_approval": True,
                "resume_context": approval["resume_context"],
                "task_briefs_by_member": approval["task_briefs_by_member"],
            },
        ]

    @pytest.mark.asyncio
    async def test_delegate_mode_preserves_worktrees_and_emits_follow_up_contract(self):
        """Delegate mode should preserve worktrees and emit a concrete follow-up queue when blocked."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        fake_runtime.cleanup.assert_not_called()
        follow_up = result["delegate_follow_up_contract"]
        assert follow_up["next_action"] == "fix_validation"
        assert follow_up["preserve_worktrees"] is True
        assert follow_up["fix_validation_queue"] == [
            {
                "member_id": "tester",
                "validation_command": "python -m pytest tests/unit/auth/test_service.py",
                "validation_summary": "1 failed",
                "changed_files": ["tests/auth/test_service.py"],
            }
        ]
        assert follow_up["review_queue"] == [
            {
                "member_id": "tester",
                "merge_risk_level": "low",
                "merge_risk_reasons": [],
                "changed_files": ["tests/auth/test_service.py"],
                "task_summary": "Validated auth tests and found one failing assertion.",
            }
        ]
        assert follow_up["reentry_contract"] == {
            "mode": "delegate",
            "next_action": "fix_validation",
            "retry_member_ids": ["tester"],
            "resume_worktree_paths": {"tester": "/tmp/feature-tester"},
            "retry_tasks_by_member": {
                "tester": (
                    "Fix the failing validation run for tester. Re-run "
                    "`python -m pytest tests/unit/auth/test_service.py`. "
                    "Last result: 1 failed. Focus on: tests/auth/test_service.py."
                )
            },
            "resume_member_context_overrides": {
                "tester": {
                    "isolation_mode": "worktree",
                    "workspace_root": "/tmp/feature-tester",
                    "materialized_worktree": True,
                    "worktree_path": "/tmp/feature-tester",
                    "branch_name": "victor/feature/tester-1",
                    "worktree_assignment": {
                        "member_id": "tester",
                        "branch_name": "victor/feature/tester-1",
                        "worktree_path": "/tmp/feature-tester",
                        "materialized": True,
                    },
                }
            },
            "context_overrides": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": False,
                "cleanup_worktrees": False,
            },
        }
        approval = follow_up["approval_contract"]
        assert approval["required"] is False
        assert approval["reason"] == "validation_failed"
        assert approval["recommended_action"] == "retry"
        assert approval["recommended_mode"] == "manual_review"
        assert approval["resume_ready"] is True
        assert approval["auto_retry_eligible"] is True
        assert approval["merge_executed"] is False
        assert approval["target_member_ids"] == ["tester"]
        assert approval["summary"] == (
            "Resume preserved worktrees to fix failing validation for: tester."
        )
        assert approval["resume_context"] == {
            "mode": "delegate",
            "delegate_reentry_contract": follow_up["reentry_contract"],
        }
        assert approval["task_briefs_by_member"] == {
            "tester": (
                "Fix the failing validation run for tester. Re-run "
                "`python -m pytest tests/unit/auth/test_service.py`. "
                "Last result: 1 failed. Focus on: tests/auth/test_service.py."
            )
        }
        assert approval["next_steps"] == [
            {
                "step": "resume_delegate_retry",
                "step_id": "resume_delegate_retry",
                "instruction": "Resume preserved worktrees to fix failing validation for: tester.",
                "target_member_ids": ["tester"],
                "requires_approval": False,
                "resume_context": approval["resume_context"],
                "task_briefs_by_member": approval["task_briefs_by_member"],
            }
        ]
        assert follow_up["next_steps"] == approval["next_steps"]
        assert follow_up["primary_step_id"] == "resume_delegate_retry"
        assert follow_up["step_requests"] == {
            "resume_delegate_retry": {
                "mode": "delegate",
                "delegate_follow_up_contract": {
                    "next_steps": follow_up["next_steps"],
                    "primary_step_id": "resume_delegate_retry",
                },
                "delegate_next_step_id": "resume_delegate_retry",
            }
        }
        assert (
            follow_up["primary_step_request"] == follow_up["step_requests"]["resume_delegate_retry"]
        )
        assert follow_up["step_execution_requests"] == {
            "resume_delegate_retry": {
                "task": "Resume preserved worktrees to fix failing validation for: tester.",
                "context": follow_up["step_requests"]["resume_delegate_retry"],
            }
        }
        assert (
            follow_up["primary_step_execution_request"]
            == follow_up["step_execution_requests"]["resume_delegate_retry"]
        )
        expected_suggestions = [
            {
                "tool": "delegate_follow_up",
                "command": "delegate_follow_up(step_id='resume_delegate_retry')",
                "description": (
                    "resume_delegate_retry: Resume preserved worktrees to fix failing "
                    "validation for: tester."
                ),
                "step_id": "resume_delegate_retry",
                "requires_approval": False,
            }
        ]
        assert follow_up["follow_up_suggestions"] == expected_suggestions
        assert result["delegate_follow_up_suggestions"] == expected_suggestions
        assert result["follow_up_suggestions"] == expected_suggestions
        assert result["worktree_cleanup"]["removed"] == []
        assert result["worktree_cleanup"]["errors"] == []
        assert result["worktree_cleanup"]["skipped"] == [
            "/tmp/feature-planner",
            "/tmp/feature-tester",
        ]
        assert result["worktree_cleanup"]["reason"] == "preserved_for_follow_up"

    @pytest.mark.asyncio
    async def test_delegate_reentry_contract_reuses_targeted_member_worktree_context(
        self,
    ):
        """Delegate re-entry should resume only the targeted members without rematerializing worktrees."""

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(),
            collect_changed_files=MagicMock(),
            cleanup=MagicMock(),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember("planner", "Planner done")
        tester = StructuredMember("tester", "Tester retried validation")
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task(
            "Retry failed validation",
            {
                "mode": "delegate",
                "delegate_reentry_contract": {
                    "mode": "delegate",
                    "next_action": "fix_validation",
                    "retry_member_ids": ["tester"],
                    "resume_worktree_paths": {"tester": "/tmp/feature-tester"},
                    "retry_tasks_by_member": {
                        "tester": (
                            "Fix the failing validation run for tester. Re-run "
                            "`python -m pytest tests/unit/auth/test_service.py`. "
                            "Last result: 1 failed. Focus on: tests/auth/test_service.py."
                        )
                    },
                    "resume_member_context_overrides": {
                        "tester": {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "worktree_path": "/tmp/feature-tester",
                            "branch_name": "victor/feature/tester-1",
                            "claimed_paths": ["tests/auth"],
                            "readonly_paths": ["docs"],
                            "worktree_assignment": {
                                "member_id": "tester",
                                "branch_name": "victor/feature/tester-1",
                                "worktree_path": "/tmp/feature-tester",
                            },
                            "materialized_worktree": True,
                        }
                    },
                    "context_overrides": {
                        "mode": "delegate",
                        "worktree_isolation": True,
                        "materialize_worktrees": False,
                        "cleanup_worktrees": False,
                    },
                },
            },
        )

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        fake_runtime.cleanup.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["workspace_root"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["claimed_paths"] == ["tests/auth"]
        assert tester.seen_contexts[0]["readonly_paths"] == ["docs"]
        assert tester.seen_contexts[0]["materialized_worktree"] is True
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["follow_up_task_brief"] == (
            "Fix the failing validation run for tester. Re-run "
            "`python -m pytest tests/unit/auth/test_service.py`. "
            "Last result: 1 failed. Focus on: tests/auth/test_service.py."
        )
        assert set(result["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_delegate_next_step_resume_reuses_nested_retry_context(self):
        """Selected retry steps should unwrap their nested resume context automatically."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        retry_step = first["delegate_follow_up_contract"]["approval_contract"]["next_steps"][0]
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_task(
            "Retry failed validation",
            {
                "mode": "delegate",
                "delegate_next_step": retry_step,
            },
        )

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["follow_up_task_brief"] == (
            "Fix the failing validation run for tester. Re-run "
            "`python -m pytest tests/unit/auth/test_service.py`. "
            "Last result: 1 failed. Focus on: tests/auth/test_service.py."
        )
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert tester.seen_contexts[0]["delegate_selected_instruction"] == (
            "Resume preserved worktrees to fix failing validation for: tester."
        )
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_delegate_primary_step_request_executes_retry_from_follow_up_contract(
        self,
    ):
        """Retry selection should work from the surface-ready primary step request."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        follow_up_contract = first["delegate_follow_up_contract"]
        primary_step_request = follow_up_contract["primary_step_request"]
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_task(
            "Retry failed validation",
            primary_step_request,
        )

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_delegate_primary_step_execution_request_executes_retry_from_follow_up_contract(
        self,
    ):
        """Retry selection should work from the task-plus-context execution request."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        execution_request = first["delegate_follow_up_contract"]["primary_step_execution_request"]
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_task(
            execution_request["task"],
            execution_request["context"],
        )

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_execute_follow_up_request_executes_retry_from_follow_up_contract(
        self,
    ):
        """Coordinator should execute retry follow-up requests directly."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        execution_request = first["delegate_follow_up_contract"]["primary_step_execution_request"]
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_follow_up_request(execution_request)

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_execute_follow_up_request_requires_task_and_context(self):
        """Coordinator should reject malformed direct follow-up requests."""

        coordinator = UnifiedTeamCoordinator(enable_observability=False)

        with pytest.raises(ValueError, match="non-empty task"):
            await coordinator.execute_follow_up_request({"context": {}})

        with pytest.raises(ValueError, match="mapping context"):
            await coordinator.execute_follow_up_request({"task": "Retry failed validation"})

    @pytest.mark.asyncio
    async def test_execute_follow_up_contract_executes_retry_from_follow_up_contract(
        self,
    ):
        """Coordinator should execute retry directly from the full follow-up contract."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        follow_up_contract = first["delegate_follow_up_contract"]
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_follow_up_contract(follow_up_contract)

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_execute_follow_up_contract_backfills_request_from_step_requests(
        self,
    ):
        """Coordinator should rebuild runnable requests from older follow-up contract shapes."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        follow_up_contract = dict(first["delegate_follow_up_contract"])
        follow_up_contract.pop("primary_step_execution_request", None)
        follow_up_contract.pop("step_execution_requests", None)
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_follow_up_contract(follow_up_contract)

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_delegate_next_step_id_resolves_from_approval_contract(self):
        """Retry selection should work from a compact approval contract plus step id."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(merge_order=("planner", "tester"))
                self.assignments = [
                    SimpleNamespace(
                        member_id="planner",
                        branch_name="victor/feature/planner-1",
                        worktree_path="/tmp/feature-planner",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-planner",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "planner",
                            "branch_name": "victor/feature/planner-1",
                            "worktree_path": "/tmp/feature-planner",
                            "materialized": True,
                        },
                    ),
                    SimpleNamespace(
                        member_id="tester",
                        branch_name="victor/feature/tester-1",
                        worktree_path="/tmp/feature-tester",
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-tester",
                            "materialized_worktree": True,
                        },
                        to_dict=lambda: {
                            "member_id": "tester",
                            "branch_name": "victor/feature/tester-1",
                            "worktree_path": "/tmp/feature-tester",
                            "materialized": True,
                        },
                    ),
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [assignment.to_dict() for assignment in self.assignments],
                }

            def assignment_for(self, member_id: str):
                for assignment in self.assignments:
                    if assignment.member_id == member_id:
                        return assignment
                return None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(
                side_effect=lambda _session, member_id: {
                    "planner": ("src/auth/service.py",),
                    "tester": ("tests/auth/test_service.py",),
                }[member_id]
            ),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["planner", "tester"],
                    "materialized": True,
                    "merge_execution_eligible": False,
                    "merge_risk_level": "medium",
                }
            ),
            cleanup=MagicMock(return_value={"removed": [], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        planner = StructuredMember(
            "planner",
            "Patched auth service and documented the change.",
            changed_files=[],
            metadata={
                "task_summary": "Patched auth service",
                "validation_run": {
                    "status": "passed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 passed",
                },
            },
        )
        tester = StructuredMember(
            "tester",
            "Validated auth tests and found one failing assertion.",
            changed_files=[],
            metadata={
                "validation_run": {
                    "status": "failed",
                    "command": "python -m pytest tests/unit/auth/test_service.py",
                    "summary": "1 failed",
                }
            },
        )
        coordinator.add_member(planner)
        coordinator.add_member(tester)
        coordinator.set_formation(TeamFormation.PARALLEL)

        first = await coordinator.execute_task(
            "Implement auth flow",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "member_write_scopes": {
                    "planner": ["src/auth"],
                    "tester": ["tests/auth"],
                },
            },
        )

        approval_contract = first["delegate_follow_up_contract"]["approval_contract"]
        retry_step_id = approval_contract["next_steps"][0]["step_id"]
        fake_runtime.materialize.reset_mock()
        fake_runtime.collect_changed_files.reset_mock()
        planner.seen_contexts.clear()
        tester.seen_contexts.clear()

        second = await coordinator.execute_task(
            "Retry failed validation",
            {
                "mode": "delegate",
                "delegate_approval_contract": approval_contract,
                "delegate_next_step_id": retry_step_id,
            },
        )

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        assert planner.seen_contexts == []
        assert len(tester.seen_contexts) == 1
        assert tester.seen_contexts[0]["worktree_path"] == "/tmp/feature-tester"
        assert tester.seen_contexts[0]["delegate_reentry_next_action"] == "fix_validation"
        assert tester.seen_contexts[0]["delegate_selected_step"] == "resume_delegate_retry"
        assert tester.seen_contexts[0]["delegate_selected_instruction"] == (
            "Resume preserved worktrees to fix failing validation for: tester."
        )
        assert set(second["member_results"]) == {"tester"}

    @pytest.mark.asyncio
    async def test_delegate_merge_contract_executes_approved_merge_without_rerunning_members(
        self,
    ):
        """Approved merge contracts should execute directly without rerunning workers."""

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(),
            collect_changed_files=MagicMock(),
            build_merge_orchestration=MagicMock(),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done")
        coordinator.add_member(member)

        merge_contract = {
            "mode": "delegate",
            "next_action": "merge",
            "worktree_session": {
                "plan": {
                    "team_name": "feature_team",
                    "repo_root": "/repo/project",
                    "parent_dir": "/tmp/worktrees",
                    "base_ref": "HEAD",
                    "branch_prefix": "victor/feature_team",
                    "formation": "parallel",
                    "assignments": [
                        {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                        }
                    ],
                    "merge_order": ["m1"],
                    "shared_readonly_paths": ["docs"],
                    "rationale": "delegate merge approval",
                    "metadata": {"member_count": 1, "scoped_members": 1},
                },
                "materialized": True,
                "dry_run": False,
                "assignments": [
                    {
                        "member_id": "m1",
                        "branch_name": "victor/feature/m1-1",
                        "worktree_name": "feature-team-m1",
                        "worktree_path": "/tmp/feature-m1",
                        "claimed_paths": ["src/auth"],
                        "readonly_paths": ["docs"],
                        "merge_priority": 0,
                        "metadata": {"member_index": 0, "formation": "parallel"},
                        "materialized": True,
                        "cleanup_required": True,
                        "runtime_metadata": {
                            "repo_root": "/repo/project",
                            "base_ref": "HEAD",
                        },
                    }
                ],
                "metadata": {"created_paths": ["/tmp/feature-m1"]},
            },
            "merge_analysis": {
                "risk_level": "low",
                "recommended_merge_order": ["m1"],
                "conflict_count": 0,
                "overlapping_files": [],
                "member_changed_files": {"m1": ["src/auth/service.py"]},
            },
            "merge_orchestration": {
                "recommended_merge_order": ["m1"],
                "materialized": True,
                "merge_execution_eligible": True,
                "merge_risk_level": "low",
                "recommended_mode": "auto_apply_safe",
                "worktree_paths": {"m1": "/tmp/feature-m1"},
                "branches": {"m1": "victor/feature/m1-1"},
                "merge_base": "HEAD",
                "conflict_paths": [],
            },
            "merge_review_contract": {
                "merge_ready": True,
                "review_required": False,
                "recommended_merge_order": ["m1"],
                "review_required_members": [],
                "validation_failed_members": [],
                "blocking_issues": [],
                "merge_risk_level": "low",
                "merge_execution_eligible": True,
                "recommended_mode": "auto_apply_safe",
                "next_action": "merge",
            },
            "worker_return_contracts": {
                "m1": {
                    "member_id": "m1",
                    "success": True,
                    "task_summary": "Patched auth service",
                    "changed_files": ["src/auth/service.py"],
                    "validation_run": {"status": "passed", "summary": "1 passed"},
                    "merge_risk": {
                        "level": "low",
                        "reasons": [],
                        "overlapping_files": [],
                        "out_of_scope_writes": [],
                        "readonly_violations": [],
                    },
                }
            },
            "context_overrides": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": False,
                "cleanup_worktrees": True,
            },
        }

        result = await coordinator.execute_task(
            "Approve merge execution",
            {
                "mode": "delegate",
                "delegate_merge_contract": merge_contract,
            },
        )

        fake_runtime.materialize.assert_not_called()
        fake_runtime.collect_changed_files.assert_not_called()
        fake_runtime.build_merge_orchestration.assert_not_called()
        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert result["success"] is True
        assert result["merge_execution"]["status"] == "success"
        assert (
            result["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )
        assert result["delegate_follow_up_contract"]["approval_contract"]["next_steps"] == [
            {
                "step": "status_merged",
                "step_id": "status_merged",
                "instruction": "Merge orchestration already executed for: m1.",
                "target_member_ids": ["m1"],
                "requires_approval": False,
            }
        ]
        assert result["worktree_cleanup"] == {
            "removed": ["/tmp/feature-m1"],
            "errors": [],
        }

    @pytest.mark.asyncio
    async def test_delegate_next_step_executes_nested_merge_context(self):
        """Selected merge approval steps should unwrap nested execution context automatically."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        first = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        merge_step = first["delegate_follow_up_contract"]["approval_contract"]["next_steps"][0]
        fake_runtime.execute_merge_orchestration.reset_mock()
        fake_runtime.cleanup.reset_mock()
        member.seen_contexts.clear()

        second = await coordinator.execute_task(
            "Approve merge execution",
            {
                "mode": "delegate",
                "delegate_next_step": merge_step,
            },
        )

        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert second["success"] is True
        assert second["merge_execution"]["status"] == "success"
        assert (
            second["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )

    @pytest.mark.asyncio
    async def test_delegate_primary_step_request_executes_merge_from_follow_up_contract(
        self,
    ):
        """Merge approval should work from the surface-ready primary step request."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        first = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        follow_up_contract = first["delegate_follow_up_contract"]
        primary_step_request = follow_up_contract["primary_step_request"]
        fake_runtime.execute_merge_orchestration.reset_mock()
        fake_runtime.cleanup.reset_mock()
        member.seen_contexts.clear()

        second = await coordinator.execute_task(
            "Approve merge execution",
            primary_step_request,
        )

        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert second["success"] is True
        assert second["merge_execution"]["status"] == "success"
        assert (
            second["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )

    @pytest.mark.asyncio
    async def test_delegate_primary_step_execution_request_executes_merge_from_follow_up_contract(
        self,
    ):
        """Merge approval should work from the task-plus-context execution request."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        first = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        execution_request = first["delegate_follow_up_contract"]["primary_step_execution_request"]
        fake_runtime.execute_merge_orchestration.reset_mock()
        fake_runtime.cleanup.reset_mock()
        member.seen_contexts.clear()

        second = await coordinator.execute_task(
            execution_request["task"],
            execution_request["context"],
        )

        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert second["success"] is True
        assert second["merge_execution"]["status"] == "success"
        assert (
            second["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )

    @pytest.mark.asyncio
    async def test_execute_follow_up_request_executes_merge_from_follow_up_contract(
        self,
    ):
        """Coordinator should execute merge follow-up requests directly."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        first = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        execution_request = first["delegate_follow_up_contract"]["primary_step_execution_request"]
        fake_runtime.execute_merge_orchestration.reset_mock()
        fake_runtime.cleanup.reset_mock()
        member.seen_contexts.clear()

        second = await coordinator.execute_follow_up_request(execution_request)

        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert second["success"] is True
        assert second["merge_execution"]["status"] == "success"
        assert (
            second["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )

    @pytest.mark.asyncio
    async def test_execute_follow_up_contract_executes_merge_from_follow_up_contract(
        self,
    ):
        """Coordinator should execute merge directly from the full follow-up contract."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        first = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        follow_up_contract = first["delegate_follow_up_contract"]
        fake_runtime.execute_merge_orchestration.reset_mock()
        fake_runtime.cleanup.reset_mock()
        member.seen_contexts.clear()

        second = await coordinator.execute_follow_up_contract(follow_up_contract)

        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert second["success"] is True
        assert second["merge_execution"]["status"] == "success"
        assert (
            second["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )

    @pytest.mark.asyncio
    async def test_delegate_next_step_id_resolves_from_follow_up_contract(self):
        """Merge approval should work from a follow-up contract plus step id."""

        class FakeSession:
            def __init__(self) -> None:
                self.materialized = True
                self.dry_run = False
                self.plan = MagicMock(
                    team_name="feature_team",
                    repo_root="/repo/project",
                    parent_dir="/tmp/worktrees",
                    base_ref="HEAD",
                    branch_prefix="victor/feature_team",
                    merge_order=("m1",),
                    formation=TeamFormation.PARALLEL,
                )
                self.assignments = [
                    SimpleNamespace(
                        member_id="m1",
                        branch_name="victor/feature/m1-1",
                        worktree_path="/tmp/feature-m1",
                        cleanup_required=True,
                        materialized=True,
                        metadata={"repo_root": "/repo/project", "base_ref": "HEAD"},
                        assignment=SimpleNamespace(
                            member_id="m1",
                            branch_name="victor/feature/m1-1",
                            worktree_name="feature-team-m1",
                            worktree_path="/tmp/feature-m1",
                            claimed_paths=("src/auth",),
                            readonly_paths=("docs",),
                            merge_priority=0,
                            metadata={"member_index": 0, "formation": "parallel"},
                        ),
                        to_context_overrides=lambda: {
                            "isolation_mode": "worktree",
                            "workspace_root": "/tmp/feature-m1",
                            "materialized_worktree": True,
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                        },
                        to_dict=lambda: {
                            "member_id": "m1",
                            "branch_name": "victor/feature/m1-1",
                            "worktree_name": "feature-team-m1",
                            "worktree_path": "/tmp/feature-m1",
                            "claimed_paths": ["src/auth"],
                            "readonly_paths": ["docs"],
                            "merge_priority": 0,
                            "metadata": {"member_index": 0, "formation": "parallel"},
                            "materialized": True,
                            "cleanup_required": True,
                            "runtime_metadata": {
                                "repo_root": "/repo/project",
                                "base_ref": "HEAD",
                            },
                        },
                    )
                ]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "plan": {
                        "team_name": "feature_team",
                        "repo_root": "/repo/project",
                        "parent_dir": "/tmp/worktrees",
                        "base_ref": "HEAD",
                        "branch_prefix": "victor/feature_team",
                        "formation": "parallel",
                        "assignments": [
                            {
                                "member_id": "m1",
                                "branch_name": "victor/feature/m1-1",
                                "worktree_name": "feature-team-m1",
                                "worktree_path": "/tmp/feature-m1",
                                "claimed_paths": ["src/auth"],
                                "readonly_paths": ["docs"],
                                "merge_priority": 0,
                                "metadata": {
                                    "member_index": 0,
                                    "formation": "parallel",
                                },
                            }
                        ],
                        "merge_order": ["m1"],
                        "shared_readonly_paths": ["docs"],
                        "rationale": "delegate merge approval",
                        "metadata": {"member_count": 1, "scoped_members": 1},
                    },
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [self.assignments[0].to_dict()],
                    "metadata": {"created_paths": ["/tmp/feature-m1"]},
                }

            def assignment_for(self, member_id: str):
                return self.assignments[0] if member_id == "m1" else None

        fake_runtime = SimpleNamespace(
            materialize=MagicMock(return_value=FakeSession()),
            collect_changed_files=MagicMock(return_value=("src/auth/service.py",)),
            build_merge_orchestration=MagicMock(
                return_value={
                    "recommended_merge_order": ["m1"],
                    "materialized": True,
                    "merge_execution_eligible": True,
                    "merge_risk_level": "low",
                    "recommended_mode": "auto_apply_safe",
                    "worktree_paths": {"m1": "/tmp/feature-m1"},
                    "branches": {"m1": "victor/feature/m1-1"},
                    "merge_base": "HEAD",
                    "conflict_paths": [],
                }
            ),
            execute_merge_orchestration=MagicMock(
                return_value={
                    "status": "success",
                    "executed": True,
                    "merged_members": ["m1"],
                    "cleanup": {
                        "removed": ["/tmp/feature-team-integration"],
                        "errors": [],
                    },
                }
            ),
            cleanup=MagicMock(return_value={"removed": ["/tmp/feature-m1"], "errors": []}),
        )
        coordinator = UnifiedTeamCoordinator(
            enable_observability=False,
            worktree_runtime=fake_runtime,
        )
        member = StructuredMember("m1", "Done", changed_files=[])
        coordinator.add_member(member)

        first = await coordinator.execute_task(
            "Implement feature",
            {
                "mode": "delegate",
                "team_name": "feature_team",
                "repo_root": "/repo/project",
                "worktree_isolation": True,
                "auto_merge_worktrees": False,
            },
        )

        follow_up_contract = first["delegate_follow_up_contract"]
        merge_step_id = follow_up_contract["approval_contract"]["next_steps"][0]["step_id"]
        fake_runtime.execute_merge_orchestration.reset_mock()
        fake_runtime.cleanup.reset_mock()
        member.seen_contexts.clear()

        second = await coordinator.execute_task(
            "Approve merge execution",
            {
                "mode": "delegate",
                "delegate_follow_up_contract": follow_up_contract,
                "delegate_next_step_id": merge_step_id,
            },
        )

        fake_runtime.execute_merge_orchestration.assert_called_once()
        fake_runtime.cleanup.assert_called_once()
        assert member.seen_contexts == []
        assert second["success"] is True
        assert second["merge_execution"]["status"] == "success"
        assert (
            second["delegate_follow_up_contract"]["approval_contract"]["reason"] == "merge_executed"
        )


class TestBroadcast:
    """Tests for message broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self):
        """Broadcast should send to all members."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        m1 = MockTeamMember("m1")
        m2 = MockTeamMember("m2")
        coordinator.add_member(m1)
        coordinator.add_member(m2)

        message = AgentMessage(
            sender_id="external",
            recipient_id=None,
            content="Hello all",
            message_type=MessageType.STATUS,
        )

        responses = await coordinator.broadcast(message)

        assert len(responses) == 2
        assert len(m1._messages) == 1
        assert len(m2._messages) == 1


class TestObservability:
    """Tests for observability features."""

    def test_set_execution_context(self):
        """Should set execution context."""
        coordinator = UnifiedTeamCoordinator()
        coordinator.set_execution_context(
            task_type="feature",
            complexity="high",
            vertical="coding",
            trigger="manual",
        )

        ctx = coordinator._get_observability_context()
        assert ctx["task_type"] == "feature"
        assert ctx["complexity"] == "high"
        assert ctx["vertical"] == "coding"
        assert ctx["trigger"] == "manual"

    def test_progress_callback(self):
        """Should support progress callbacks."""
        coordinator = UnifiedTeamCoordinator()
        progress_calls = []

        def callback(member_id, status, progress):
            progress_calls.append((member_id, status, progress))

        coordinator.set_progress_callback(callback)
        coordinator._report_progress("m1", "running", 0.5)

        assert len(progress_calls) == 1
        assert progress_calls[0] == ("m1", "running", 0.5)

    def test_emit_team_event_uses_current_observability_bus_api(self, caplog):
        """Team event emission should not call removed EventBus singleton APIs."""
        coordinator = UnifiedTeamCoordinator()

        with caplog.at_level("WARNING", logger="victor.teams.mixins.observability"):
            coordinator._emit_team_event("started", {"team_id": "team_1"})

        assert "Failed to emit team event" not in caplog.text


class TestFactoryFunction:
    """Tests for create_coordinator factory."""

    def test_default_requires_orchestrator(self):
        """Default (non-lightweight) should require orchestrator."""
        with pytest.raises(ValueError, match="orchestrator is required"):
            create_coordinator()

    def test_lightweight_creates_coordinator(self):
        """Lightweight should create a coordinator without orchestrator."""
        coordinator = create_coordinator(lightweight=True)
        assert coordinator is not None
        assert hasattr(coordinator, "add_member") or hasattr(coordinator, "members")

    def test_enable_observability(self):
        """Should accept enable_observability flag."""
        mock_orch = MagicMock()
        coordinator = create_coordinator(orchestrator=mock_orch, enable_observability=True)
        assert isinstance(coordinator, UnifiedTeamCoordinator)

    def test_enable_rl(self):
        """Should accept enable_rl flag."""
        mock_orch = MagicMock()
        coordinator = create_coordinator(orchestrator=mock_orch, enable_rl=True)
        assert isinstance(coordinator, UnifiedTeamCoordinator)


class TestClearAndReset:
    """Tests for clear/reset functionality."""

    def test_clear_members(self):
        """Clear should remove all members."""
        coordinator = UnifiedTeamCoordinator()
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.add_member(MockTeamMember("m2"))

        assert len(coordinator.members) == 2
        coordinator.clear()
        assert len(coordinator.members) == 0

    def test_clear_returns_self(self):
        """Clear should return self for chaining."""
        coordinator = UnifiedTeamCoordinator()
        result = coordinator.clear()
        assert result is coordinator


class TestStateGraphNodeIntegration:
    """Tests for UnifiedTeamCoordinator usage as a StateGraph node (`__call__`).

    The coordinator implements ``async __call__(state) -> state`` so it can be
    passed directly to ``StateGraph.add_node``. This class covers the dict-state
    contract:

    - Reads task from ``state['task']`` (or falls back to ``state['query']``).
    - Builds context excluding the task/query keys.
    - Returns a new dict (does NOT mutate the caller's input).
    - On success: writes ``result`` and ``team_output``.
    - On failure: writes ``error`` and ``team_output``.
    """

    @pytest.mark.asyncio
    async def test_success_writes_result_and_team_output(self):
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "First"))
        coordinator.add_member(MockTeamMember("m2", "Second"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        out = await coordinator({"task": "demo"})

        assert "result" in out
        assert "team_output" in out
        assert out["team_output"]["success"] is True
        assert "error" not in out

    @pytest.mark.asyncio
    async def test_failure_writes_error_and_team_output(self):
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        # No members → execute_task returns success=False
        out = await coordinator({"task": "demo"})

        assert "error" in out
        assert "team_output" in out
        assert out["team_output"]["success"] is False
        assert "result" not in out

    @pytest.mark.asyncio
    async def test_uses_task_key(self):
        captured: list[str] = []

        class CapturingMember(MockTeamMember):
            async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
                captured.append(task)
                return "ok"

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(CapturingMember("m1"))

        await coordinator({"task": "build the thing"})

        assert captured == ["build the thing"]

    @pytest.mark.asyncio
    async def test_query_key_fallback(self):
        captured: list[str] = []

        class CapturingMember(MockTeamMember):
            async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
                captured.append(task)
                return "ok"

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(CapturingMember("m1"))

        await coordinator({"query": "find auth code"})

        assert captured == ["find auth code"]

    @pytest.mark.asyncio
    async def test_context_excludes_task_and_query_keys(self):
        captured_contexts: list[Dict[str, Any]] = []

        class CapturingMember(MockTeamMember):
            async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
                captured_contexts.append(dict(context))
                return "ok"

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(CapturingMember("m1"))

        await coordinator({"task": "T", "query": "Q", "scope": "auth", "max_workers": 3})

        assert captured_contexts, "member.execute_task was not invoked"
        ctx = captured_contexts[0]
        assert "task" not in ctx
        assert "query" not in ctx
        assert ctx["scope"] == "auth"
        assert ctx["max_workers"] == 3

    @pytest.mark.asyncio
    async def test_no_members_returns_error_state(self):
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        out = await coordinator({"task": "anything"})

        assert "error" in out
        assert out["team_output"]["success"] is False

    @pytest.mark.asyncio
    async def test_does_not_mutate_caller_state(self):
        """The caller's dict must not be mutated; __call__ returns a new dict."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "ok"))

        original = {"task": "build", "scope": "auth"}
        snapshot = dict(original)

        out = await coordinator(original)

        assert original == snapshot, "input dict was mutated"
        assert out is not original
        assert out["task"] == "build"  # input keys preserved in output
        assert out["scope"] == "auth"


class TestStateGraphNodeConfig:
    """Tests for ``StateGraphNodeConfig`` and the ``with_state_graph_config``
    fluent setter.

    Configuration is the seam that keeps the StateGraph node usable across
    different graph schemas without forcing every consumer to rename their
    state keys to match an internal convention.
    """

    def test_default_config_preserves_keys(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        config = StateGraphNodeConfig()
        assert config.task_key == "task"
        assert config.query_key == "query"
        assert config.result_key == "result"
        assert config.output_key == "team_output"
        assert config.error_key == "error"
        assert config.formation_strategy is None

    def test_with_state_graph_config_returns_self_for_chaining(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        result = coordinator.with_state_graph_config(StateGraphNodeConfig())
        assert result is coordinator

    @pytest.mark.asyncio
    async def test_custom_keys_honored_on_input(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        captured: list[str] = []

        class CapturingMember(MockTeamMember):
            async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
                captured.append(task)
                return "ok"

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(CapturingMember("m1"))
        coordinator.with_state_graph_config(
            StateGraphNodeConfig(task_key="instruction", query_key="prompt")
        )

        await coordinator({"instruction": "do work", "prompt": "ignored"})
        assert captured == ["do work"]

    @pytest.mark.asyncio
    async def test_custom_keys_honored_on_output(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Done"))
        coordinator.with_state_graph_config(
            StateGraphNodeConfig(
                result_key="final",
                output_key="team_run",
                error_key="failure",
            )
        )

        out = await coordinator({"task": "run"})

        assert out["final"] == "Done"
        assert "team_run" in out
        # Default keys MUST NOT be written when overrides are configured.
        assert "result" not in out
        assert "team_output" not in out

    @pytest.mark.asyncio
    async def test_custom_error_key_used_on_failure(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        # No members → guaranteed failure
        coordinator.with_state_graph_config(StateGraphNodeConfig(error_key="failure"))

        out = await coordinator({"task": "run"})

        assert "failure" in out
        assert "error" not in out


class TestExecuteTeamConfig:
    """Tests for ``execute_team_config(config, members=...)`` and the
    ``_execute_with`` parameterized core.

    ``execute_team_config`` exists so callers (notably ``TeamStep`` in
    ``victor.framework.workflows.nodes``) can hand a ``TeamConfig`` to the
    coordinator without first having to mutate ``self._formation`` or
    ``self._members``. It must be safe to call concurrently on a shared
    coordinator.
    """

    @staticmethod
    def _make_config(members, formation=TeamFormation.SEQUENTIAL):
        from victor.teams.types import TeamConfig, TeamMember
        from victor.core.shared_types import SubAgentRole

        # TeamConfig.members type is List[TeamMember]; for these tests we
        # pass MockTeamMember instances via the ``members`` override
        # parameter on execute_team_config, so config.members can be empty
        # placeholders or real TeamMembers.
        placeholder_members = [
            TeamMember(id=m.id, role=SubAgentRole.RESEARCHER, name=m.id, goal="test")
            for m in members
        ]
        return TeamConfig(
            name="TestTeam",
            goal="Test goal",
            members=placeholder_members,
            formation=formation,
        )

    @pytest.mark.asyncio
    async def test_returns_team_result(self):
        from victor.teams.types import TeamResult

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        members = [MockTeamMember("m1", "Done")]
        config = self._make_config(members)

        result = await coordinator.execute_team_config(config, members=members)

        assert isinstance(result, TeamResult)
        assert result.success is True
        assert result.formation == TeamFormation.SEQUENTIAL
        assert "m1" in result.member_results

    @pytest.mark.asyncio
    async def test_uses_config_formation(self):
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        # Set a different default formation; config should override it
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        members = [MockTeamMember("m1"), MockTeamMember("m2")]
        config = self._make_config(members, formation=TeamFormation.PARALLEL)

        result = await coordinator.execute_team_config(config, members=members)

        assert result.formation == TeamFormation.PARALLEL
        # And self._formation must NOT have been changed
        assert coordinator.formation == TeamFormation.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_does_not_mutate_self_state(self):
        """Crucial: execute_team_config must never alter self._members or
        self._formation, otherwise concurrent graph runs corrupt each other."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("baseline"))
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        baseline_members = list(coordinator.members)
        baseline_formation = coordinator.formation

        members = [MockTeamMember("m1"), MockTeamMember("m2")]
        config = self._make_config(members, formation=TeamFormation.PARALLEL)
        await coordinator.execute_team_config(config, members=members)

        assert list(coordinator.members) == baseline_members
        assert coordinator.formation == baseline_formation

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_isolated(self):
        """Two simultaneous execute_team_config calls must each see their
        own formation/members — no cross-contamination."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)

        members_a = [MockTeamMember("a1"), MockTeamMember("a2")]
        members_b = [MockTeamMember("b1"), MockTeamMember("b2")]
        config_a = self._make_config(members_a, formation=TeamFormation.PARALLEL)
        config_b = self._make_config(members_b, formation=TeamFormation.SEQUENTIAL)

        result_a, result_b = await asyncio.gather(
            coordinator.execute_team_config(config_a, members=members_a),
            coordinator.execute_team_config(config_b, members=members_b),
        )

        assert result_a.formation == TeamFormation.PARALLEL
        assert result_b.formation == TeamFormation.SEQUENTIAL
        assert set(result_a.member_results.keys()) == {"a1", "a2"}
        assert set(result_b.member_results.keys()) == {"b1", "b2"}

    @pytest.mark.asyncio
    async def test_raises_without_members_or_orchestrator(self):
        """If config.members can't be adapted (no orchestrator) and no
        ``members=`` override is provided, the error must be explicit."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        members = [MockTeamMember("m1")]
        config = self._make_config(members)

        with pytest.raises(ValueError, match=r"orchestrator"):
            # Pass no members override, force the adapter path
            await coordinator.execute_team_config(config)

    @pytest.mark.asyncio
    async def test_team_step_adapter_calls_execute_team_config(self):
        """``TeamStep._execute_team`` calls ``coordinator.execute_team_config``
        at workflows/nodes.py:392. Before this refactor, that method did not
        exist. This test exercises that exact code path against a stub
        coordinator and asserts the call routes cleanly.
        """
        from victor.framework.workflows.nodes import TeamStep, TeamStepConfig
        from victor.teams.types import TeamMember, TeamResult, MemberResult
        from victor.core.shared_types import SubAgentRole

        # Build a UnifiedTeamCoordinator and stub execute_team_config so we
        # don't need a real orchestrator to validate the path.
        recorded: Dict[str, Any] = {}

        coordinator = UnifiedTeamCoordinator(enable_observability=False)

        async def _stub_execute_team_config(config, members=None):  # pragma: no cover - thin stub
            recorded["called"] = True
            recorded["config_name"] = config.name
            recorded["formation"] = config.formation
            return TeamResult(
                success=True,
                final_output="stub-output",
                member_results={
                    m.id: MemberResult(member_id=m.id, success=True, output="ok")
                    for m in config.members
                },
                formation=config.formation,
            )

        # Monkey-patch onto this instance only.
        coordinator.execute_team_config = _stub_execute_team_config  # type: ignore[assignment]

        # Replace the create_coordinator factory so TeamStep picks up our stub.
        import victor.framework.workflows.nodes as nodes_mod

        original_create = nodes_mod.__dict__.get("create_coordinator")

        def _stub_create_coordinator(**kwargs):  # noqa: ARG001
            return coordinator

        # TeamStep imports create_coordinator inside execute_async — patch the
        # ``victor.teams`` module re-export so the late import sees our stub.
        import victor.teams as teams_mod

        original_teams_create = teams_mod.create_coordinator
        teams_mod.create_coordinator = _stub_create_coordinator  # type: ignore[assignment]
        try:
            node = TeamStep(
                id="t1",
                name="TestTeam",
                goal="Goal",
                team_formation=TeamFormation.PARALLEL,
                members=[
                    TeamMember(
                        id="m1",
                        role=SubAgentRole.RESEARCHER,
                        name="m1",
                        goal="test",
                    )
                ],
                config=TeamStepConfig(timeout_seconds=None),
            )
            graph_state = {"task": "execute"}
            out = await node.execute_async(orchestrator=None, graph_state=graph_state)
        finally:
            teams_mod.create_coordinator = original_teams_create  # type: ignore[assignment]
            if original_create is not None:
                nodes_mod.__dict__["create_coordinator"] = original_create

        assert recorded.get("called") is True, "execute_team_config not invoked"
        assert recorded["config_name"] == "TestTeam"
        assert recorded["formation"] == TeamFormation.PARALLEL
        # Result merged into graph state under TeamStepConfig.output_key
        assert "team_result" in out


class TestPydanticAndCopyOnWriteState:
    """Tests that ``__call__`` cleanly handles non-dict state types.

    StateGraph (`victor/framework/graph.py`) accepts dict, ``CopyOnWriteState``
    wrapper, and arbitrary Pydantic ``BaseModel`` schemas. The coordinator
    must work in all three without forcing the caller's hand.
    """

    @pytest.mark.asyncio
    async def test_call_with_pydantic_extra_allow_state(self):
        """A BaseModel with ``extra='allow'`` must accept the result/output
        keys via model_copy and return a model of the same type."""
        from pydantic import BaseModel, ConfigDict

        class FlexibleState(BaseModel):
            model_config = ConfigDict(extra="allow")
            task: str = ""

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Done"))

        state = FlexibleState(task="run analysis")
        out = await coordinator(state)

        assert isinstance(out, FlexibleState), "Pydantic input must yield Pydantic output"
        assert getattr(out, "result", None) == "Done"
        assert getattr(out, "team_output", None) is not None
        assert out.task == "run analysis"  # original field preserved

    @pytest.mark.asyncio
    async def test_call_with_strict_pydantic_raises_clear_error(self):
        """A strict (extra='forbid') BaseModel with no result/output fields
        must raise a clear error naming the missing key."""
        from pydantic import BaseModel, ConfigDict

        class StrictState(BaseModel):
            model_config = ConfigDict(extra="forbid")
            task: str = ""

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Done"))

        with pytest.raises(ValueError, match=r"(result|team_output|extra='allow')"):
            await coordinator(StrictState(task="x"))

    @pytest.mark.asyncio
    async def test_call_with_copy_on_write_state(self):
        """When the StateGraph executor wraps state in CopyOnWriteState the
        coordinator must still find the task and write the results so that
        the wrapper sees the mutation."""
        from victor.framework.graph import CopyOnWriteState

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Done"))

        cow = CopyOnWriteState({"task": "build", "scope": "auth"})
        out = await coordinator(cow)

        # Output is the same wrapper, mutated via __setitem__.
        assert isinstance(out, CopyOnWriteState)
        assert out["result"] == "Done"
        assert out["team_output"]["success"] is True
        assert out["scope"] == "auth"  # passthrough preserved

    @pytest.mark.asyncio
    async def test_call_with_agentic_loop_state_model_uses_query_field(self):
        """``AgenticLoopStateModel`` has a ``query`` field but no ``task``
        field. Default config falls back to query — and writing to the
        coordinator's default ``result_key`` requires extra='allow'."""
        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        captured: list[str] = []

        class CapturingMember(MockTeamMember):
            async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
                captured.append(task)
                return "ok"

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(CapturingMember("m1"))
        # Map outputs into the model's existing ``context`` dict field so the
        # strict schema doesn't reject them.
        coordinator.with_state_graph_config(
            StateGraphNodeConfig(
                result_key="context",
                output_key="evaluation",
                error_key="evaluation",
            )
        )

        state = AgenticLoopStateModel(query="explore the repo")
        out = await coordinator(state)

        assert captured == ["explore the repo"]
        assert isinstance(out, AgenticLoopStateModel)


class TestFormationStrategy:
    """Tests for the optional ``formation_strategy`` slot on
    ``StateGraphNodeConfig``.

    The strategy turns ``select_formation`` (an existing utility in
    ``victor/framework/agentic_graph/team_selector.py``) into a real
    production seam — its result is injected into the per-call context as
    ``formation_hint``, which the existing ``_resolve_effective_formation``
    consumes. Crucially this does NOT mutate ``self._formation``, so two
    concurrent ``__call__`` invocations can still pick different formations
    without corrupting each other.
    """

    @pytest.mark.asyncio
    async def test_no_strategy_uses_self_formation(self):
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.add_member(MockTeamMember("m2"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        out = await coordinator({"task": "x"})

        assert out["team_output"]["formation"] == "parallel"

    @pytest.mark.asyncio
    async def test_strategy_overrides_default_via_hint(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.add_member(MockTeamMember("m2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        coordinator.with_state_graph_config(
            StateGraphNodeConfig(formation_strategy=lambda _state: TeamFormation.PARALLEL)
        )

        out = await coordinator({"task": "x"})

        assert out["team_output"]["formation"] == "parallel"
        # Self-formation must NOT be mutated.
        assert coordinator.formation == TeamFormation.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_select_formation_works_as_strategy(self):
        """The shipped ``select_formation`` utility is plug-and-play: pass it
        as the ``formation_strategy`` and it picks based on dict context."""
        from victor.framework.agentic_graph.team_selector import select_formation
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.add_member(MockTeamMember("m2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        coordinator.with_state_graph_config(
            StateGraphNodeConfig(formation_strategy=select_formation)
        )

        # task_type="research" is a select_formation override → PARALLEL
        out = await coordinator({"task": "x", "context": {"task_type": "research", "team_size": 2}})

        assert out["team_output"]["formation"] == "parallel"

    @pytest.mark.asyncio
    async def test_async_strategy_is_rejected(self):
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        async def async_strategy(_state):
            return TeamFormation.PARALLEL

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.with_state_graph_config(StateGraphNodeConfig(formation_strategy=async_strategy))

        with pytest.raises(TypeError, match="synchronously"):
            await coordinator({"task": "x"})

    @pytest.mark.asyncio
    async def test_strategy_works_with_copy_on_write_state(self):
        from victor.framework.agentic_graph.team_selector import select_formation
        from victor.framework.graph import CopyOnWriteState
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.add_member(MockTeamMember("m2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        coordinator.with_state_graph_config(
            StateGraphNodeConfig(formation_strategy=select_formation)
        )

        out = await coordinator(
            CopyOnWriteState({"task": "x", "context": {"task_type": "research", "team_size": 2}})
        )

        assert out["team_output"]["formation"] == "parallel"


class TestEndToEndStateGraphIntegration:
    """End-to-end coverage for using the coordinator directly as a graph node."""

    @pytest.mark.asyncio
    async def test_state_graph_invokes_coordinator_node(self):
        from victor.framework.graph import END, StateGraph

        class TeamState(TypedDict, total=False):
            task: str
            result: str
            team_output: Dict[str, Any]

        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Research complete"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        graph = StateGraph(TeamState)
        graph.add_node("research_team", coordinator)
        graph.add_edge("research_team", END)
        graph.set_entry_point("research_team")

        compiled = graph.compile()
        result = await compiled.invoke({"task": "Inspect the auth flow"})
        final_state = result.state

        assert final_state["result"] == "Research complete"
        assert final_state["team_output"]["success"] is True
        assert final_state["team_output"]["formation"] == "sequential"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

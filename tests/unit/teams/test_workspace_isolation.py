from types import SimpleNamespace

from victor.teams.types import MemberResult, TeamFormation
from victor.teams.workspace_isolation import (
    WorkspaceIsolationPolicy,
    WorkspaceIsolationService,
)
from victor.teams.worktree_runtime import (
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
    WorktreeExecutionPlan,
    WorktreeMaterializationSession,
    WorktreeRuntimeError,
)


def _plan() -> WorktreeExecutionPlan:
    assignment = WorktreeAssignment(
        member_id="worker",
        branch_name="victor/team/worker-1",
        worktree_name="team-worker",
        worktree_path="/tmp/team-worker",
    )
    return WorktreeExecutionPlan(
        team_name="team",
        repo_root="/repo",
        parent_dir="/tmp",
        base_ref="HEAD",
        branch_prefix="victor/team",
        formation=TeamFormation.PARALLEL,
        assignments=(assignment,),
        merge_order=("worker",),
    )


def _session() -> WorktreeMaterializationSession:
    plan = _plan()
    return WorktreeMaterializationSession(
        plan=plan,
        assignments=(
            MaterializedWorktreeAssignment(
                assignment=plan.assignments[0],
                materialized=True,
                cleanup_required=True,
            ),
        ),
        materialized=True,
    )


def test_workspace_isolation_service_plans_and_materializes_delegate_context():
    plan = _plan()
    session = _session()
    planner = SimpleNamespace(plan=lambda members, *, context, formation: plan)

    class Runtime:
        def __init__(self):
            self.materialize_calls = []

        def materialize(self, received_plan, *, dry_run=False):
            self.materialize_calls.append((received_plan, dry_run))
            return session

    runtime = Runtime()
    service = WorkspaceIsolationService(planner=planner, runtime=runtime, merge_analyzer=None)

    resolved_plan = service.plan(
        [SimpleNamespace(id="worker")],
        context={"worktree_isolation": True, "repo_root": "/repo"},
        formation=TeamFormation.PARALLEL,
    )
    resolved_session = service.materialize(
        resolved_plan,
        context={"mode": "delegate", "worktree_isolation": True},
    )

    assert resolved_plan is plan
    assert resolved_session is session
    assert runtime.materialize_calls == [(plan, False)]


def test_workspace_isolation_policy_resolves_delegate_defaults_and_overrides():
    policy = WorkspaceIsolationPolicy.from_context({"mode": "delegate", "worktree_isolation": True})
    explicit_policy = WorkspaceIsolationPolicy.from_context(
        {
            "mode": "delegate",
            "worktree_isolation": True,
            "materialize_worktrees": False,
            "dry_run_worktrees": True,
            "auto_merge_worktrees": "yes",
            "allow_risky_worktree_merge": "true",
            "preserve_merge_workspace": "1",
            "cleanup_worktrees": "off",
        }
    )

    assert policy.mode == "delegate"
    assert policy.worktree_isolation is True
    assert policy.materialize_worktrees is True
    assert policy.dry_run_worktrees is False
    assert policy.should_materialize is True
    assert explicit_policy.materialize_worktrees is False
    assert explicit_policy.dry_run_worktrees is True
    assert explicit_policy.should_materialize is True
    assert explicit_policy.auto_merge_worktrees is True
    assert explicit_policy.allow_risky_worktree_merge is True
    assert explicit_policy.preserve_merge_workspace is True
    assert explicit_policy.cleanup_worktrees is False


def test_workspace_isolation_service_materialize_uses_policy_once():
    plan = _plan()
    session = _session()

    class Runtime:
        def __init__(self):
            self.materialize_calls = []

        def materialize(self, received_plan, *, dry_run=False):
            self.materialize_calls.append((received_plan, dry_run))
            return session

    runtime = Runtime()
    service = WorkspaceIsolationService(runtime=runtime)

    skipped = service.materialize(
        plan,
        context={"mode": "build", "worktree_isolation": True},
    )
    dry_run = service.materialize(
        plan,
        context={
            "mode": "build",
            "worktree_isolation": True,
            "dry_run_worktrees": True,
        },
    )

    assert skipped is None
    assert dry_run is session
    assert runtime.materialize_calls == [(plan, True)]


def test_workspace_isolation_service_materialize_reports_runtime_diagnostics():
    plan = _plan()

    class Runtime:
        def materialize(self, received_plan, *, dry_run=False):
            assert received_plan is plan
            assert dry_run is False
            raise WorktreeRuntimeError(
                "branch already exists",
                reason="branch_exists",
                details={"branch_name": "victor/team/worker-1", "member_id": "worker"},
            )

    service = WorkspaceIsolationService(runtime=Runtime())

    outcome = service.materialize_with_diagnostics(
        plan,
        context={"mode": "delegate", "worktree_isolation": True},
    )

    assert outcome.session is None
    diagnostic = outcome.diagnostics_payload()[0]
    assert diagnostic["operation"] == "materialize"
    assert diagnostic["reason"] == "branch_exists"
    assert diagnostic["message"] == "branch already exists"
    assert diagnostic["details"]["branch_name"] == "victor/team/worker-1"
    assert diagnostic["details"]["member_id"] == "worker"


def test_workspace_isolation_service_execute_merge_preserves_blocking_diagnostics():
    session = _session()

    class Runtime:
        def execute_merge_orchestration(
            self,
            received_session,
            *,
            merge_analysis=None,
            allow_risky=False,
            preserve_artifacts=False,
        ):
            assert received_session is session
            raise WorktreeRuntimeError(
                "integration workspace already exists",
                reason="integration_workspace_exists",
                details={"path": "/tmp/victor-integration"},
            )

    service = WorkspaceIsolationService(runtime=Runtime())

    result = service.execute_merge(session, context={})

    assert result["executed"] is False
    assert result["blocked_reason"] == "integration_workspace_exists"
    assert result["error_details"]["path"] == "/tmp/victor-integration"
    assert result["diagnostics"][0]["operation"] == "merge_execute"
    assert result["diagnostics"][0]["reason"] == "integration_workspace_exists"


def test_workspace_isolation_policy_controls_merge_and_cleanup_decisions():
    service = WorkspaceIsolationService()

    default_delegate_merge = service.should_execute_merge(
        {"mode": "delegate", "worktree_isolation": True},
        merge_orchestration={"merge_execution_eligible": True},
    )
    explicit_merge_override = service.should_execute_merge(
        {"mode": "build", "auto_merge_worktrees": True},
        merge_orchestration={"merge_execution_eligible": False},
    )
    cleanup_override = service.should_cleanup(
        {"cleanup_worktrees": False},
        result_dict={"delegate_follow_up_contract": {}},
    )

    assert default_delegate_merge is True
    assert explicit_merge_override is True
    assert cleanup_override is False


def test_workspace_isolation_service_injects_changed_files_without_overwriting_metadata():
    session = _session()

    class Runtime:
        def collect_changed_files(self, received_session, member_id):
            assert received_session is session
            assert member_id == "worker"
            return ("src/service.py",)

    service = WorkspaceIsolationService(runtime=Runtime())
    member_results = {
        "worker": MemberResult(
            member_id="worker",
            success=True,
            output="done",
            metadata={},
        ),
        "reported": MemberResult(
            member_id="reported",
            success=True,
            output="done",
            metadata={"changed_files": ["already.py"]},
        ),
    }

    service.inject_changed_files(member_results, worktree_session=session)

    assert member_results["worker"].metadata["changed_files"] == ["src/service.py"]
    assert member_results["reported"].metadata["changed_files"] == ["already.py"]


def test_workspace_isolation_service_preserves_cleanup_for_follow_up_contracts():
    session = _session()
    service = WorkspaceIsolationService()

    should_cleanup = service.should_cleanup(
        {},
        result_dict={"delegate_follow_up_contract": {"preserve_worktrees": True}},
    )
    summary = service.preserved_cleanup_summary(session, reason="preserved_for_follow_up")

    assert should_cleanup is False
    assert summary == {
        "removed": [],
        "skipped": ["/tmp/team-worker"],
        "errors": [],
        "reason": "preserved_for_follow_up",
    }


def test_workspace_isolation_service_builds_merge_review_contract_from_worker_returns():
    service = WorkspaceIsolationService()
    worker_contracts = {
        "worker": {
            "member_id": "worker",
            "changed_files": ["src/service.py"],
            "validation_run": {
                "status": "failed",
                "command": "pytest tests/unit",
                "summary": "unit failure",
            },
            "merge_risk": {
                "level": "high",
                "reasons": ["readonly violation"],
            },
        },
        "reviewer": {
            "member_id": "reviewer",
            "changed_files": ["tests/test_service.py"],
            "validation_run": {"status": "passed"},
            "merge_risk": {"level": "low"},
        },
    }

    contract = service.build_merge_review_contract(
        worker_contracts,
        merge_analysis={
            "risk_level": "high",
            "recommended_merge_order": ["worker", "reviewer"],
        },
        merge_orchestration={
            "merge_execution_eligible": False,
            "recommended_mode": "manual_review",
        },
    )

    assert contract["merge_ready"] is False
    assert contract["review_required"] is True
    assert contract["next_action"] == "fix_validation"
    assert contract["recommended_merge_order"] == ["worker", "reviewer"]
    assert contract["validation_failed_members"] == ["worker"]
    assert contract["review_required_members"] == ["worker"]
    assert contract["merge_execution_eligible"] is False
    assert contract["recommended_mode"] == "manual_review"
    assert {
        "type": "validation_failed",
        "member_id": "worker",
        "status": "failed",
        "summary": "unit failure",
        "command": "pytest tests/unit",
    } in contract["blocking_issues"]
    assert any(issue["type"] == "merge_risk_high" for issue in contract["blocking_issues"])


def test_workspace_isolation_service_marks_low_risk_merge_ready():
    service = WorkspaceIsolationService()

    contract = service.build_merge_review_contract(
        {
            "worker": {
                "member_id": "worker",
                "changed_files": ["src/service.py"],
                "validation_run": {"status": "passed"},
                "merge_risk": {"level": "low"},
            }
        },
        merge_analysis={"risk_level": "low", "recommended_merge_order": ["worker"]},
        merge_orchestration={
            "merge_execution_eligible": True,
            "recommended_mode": "auto_apply_safe",
        },
    )

    assert contract["merge_ready"] is True
    assert contract["review_required"] is False
    assert contract["next_action"] == "merge"
    assert contract["review_required_members"] == []
    assert contract["blocking_issues"] == []


def test_policy_flags_accept_bool_and_truthy_strings():
    policy = WorkspaceIsolationPolicy.from_context(
        {
            "worktree_isolation": " Yes ",
            "materialize_worktrees": "off",
            "dry_run_worktrees": True,
            "allow_risky_worktree_merge": "1",
        }
    )
    assert policy.worktree_isolation is True
    assert policy.materialize_worktrees is False
    assert policy.dry_run_worktrees is True
    assert policy.allow_risky_worktree_merge is True
    assert WorkspaceIsolationPolicy.from_context({}).worktree_isolation is False

"""Pure delegate-contract and worktree-plan builders, tested without a coordinator."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.teams import delegate_contracts as dc
from victor.teams.types import TeamFormation
from victor.teams.worktree_runtime import (
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
    WorktreeExecutionPlan,
    WorktreeMaterializationSession,
)


class TestNormalizers:
    def test_text_and_lists_strip_dedupe_and_drop_blanks(self):
        assert dc.coerce_optional_text("  x ") == "x"
        assert dc.coerce_optional_text("   ") is None
        assert dc.coerce_optional_text(None) is None
        assert dc.normalize_path_list("src/") == ["src"]
        assert dc.normalize_path_list(["a/", "a", " ", None, "b"]) == ["a", "b"]
        assert dc.normalize_member_id_list(["m1", "m1", "", "m2"]) == ["m1", "m2"]
        assert dc.normalize_path_map({"m1": ["x/"], " ": ["y"], "m2": None}) == {
            "m1": ["x"],
            "m2": [],
        }
        assert dc.normalize_path_map(["not", "a", "map"]) == {}

    def test_context_mode_prefers_first_non_blank_key(self):
        assert dc.resolve_context_mode({"mode": " ", "current_mode": "Delegate"}) == "delegate"
        assert dc.resolve_context_mode({}) is None

    def test_workspace_diagnostics_fill_defaults(self):
        (diag,) = dc.normalize_workspace_diagnostics({"blocked_reason": "dirty", "details": 3})
        assert diag == {
            "blocked_reason": "dirty",
            "operation": "workspace_isolation",
            "reason": "dirty",
            "message": "dirty",
            "severity": "warning",
            "details": {},
        }
        assert dc.normalize_workspace_diagnostics([1, "x"]) == []

    def test_next_steps_get_unique_ids(self):
        steps = dc.normalize_delegate_next_steps(
            [{"step": "review"}, {"step": "review"}, {}, "junk"]
        )
        assert [s["step_id"] for s in steps] == ["review", "review_2", "delegate_step_3"]
        assert dc.normalize_delegate_next_steps("not a list") == []


class TestContractExtraction:
    def test_approval_contract_falls_back_to_follow_up_nesting(self):
        nested = {"delegate_follow_up_contract": {"approval_contract": {"a": 1}}}
        assert dc.extract_delegate_approval_contract(nested) == {"a": 1}
        assert dc.extract_delegate_approval_contract({"delegate_approval_contract": {"b": 2}}) == {
            "b": 2
        }
        assert dc.extract_delegate_reentry_contract({"delegate_reentry_contract": "x"}) == {}

    def test_next_steps_fall_back_to_approval_contract(self):
        context = {"delegate_approval_contract": {"next_steps": [{"step": "merge"}]}}
        assert dc.extract_delegate_follow_up_next_steps(context)[0]["step_id"] == "merge"

    def test_next_step_resolution_by_id_or_name(self):
        context = {
            "delegate_follow_up_contract": {"next_steps": [{"step_id": "s1", "step": "fix"}]},
            "delegate_next_step_id": "fix",
        }
        assert dc.extract_delegate_next_step(context)["step_id"] == "s1"
        assert dc.resolve_delegate_next_step_by_id(context, step_id=" ") == {}
        assert dc.extract_delegate_next_step({}) == {}


class TestRequests:
    def test_step_requests_are_deep_copied(self):
        steps = [{"step": "fix", "instruction": "Fix it", "meta": {"k": 1}}]
        requests = dc.build_delegate_follow_up_step_requests(
            next_steps=steps, primary_step_id="fix"
        )
        request = requests["fix"]
        assert request["delegate_follow_up_contract"]["primary_step_id"] == "fix"
        request["delegate_follow_up_contract"]["next_steps"][0]["meta"]["k"] = 2
        assert steps[0]["meta"]["k"] == 1
        executions = dc.build_delegate_follow_up_step_execution_requests(
            next_steps=steps, step_requests=requests
        )
        assert executions["fix"]["task"] == "Fix it"

    @pytest.mark.parametrize(
        "request_payload,error",
        [
            ("nope", TypeError),
            ({"task": " ", "context": {}}, ValueError),
            ({"task": "go", "context": "x"}, ValueError),
        ],
    )
    def test_execution_request_validation(self, request_payload, error):
        with pytest.raises(error):
            dc.normalize_delegate_execution_request(request_payload)

    def test_valid_execution_request_copies_context(self):
        ctx = {"k": [1]}
        task, copied = dc.normalize_delegate_execution_request({"task": "go", "context": ctx})
        assert task == "go" and copied == ctx and copied["k"] is not ctx["k"]


class TestSuggestions:
    def test_commands_are_escaped_and_descriptions_truncated(self):
        (s,) = dc.build_delegate_follow_up_suggestions(
            [{"step_id": "o'k", "instruction": "x" * 200, "requires_approval": 1}]
        )
        assert s["command"] == "delegate_follow_up(step_id='o\\'k')"
        assert len(s["description"]) == 120 and s["description"].endswith("...")
        assert s["requires_approval"] is True

    def test_attach_prefers_explicit_suggestions_and_skips_empty(self):
        result: dict = {}
        dc.attach_delegate_follow_up_suggestions(result, {"follow_up_suggestions": [{"a": 1}]})
        assert result["follow_up_suggestions"] == [{"a": 1}]
        empty: dict = {}
        dc.attach_delegate_follow_up_suggestions(empty, {})
        assert empty == {}


class TestContextMerging:
    def test_caller_context_wins_over_contract_overrides(self):
        merged = dc.apply_delegate_merge_context(
            {"mode": "user"}, delegate_merge_contract={"context_overrides": {"mode": "x", "y": 1}}
        )
        assert merged == {"mode": "user", "y": 1}
        reentry = dc.apply_delegate_reentry_context(
            {"a": 1}, delegate_reentry_contract={"context_overrides": "bad"}
        )
        assert reentry == {"a": 1}

    def test_next_step_context_layers_resume_execution_then_selection(self):
        merged = dc.apply_delegate_next_step_context(
            {"k": "ctx"},
            delegate_next_step={
                "step": "fix",
                "instruction": "do",
                "resume_context": {"k": "resume", "r": 1},
                "execution_context": {"e": 2},
            },
        )
        assert merged == {
            "k": "ctx",
            "r": 1,
            "e": 2,
            "delegate_selected_step": "fix",
            "delegate_selected_instruction": "do",
        }
        assert dc.apply_delegate_next_step_context({"a": 1}, delegate_next_step={}) == {"a": 1}


class TestReentryAndApproval:
    def test_reentry_member_resolution(self):
        assert dc.resolve_delegate_reentry_member_ids(
            next_action="fix_validation",
            validation_failed_members=["a"],
            review_required_members=["b"],
        ) == ["a"]
        assert dc.resolve_delegate_reentry_member_ids(
            next_action="other", validation_failed_members=["a", "b"], review_required_members=["b"]
        ) == ["a", "b"]
        assert dc.extract_delegate_reentry_member_ids(
            {"resume_member_context_overrides": {"m2": {}, "m1": {}}}
        ) == ["m2", "m1"]

    def test_reentry_overrides_carry_action_and_brief(self):
        overrides = dc.extract_delegate_reentry_member_context_overrides(
            {
                "resume_member_context_overrides": {"m1": {"x": 1}, "m2": "bad"},
                "next_action": "review",
                "retry_tasks_by_member": {"m1": "redo tests"},
            }
        )
        assert overrides == {
            "m1": {
                "x": 1,
                "delegate_reentry_next_action": "review",
                "follow_up_task_brief": "redo tests",
            }
        }

    def test_approval_resume_context_and_briefs(self):
        assert dc.build_delegate_approval_resume_context({}, target_member_ids=["m1"]) is None
        ctx = dc.build_delegate_approval_resume_context(
            {"resume_worktree_paths": {"m1": ["a"]}}, target_member_ids=["m1"]
        )
        assert ctx["delegate_reentry_contract"]["retry_member_ids"] == ["m1"]
        briefs = dc.build_delegate_approval_task_briefs(
            {"retry_tasks_by_member": {"m1": "a", "m2": " "}}, target_member_ids=[]
        )
        assert briefs == {"m1": "a"}
        assert (
            dc.build_delegate_approval_summary("Approve", target_member_ids=["m1"])
            == "Approve: m1."
        )
        assert dc.build_delegate_approval_summary("Approve", target_member_ids=[]) == (
            "Approve the delegate worktree set."
        )


class TestMemberHelpers:
    def test_member_filter_falls_back_to_everyone(self):
        members = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
        assert dc.filter_execution_members(members, member_ids=["b"]) == [members[1]]
        assert dc.filter_execution_members(members, member_ids=["zzz"]) == members
        assert dc.filter_execution_members(members, member_ids=[]) == members

    def test_output_summary_and_validation_run(self):
        assert dc.summarize_member_output("a \n b") == "a b"
        assert len(dc.summarize_member_output("w " * 400)) == 280
        assert dc.summarize_member_output(" ") is None
        assert dc.normalize_validation_run(
            {"validation_run": {"outcome": "passed"}, "test_command": "pytest"}
        ) == {"status": "passed", "command": "pytest"}
        assert dc.normalize_validation_run({}) is None


class TestWorktreeRebuild:
    def test_session_round_trips_through_to_dict(self):
        assignment = WorktreeAssignment(
            member_id="m1",
            branch_name="team/m1",
            worktree_name="m1",
            worktree_path="/tmp/wt/m1",
            claimed_paths=("src",),
            readonly_paths=("docs",),
            merge_priority=2,
            metadata={"k": "v"},
        )
        plan = WorktreeExecutionPlan(
            team_name="t",
            repo_root="/repo",
            parent_dir="/tmp/wt",
            base_ref="develop",
            branch_prefix="team",
            formation=TeamFormation.PARALLEL,
            assignments=(assignment,),
            merge_order=("m1",),
            shared_readonly_paths=("docs",),
            rationale="why",
            metadata={},
        )
        session = WorktreeMaterializationSession(
            plan=plan,
            assignments=(
                MaterializedWorktreeAssignment(
                    assignment=assignment, materialized=True, cleanup_required=True, metadata={}
                ),
            ),
            materialized=True,
            dry_run=False,
            metadata={"s": 1},
        )
        rebuilt = dc.rebuild_worktree_session(session.to_dict())
        assert rebuilt is not None
        assert rebuilt.to_dict() == session.to_dict()

    def test_incomplete_payloads_are_rejected(self):
        assert dc.rebuild_worktree_assignment({"member_id": "m1"}) is None
        assert dc.rebuild_worktree_plan({"team_name": "t"}) is None
        assert dc.rebuild_worktree_session({}) is None
        plan = dc.rebuild_worktree_plan(
            {
                "team_name": "t",
                "repo_root": "/r",
                "parent_dir": "/p",
                "base_ref": "b",
                "branch_prefix": "x",
                "formation": "not-a-formation",
            }
        )
        assert plan is not None and plan.formation is TeamFormation.SEQUENTIAL


def test_module_is_independent_of_the_coordinator():
    tree = ast.parse(Path(dc.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "victor.teams.unified_coordinator" not in imported
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "cls"]

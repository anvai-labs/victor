# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Worktree-isolation planning helpers for team execution.

This module does not materialize git worktrees directly. It provides the
stable planning contract and per-member context overlays that the runtime can
apply before shell-backed worktree creation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Optional, Sequence

from victor.teams.types import TeamFormation


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slug(value: Any) -> str:
    text = _normalize_text(value) or "member"
    normalized = re.sub(r"[^a-zA-Z0-9._/-]+", "-", text).strip("-./")
    normalized = normalized.replace("/", "-").replace("\\", "-")
    return normalized or "member"


def _normalize_paths(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in list(values or []):
        text = _normalize_text(value)
        if text is None:
            continue
        normalized.append(text.rstrip("/"))
    return tuple(dict.fromkeys(normalized))


def _normalize_scope_map(raw_map: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_map, dict):
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for member_id, scopes in raw_map.items():
        key = _normalize_text(member_id)
        if key is None:
            continue
        normalized[key] = _normalize_paths(scopes)
    return normalized


def _path_for_member(parent_dir: str, team_name: str, member_id: str) -> str:
    return str(Path(parent_dir) / f"{_slug(team_name)}-{_slug(member_id)}")


@dataclass(frozen=True)
class WorktreeAssignment:
    """One member's isolated workspace assignment."""

    member_id: str
    branch_name: str
    worktree_name: str
    worktree_path: str
    claimed_paths: tuple[str, ...] = ()
    readonly_paths: tuple[str, ...] = ()
    merge_priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_id": self.member_id,
            "branch_name": self.branch_name,
            "worktree_name": self.worktree_name,
            "worktree_path": self.worktree_path,
            "claimed_paths": list(self.claimed_paths),
            "readonly_paths": list(self.readonly_paths),
            "merge_priority": self.merge_priority,
            "metadata": dict(self.metadata),
        }

    def to_context_overrides(self) -> Dict[str, Any]:
        return {
            "isolation_mode": "worktree",
            "workspace_root": self.worktree_path,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "claimed_paths": list(self.claimed_paths),
            "readonly_paths": list(self.readonly_paths),
            "worktree_assignment": self.to_dict(),
        }


@dataclass(frozen=True)
class WorktreeExecutionPlan:
    """Execution-scoped isolation plan for a team run."""

    team_name: str
    repo_root: str
    parent_dir: str
    base_ref: str
    branch_prefix: str
    formation: TeamFormation
    assignments: tuple[WorktreeAssignment, ...]
    merge_order: tuple[str, ...]
    shared_readonly_paths: tuple[str, ...] = ()
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assignment_for(self, member_id: str) -> Optional[WorktreeAssignment]:
        normalized = _normalize_text(member_id)
        if normalized is None:
            return None
        for assignment in self.assignments:
            if assignment.member_id == normalized:
                return assignment
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_name": self.team_name,
            "repo_root": self.repo_root,
            "parent_dir": self.parent_dir,
            "base_ref": self.base_ref,
            "branch_prefix": self.branch_prefix,
            "formation": self.formation.value,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "merge_order": list(self.merge_order),
            "shared_readonly_paths": list(self.shared_readonly_paths),
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MaterializedWorktreeAssignment:
    """One worktree assignment after optional git-backed materialization."""

    assignment: WorktreeAssignment
    materialized: bool = False
    cleanup_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def member_id(self) -> str:
        return self.assignment.member_id

    @property
    def worktree_path(self) -> str:
        return self.assignment.worktree_path

    @property
    def branch_name(self) -> str:
        return self.assignment.branch_name

    def to_dict(self) -> Dict[str, Any]:
        payload = self.assignment.to_dict()
        payload.update(
            {
                "materialized": self.materialized,
                "cleanup_required": self.cleanup_required,
                "runtime_metadata": dict(self.metadata),
            }
        )
        return payload

    def to_context_overrides(self) -> Dict[str, Any]:
        payload = self.assignment.to_context_overrides()
        payload["materialized_worktree"] = self.materialized
        if self.metadata:
            payload["worktree_runtime"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class WorktreeMaterializationSession:
    """Runtime session for a planned set of isolated worktrees."""

    plan: WorktreeExecutionPlan
    assignments: tuple[MaterializedWorktreeAssignment, ...]
    materialized: bool
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assignment_for(self, member_id: str) -> Optional[MaterializedWorktreeAssignment]:
        normalized = _normalize_text(member_id)
        if normalized is None:
            return None
        for assignment in self.assignments:
            if assignment.member_id == normalized:
                return assignment
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "materialized": self.materialized,
            "dry_run": self.dry_run,
            "metadata": dict(self.metadata),
        }


class WorktreeRuntimeError(RuntimeError):
    """Raised when git-backed worktree orchestration fails."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "worktree_runtime_error",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = dict(details or {})


class WorktreeIsolationPlanner:
    """Plan isolated per-member workspaces for team execution."""

    def plan(
        self,
        members: Sequence[Any],
        *,
        context: Dict[str, Any],
        formation: TeamFormation,
    ) -> Optional[WorktreeExecutionPlan]:
        if not self._is_enabled(context) or not members:
            return None

        repo_root = _normalize_text(
            context.get("repo_root")
            or context.get("workspace_root")
            or context.get("worktree_root")
        )
        if repo_root is None:
            return None

        team_name = _normalize_text(context.get("team_name")) or "UnifiedTeam"
        parent_dir = _normalize_text(context.get("worktree_parent"))
        if parent_dir is None:
            parent_dir = str(Path(repo_root) / ".victor" / "team_worktrees")
        branch_prefix = (
            _normalize_text(context.get("branch_prefix")) or f"victor/{_slug(team_name)}"
        )
        base_ref = _normalize_text(context.get("base_ref")) or "HEAD"
        shared_readonly_paths = _normalize_paths(
            context.get("shared_readonly_paths") or context.get("readonly_paths")
        )
        scope_map = _normalize_scope_map(
            context.get("member_write_scopes") or context.get("member_scopes")
        )
        explicit_merge_order = [
            member_id
            for member_id in (_normalize_paths(context.get("member_merge_order")) or ())
            if member_id
        ]

        assignments: list[WorktreeAssignment] = []
        for index, member in enumerate(members):
            member_id = _normalize_text(getattr(member, "id", None)) or f"member-{index + 1}"
            claimed_paths = scope_map.get(member_id, ())
            merge_priority = self._resolve_merge_priority(
                member=member,
                index=index,
                formation=formation,
                explicit_merge_order=explicit_merge_order,
            )
            assignments.append(
                WorktreeAssignment(
                    member_id=member_id,
                    branch_name=f"{branch_prefix}/{_slug(member_id)}-{index + 1}",
                    worktree_name=f"{_slug(team_name)}-{_slug(member_id)}",
                    worktree_path=_path_for_member(parent_dir, team_name, member_id),
                    claimed_paths=claimed_paths,
                    readonly_paths=shared_readonly_paths,
                    merge_priority=merge_priority,
                    metadata={
                        "member_index": index,
                        "formation": formation.value,
                        "is_manager": bool(getattr(member, "is_manager", False)),
                        "is_supervisor": _is_supervisor_member(member),
                    },
                )
            )

        merge_order = tuple(
            assignment.member_id
            for assignment in sorted(
                assignments, key=lambda item: (item.merge_priority, item.member_id)
            )
        )
        return WorktreeExecutionPlan(
            team_name=team_name,
            repo_root=repo_root,
            parent_dir=parent_dir,
            base_ref=base_ref,
            branch_prefix=branch_prefix,
            formation=formation,
            assignments=tuple(assignments),
            merge_order=merge_order,
            shared_readonly_paths=shared_readonly_paths,
            rationale="worktree isolation enabled by runtime context",
            metadata={
                "member_count": len(assignments),
                "scoped_members": sum(1 for assignment in assignments if assignment.claimed_paths),
            },
        )

    @staticmethod
    def _is_enabled(context: Dict[str, Any]) -> bool:
        raw_value = context.get("worktree_isolation")
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is not None:
            text = str(raw_value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return _normalize_text(context.get("isolation_mode")) == "worktree"

    @staticmethod
    def _resolve_merge_priority(
        *,
        member: Any,
        index: int,
        formation: TeamFormation,
        explicit_merge_order: Sequence[str],
    ) -> int:
        member_id = _normalize_text(getattr(member, "id", None)) or f"member-{index + 1}"
        if member_id in explicit_merge_order:
            return explicit_merge_order.index(member_id)
        if formation == TeamFormation.HIERARCHICAL and _is_supervisor_member(member):
            return len(explicit_merge_order) + 1000
        return len(explicit_merge_order) + index


def _is_supervisor_member(member: Any) -> bool:
    """Return whether a member coordinates a hierarchical team."""
    return bool(getattr(member, "is_supervisor", False) or getattr(member, "is_manager", False))


class GitWorktreeRuntime:
    """Materialize planned worktrees and collect merge-time repo signals."""

    def materialize(
        self,
        plan: WorktreeExecutionPlan,
        *,
        dry_run: bool = False,
    ) -> WorktreeMaterializationSession:
        if dry_run:
            assignments = tuple(
                MaterializedWorktreeAssignment(
                    assignment=assignment,
                    materialized=False,
                    cleanup_required=False,
                    metadata={"dry_run": True},
                )
                for assignment in plan.assignments
            )
            return WorktreeMaterializationSession(
                plan=plan,
                assignments=assignments,
                materialized=False,
                dry_run=True,
                metadata={"dry_run": True},
            )

        Path(plan.parent_dir).mkdir(parents=True, exist_ok=True)
        created_paths: list[str] = []
        assignments: list[MaterializedWorktreeAssignment] = []
        try:
            for assignment in plan.assignments:
                worktree_path = Path(assignment.worktree_path)
                if worktree_path.exists():
                    raise WorktreeRuntimeError(
                        f"Worktree path already exists: {assignment.worktree_path}",
                        reason="worktree_path_exists",
                        details={
                            "member_id": assignment.member_id,
                            "worktree_path": assignment.worktree_path,
                            "branch_name": assignment.branch_name,
                            "stale_path": True,
                        },
                    )
                if self._branch_exists(plan.repo_root, assignment.branch_name):
                    raise WorktreeRuntimeError(
                        f"Worktree branch already exists: {assignment.branch_name}",
                        reason="branch_exists",
                        details={
                            "member_id": assignment.member_id,
                            "branch_name": assignment.branch_name,
                            "worktree_path": assignment.worktree_path,
                        },
                    )
                self._run_git(
                    plan.repo_root,
                    "worktree",
                    "add",
                    "-b",
                    assignment.branch_name,
                    assignment.worktree_path,
                    plan.base_ref,
                )
                created_paths.append(assignment.worktree_path)
                assignments.append(
                    MaterializedWorktreeAssignment(
                        assignment=assignment,
                        materialized=True,
                        cleanup_required=True,
                        metadata={
                            "repo_root": plan.repo_root,
                            "base_ref": plan.base_ref,
                        },
                    )
                )
        except Exception:
            for path in reversed(created_paths):
                try:
                    self._run_git(plan.repo_root, "worktree", "remove", "--force", path)
                except Exception:
                    pass
            raise

        return WorktreeMaterializationSession(
            plan=plan,
            assignments=tuple(assignments),
            materialized=True,
            dry_run=False,
            metadata={"created_paths": list(created_paths)},
        )

    def collect_changed_files(
        self,
        session: WorktreeMaterializationSession,
        member_id: str,
    ) -> tuple[str, ...]:
        assignment = session.assignment_for(member_id)
        if assignment is None or not assignment.materialized:
            return ()
        stdout = self._run_git_stdout(
            assignment.worktree_path,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        files: list[str] = []
        for line in stdout.splitlines():
            if not line.strip():
                continue
            filename = line[3:].strip()
            if " -> " in filename:
                filename = filename.split(" -> ", 1)[1]
            if filename:
                files.append(filename)
        return tuple(dict.fromkeys(files))

    def build_merge_orchestration(
        self,
        session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        recommended_merge_order = (
            list(merge_analysis.get("recommended_merge_order") or ())
            if isinstance(merge_analysis, dict)
            else list(session.plan.merge_order)
        )
        merge_risk_level = (
            merge_analysis.get("risk_level") if isinstance(merge_analysis, dict) else None
        )
        executable_without_override = (
            bool(session.materialized) and not bool(session.dry_run) and merge_risk_level == "low"
        )
        return {
            "materialized": session.materialized,
            "dry_run": session.dry_run,
            "recommended_merge_order": recommended_merge_order,
            "branches": {item.member_id: item.branch_name for item in session.assignments},
            "worktree_paths": {item.member_id: item.worktree_path for item in session.assignments},
            "merge_base": session.plan.base_ref,
            "merge_risk_level": merge_risk_level,
            "merge_execution_eligible": executable_without_override,
            "recommended_mode": (
                "auto_apply_safe"
                if executable_without_override
                else ("preview_only" if session.dry_run else "manual_review")
            ),
            "conflict_paths": [
                item.get("path")
                for item in (
                    merge_analysis.get("overlapping_files")
                    if isinstance(merge_analysis, dict)
                    else []
                )
                or []
                if item.get("path")
            ],
        }

    def execute_merge_orchestration(
        self,
        session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
        allow_risky: bool = False,
        preserve_artifacts: bool = False,
    ) -> Dict[str, Any]:
        """Execute a guarded merge plan in an isolated integration worktree."""
        orchestration = self.build_merge_orchestration(session, merge_analysis=merge_analysis)
        risk_level = str(orchestration.get("merge_risk_level") or "").strip().lower()
        recommended_merge_order = list(orchestration.get("recommended_merge_order") or [])

        result: Dict[str, Any] = {
            "status": "skipped",
            "executed": False,
            "eligible": bool(orchestration.get("merge_execution_eligible")),
            "materialized": session.materialized,
            "dry_run": session.dry_run,
            "risk_level": risk_level or None,
            "recommended_merge_order": recommended_merge_order,
            "merged_members": [],
            "attempted_members": [],
            "blocked_reason": None,
            "integration_branch": None,
            "integration_worktree_path": None,
            "missing_members": [],
            "integration_artifacts": {},
            "conflict_paths": list(orchestration.get("conflict_paths") or []),
            "cleanup": {
                "removed": [],
                "skipped": [],
                "errors": [],
                "branch_deleted": False,
            },
        }

        if not session.materialized:
            result["blocked_reason"] = "worktrees_not_materialized"
            return result
        if session.dry_run:
            result["status"] = "dry_run"
            result["blocked_reason"] = "dry_run_session"
            return result
        if risk_level and risk_level != "low" and not allow_risky:
            result["status"] = "blocked"
            result["blocked_reason"] = f"merge_risk_{risk_level}"
            return result

        missing_members = [
            member_id
            for member_id in recommended_merge_order
            if session.assignment_for(member_id) is None
        ]
        if missing_members:
            result["status"] = "blocked"
            result["blocked_reason"] = "merge_member_missing"
            result["missing_members"] = missing_members
            return result

        integration_branch = f"{session.plan.branch_prefix}/integration"
        integration_worktree_path = str(
            Path(session.plan.parent_dir) / f"{_slug(session.plan.team_name)}-integration"
        )
        result["integration_branch"] = integration_branch
        result["integration_worktree_path"] = integration_worktree_path

        integration_artifacts = self._inspect_integration_artifacts(
            repo_root=session.plan.repo_root,
            worktree_path=integration_worktree_path,
            branch_name=integration_branch,
        )
        result["integration_artifacts"] = integration_artifacts
        if integration_artifacts["worktree_path_exists"] or integration_artifacts["branch_exists"]:
            result["status"] = "blocked"
            result["blocked_reason"] = "integration_artifacts_exist"
            return result

        created_worktree = False
        try:
            self._run_git(
                session.plan.repo_root,
                "worktree",
                "add",
                "-B",
                integration_branch,
                integration_worktree_path,
                session.plan.base_ref,
            )
            created_worktree = True

            for member_id in recommended_merge_order:
                assignment = session.assignment_for(member_id)
                if assignment is None:
                    continue
                result["attempted_members"].append(member_id)
                try:
                    self._run_git(
                        integration_worktree_path,
                        "merge",
                        "--no-ff",
                        "--no-edit",
                        assignment.branch_name,
                    )
                except Exception as exc:
                    result["status"] = "conflict"
                    result["blocked_reason"] = "merge_conflict"
                    result["error"] = str(exc)
                    try:
                        self._run_git(integration_worktree_path, "merge", "--abort")
                    except Exception:
                        pass
                    return result
                result["merged_members"].append(member_id)

            result["status"] = "success"
            result["executed"] = True
            return result
        except Exception as exc:
            result["status"] = "error"
            result["blocked_reason"] = "merge_execution_failed"
            result["error"] = str(exc)
            return result
        finally:
            if created_worktree and not preserve_artifacts:
                result["cleanup"] = self._cleanup_integration_artifacts(
                    repo_root=session.plan.repo_root,
                    worktree_path=integration_worktree_path,
                    branch_name=integration_branch,
                )

    def _cleanup_integration_artifacts(
        self,
        *,
        repo_root: str,
        worktree_path: str,
        branch_name: str,
    ) -> Dict[str, Any]:
        summary = {"removed": [], "skipped": [], "errors": [], "branch_deleted": False}
        if Path(worktree_path).exists():
            try:
                self._run_git(repo_root, "worktree", "remove", "--force", worktree_path)
                summary["removed"].append(worktree_path)
            except Exception as exc:
                summary["errors"].append(f"{worktree_path}: {exc}")
        else:
            summary["skipped"].append(worktree_path)

        if self._branch_exists(repo_root, branch_name):
            try:
                self._run_git(repo_root, "branch", "-D", branch_name)
                summary["branch_deleted"] = True
            except Exception as exc:
                summary["errors"].append(f"{branch_name}: {exc}")
        return summary

    def _inspect_integration_artifacts(
        self,
        *,
        repo_root: str,
        worktree_path: str,
        branch_name: str,
    ) -> Dict[str, Any]:
        return {
            "worktree_path": worktree_path,
            "branch_name": branch_name,
            "worktree_path_exists": Path(worktree_path).exists(),
            "branch_exists": self._branch_exists(repo_root, branch_name),
        }

    @staticmethod
    def _branch_exists(repo_root: str, branch_name: str) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0

    def cleanup(
        self,
        session: WorktreeMaterializationSession,
        *,
        force: bool = True,
    ) -> Dict[str, Any]:
        removed: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        branch_deleted: list[str] = []
        branch_skipped: list[str] = []
        if not session.materialized:
            return {
                "removed": removed,
                "skipped": skipped,
                "errors": errors,
                "branch_deleted": branch_deleted,
                "branch_skipped": branch_skipped,
            }

        for assignment in session.assignments:
            if not assignment.cleanup_required:
                skipped.append(assignment.worktree_path)
            elif not Path(assignment.worktree_path).exists():
                skipped.append(assignment.worktree_path)
            else:
                args = ["worktree", "remove"]
                if force:
                    args.append("--force")
                args.append(assignment.worktree_path)
                try:
                    self._run_git(session.plan.repo_root, *args)
                    removed.append(assignment.worktree_path)
                except Exception as exc:
                    errors.append(f"{assignment.worktree_path}: {exc}")

            if self._branch_exists(session.plan.repo_root, assignment.branch_name):
                try:
                    self._run_git(session.plan.repo_root, "branch", "-D", assignment.branch_name)
                    branch_deleted.append(assignment.branch_name)
                except Exception as exc:
                    errors.append(f"{assignment.branch_name}: {exc}")
            else:
                branch_skipped.append(assignment.branch_name)
        return {
            "removed": removed,
            "skipped": skipped,
            "errors": errors,
            "branch_deleted": branch_deleted,
            "branch_skipped": branch_skipped,
        }

    @staticmethod
    def _run_git(repo_root: str, *args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise WorktreeRuntimeError(
                result.stderr.strip() or result.stdout.strip() or "git failed",
                reason="git_command_failed",
                details={
                    "repo_root": repo_root,
                    "args": list(args),
                    "return_code": result.returncode,
                },
            )

    @staticmethod
    def _run_git_stdout(repo_root: str, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise WorktreeRuntimeError(
                result.stderr.strip() or result.stdout.strip() or "git failed",
                reason="git_command_failed",
                details={
                    "repo_root": repo_root,
                    "args": list(args),
                    "return_code": result.returncode,
                },
            )
        return result.stdout


__all__ = [
    "GitWorktreeRuntime",
    "MaterializedWorktreeAssignment",
    "WorktreeAssignment",
    "WorktreeExecutionPlan",
    "WorktreeIsolationPlanner",
    "WorktreeMaterializationSession",
    "WorktreeRuntimeError",
]

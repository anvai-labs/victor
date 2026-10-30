from __future__ import annotations

"""Team/worktree feedback summarization helpers for evaluation artifacts."""

from collections import Counter
from typing import Any, Iterable, Mapping, Optional

from victor.evaluation.summary_mapping import extract_summary_mapping
from victor.evaluation.summary_mapping import as_event_mapping as _as_event_mapping
from victor.evaluation.summary_mapping import coerce_optional_int as _coerce_int
from victor.evaluation.summary_mapping import coerce_optional_text as _coerce_optional_text
from victor.evaluation.summary_mapping import extract_sequence as _extract_sequence
from victor.evaluation.summary_mapping import extract_value as _extract_value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    mapping = _as_event_mapping(value)
    return dict(mapping) if mapping is not None else {}


def _extract_mapping(value: Any, key: str) -> dict[str, Any]:
    return _coerce_mapping(_extract_value(value, key))


def _normalize_path_map(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for member_id, paths in value.items():
        key = _coerce_optional_text(member_id)
        if key is None:
            continue
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, (list, tuple)):
            continue
        normalized[key] = tuple(
            text for text in (_coerce_optional_text(path) for path in paths) if text is not None
        )
    return normalized


def _normalize_workspace_diagnostics(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        diagnostic = dict(item)
        reason = (
            _coerce_optional_text(diagnostic.get("reason"))
            or _coerce_optional_text(diagnostic.get("blocked_reason"))
            or _coerce_optional_text(diagnostic.get("type"))
            or "workspace_isolation"
        )
        message = (
            _coerce_optional_text(diagnostic.get("message"))
            or _coerce_optional_text(diagnostic.get("error"))
            or reason
        )
        operation = _coerce_optional_text(diagnostic.get("operation")) or "workspace_isolation"
        severity = _coerce_optional_text(diagnostic.get("severity")) or "warning"
        details = diagnostic.get("details")
        diagnostic["reason"] = reason
        diagnostic["message"] = message
        diagnostic["operation"] = operation
        diagnostic["severity"] = severity
        diagnostic["details"] = dict(details) if isinstance(details, Mapping) else {}
        normalized.append(diagnostic)
    return normalized


def _iter_metadata_containers(value: Any) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        containers.append(value)
    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            containers.append(container)
    for container in list(containers):
        task_report = container.get("task_report")
        if isinstance(task_report, Mapping):
            report_metadata = task_report.get("metadata")
            if isinstance(report_metadata, Mapping):
                containers.append(report_metadata)
    return containers


def _extract_workspace_diagnostics(value: Any) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for container in _iter_metadata_containers(value):
        candidates = [container.get("workspace_isolation_diagnostics")]
        follow_up = container.get("delegate_follow_up_contract")
        if isinstance(follow_up, Mapping):
            candidates.append(follow_up.get("workspace_isolation_diagnostics"))
            approval = follow_up.get("approval_contract")
            if isinstance(approval, Mapping):
                candidates.append(approval.get("workspace_isolation_diagnostics"))
        for candidate in candidates:
            for diagnostic in _normalize_workspace_diagnostics(candidate):
                identity = (
                    str(diagnostic.get("operation")),
                    str(diagnostic.get("reason")),
                    str(diagnostic.get("message")),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                diagnostics.append(diagnostic)
    return diagnostics


def _extract_workspace_policy(value: Any) -> dict[str, Any]:
    for container in _iter_metadata_containers(value):
        for key in ("workspace_isolation_policy", "workspace_policy"):
            policy = container.get(key)
            if isinstance(policy, Mapping):
                return dict(policy)
    return {}


def _derive_delegate_approval_contract(
    *,
    next_action: Optional[str],
    merge_next_action: Optional[str],
    delegate_follow_up_contract: Mapping[str, Any],
    delegate_reentry_contract: Mapping[str, Any],
    delegate_merge_contract: Mapping[str, Any],
    fix_validation_queue: list[Any],
    review_queue: list[Any],
    merge_order: list[Any],
) -> dict[str, Any]:
    resolved_action = next_action or merge_next_action or "inspect"
    retry_member_ids = _extract_sequence(delegate_reentry_contract, "retry_member_ids")
    review_member_ids = [
        member_id
        for member_id in (
            _coerce_optional_text(_extract_value(item, "member_id")) for item in review_queue
        )
        if member_id is not None
    ]
    validation_member_ids = [
        member_id
        for member_id in (
            _coerce_optional_text(_extract_value(item, "member_id"))
            for item in fix_validation_queue
        )
        if member_id is not None
    ]
    if resolved_action == "fix_validation":
        target_member_ids = retry_member_ids or validation_member_ids
        resume_ready = bool(delegate_reentry_contract) and bool(target_member_ids)
        return _enrich_delegate_approval_contract(
            {
                "required": not resume_ready,
                "reason": "validation_failed",
                "recommended_action": "retry" if resume_ready else "approve_retry",
                "resume_ready": resume_ready,
                "auto_retry_eligible": resume_ready,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
            },
            delegate_reentry_contract=delegate_reentry_contract,
            delegate_merge_contract=delegate_merge_contract,
        )
    if resolved_action == "review":
        target_member_ids = retry_member_ids or review_member_ids
        return _enrich_delegate_approval_contract(
            {
                "required": True,
                "reason": "review_required",
                "recommended_action": "review_then_retry",
                "resume_ready": bool(delegate_reentry_contract),
                "auto_retry_eligible": False,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
            },
            delegate_reentry_contract=delegate_reentry_contract,
            delegate_merge_contract=delegate_merge_contract,
        )
    if resolved_action == "merge":
        return _enrich_delegate_approval_contract(
            {
                "required": True,
                "reason": "merge_ready",
                "recommended_action": "approve_merge",
                "resume_ready": False,
                "auto_retry_eligible": False,
                "merge_executed": False,
                "target_member_ids": list(merge_order),
            },
            delegate_reentry_contract=delegate_reentry_contract,
            delegate_merge_contract=delegate_merge_contract,
        )
    target_member_ids = retry_member_ids or list(
        dict.fromkeys([*validation_member_ids, *review_member_ids])
    )
    return _enrich_delegate_approval_contract(
        {
            "required": True,
            "reason": "inspect_required",
            "recommended_action": "inspect_worktrees",
            "resume_ready": bool(delegate_reentry_contract),
            "auto_retry_eligible": False,
            "merge_executed": False,
            "target_member_ids": target_member_ids,
        },
        delegate_reentry_contract=delegate_reentry_contract,
        delegate_merge_contract=delegate_merge_contract,
    )


def _build_delegate_approval_resume_context(
    delegate_reentry_contract: Mapping[str, Any],
    *,
    target_member_ids: list[Any],
) -> Optional[dict[str, Any]]:
    if not delegate_reentry_contract:
        return None
    normalized_payload = dict(delegate_reentry_contract)
    retry_member_ids = _extract_sequence(normalized_payload, "retry_member_ids")
    resume_paths = _normalize_path_map(normalized_payload.get("resume_worktree_paths"))
    resume_overrides = _extract_mapping(normalized_payload, "resume_member_context_overrides")
    has_resume_details = bool(retry_member_ids) or bool(resume_paths) or bool(resume_overrides)
    if not has_resume_details:
        return None
    normalized_targets = [
        member_id
        for member_id in (_coerce_optional_text(item) for item in target_member_ids)
        if member_id is not None
    ]
    if not retry_member_ids and normalized_targets:
        normalized_payload["retry_member_ids"] = normalized_targets
    return {
        "mode": "delegate",
        "delegate_reentry_contract": normalized_payload,
    }


def _build_delegate_approval_task_briefs(
    delegate_reentry_contract: Mapping[str, Any],
    *,
    target_member_ids: list[Any],
) -> dict[str, str]:
    raw_briefs = delegate_reentry_contract.get("retry_tasks_by_member")
    if not isinstance(raw_briefs, Mapping):
        return {}
    prioritized_ids = [
        member_id
        for member_id in (_coerce_optional_text(item) for item in target_member_ids)
        if member_id is not None
    ]
    if not prioritized_ids:
        prioritized_ids = [
            member_id
            for member_id in (_coerce_optional_text(item) for item in raw_briefs.keys())
            if member_id is not None
        ]
    task_briefs: dict[str, str] = {}
    for member_id in prioritized_ids:
        task_brief = _coerce_optional_text(raw_briefs.get(member_id))
        if task_brief is None:
            continue
        task_briefs[member_id] = task_brief
    return task_briefs


def _build_delegate_approval_summary(
    prefix: str,
    *,
    target_member_ids: list[Any],
) -> str:
    normalized_targets = [
        member_id
        for member_id in (_coerce_optional_text(item) for item in target_member_ids)
        if member_id is not None
    ]
    if normalized_targets:
        return f"{prefix}: {', '.join(normalized_targets)}."
    return f"{prefix} the delegate worktree set."


def _build_delegate_approval_next_steps(
    contract: Mapping[str, Any],
    *,
    target_member_ids: list[Any],
    resume_context: Mapping[str, Any],
    task_briefs: Mapping[str, str],
    merge_execution_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    recommended_action = _coerce_optional_text(contract.get("recommended_action"))
    summary = _coerce_optional_text(contract.get("summary"))
    requires_approval = bool(contract.get("required", False))
    normalized_targets = [
        member_id
        for member_id in (_coerce_optional_text(item) for item in target_member_ids)
        if member_id is not None
    ]

    def build_step(
        step: str,
        instruction: Optional[str],
        *,
        step_requires_approval: bool,
        include_resume: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": step,
            "instruction": instruction
            or _build_delegate_approval_summary(
                "Continue delegate follow-up for",
                target_member_ids=normalized_targets,
            ),
            "target_member_ids": list(normalized_targets),
            "requires_approval": step_requires_approval,
        }
        if include_resume and resume_context:
            payload["resume_context"] = dict(resume_context)
        if include_resume and task_briefs:
            payload["task_briefs_by_member"] = dict(task_briefs)
        return payload

    if recommended_action == "merged" or bool(contract.get("merge_executed", False)):
        return [build_step("status_merged", summary, step_requires_approval=False)]
    if recommended_action == "retry":
        return [
            build_step(
                "resume_delegate_retry",
                summary,
                step_requires_approval=False,
                include_resume=True,
            )
        ]
    if recommended_action == "approve_retry":
        return [build_step("approve_delegate_retry", summary, step_requires_approval=True)]
    if recommended_action == "review_then_retry":
        steps = [build_step("review_worktrees", summary, step_requires_approval=True)]
        if resume_context:
            steps.append(
                build_step(
                    "resume_delegate_retry",
                    _build_delegate_approval_summary(
                        "Resume preserved worktrees after review for",
                        target_member_ids=normalized_targets,
                    ),
                    step_requires_approval=requires_approval,
                    include_resume=True,
                )
            )
        return steps
    if recommended_action == "approve_merge":
        step = build_step("approve_merge_execution", summary, step_requires_approval=True)
        if merge_execution_contract:
            step["execution_context"] = {
                "mode": "delegate",
                "delegate_merge_contract": dict(merge_execution_contract),
            }
        return [step]
    if recommended_action == "inspect_worktrees":
        steps = [build_step("inspect_worktrees", summary, step_requires_approval=True)]
        if resume_context:
            steps.append(
                build_step(
                    "resume_delegate_retry",
                    _build_delegate_approval_summary(
                        "Resume preserved worktrees after inspection for",
                        target_member_ids=normalized_targets,
                    ),
                    step_requires_approval=requires_approval,
                    include_resume=True,
                )
            )
        return steps
    return []


def _enrich_delegate_approval_contract(
    contract: Mapping[str, Any],
    *,
    delegate_reentry_contract: Mapping[str, Any],
    delegate_merge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    enriched_contract = dict(contract)
    target_member_ids = _extract_sequence(enriched_contract, "target_member_ids")
    if "resume_context" not in enriched_contract:
        resume_context = _build_delegate_approval_resume_context(
            delegate_reentry_contract,
            target_member_ids=target_member_ids,
        )
        if resume_context is not None:
            enriched_contract["resume_context"] = resume_context
    if "task_briefs_by_member" not in enriched_contract:
        task_briefs = _build_delegate_approval_task_briefs(
            delegate_reentry_contract,
            target_member_ids=target_member_ids,
        )
        if task_briefs:
            enriched_contract["task_briefs_by_member"] = task_briefs
    if "next_steps" not in enriched_contract:
        next_steps = _build_delegate_approval_next_steps(
            enriched_contract,
            target_member_ids=target_member_ids,
            resume_context=_extract_mapping(enriched_contract, "resume_context"),
            task_briefs=_extract_mapping(enriched_contract, "task_briefs_by_member"),
            merge_execution_contract=delegate_merge_contract,
        )
        if next_steps:
            enriched_contract["next_steps"] = next_steps
    return enriched_contract


def extract_team_feedback_artifacts(value: Any) -> dict[str, dict[str, Any]]:
    """Extract normalized team/worktree artifacts from task, trace, or payload objects."""
    artifacts: dict[str, dict[str, Any]] = {}
    for key in (
        "worktree_plan",
        "worktree_session",
        "merge_analysis",
        "merge_orchestration",
        "worktree_cleanup",
        "worker_return_contracts",
        "merge_review_contract",
        "delegate_follow_up_contract",
        "workspace_isolation_policy",
    ):
        mapping = _extract_mapping(value, key)
        if mapping:
            artifacts[key] = mapping

    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if not isinstance(container, Mapping):
            continue
        for key in (
            "worktree_plan",
            "worktree_session",
            "merge_analysis",
            "merge_orchestration",
            "worktree_cleanup",
            "worker_return_contracts",
            "merge_review_contract",
            "delegate_follow_up_contract",
            "workspace_isolation_policy",
        ):
            if key in artifacts:
                continue
            mapping = _extract_mapping(container, key)
            if mapping:
                artifacts[key] = mapping

    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        for key, mapping in extract_team_feedback_artifacts(trace).items():
            artifacts.setdefault(key, mapping)
    return artifacts


def summarize_team_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level team/worktree summary suitable for benchmark artifacts."""
    existing_summary = extract_summary_mapping(value, "team_feedback_summary")
    if existing_summary:
        return existing_summary

    artifacts = extract_team_feedback_artifacts(value)
    workspace_policy = _extract_workspace_policy(value)
    workspace_diagnostics = _extract_workspace_diagnostics(value)
    if not artifacts and not workspace_policy and not workspace_diagnostics:
        return None

    plan = artifacts.get("worktree_plan", {})
    session = artifacts.get("worktree_session", {})
    merge_analysis = artifacts.get("merge_analysis", {})
    merge_orchestration = artifacts.get("merge_orchestration", {})
    cleanup = artifacts.get("worktree_cleanup", {})
    worker_return_contracts = _coerce_mapping(artifacts.get("worker_return_contracts"))
    merge_review_contract = _coerce_mapping(artifacts.get("merge_review_contract"))
    delegate_follow_up_contract = _coerce_mapping(artifacts.get("delegate_follow_up_contract"))
    workspace_policy = workspace_policy or _coerce_mapping(
        artifacts.get("workspace_isolation_policy")
    )
    workspace_diagnostic_reasons = Counter(
        diagnostic.get("reason") for diagnostic in workspace_diagnostics if diagnostic.get("reason")
    )
    workspace_diagnostic_operations = Counter(
        diagnostic.get("operation")
        for diagnostic in workspace_diagnostics
        if diagnostic.get("operation")
    )

    plan_assignments = _extract_sequence(plan, "assignments")
    session_assignments = _extract_sequence(session, "assignments")
    assignments = plan_assignments or session_assignments
    assignment_count = len(assignments)
    scoped_member_count = sum(
        1 for assignment in assignments if _extract_sequence(assignment, "claimed_paths")
    )
    readonly_shared_path_count = len(_extract_sequence(plan, "shared_readonly_paths"))
    materialized_assignment_count = sum(
        1 for assignment in session_assignments if bool(_extract_value(assignment, "materialized"))
    )
    merge_risk_level = _coerce_optional_text(
        _extract_value(value, "merge_risk_level") or merge_analysis.get("risk_level")
    )
    member_changed_files = _normalize_path_map(merge_analysis.get("member_changed_files"))
    out_of_scope_writes = _normalize_path_map(merge_analysis.get("out_of_scope_writes"))
    readonly_violations = _normalize_path_map(merge_analysis.get("readonly_violations"))
    merge_order = (
        _extract_sequence(merge_orchestration, "recommended_merge_order")
        or _extract_sequence(merge_analysis, "recommended_merge_order")
        or _extract_sequence(plan, "merge_order")
    )
    cleanup_removed = _extract_sequence(cleanup, "removed")
    cleanup_errors = _extract_sequence(cleanup, "errors")
    cleanup_skipped = _extract_sequence(cleanup, "skipped")
    changed_file_count = sum(len(paths) for paths in member_changed_files.values())
    members_with_changes = sum(1 for paths in member_changed_files.values() if paths)
    out_of_scope_count = sum(len(paths) for paths in out_of_scope_writes.values())
    readonly_violation_count = sum(len(paths) for paths in readonly_violations.values())
    worker_contract_count = len(worker_return_contracts)
    worker_validation_count = sum(
        1
        for contract in worker_return_contracts.values()
        if _extract_mapping(contract, "validation_run")
    )
    worker_high_risk_count = sum(
        1
        for contract in worker_return_contracts.values()
        if _coerce_optional_text(_extract_mapping(contract, "merge_risk").get("level")) == "high"
    )
    worker_medium_risk_count = sum(
        1
        for contract in worker_return_contracts.values()
        if _coerce_optional_text(_extract_mapping(contract, "merge_risk").get("level")) == "medium"
    )
    review_required_members = _extract_sequence(merge_review_contract, "review_required_members")
    blocking_issues = _extract_sequence(merge_review_contract, "blocking_issues")
    merge_next_action = _coerce_optional_text(_extract_value(merge_review_contract, "next_action"))
    delegate_follow_up_next_action = _coerce_optional_text(
        _extract_value(delegate_follow_up_contract, "next_action")
    )
    if delegate_follow_up_next_action is None:
        delegate_follow_up_next_action = merge_next_action
    fix_validation_queue = _extract_sequence(delegate_follow_up_contract, "fix_validation_queue")
    review_queue = _extract_sequence(delegate_follow_up_contract, "review_queue")
    delegate_reentry_contract = _extract_mapping(delegate_follow_up_contract, "reentry_contract")
    delegate_merge_contract = _extract_mapping(
        delegate_follow_up_contract, "merge_execution_contract"
    )
    delegate_approval_contract = _extract_mapping(delegate_follow_up_contract, "approval_contract")
    delegate_reentry_next_action = _coerce_optional_text(
        _extract_value(delegate_reentry_contract, "next_action")
    )
    if delegate_reentry_next_action is None:
        delegate_reentry_next_action = delegate_follow_up_next_action
    delegate_reentry_member_ids = _extract_sequence(delegate_reentry_contract, "retry_member_ids")
    delegate_reentry_resume_worktree_paths = _normalize_path_map(
        delegate_reentry_contract.get("resume_worktree_paths")
    )
    if merge_next_action is None:
        if bool(merge_review_contract.get("merge_ready", False)):
            merge_next_action = "merge"
        elif _extract_sequence(merge_review_contract, "validation_failed_members"):
            merge_next_action = "fix_validation"
        elif review_required_members or blocking_issues:
            merge_next_action = "review"
        else:
            merge_next_action = "inspect"
    if not delegate_approval_contract and delegate_follow_up_contract:
        delegate_approval_contract = _derive_delegate_approval_contract(
            next_action=delegate_follow_up_next_action,
            merge_next_action=merge_next_action,
            delegate_follow_up_contract=delegate_follow_up_contract,
            delegate_reentry_contract=delegate_reentry_contract,
            delegate_merge_contract=delegate_merge_contract,
            fix_validation_queue=fix_validation_queue,
            review_queue=review_queue,
            merge_order=list(merge_order),
        )
    elif delegate_approval_contract:
        delegate_approval_contract = _enrich_delegate_approval_contract(
            delegate_approval_contract,
            delegate_reentry_contract=delegate_reentry_contract,
            delegate_merge_contract=delegate_merge_contract,
        )
    delegate_approval_required = bool(delegate_approval_contract.get("required", False))
    delegate_approval_reason = _coerce_optional_text(delegate_approval_contract.get("reason"))
    delegate_approval_action = _coerce_optional_text(
        delegate_approval_contract.get("recommended_action")
    )
    delegate_approval_target_ids = _extract_sequence(
        delegate_approval_contract, "target_member_ids"
    )
    delegate_approval_resume_ready = bool(delegate_approval_contract.get("resume_ready", False))
    delegate_auto_retry_eligible = bool(
        delegate_approval_contract.get("auto_retry_eligible", False)
    )
    delegate_approval_resume_context = _extract_mapping(
        delegate_approval_contract, "resume_context"
    )
    delegate_approval_task_briefs = _extract_mapping(
        delegate_approval_contract, "task_briefs_by_member"
    )
    delegate_approval_next_steps = _extract_sequence(delegate_approval_contract, "next_steps")
    delegate_approval_executable_steps = [
        step for step in delegate_approval_next_steps if _extract_mapping(step, "execution_context")
    ]
    delegate_approval_primary_step = (
        _coerce_optional_text(_extract_value(delegate_approval_next_steps[0], "step"))
        if delegate_approval_next_steps
        else None
    )

    return {
        "has_worktree_plan": bool(plan),
        "has_worktree_session": bool(session),
        "has_merge_analysis": bool(merge_analysis),
        "has_merge_orchestration": bool(merge_orchestration),
        "has_worktree_cleanup": bool(cleanup),
        "has_worker_return_contracts": bool(worker_return_contracts),
        "has_merge_review_contract": bool(merge_review_contract),
        "has_delegate_follow_up_contract": bool(delegate_follow_up_contract),
        "has_delegate_reentry_contract": bool(delegate_reentry_contract),
        "has_delegate_approval_contract": bool(delegate_approval_contract),
        "has_workspace_isolation_policy": bool(workspace_policy),
        "has_workspace_isolation_diagnostics": bool(workspace_diagnostics),
        "team_name": _coerce_optional_text(plan.get("team_name")),
        "formation": _coerce_optional_text(plan.get("formation")),
        "workspace_policy_mode": _coerce_optional_text(workspace_policy.get("mode")),
        "workspace_policy_worktree_isolation": bool(
            workspace_policy.get("worktree_isolation", False)
        ),
        "workspace_policy_materialize_worktrees": bool(
            workspace_policy.get("materialize_worktrees", False)
        ),
        "workspace_policy_dry_run_worktrees": bool(
            workspace_policy.get("dry_run_worktrees", False)
        ),
        "workspace_policy_auto_merge_worktrees": bool(
            workspace_policy.get("auto_merge_worktrees", False)
        ),
        "workspace_policy_cleanup_worktrees": workspace_policy.get("cleanup_worktrees"),
        "workspace_diagnostic_count": len(workspace_diagnostics),
        "workspace_diagnostic_reasons": dict(workspace_diagnostic_reasons),
        "workspace_diagnostic_operations": dict(workspace_diagnostic_operations),
        "assignment_count": assignment_count,
        "scoped_member_count": scoped_member_count,
        "readonly_shared_path_count": readonly_shared_path_count,
        "materialized": bool(session.get("materialized", False)),
        "dry_run": bool(session.get("dry_run", False)),
        "materialized_assignment_count": materialized_assignment_count,
        "merge_risk_level": merge_risk_level,
        "merge_conflict_count": _coerce_int(merge_analysis.get("conflict_count")) or 0,
        "merge_overlap_count": len(_extract_sequence(merge_analysis, "overlapping_files")),
        "out_of_scope_write_count": out_of_scope_count,
        "readonly_violation_count": readonly_violation_count,
        "members_with_changes": members_with_changes,
        "changed_file_count": changed_file_count,
        "cleanup_removed_count": len(cleanup_removed),
        "cleanup_error_count": len(cleanup_errors),
        "cleanup_skipped_count": len(cleanup_skipped),
        "merge_order_length": len(merge_order),
        "merge_order": list(merge_order),
        "worker_contract_count": worker_contract_count,
        "worker_validation_count": worker_validation_count,
        "worker_high_risk_count": worker_high_risk_count,
        "worker_medium_risk_count": worker_medium_risk_count,
        "merge_ready": bool(merge_review_contract.get("merge_ready", False)),
        "review_required": bool(merge_review_contract.get("review_required", False)),
        "merge_next_action": merge_next_action,
        "delegate_follow_up_next_action": delegate_follow_up_next_action,
        "delegate_follow_up_preserve_worktrees": bool(
            delegate_follow_up_contract.get("preserve_worktrees", False)
        ),
        "delegate_approval_required": delegate_approval_required,
        "delegate_approval_reason": delegate_approval_reason,
        "delegate_approval_action": delegate_approval_action,
        "delegate_auto_retry_eligible": delegate_auto_retry_eligible,
        "delegate_resume_ready": delegate_approval_resume_ready,
        "delegate_approval_target_count": len(delegate_approval_target_ids),
        "delegate_approval_has_resume_context": bool(delegate_approval_resume_context),
        "delegate_approval_has_execution_context": bool(delegate_approval_executable_steps),
        "delegate_approval_task_brief_count": len(delegate_approval_task_briefs),
        "delegate_approval_step_count": len(delegate_approval_next_steps),
        "delegate_approval_executable_step_count": len(delegate_approval_executable_steps),
        "delegate_approval_primary_step": delegate_approval_primary_step,
        "delegate_reentry_next_action": delegate_reentry_next_action,
        "delegate_reentry_member_count": len(delegate_reentry_member_ids),
        "delegate_reentry_resume_worktree_count": len(delegate_reentry_resume_worktree_paths),
        "fix_validation_queue_count": len(fix_validation_queue),
        "review_queue_count": len(review_queue),
        "review_required_member_count": len(review_required_members),
        "merge_blocker_count": len(blocking_issues),
        "member_changed_files": {
            member_id: list(paths) for member_id, paths in member_changed_files.items()
        },
        "out_of_scope_writes": {
            member_id: list(paths) for member_id, paths in out_of_scope_writes.items()
        },
        "readonly_violations": {
            member_id: list(paths) for member_id, paths in readonly_violations.items()
        },
    }


def aggregate_team_feedback(
    values: Iterable[Any],
    *,
    total_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate team/worktree summaries across benchmark task results."""
    summaries = [summary for value in values if (summary := summarize_team_feedback(value))]
    if not summaries:
        return {
            "tasks_with_team_feedback": 0,
            "team_feedback_coverage": 0.0,
            "team_formations": {},
            "team_merge_risk_levels": {},
            "team_worktree_plan_count": 0,
            "team_worktree_materialized_count": 0,
            "team_worktree_dry_run_count": 0,
            "team_workspace_policy_task_count": 0,
            "team_workspace_policy_modes": {},
            "team_workspace_policy_materialize_count": 0,
            "team_workspace_policy_dry_run_count": 0,
            "team_workspace_policy_auto_merge_count": 0,
            "team_workspace_policy_cleanup_disabled_count": 0,
            "team_workspace_diagnostic_task_count": 0,
            "team_workspace_diagnostic_count": 0,
            "team_workspace_diagnostic_reasons": {},
            "team_workspace_diagnostic_operations": {},
            "team_low_risk_task_count": 0,
            "team_medium_risk_task_count": 0,
            "team_high_risk_task_count": 0,
            "team_merge_conflict_task_count": 0,
            "team_merge_conflict_count": 0,
            "team_merge_overlap_task_count": 0,
            "team_out_of_scope_write_task_count": 0,
            "team_out_of_scope_write_count": 0,
            "team_readonly_violation_task_count": 0,
            "team_readonly_violation_count": 0,
            "team_cleanup_task_count": 0,
            "team_cleanup_error_task_count": 0,
            "team_cleanup_error_count": 0,
            "avg_team_assignments": 0.0,
            "avg_team_scoped_members": 0.0,
            "avg_team_members_with_changes": 0.0,
            "avg_team_changed_file_count": 0.0,
            "team_materialized_assignment_total": 0,
            "team_worker_contract_task_count": 0,
            "team_worker_contract_count": 0,
            "team_worker_validation_count": 0,
            "team_worker_high_risk_count": 0,
            "team_worker_medium_risk_count": 0,
            "team_merge_review_contract_task_count": 0,
            "team_merge_ready_task_count": 0,
            "team_merge_ready_rate": 0.0,
            "team_merge_next_actions": {},
            "team_delegate_follow_up_task_count": 0,
            "team_delegate_follow_up_actions": {},
            "team_delegate_approval_task_count": 0,
            "team_delegate_approval_required_task_count": 0,
            "team_delegate_auto_retry_eligible_task_count": 0,
            "team_delegate_resume_context_task_count": 0,
            "team_delegate_execution_context_task_count": 0,
            "team_delegate_approval_actions": {},
            "team_delegate_approval_reasons": {},
            "team_delegate_approval_primary_steps": {},
            "team_delegate_reentry_task_count": 0,
            "team_delegate_reentry_actions": {},
            "team_preserved_worktree_task_count": 0,
            "team_review_required_task_count": 0,
            "team_review_required_rate": 0.0,
            "team_merge_blocker_count": 0,
            "avg_team_materialized_assignments": 0.0,
            "avg_worker_validations_per_task": 0.0,
            "avg_team_merge_blockers": 0.0,
            "avg_fix_validation_queue_length": 0.0,
            "avg_review_queue_length": 0.0,
            "avg_delegate_approval_target_count": 0.0,
            "avg_delegate_approval_task_brief_count": 0.0,
            "avg_delegate_approval_step_count": 0.0,
            "avg_delegate_approval_executable_step_count": 0.0,
            "avg_delegate_reentry_member_count": 0.0,
            "avg_delegate_reentry_resume_worktree_count": 0.0,
            "avg_changed_files_per_materialized_assignment": 0.0,
        }

    task_count = total_tasks if total_tasks is not None else len(summaries)
    formations = Counter(summary["formation"] for summary in summaries if summary.get("formation"))
    risk_levels = Counter(
        summary["merge_risk_level"] for summary in summaries if summary.get("merge_risk_level")
    )
    next_actions = Counter(
        summary["merge_next_action"] for summary in summaries if summary.get("merge_next_action")
    )
    policy_modes = Counter(
        summary["workspace_policy_mode"]
        for summary in summaries
        if summary.get("workspace_policy_mode")
    )
    diagnostic_reasons: Counter[str] = Counter()
    diagnostic_operations: Counter[str] = Counter()
    for summary in summaries:
        diagnostic_reasons.update(dict(summary.get("workspace_diagnostic_reasons") or {}))
        diagnostic_operations.update(dict(summary.get("workspace_diagnostic_operations") or {}))
    follow_up_actions = Counter(
        summary["delegate_follow_up_next_action"]
        for summary in summaries
        if summary.get("delegate_follow_up_next_action")
    )
    approval_actions = Counter(
        summary["delegate_approval_action"]
        for summary in summaries
        if summary.get("delegate_approval_action")
    )
    approval_reasons = Counter(
        summary["delegate_approval_reason"]
        for summary in summaries
        if summary.get("delegate_approval_reason")
    )
    approval_primary_steps = Counter(
        summary["delegate_approval_primary_step"]
        for summary in summaries
        if summary.get("delegate_approval_primary_step")
    )
    reentry_actions = Counter(
        summary["delegate_reentry_next_action"]
        for summary in summaries
        if summary.get("has_delegate_reentry_contract")
        and summary.get("delegate_reentry_next_action")
    )
    plan_count = sum(1 for summary in summaries if summary.get("has_worktree_plan"))
    materialized_count = sum(1 for summary in summaries if summary.get("materialized"))
    dry_run_count = sum(1 for summary in summaries if summary.get("dry_run"))
    worker_contract_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_worker_return_contracts"))
    )
    merge_review_contract_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_merge_review_contract"))
    )
    delegate_follow_up_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_delegate_follow_up_contract"))
    )
    delegate_approval_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_delegate_approval_contract"))
    )
    delegate_reentry_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_delegate_reentry_contract"))
    )
    merge_conflict_task_count = sum(
        1 for summary in summaries if int(summary.get("merge_conflict_count", 0) or 0) > 0
    )
    merge_overlap_task_count = sum(
        1 for summary in summaries if int(summary.get("merge_overlap_count", 0) or 0) > 0
    )
    out_of_scope_task_count = sum(
        1 for summary in summaries if int(summary.get("out_of_scope_write_count", 0) or 0) > 0
    )
    readonly_violation_task_count = sum(
        1 for summary in summaries if int(summary.get("readonly_violation_count", 0) or 0) > 0
    )
    cleanup_task_count = sum(1 for summary in summaries if summary.get("has_worktree_cleanup"))
    cleanup_error_task_count = sum(
        1 for summary in summaries if int(summary.get("cleanup_error_count", 0) or 0) > 0
    )
    preserved_worktree_task_count = sum(
        1 for summary in summaries if bool(summary.get("delegate_follow_up_preserve_worktrees"))
    )
    summary_count = max(1, len(summaries))

    return {
        "tasks_with_team_feedback": len(summaries),
        "team_feedback_coverage": round(len(summaries) / max(1, task_count), 4),
        "team_formations": dict(formations),
        "team_merge_risk_levels": dict(risk_levels),
        "team_worktree_plan_count": plan_count,
        "team_worktree_materialized_count": materialized_count,
        "team_worktree_dry_run_count": dry_run_count,
        "team_workspace_policy_task_count": sum(
            1 for summary in summaries if bool(summary.get("has_workspace_isolation_policy"))
        ),
        "team_workspace_policy_modes": dict(policy_modes),
        "team_workspace_policy_materialize_count": sum(
            1
            for summary in summaries
            if bool(summary.get("workspace_policy_materialize_worktrees"))
        ),
        "team_workspace_policy_dry_run_count": sum(
            1 for summary in summaries if bool(summary.get("workspace_policy_dry_run_worktrees"))
        ),
        "team_workspace_policy_auto_merge_count": sum(
            1 for summary in summaries if bool(summary.get("workspace_policy_auto_merge_worktrees"))
        ),
        "team_workspace_policy_cleanup_disabled_count": sum(
            1 for summary in summaries if summary.get("workspace_policy_cleanup_worktrees") is False
        ),
        "team_workspace_diagnostic_task_count": sum(
            1 for summary in summaries if bool(summary.get("has_workspace_isolation_diagnostics"))
        ),
        "team_workspace_diagnostic_count": sum(
            int(summary.get("workspace_diagnostic_count", 0) or 0) for summary in summaries
        ),
        "team_workspace_diagnostic_reasons": dict(diagnostic_reasons),
        "team_workspace_diagnostic_operations": dict(diagnostic_operations),
        "team_low_risk_task_count": int(risk_levels.get("low", 0) or 0),
        "team_medium_risk_task_count": int(risk_levels.get("medium", 0) or 0),
        "team_high_risk_task_count": int(risk_levels.get("high", 0) or 0),
        "team_merge_conflict_task_count": merge_conflict_task_count,
        "team_merge_conflict_count": sum(
            int(summary.get("merge_conflict_count", 0) or 0) for summary in summaries
        ),
        "team_merge_overlap_task_count": merge_overlap_task_count,
        "team_out_of_scope_write_task_count": out_of_scope_task_count,
        "team_out_of_scope_write_count": sum(
            int(summary.get("out_of_scope_write_count", 0) or 0) for summary in summaries
        ),
        "team_readonly_violation_task_count": readonly_violation_task_count,
        "team_readonly_violation_count": sum(
            int(summary.get("readonly_violation_count", 0) or 0) for summary in summaries
        ),
        "team_cleanup_task_count": cleanup_task_count,
        "team_cleanup_error_task_count": cleanup_error_task_count,
        "team_cleanup_error_count": sum(
            int(summary.get("cleanup_error_count", 0) or 0) for summary in summaries
        ),
        "avg_team_assignments": round(
            sum(int(summary.get("assignment_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_scoped_members": round(
            sum(int(summary.get("scoped_member_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_members_with_changes": round(
            sum(int(summary.get("members_with_changes", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_changed_file_count": round(
            sum(int(summary.get("changed_file_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "team_materialized_assignment_total": sum(
            int(summary.get("materialized_assignment_count", 0) or 0) for summary in summaries
        ),
        "team_worker_contract_task_count": worker_contract_task_count,
        "team_worker_contract_count": sum(
            int(summary.get("worker_contract_count", 0) or 0) for summary in summaries
        ),
        "team_worker_validation_count": sum(
            int(summary.get("worker_validation_count", 0) or 0) for summary in summaries
        ),
        "team_worker_high_risk_count": sum(
            int(summary.get("worker_high_risk_count", 0) or 0) for summary in summaries
        ),
        "team_worker_medium_risk_count": sum(
            int(summary.get("worker_medium_risk_count", 0) or 0) for summary in summaries
        ),
        "team_merge_review_contract_task_count": merge_review_contract_task_count,
        "team_merge_ready_task_count": sum(
            1 for summary in summaries if bool(summary.get("merge_ready"))
        ),
        "team_merge_ready_rate": round(
            sum(1 for summary in summaries if bool(summary.get("merge_ready")))
            / max(1, merge_review_contract_task_count),
            4,
        ),
        "team_merge_next_actions": dict(next_actions),
        "team_delegate_follow_up_task_count": delegate_follow_up_task_count,
        "team_delegate_follow_up_actions": dict(follow_up_actions),
        "team_delegate_approval_task_count": delegate_approval_task_count,
        "team_delegate_approval_required_task_count": sum(
            1 for summary in summaries if bool(summary.get("delegate_approval_required"))
        ),
        "team_delegate_auto_retry_eligible_task_count": sum(
            1 for summary in summaries if bool(summary.get("delegate_auto_retry_eligible"))
        ),
        "team_delegate_resume_context_task_count": sum(
            1 for summary in summaries if bool(summary.get("delegate_approval_has_resume_context"))
        ),
        "team_delegate_execution_context_task_count": sum(
            1
            for summary in summaries
            if bool(summary.get("delegate_approval_has_execution_context"))
        ),
        "team_delegate_approval_actions": dict(approval_actions),
        "team_delegate_approval_reasons": dict(approval_reasons),
        "team_delegate_approval_primary_steps": dict(approval_primary_steps),
        "team_delegate_reentry_task_count": delegate_reentry_task_count,
        "team_delegate_reentry_actions": dict(reentry_actions),
        "team_preserved_worktree_task_count": preserved_worktree_task_count,
        "team_review_required_task_count": sum(
            1 for summary in summaries if bool(summary.get("review_required"))
        ),
        "team_review_required_rate": round(
            sum(1 for summary in summaries if bool(summary.get("review_required")))
            / max(1, merge_review_contract_task_count),
            4,
        ),
        "team_merge_blocker_count": sum(
            int(summary.get("merge_blocker_count", 0) or 0) for summary in summaries
        ),
        "avg_team_materialized_assignments": round(
            sum(int(summary.get("materialized_assignment_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_worker_validations_per_task": round(
            sum(int(summary.get("worker_validation_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_merge_blockers": round(
            sum(int(summary.get("merge_blocker_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_fix_validation_queue_length": round(
            sum(int(summary.get("fix_validation_queue_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_review_queue_length": round(
            sum(int(summary.get("review_queue_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_approval_target_count": round(
            sum(int(summary.get("delegate_approval_target_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_approval_task_brief_count": round(
            sum(
                int(summary.get("delegate_approval_task_brief_count", 0) or 0)
                for summary in summaries
            )
            / summary_count,
            4,
        ),
        "avg_delegate_approval_step_count": round(
            sum(int(summary.get("delegate_approval_step_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_approval_executable_step_count": round(
            sum(
                int(summary.get("delegate_approval_executable_step_count", 0) or 0)
                for summary in summaries
            )
            / summary_count,
            4,
        ),
        "avg_delegate_reentry_member_count": round(
            sum(int(summary.get("delegate_reentry_member_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_reentry_resume_worktree_count": round(
            sum(
                int(summary.get("delegate_reentry_resume_worktree_count", 0) or 0)
                for summary in summaries
            )
            / summary_count,
            4,
        ),
        "avg_changed_files_per_materialized_assignment": round(
            (
                sum(int(summary.get("changed_file_count", 0) or 0) for summary in summaries)
                / max(
                    1,
                    sum(
                        int(summary.get("materialized_assignment_count", 0) or 0)
                        for summary in summaries
                    ),
                )
            ),
            4,
        ),
    }


__all__ = [
    "aggregate_team_feedback",
    "extract_team_feedback_artifacts",
    "summarize_team_feedback",
]

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

"""Pure builders for team delegate contracts and worktree plans.

Extracted from :class:`victor.teams.unified_coordinator.UnifiedTeamCoordinator`
(co-design backlog item 30, ``DelegateContractBuilder``): functions over member
results, contract payloads and worktree plans with no coordinator state, so they
can be reasoned about and tested on their own. The coordinator calls them.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from victor.teams.types import MemberResult, TeamFormation
from victor.teams.worktree_runtime import (
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
    WorktreeExecutionPlan,
    WorktreeMaterializationSession,
)

if TYPE_CHECKING:
    from victor.protocols.team import ITeamMember


def resolve_context_mode(context: Dict[str, Any]) -> Optional[str]:
    for key in ("mode", "current_mode", "active_mode"):
        raw_value = context.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip().lower()
        if value:
            return value
    return None


def coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_path_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    normalized: list[str] = []
    for item in list(value or []):
        text = coerce_optional_text(item)
        if text is None:
            continue
        normalized.append(text.rstrip("/"))
    return list(dict.fromkeys(normalized))


def normalize_path_map(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, List[str]] = {}
    for member_id, paths in value.items():
        key = coerce_optional_text(member_id)
        if key is None:
            continue
        normalized[key] = normalize_path_list(paths)
    return normalized


def normalize_member_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    normalized: list[str] = []
    for item in list(value or []):
        text = coerce_optional_text(item)
        if text is None:
            continue
        normalized.append(text)
    return list(dict.fromkeys(normalized))


def normalize_workspace_diagnostics(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = [value]
    normalized: list[Dict[str, Any]] = []
    for item in list(value or []):
        if not isinstance(item, Mapping):
            continue
        diagnostic = dict(item)
        reason = (
            coerce_optional_text(diagnostic.get("reason"))
            or coerce_optional_text(diagnostic.get("blocked_reason"))
            or coerce_optional_text(diagnostic.get("type"))
            or "workspace_isolation"
        )
        message = (
            coerce_optional_text(diagnostic.get("message"))
            or coerce_optional_text(diagnostic.get("error"))
            or reason
        )
        operation = coerce_optional_text(diagnostic.get("operation")) or "workspace_isolation"
        severity = coerce_optional_text(diagnostic.get("severity")) or "warning"
        details = diagnostic.get("details")
        diagnostic["operation"] = operation
        diagnostic["reason"] = reason
        diagnostic["message"] = message
        diagnostic["severity"] = severity
        diagnostic["details"] = dict(details) if isinstance(details, Mapping) else {}
        normalized.append(diagnostic)
    return normalized


def extract_delegate_reentry_contract(context: Mapping[str, Any]) -> Dict[str, Any]:
    raw_value = context.get("delegate_reentry_contract")
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def extract_delegate_merge_contract(context: Mapping[str, Any]) -> Dict[str, Any]:
    raw_value = context.get("delegate_merge_contract")
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def extract_delegate_follow_up_contract(context: Mapping[str, Any]) -> Dict[str, Any]:
    raw_value = context.get("delegate_follow_up_contract")
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def extract_delegate_approval_contract(context: Mapping[str, Any]) -> Dict[str, Any]:
    raw_value = context.get("delegate_approval_contract")
    if isinstance(raw_value, Mapping):
        return dict(raw_value)
    follow_up_contract = extract_delegate_follow_up_contract(context)
    nested = follow_up_contract.get("approval_contract")
    return dict(nested) if isinstance(nested, Mapping) else {}


def extract_delegate_follow_up_next_steps(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    follow_up_contract = extract_delegate_follow_up_contract(context)
    top_level_steps = normalize_delegate_next_steps(follow_up_contract.get("next_steps"))
    if top_level_steps:
        return top_level_steps
    approval_contract = extract_delegate_approval_contract(context)
    return normalize_delegate_next_steps(approval_contract.get("next_steps"))


def normalize_delegate_next_steps(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    normalized: List[Dict[str, Any]] = []
    seen_step_ids: Dict[str, int] = {}
    for index, item in enumerate(list(value), start=1):
        if not isinstance(item, Mapping):
            continue
        payload = dict(item)
        base_step_id = (
            coerce_optional_text(payload.get("step_id"))
            or coerce_optional_text(payload.get("step"))
            or f"delegate_step_{index}"
        )
        occurrence = seen_step_ids.get(base_step_id, 0) + 1
        seen_step_ids[base_step_id] = occurrence
        payload["step_id"] = base_step_id if occurrence == 1 else f"{base_step_id}_{occurrence}"
        normalized.append(payload)
    return normalized


def build_delegate_next_step_request(
    *,
    step_id: str,
    next_steps: Sequence[Mapping[str, Any]],
    primary_step_id: Optional[str],
) -> Dict[str, Any]:
    request_payload: Dict[str, Any] = {
        "mode": "delegate",
        "delegate_next_step_id": step_id,
        "delegate_follow_up_contract": {
            "next_steps": copy.deepcopy(list(next_steps)),
        },
    }
    if primary_step_id is not None:
        request_payload["delegate_follow_up_contract"]["primary_step_id"] = primary_step_id
    return request_payload


def build_delegate_follow_up_step_requests(
    *,
    next_steps: Sequence[Mapping[str, Any]],
    primary_step_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    normalized_steps = normalize_delegate_next_steps(next_steps)
    step_requests: Dict[str, Dict[str, Any]] = {}
    for step in normalized_steps:
        step_id = coerce_optional_text(step.get("step_id"))
        if step_id is None:
            continue
        step_requests[step_id] = build_delegate_next_step_request(
            step_id=step_id,
            next_steps=normalized_steps,
            primary_step_id=primary_step_id,
        )
    return step_requests


def build_delegate_next_step_execution_request(
    *,
    step: Mapping[str, Any],
    step_request: Mapping[str, Any],
) -> Dict[str, Any]:
    task = (
        coerce_optional_text(step.get("instruction"))
        or coerce_optional_text(step.get("step"))
        or "Continue delegate follow-up"
    )
    return {
        "task": task,
        "context": copy.deepcopy(dict(step_request)),
    }


def build_delegate_follow_up_step_execution_requests(
    *,
    next_steps: Sequence[Mapping[str, Any]],
    step_requests: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    normalized_steps = normalize_delegate_next_steps(next_steps)
    step_execution_requests: Dict[str, Dict[str, Any]] = {}
    for step in normalized_steps:
        step_id = coerce_optional_text(step.get("step_id"))
        if step_id is None:
            continue
        step_request = step_requests.get(step_id)
        if not isinstance(step_request, Mapping):
            continue
        step_execution_requests[step_id] = build_delegate_next_step_execution_request(
            step=step,
            step_request=step_request,
        )
    return step_execution_requests


def build_delegate_follow_up_suggestions(
    next_steps: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build compact runtime suggestions from selectable delegate follow-up steps."""
    suggestions: List[Dict[str, Any]] = []
    for step in normalize_delegate_next_steps(next_steps):
        step_id = coerce_optional_text(step.get("step_id"))
        if step_id is None:
            continue
        summary = (
            coerce_optional_text(step.get("instruction"))
            or coerce_optional_text(step.get("description"))
            or coerce_optional_text(step.get("step"))
            or step_id
        )
        description = f"{step_id}: {summary}"
        if len(description) > 120:
            description = description[:117] + "..."
        escaped_step_id = step_id.replace("\\", "\\\\").replace("'", "\\'")
        suggestions.append(
            {
                "tool": "delegate_follow_up",
                "command": f"delegate_follow_up(step_id='{escaped_step_id}')",
                "description": description,
                "step_id": step_id,
                "requires_approval": bool(step.get("requires_approval", False)),
            }
        )
    return suggestions


def attach_delegate_follow_up_suggestions(
    result_dict: Dict[str, Any],
    delegate_follow_up_contract: Mapping[str, Any],
) -> None:
    """Mirror delegate follow-up suggestions onto result artifacts for renderers."""
    raw_suggestions = delegate_follow_up_contract.get("follow_up_suggestions")
    suggestions: List[Dict[str, Any]] = []
    if isinstance(raw_suggestions, Sequence) and not isinstance(raw_suggestions, (str, bytes)):
        suggestions = [
            dict(suggestion) for suggestion in raw_suggestions if isinstance(suggestion, Mapping)
        ]
    if not suggestions:
        suggestions = build_delegate_follow_up_suggestions(
            delegate_follow_up_contract.get("next_steps") or []
        )
    if not suggestions:
        return
    result_dict["delegate_follow_up_suggestions"] = suggestions
    result_dict["follow_up_suggestions"] = suggestions


def normalize_delegate_execution_request(
    request: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    if not isinstance(request, Mapping):
        raise TypeError("delegate follow-up request must be a mapping")
    task = coerce_optional_text(request.get("task"))
    if task is None:
        raise ValueError("delegate follow-up request must include a non-empty task")
    context = request.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("delegate follow-up request must include a mapping context")
    return task, copy.deepcopy(dict(context))


def resolve_delegate_execution_request_from_follow_up_contract(
    contract: Mapping[str, Any],
    *,
    step_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TypeError("delegate follow-up contract must be a mapping")

    follow_up_contract = dict(contract)
    normalized_step_id = coerce_optional_text(step_id) or coerce_optional_text(
        follow_up_contract.get("primary_step_id")
    )

    step_execution_requests = (
        dict(follow_up_contract.get("step_execution_requests") or {})
        if isinstance(follow_up_contract.get("step_execution_requests"), Mapping)
        else {}
    )
    if normalized_step_id is not None:
        candidate_request = step_execution_requests.get(normalized_step_id)
        if isinstance(candidate_request, Mapping):
            return copy.deepcopy(dict(candidate_request))
    else:
        primary_request = follow_up_contract.get("primary_step_execution_request")
        if isinstance(primary_request, Mapping):
            return copy.deepcopy(dict(primary_request))

    step_requests = (
        dict(follow_up_contract.get("step_requests") or {})
        if isinstance(follow_up_contract.get("step_requests"), Mapping)
        else {}
    )
    next_steps = normalize_delegate_next_steps(follow_up_contract.get("next_steps"))
    if normalized_step_id is None and next_steps:
        normalized_step_id = coerce_optional_text(next_steps[0].get("step_id"))
    if normalized_step_id is None:
        raise ValueError("delegate follow-up contract does not include a selectable step")

    candidate_step_request = step_requests.get(normalized_step_id)
    if not isinstance(candidate_step_request, Mapping):
        raise ValueError("delegate follow-up contract does not include a runnable step request")

    for step in next_steps:
        candidate_step_id = coerce_optional_text(step.get("step_id"))
        if candidate_step_id == normalized_step_id:
            return build_delegate_next_step_execution_request(
                step=step,
                step_request=candidate_step_request,
            )

    raise ValueError("delegate follow-up contract does not include the selected next step")


def resolve_delegate_next_step_by_id(
    context: Mapping[str, Any],
    *,
    step_id: str,
) -> Dict[str, Any]:
    next_steps = extract_delegate_follow_up_next_steps(context)
    normalized_step_id = coerce_optional_text(step_id)
    if normalized_step_id is None:
        return {}
    for step in next_steps:
        candidate_id = coerce_optional_text(step.get("step_id"))
        candidate_step = coerce_optional_text(step.get("step"))
        if normalized_step_id in {candidate_id, candidate_step}:
            return step
    return {}


def extract_delegate_next_step(context: Mapping[str, Any]) -> Dict[str, Any]:
    raw_value = context.get("delegate_next_step")
    if isinstance(raw_value, Mapping):
        return dict(raw_value)
    step_id = coerce_optional_text(raw_value)
    if step_id is None:
        step_id = coerce_optional_text(context.get("delegate_next_step_id"))
    if step_id is None:
        return {}
    return resolve_delegate_next_step_by_id(context, step_id=step_id)


def apply_delegate_reentry_context(
    context: Dict[str, Any],
    *,
    delegate_reentry_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    context_overrides = (
        dict(delegate_reentry_contract.get("context_overrides") or {})
        if isinstance(delegate_reentry_contract.get("context_overrides"), Mapping)
        else {}
    )
    effective_context = dict(context_overrides)
    effective_context.update(context)
    return effective_context


def apply_delegate_next_step_context(
    context: Dict[str, Any],
    *,
    delegate_next_step: Mapping[str, Any],
) -> Dict[str, Any]:
    if not delegate_next_step:
        return dict(context)
    selected_step = dict(context)
    step_name = coerce_optional_text(delegate_next_step.get("step"))
    instruction = coerce_optional_text(delegate_next_step.get("instruction"))
    if step_name is not None:
        selected_step.setdefault("delegate_selected_step", step_name)
    if instruction is not None:
        selected_step.setdefault("delegate_selected_instruction", instruction)

    resume_context = (
        dict(delegate_next_step.get("resume_context") or {})
        if isinstance(delegate_next_step.get("resume_context"), Mapping)
        else {}
    )
    execution_context = (
        dict(delegate_next_step.get("execution_context") or {})
        if isinstance(delegate_next_step.get("execution_context"), Mapping)
        else {}
    )
    effective_context = dict(resume_context)
    effective_context.update(execution_context)
    effective_context.update(selected_step)
    return effective_context


def apply_delegate_merge_context(
    context: Dict[str, Any],
    *,
    delegate_merge_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    context_overrides = (
        dict(delegate_merge_contract.get("context_overrides") or {})
        if isinstance(delegate_merge_contract.get("context_overrides"), Mapping)
        else {}
    )
    effective_context = dict(context_overrides)
    effective_context.update(context)
    return effective_context


def extract_delegate_reentry_member_ids(
    delegate_reentry_contract: Mapping[str, Any],
) -> List[str]:
    member_ids = normalize_member_id_list(delegate_reentry_contract.get("retry_member_ids"))
    if member_ids:
        return member_ids
    overrides = delegate_reentry_contract.get("resume_member_context_overrides")
    if isinstance(overrides, Mapping):
        return normalize_member_id_list(overrides.keys())
    return []


def extract_delegate_reentry_member_context_overrides(
    delegate_reentry_contract: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    overrides = delegate_reentry_contract.get("resume_member_context_overrides")
    if not isinstance(overrides, Mapping):
        return {}
    next_action = coerce_optional_text(delegate_reentry_contract.get("next_action"))
    retry_tasks_by_member = (
        dict(delegate_reentry_contract.get("retry_tasks_by_member") or {})
        if isinstance(delegate_reentry_contract.get("retry_tasks_by_member"), Mapping)
        else {}
    )
    normalized: Dict[str, Dict[str, Any]] = {}
    for member_id, payload in overrides.items():
        key = coerce_optional_text(member_id)
        if key is None or not isinstance(payload, Mapping):
            continue
        override = dict(payload)
        if next_action is not None:
            override["delegate_reentry_next_action"] = next_action
        follow_up_task_brief = coerce_optional_text(retry_tasks_by_member.get(key))
        if follow_up_task_brief is not None:
            override["follow_up_task_brief"] = follow_up_task_brief
        normalized[key] = override
    return normalized


def filter_execution_members(
    members: List["ITeamMember"],
    *,
    member_ids: List[str],
) -> List["ITeamMember"]:
    if not member_ids:
        return list(members)
    selected_ids = set(member_ids)
    filtered = [member for member in members if member.id in selected_ids]
    return filtered or list(members)


def summarize_member_output(output: str) -> Optional[str]:
    text = coerce_optional_text(output)
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= 280:
        return compact
    return compact[:277].rstrip() + "..."


def normalize_validation_run(metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw_validation = metadata.get("validation_run")
    payload = dict(raw_validation) if isinstance(raw_validation, Mapping) else {}
    status = coerce_optional_text(
        payload.get("status")
        or payload.get("result")
        or payload.get("outcome")
        or metadata.get("validation_status")
    )
    command = coerce_optional_text(
        payload.get("command")
        or payload.get("test_command")
        or metadata.get("validation_command")
        or metadata.get("test_command")
    )
    summary = coerce_optional_text(
        payload.get("summary")
        or payload.get("output")
        or payload.get("result_summary")
        or metadata.get("validation_summary")
    )
    normalized: Dict[str, Any] = {}
    if status is not None:
        normalized["status"] = status
    if command is not None:
        normalized["command"] = command
    if summary is not None:
        normalized["summary"] = summary
    return normalized or None


def build_worker_return_contracts(
    member_results: Dict[str, MemberResult],
    *,
    merge_analysis: Optional[Any],
) -> Dict[str, Dict[str, Any]]:
    if hasattr(merge_analysis, "to_dict"):
        merge_payload = merge_analysis.to_dict()
    elif isinstance(merge_analysis, Mapping):
        merge_payload = dict(merge_analysis)
    else:
        merge_payload = {}

    member_changed_files = normalize_path_map(merge_payload.get("member_changed_files"))
    out_of_scope_writes = normalize_path_map(merge_payload.get("out_of_scope_writes"))
    readonly_violations = normalize_path_map(merge_payload.get("readonly_violations"))
    overlapping_files_by_member: Dict[str, List[str]] = {}
    for conflict in list(merge_payload.get("overlapping_files") or []):
        if not isinstance(conflict, Mapping):
            continue
        path = coerce_optional_text(conflict.get("path"))
        if path is None:
            continue
        for member_id in list(conflict.get("members") or []):
            normalized_member_id = coerce_optional_text(member_id)
            if normalized_member_id is None:
                continue
            overlapping_files_by_member.setdefault(normalized_member_id, []).append(path)

    contracts: Dict[str, Dict[str, Any]] = {}
    for member_id, result in member_results.items():
        normalized_member_id = coerce_optional_text(member_id)
        if normalized_member_id is None:
            continue
        metadata = dict(result.metadata or {})
        changed_files = normalize_path_list(
            metadata.get("changed_files")
            or metadata.get("files_touched")
            or metadata.get("modified_files")
            or member_changed_files.get(normalized_member_id)
        )
        overlap_paths = normalize_path_list(overlapping_files_by_member.get(normalized_member_id))
        out_of_scope_paths = normalize_path_list(out_of_scope_writes.get(normalized_member_id))
        readonly_paths = normalize_path_list(readonly_violations.get(normalized_member_id))
        reasons: list[str] = []
        if overlap_paths:
            reasons.append("overlapping_files")
        if readonly_paths:
            reasons.append("readonly_violations")
        if out_of_scope_paths:
            reasons.append("out_of_scope_writes")
        risk_level = "low"
        if overlap_paths or readonly_paths:
            risk_level = "high"
        elif out_of_scope_paths:
            risk_level = "medium"

        task_summary = coerce_optional_text(
            metadata.get("task_summary")
            or metadata.get("summary")
            or metadata.get("result_summary")
        ) or summarize_member_output(result.output)

        contracts[normalized_member_id] = {
            "member_id": normalized_member_id,
            "success": bool(result.success),
            "task_summary": task_summary,
            "changed_files": changed_files,
            "validation_run": normalize_validation_run(metadata),
            "merge_risk": {
                "level": risk_level,
                "reasons": reasons,
                "overlapping_files": overlap_paths,
                "out_of_scope_writes": out_of_scope_paths,
                "readonly_violations": readonly_paths,
            },
        }

    return contracts


def build_delegate_follow_up_contract(
    worker_return_contracts: Mapping[str, Mapping[str, Any]],
    *,
    merge_review_contract: Mapping[str, Any],
    worktree_session: Optional[WorktreeMaterializationSession],
    merge_execution: Optional[Mapping[str, Any]] = None,
    merge_analysis: Optional[Mapping[str, Any]] = None,
    merge_orchestration: Optional[Mapping[str, Any]] = None,
    workspace_diagnostics: Optional[Sequence[Mapping[str, Any]]] = None,
    preserve_merge_follow_up: bool = False,
) -> Dict[str, Any]:
    next_action = coerce_optional_text(merge_review_contract.get("next_action")) or "inspect"
    workspace_diagnostics_payload = normalize_workspace_diagnostics(workspace_diagnostics)
    merge_execution_payload = (
        dict(merge_execution or {}) if isinstance(merge_execution, Mapping) else {}
    )
    merge_executed = bool(merge_execution_payload.get("executed", False))
    validation_failed_members = normalize_path_list(
        merge_review_contract.get("validation_failed_members")
    )
    review_required_members = normalize_path_list(
        merge_review_contract.get("review_required_members")
    )

    fix_validation_queue: list[Dict[str, Any]] = []
    for member_id in validation_failed_members:
        contract = (
            dict(worker_return_contracts.get(member_id) or {})
            if isinstance(worker_return_contracts.get(member_id), Mapping)
            else {}
        )
        validation_run = (
            dict(contract.get("validation_run") or {})
            if isinstance(contract.get("validation_run"), Mapping)
            else {}
        )
        fix_validation_queue.append(
            {
                "member_id": member_id,
                "validation_command": coerce_optional_text(validation_run.get("command")),
                "validation_summary": coerce_optional_text(validation_run.get("summary")),
                "changed_files": normalize_path_list(contract.get("changed_files")),
            }
        )

    review_queue: list[Dict[str, Any]] = []
    for member_id in review_required_members:
        contract = (
            dict(worker_return_contracts.get(member_id) or {})
            if isinstance(worker_return_contracts.get(member_id), Mapping)
            else {}
        )
        merge_risk = (
            dict(contract.get("merge_risk") or {})
            if isinstance(contract.get("merge_risk"), Mapping)
            else {}
        )
        review_queue.append(
            {
                "member_id": member_id,
                "merge_risk_level": coerce_optional_text(merge_risk.get("level")) or "low",
                "merge_risk_reasons": list(merge_risk.get("reasons") or []),
                "changed_files": normalize_path_list(contract.get("changed_files")),
                "task_summary": coerce_optional_text(contract.get("task_summary")),
            }
        )

    preserve_worktrees = bool(
        worktree_session is not None
        and (
            next_action in {"fix_validation", "review", "inspect"}
            or (preserve_merge_follow_up and next_action == "merge" and not merge_executed)
        )
    )
    preserved_worktree_paths: list[str] = []
    if preserve_worktrees and worktree_session is not None:
        assignments = getattr(worktree_session, "assignments", [])
        for assignment in list(assignments or []):
            path = coerce_optional_text(getattr(assignment, "worktree_path", None))
            if path is not None:
                preserved_worktree_paths.append(path)

    reentry_contract: Optional[Dict[str, Any]] = None
    merge_execution_contract: Optional[Dict[str, Any]] = None
    if preserve_worktrees and worktree_session is not None:
        if next_action == "merge" and not merge_executed:
            merge_execution_contract = build_delegate_merge_execution_contract(
                worktree_session=worktree_session,
                merge_analysis=merge_analysis,
                merge_orchestration=merge_orchestration,
                merge_review_contract=merge_review_contract,
                worker_return_contracts=worker_return_contracts,
            )
        else:
            retry_member_ids = resolve_delegate_reentry_member_ids(
                next_action=next_action,
                validation_failed_members=validation_failed_members,
                review_required_members=review_required_members,
            )
            retry_tasks_by_member = build_delegate_reentry_retry_tasks(
                next_action=next_action,
                fix_validation_queue=fix_validation_queue,
                review_queue=review_queue,
            )
            resume_member_context_overrides: Dict[str, Dict[str, Any]] = {}
            resume_worktree_paths: Dict[str, str] = {}
            for member_id in retry_member_ids:
                assignment = worktree_session.assignment_for(member_id)
                if assignment is None:
                    continue
                override = build_delegate_reentry_member_context_override(assignment)
                if not override:
                    continue
                resume_member_context_overrides[member_id] = override
                path = coerce_optional_text(override.get("worktree_path"))
                if path is not None:
                    resume_worktree_paths[member_id] = path
            if resume_member_context_overrides:
                reentry_contract = {
                    "mode": "delegate",
                    "next_action": next_action,
                    "retry_member_ids": retry_member_ids,
                    "resume_worktree_paths": resume_worktree_paths,
                    "retry_tasks_by_member": retry_tasks_by_member,
                    "resume_member_context_overrides": resume_member_context_overrides,
                    "context_overrides": {
                        "mode": "delegate",
                        "worktree_isolation": True,
                        "materialize_worktrees": False,
                        "cleanup_worktrees": False,
                    },
                }

    approval_contract = build_delegate_approval_contract(
        next_action=next_action,
        merge_review_contract=merge_review_contract,
        reentry_contract=reentry_contract,
        merge_execution_contract=merge_execution_contract,
        fix_validation_queue=fix_validation_queue,
        review_queue=review_queue,
        merge_execution=merge_execution,
    )
    if workspace_diagnostics_payload:
        approval_contract = dict(approval_contract)
        approval_contract["workspace_isolation_diagnostics"] = list(workspace_diagnostics_payload)
    follow_up_steps = normalize_delegate_next_steps(approval_contract.get("next_steps"))

    contract = {
        "next_action": next_action,
        "preserve_worktrees": preserve_worktrees,
        "fix_validation_queue": fix_validation_queue,
        "review_queue": review_queue,
        "review_required_members": review_required_members,
        "validation_failed_members": validation_failed_members,
        "preserved_worktree_paths": preserved_worktree_paths,
        "approval_contract": approval_contract,
    }
    if workspace_diagnostics_payload:
        contract["workspace_isolation_diagnostics"] = list(workspace_diagnostics_payload)
    if follow_up_steps:
        contract["next_steps"] = follow_up_steps
        primary_step_id = coerce_optional_text(follow_up_steps[0].get("step_id"))
        if primary_step_id is not None:
            contract["primary_step_id"] = primary_step_id
        step_requests = build_delegate_follow_up_step_requests(
            next_steps=follow_up_steps,
            primary_step_id=primary_step_id,
        )
        if step_requests:
            contract["step_requests"] = step_requests
            if primary_step_id is not None and primary_step_id in step_requests:
                contract["primary_step_request"] = copy.deepcopy(step_requests[primary_step_id])
            step_execution_requests = build_delegate_follow_up_step_execution_requests(
                next_steps=follow_up_steps,
                step_requests=step_requests,
            )
            if step_execution_requests:
                contract["step_execution_requests"] = step_execution_requests
                if primary_step_id is not None and primary_step_id in step_execution_requests:
                    contract["primary_step_execution_request"] = copy.deepcopy(
                        step_execution_requests[primary_step_id]
                    )
        follow_up_suggestions = build_delegate_follow_up_suggestions(follow_up_steps)
        if follow_up_suggestions:
            contract["follow_up_suggestions"] = follow_up_suggestions
    if reentry_contract is not None:
        contract["reentry_contract"] = reentry_contract
    if merge_execution_contract is not None:
        contract["merge_execution_contract"] = merge_execution_contract
    return contract


def build_delegate_approval_contract(
    *,
    next_action: str,
    merge_review_contract: Mapping[str, Any],
    reentry_contract: Optional[Mapping[str, Any]],
    merge_execution_contract: Optional[Mapping[str, Any]],
    fix_validation_queue: List[Dict[str, Any]],
    review_queue: List[Dict[str, Any]],
    merge_execution: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    merge_execution_payload = (
        dict(merge_execution or {}) if isinstance(merge_execution, Mapping) else {}
    )
    reentry_payload = dict(reentry_contract or {}) if isinstance(reentry_contract, Mapping) else {}
    retry_member_ids = normalize_member_id_list(reentry_payload.get("retry_member_ids"))
    recommended_merge_order = normalize_member_id_list(
        merge_review_contract.get("recommended_merge_order")
    )
    validation_target_ids = normalize_member_id_list(
        item.get("member_id") for item in fix_validation_queue if isinstance(item, Mapping)
    )
    review_target_ids = normalize_member_id_list(
        item.get("member_id") for item in review_queue if isinstance(item, Mapping)
    )
    merge_executed = bool(merge_execution_payload.get("executed"))
    if not merge_executed:
        merge_status = coerce_optional_text(merge_execution_payload.get("status"))
        if merge_status is not None:
            merge_executed = merge_status.lower() in {
                "success",
                "succeeded",
                "merged",
                "completed",
            }
    recommended_mode = coerce_optional_text(merge_review_contract.get("recommended_mode"))
    if recommended_mode is None:
        if merge_executed or bool(merge_review_contract.get("merge_execution_eligible")):
            recommended_mode = "auto_apply_safe"
        elif next_action in {"fix_validation", "review", "inspect"}:
            recommended_mode = "manual_review"

    if merge_executed:
        target_member_ids = recommended_merge_order or retry_member_ids
        return finalize_delegate_approval_contract(
            {
                "required": False,
                "reason": "merge_executed",
                "recommended_action": "merged",
                "recommended_mode": recommended_mode,
                "resume_ready": False,
                "auto_retry_eligible": False,
                "merge_executed": True,
                "target_member_ids": target_member_ids,
                "summary": build_delegate_approval_summary(
                    "Merge orchestration already executed for",
                    target_member_ids=target_member_ids,
                ),
            },
            reentry_payload=reentry_payload,
            merge_execution_contract=merge_execution_contract,
        )

    if next_action == "fix_validation":
        target_member_ids = retry_member_ids or validation_target_ids
        resume_ready = bool(reentry_payload) and bool(target_member_ids)
        return finalize_delegate_approval_contract(
            {
                "required": not resume_ready,
                "reason": "validation_failed",
                "recommended_action": "retry" if resume_ready else "approve_retry",
                "recommended_mode": recommended_mode,
                "resume_ready": resume_ready,
                "auto_retry_eligible": resume_ready,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
                "summary": build_delegate_approval_summary(
                    (
                        "Resume preserved worktrees to fix failing validation for"
                        if resume_ready
                        else "Approve a validation retry for"
                    ),
                    target_member_ids=target_member_ids,
                ),
            },
            reentry_payload=reentry_payload,
            merge_execution_contract=merge_execution_contract,
        )

    if next_action == "review":
        target_member_ids = retry_member_ids or review_target_ids
        return finalize_delegate_approval_contract(
            {
                "required": True,
                "reason": "review_required",
                "recommended_action": "review_then_retry",
                "recommended_mode": recommended_mode,
                "resume_ready": bool(reentry_payload),
                "auto_retry_eligible": False,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
                "summary": build_delegate_approval_summary(
                    "Review merge risks before retrying preserved worktrees for",
                    target_member_ids=target_member_ids,
                ),
            },
            reentry_payload=reentry_payload,
            merge_execution_contract=merge_execution_contract,
        )

    if next_action == "merge":
        target_member_ids = recommended_merge_order or retry_member_ids
        return finalize_delegate_approval_contract(
            {
                "required": True,
                "reason": "merge_ready",
                "recommended_action": "approve_merge",
                "recommended_mode": recommended_mode,
                "resume_ready": False,
                "auto_retry_eligible": False,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
                "summary": build_delegate_approval_summary(
                    "Review and approve merge execution for",
                    target_member_ids=target_member_ids,
                ),
            },
            reentry_payload=reentry_payload,
            merge_execution_contract=merge_execution_contract,
        )

    target_member_ids = retry_member_ids or list(
        dict.fromkeys([*validation_target_ids, *review_target_ids])
    )
    return finalize_delegate_approval_contract(
        {
            "required": True,
            "reason": "inspect_required",
            "recommended_action": "inspect_worktrees",
            "recommended_mode": recommended_mode,
            "resume_ready": bool(reentry_payload),
            "auto_retry_eligible": False,
            "merge_executed": False,
            "target_member_ids": target_member_ids,
            "summary": build_delegate_approval_summary(
                "Inspect preserved worktrees before retrying work for",
                target_member_ids=target_member_ids,
            ),
        },
        reentry_payload=reentry_payload,
        merge_execution_contract=merge_execution_contract,
    )


def finalize_delegate_approval_contract(
    contract: Mapping[str, Any],
    *,
    reentry_payload: Mapping[str, Any],
    merge_execution_contract: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    approval_contract = dict(contract)
    target_member_ids = normalize_member_id_list(approval_contract.get("target_member_ids"))
    resume_context = build_delegate_approval_resume_context(
        reentry_payload,
        target_member_ids=target_member_ids,
    )
    if resume_context is not None:
        approval_contract["resume_context"] = resume_context
    task_briefs = build_delegate_approval_task_briefs(
        reentry_payload,
        target_member_ids=target_member_ids,
    )
    if task_briefs:
        approval_contract["task_briefs_by_member"] = task_briefs
    if "next_steps" not in approval_contract:
        next_steps = build_delegate_approval_next_steps(
            approval_contract,
            target_member_ids=target_member_ids,
            resume_context=resume_context,
            task_briefs=task_briefs,
            merge_execution_contract=merge_execution_contract,
        )
        if next_steps:
            approval_contract["next_steps"] = next_steps
    normalized_next_steps = normalize_delegate_next_steps(approval_contract.get("next_steps"))
    if normalized_next_steps:
        approval_contract["next_steps"] = normalized_next_steps
    return approval_contract


def build_delegate_approval_resume_context(
    reentry_payload: Mapping[str, Any],
    *,
    target_member_ids: List[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(reentry_payload, Mapping) or not reentry_payload:
        return None
    normalized_payload = dict(reentry_payload)
    retry_member_ids = normalize_member_id_list(normalized_payload.get("retry_member_ids"))
    resume_overrides = normalized_payload.get("resume_member_context_overrides")
    resume_paths = normalized_payload.get("resume_worktree_paths")
    has_resume_details = bool(retry_member_ids)
    if isinstance(resume_overrides, Mapping):
        has_resume_details = has_resume_details or bool(resume_overrides)
    if isinstance(resume_paths, Mapping):
        has_resume_details = has_resume_details or bool(resume_paths)
    if not has_resume_details:
        return None
    if not retry_member_ids and target_member_ids:
        normalized_payload["retry_member_ids"] = list(target_member_ids)
    return {
        "mode": "delegate",
        "delegate_reentry_contract": normalized_payload,
    }


def build_delegate_approval_task_briefs(
    reentry_payload: Mapping[str, Any],
    *,
    target_member_ids: List[str],
) -> Dict[str, str]:
    if not isinstance(reentry_payload, Mapping):
        return {}
    raw_briefs = reentry_payload.get("retry_tasks_by_member")
    if not isinstance(raw_briefs, Mapping):
        return {}
    prioritized_ids = normalize_member_id_list(target_member_ids)
    if not prioritized_ids:
        prioritized_ids = normalize_member_id_list(raw_briefs.keys())
    task_briefs: Dict[str, str] = {}
    for member_id in prioritized_ids:
        task_brief = coerce_optional_text(raw_briefs.get(member_id))
        if task_brief is None:
            continue
        task_briefs[member_id] = task_brief
    return task_briefs


def build_delegate_approval_summary(
    prefix: str,
    *,
    target_member_ids: List[str],
) -> str:
    normalized_targets = normalize_member_id_list(target_member_ids)
    if normalized_targets:
        return f"{prefix}: {', '.join(normalized_targets)}."
    return f"{prefix} the delegate worktree set."


def build_delegate_approval_next_steps(
    approval_contract: Mapping[str, Any],
    *,
    target_member_ids: List[str],
    resume_context: Optional[Mapping[str, Any]],
    task_briefs: Mapping[str, str],
    merge_execution_contract: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    recommended_action = coerce_optional_text(approval_contract.get("recommended_action"))
    summary = coerce_optional_text(approval_contract.get("summary"))
    requires_approval = bool(approval_contract.get("required", False))
    normalized_targets = normalize_member_id_list(target_member_ids)

    def build_step(
        step: str,
        instruction: Optional[str],
        *,
        step_requires_approval: bool,
        include_resume: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step": step,
            "instruction": instruction
            or build_delegate_approval_summary(
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

    if recommended_action == "merged" or bool(approval_contract.get("merge_executed", False)):
        return [
            build_step(
                "status_merged",
                summary,
                step_requires_approval=False,
            )
        ]
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
        return [
            build_step(
                "approve_delegate_retry",
                summary,
                step_requires_approval=True,
            )
        ]
    if recommended_action == "review_then_retry":
        steps = [
            build_step(
                "review_worktrees",
                summary,
                step_requires_approval=True,
            )
        ]
        if resume_context:
            steps.append(
                build_step(
                    "resume_delegate_retry",
                    build_delegate_approval_summary(
                        "Resume preserved worktrees after review for",
                        target_member_ids=normalized_targets,
                    ),
                    step_requires_approval=requires_approval,
                    include_resume=True,
                )
            )
        return steps
    if recommended_action == "approve_merge":
        step = build_step(
            "approve_merge_execution",
            summary,
            step_requires_approval=True,
        )
        if merge_execution_contract:
            step["execution_context"] = {
                "mode": "delegate",
                "delegate_merge_contract": dict(merge_execution_contract),
            }
        return [step]
    if recommended_action == "inspect_worktrees":
        steps = [
            build_step(
                "inspect_worktrees",
                summary,
                step_requires_approval=True,
            )
        ]
        if resume_context:
            steps.append(
                build_step(
                    "resume_delegate_retry",
                    build_delegate_approval_summary(
                        "Resume preserved worktrees after inspection for",
                        target_member_ids=normalized_targets,
                    ),
                    step_requires_approval=requires_approval,
                    include_resume=True,
                )
            )
        return steps
    return []


def resolve_delegate_reentry_member_ids(
    *,
    next_action: str,
    validation_failed_members: List[str],
    review_required_members: List[str],
) -> List[str]:
    if next_action == "fix_validation":
        return list(validation_failed_members)
    if next_action == "review":
        return list(review_required_members)
    combined = list(dict.fromkeys([*validation_failed_members, *review_required_members]))
    return combined


def build_delegate_reentry_retry_tasks(
    *,
    next_action: str,
    fix_validation_queue: List[Dict[str, Any]],
    review_queue: List[Dict[str, Any]],
) -> Dict[str, str]:
    retry_tasks: Dict[str, str] = {}
    for item in fix_validation_queue:
        if not isinstance(item, Mapping):
            continue
        member_id = coerce_optional_text(item.get("member_id"))
        if member_id is None:
            continue
        command = coerce_optional_text(item.get("validation_command"))
        summary = coerce_optional_text(item.get("validation_summary"))
        changed_files = normalize_path_list(item.get("changed_files"))
        parts = [f"Fix the failing validation run for {member_id}."]
        if command is not None:
            parts.append(f"Re-run `{command}`.")
        if summary is not None:
            parts.append(f"Last result: {summary}.")
        if changed_files:
            parts.append(f"Focus on: {', '.join(changed_files)}.")
        retry_tasks[member_id] = " ".join(parts)

    for item in review_queue:
        if not isinstance(item, Mapping):
            continue
        member_id = coerce_optional_text(item.get("member_id"))
        if member_id is None or member_id in retry_tasks:
            continue
        risk_level = coerce_optional_text(item.get("merge_risk_level")) or "low"
        reasons = normalize_path_list(item.get("merge_risk_reasons"))
        changed_files = normalize_path_list(item.get("changed_files"))
        task_summary = coerce_optional_text(item.get("task_summary"))
        parts = [f"Review the pending merge risk for {member_id} ({risk_level})."]
        if reasons:
            parts.append(f"Address: {', '.join(reasons)}.")
        if changed_files:
            parts.append(f"Inspect: {', '.join(changed_files)}.")
        if task_summary is not None:
            parts.append(f"Prior output: {task_summary}.")
        retry_tasks[member_id] = " ".join(parts)

    if not retry_tasks and next_action == "inspect":
        for item in review_queue:
            if not isinstance(item, Mapping):
                continue
            member_id = coerce_optional_text(item.get("member_id"))
            if member_id is None:
                continue
            retry_tasks[member_id] = f"Inspect the preserved worktree state for {member_id}."
    return retry_tasks


def build_delegate_reentry_member_context_override(
    assignment: Any,
) -> Dict[str, Any]:
    override = {}
    to_context_overrides = getattr(assignment, "to_context_overrides", None)
    if callable(to_context_overrides):
        raw_override = to_context_overrides()
        if isinstance(raw_override, Mapping):
            override.update(dict(raw_override))

    worktree_path = coerce_optional_text(getattr(assignment, "worktree_path", None))
    branch_name = coerce_optional_text(getattr(assignment, "branch_name", None))
    if worktree_path is not None:
        override.setdefault("workspace_root", worktree_path)
        override["worktree_path"] = worktree_path
    if branch_name is not None:
        override["branch_name"] = branch_name

    assignment_payload: Dict[str, Any] = {}
    to_dict = getattr(assignment, "to_dict", None)
    if callable(to_dict):
        raw_payload = to_dict()
        if isinstance(raw_payload, Mapping):
            assignment_payload.update(dict(raw_payload))
    if not assignment_payload:
        member_id = coerce_optional_text(getattr(assignment, "member_id", None))
        if member_id is not None:
            assignment_payload["member_id"] = member_id
        if branch_name is not None:
            assignment_payload["branch_name"] = branch_name
        if worktree_path is not None:
            assignment_payload["worktree_path"] = worktree_path
    if assignment_payload:
        override["worktree_assignment"] = assignment_payload

    return override


def rebuild_worktree_assignment(
    payload: Mapping[str, Any],
) -> Optional[WorktreeAssignment]:
    member_id = coerce_optional_text(payload.get("member_id"))
    branch_name = coerce_optional_text(payload.get("branch_name"))
    worktree_name = coerce_optional_text(payload.get("worktree_name"))
    worktree_path = coerce_optional_text(payload.get("worktree_path"))
    if None in (member_id, branch_name, worktree_name, worktree_path):
        return None
    return WorktreeAssignment(
        member_id=member_id,
        branch_name=branch_name,
        worktree_name=worktree_name,
        worktree_path=worktree_path,
        claimed_paths=tuple(normalize_path_list(payload.get("claimed_paths"))),
        readonly_paths=tuple(normalize_path_list(payload.get("readonly_paths"))),
        merge_priority=int(payload.get("merge_priority") or 0),
        metadata=(
            dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        ),
    )


def rebuild_worktree_plan(
    payload: Mapping[str, Any],
) -> Optional[WorktreeExecutionPlan]:
    team_name = coerce_optional_text(payload.get("team_name"))
    repo_root = coerce_optional_text(payload.get("repo_root"))
    parent_dir = coerce_optional_text(payload.get("parent_dir"))
    base_ref = coerce_optional_text(payload.get("base_ref"))
    branch_prefix = coerce_optional_text(payload.get("branch_prefix"))
    formation_name = (
        coerce_optional_text(payload.get("formation")) or TeamFormation.SEQUENTIAL.value
    )
    if None in (team_name, repo_root, parent_dir, base_ref, branch_prefix):
        return None
    try:
        formation = TeamFormation(formation_name)
    except ValueError:
        formation = TeamFormation.SEQUENTIAL
    assignments = tuple(
        assignment
        for assignment in (
            rebuild_worktree_assignment(item)
            for item in list(payload.get("assignments") or [])
            if isinstance(item, Mapping)
        )
        if assignment is not None
    )
    return WorktreeExecutionPlan(
        team_name=team_name,
        repo_root=repo_root,
        parent_dir=parent_dir,
        base_ref=base_ref,
        branch_prefix=branch_prefix,
        formation=formation,
        assignments=assignments,
        merge_order=tuple(normalize_member_id_list(payload.get("merge_order"))),
        shared_readonly_paths=tuple(normalize_path_list(payload.get("shared_readonly_paths"))),
        rationale=coerce_optional_text(payload.get("rationale")),
        metadata=(
            dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        ),
    )


def rebuild_materialized_worktree_assignment(
    payload: Mapping[str, Any],
) -> Optional[MaterializedWorktreeAssignment]:
    assignment = rebuild_worktree_assignment(payload)
    if assignment is None:
        return None
    runtime_metadata = payload.get("runtime_metadata")
    return MaterializedWorktreeAssignment(
        assignment=assignment,
        materialized=bool(payload.get("materialized", False)),
        cleanup_required=bool(payload.get("cleanup_required", False)),
        metadata=(dict(runtime_metadata or {}) if isinstance(runtime_metadata, Mapping) else {}),
    )


def rebuild_worktree_session(
    payload: Mapping[str, Any],
) -> Optional[WorktreeMaterializationSession]:
    plan = rebuild_worktree_plan(
        dict(payload.get("plan") or {}) if isinstance(payload.get("plan"), Mapping) else {}
    )
    if plan is None:
        return None
    assignments = tuple(
        assignment
        for assignment in (
            rebuild_materialized_worktree_assignment(item)
            for item in list(payload.get("assignments") or [])
            if isinstance(item, Mapping)
        )
        if assignment is not None
    )
    metadata = payload.get("metadata")
    return WorktreeMaterializationSession(
        plan=plan,
        assignments=assignments,
        materialized=bool(payload.get("materialized", False)),
        dry_run=bool(payload.get("dry_run", False)),
        metadata=dict(metadata or {}) if isinstance(metadata, Mapping) else {},
    )


def build_delegate_merge_execution_contract(
    *,
    worktree_session: WorktreeMaterializationSession,
    merge_analysis: Optional[Mapping[str, Any]],
    merge_orchestration: Optional[Mapping[str, Any]],
    merge_review_contract: Mapping[str, Any],
    worker_return_contracts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "mode": "delegate",
        "next_action": "merge",
        "worktree_session": worktree_session.to_dict(),
        "merge_analysis": dict(merge_analysis or {}),
        "merge_orchestration": dict(merge_orchestration or {}),
        "merge_review_contract": dict(merge_review_contract or {}),
        "worker_return_contracts": {
            member_id: dict(contract)
            for member_id, contract in worker_return_contracts.items()
            if isinstance(contract, Mapping)
        },
        "context_overrides": {
            "mode": "delegate",
            "worktree_isolation": True,
            "materialize_worktrees": False,
            "cleanup_worktrees": True,
        },
    }

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Merge-risk analysis helpers for isolated team execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

from victor.teams.types import MemberResult
from victor.teams.worktree_runtime import WorktreeExecutionPlan


def _normalize_text(value: Any) -> Optional[str]:
    """Coerce any value to a stripped string, returning None for blank/None input."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_paths(values: Any) -> tuple[str, ...]:
    """Normalize a path or sequence of paths to a deduplicated tuple with trailing slashes removed."""
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


def _path_matches_any(path: str, scopes: Sequence[str]) -> bool:
    """Return True if path equals, is under, or is a parent of any scope in scopes."""
    normalized_path = path.rstrip("/")
    for scope in scopes:
        normalized_scope = scope.rstrip("/")
        if (
            normalized_path == normalized_scope
            or normalized_path.startswith(f"{normalized_scope}/")
            or normalized_scope.startswith(f"{normalized_path}/")
        ):
            return True
    return False


class MergeRiskLevel(str, Enum):
    """Risk bands for merging isolated member outputs."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MergeConflict:
    """One concrete merge conflict or policy violation."""

    path: str
    members: tuple[str, ...]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "members": list(self.members),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MergeAnalysis:
    """Summary of merge risk across isolated member executions."""

    risk_level: MergeRiskLevel
    conflict_count: int
    overlapping_files: tuple[MergeConflict, ...] = ()
    readonly_violations: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    out_of_scope_writes: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    potential_scope_overlaps: tuple[str, ...] = ()
    member_changed_files: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    recommended_merge_order: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "conflict_count": self.conflict_count,
            "overlapping_files": [conflict.to_dict() for conflict in self.overlapping_files],
            "readonly_violations": {
                member_id: list(paths) for member_id, paths in self.readonly_violations.items()
            },
            "out_of_scope_writes": {
                member_id: list(paths) for member_id, paths in self.out_of_scope_writes.items()
            },
            "potential_scope_overlaps": list(self.potential_scope_overlaps),
            "member_changed_files": {
                member_id: list(paths) for member_id, paths in self.member_changed_files.items()
            },
            "recommended_merge_order": list(self.recommended_merge_order),
            "notes": list(self.notes),
        }


class MergeAnalyzer:
    """Derive merge risk from member outputs and optional worktree plan."""

    def analyze(
        self,
        member_results: Mapping[str, MemberResult],
        *,
        worktree_plan: Optional[WorktreeExecutionPlan] = None,
    ) -> MergeAnalysis:
        assignments = (
            {assignment.member_id: assignment for assignment in worktree_plan.assignments}
            if worktree_plan is not None
            else {}
        )

        member_changed_files: Dict[str, tuple[str, ...]] = {}
        out_of_scope_writes: Dict[str, tuple[str, ...]] = {}
        readonly_violations: Dict[str, tuple[str, ...]] = {}
        overlapping_files: list[MergeConflict] = []
        path_writers: dict[str, list[str]] = defaultdict(list)

        for member_id, result in member_results.items():
            metadata = dict(getattr(result, "metadata", {}) or {})
            changed_files = _normalize_paths(
                metadata.get("changed_files")
                or metadata.get("files_touched")
                or metadata.get("modified_files")
            )
            member_changed_files[str(member_id)] = changed_files
            assignment = assignments.get(str(member_id))

            if assignment is not None and changed_files:
                if assignment.claimed_paths:
                    outside_scope = tuple(
                        path
                        for path in changed_files
                        if not _path_matches_any(path, assignment.claimed_paths)
                    )
                    if outside_scope:
                        out_of_scope_writes[str(member_id)] = outside_scope
                readonly_hits = tuple(
                    path
                    for path in changed_files
                    if _path_matches_any(path, assignment.readonly_paths)
                )
                if readonly_hits:
                    readonly_violations[str(member_id)] = readonly_hits

            for path in changed_files:
                path_writers[path].append(str(member_id))

        for path, member_ids in path_writers.items():
            unique_members = tuple(sorted(dict.fromkeys(member_ids)))
            if len(unique_members) > 1:
                overlapping_files.append(
                    MergeConflict(
                        path=path,
                        members=unique_members,
                        reason="multiple_members_changed_same_path",
                    )
                )

        potential_scope_overlaps = self._find_scope_overlaps(assignments)
        notes: list[str] = []
        if overlapping_files:
            notes.append("Multiple members changed the same file paths.")
        if out_of_scope_writes:
            notes.append("Some members wrote outside their claimed worktree scope.")
        if readonly_violations:
            notes.append("Readonly shared paths were modified during isolated execution.")
        if potential_scope_overlaps:
            notes.append("Claimed write scopes overlap and may require manual merge ordering.")

        risk_level = self._classify_risk(
            overlap_count=len(overlapping_files),
            out_of_scope_count=sum(len(paths) for paths in out_of_scope_writes.values()),
            readonly_violation_count=sum(len(paths) for paths in readonly_violations.values()),
            scope_overlap_count=len(potential_scope_overlaps),
        )
        conflict_count = (
            len(overlapping_files)
            + sum(len(paths) for paths in out_of_scope_writes.values())
            + sum(len(paths) for paths in readonly_violations.values())
        )

        return MergeAnalysis(
            risk_level=risk_level,
            conflict_count=conflict_count,
            overlapping_files=tuple(overlapping_files),
            readonly_violations=readonly_violations,
            out_of_scope_writes=out_of_scope_writes,
            potential_scope_overlaps=tuple(potential_scope_overlaps),
            member_changed_files=member_changed_files,
            recommended_merge_order=(
                worktree_plan.merge_order
                if worktree_plan is not None
                else tuple(member_results.keys())
            ),
            notes=tuple(notes),
        )

    @staticmethod
    def _find_scope_overlaps(assignments: Mapping[str, Any]) -> list[str]:
        members = list(assignments.values())
        overlaps: list[str] = []
        for index, assignment in enumerate(members):
            for other in members[index + 1 :]:
                for left_scope in assignment.claimed_paths:
                    for right_scope in other.claimed_paths:
                        if _path_matches_any(left_scope, (right_scope,)):
                            overlaps.append(
                                f"{assignment.member_id}:{left_scope}<->{other.member_id}:{right_scope}"
                            )
        return overlaps

    @staticmethod
    def _classify_risk(
        *,
        overlap_count: int,
        out_of_scope_count: int,
        readonly_violation_count: int,
        scope_overlap_count: int,
    ) -> MergeRiskLevel:
        if overlap_count > 0 or readonly_violation_count > 0:
            return MergeRiskLevel.HIGH
        if out_of_scope_count > 0 or scope_overlap_count > 0:
            return MergeRiskLevel.MEDIUM
        return MergeRiskLevel.LOW


__all__ = [
    "MergeAnalysis",
    "MergeAnalyzer",
    "MergeConflict",
    "MergeRiskLevel",
]

# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

from __future__ import annotations

import enum

"""Merge Conflict Resolution Protocol - Unified interface for conflict analysis.

This module defines the abstract interface and data structures for
intelligent merge conflict resolution and analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ConflictType(str, Enum):
    """Types of merge conflicts."""

    CONTENT = "content"  # Both sides modified same lines
    DELETE_MODIFY = "delete_modify"  # One side deleted, other modified
    ADD_ADD = "add_add"  # Both sides added different content at same location
    RENAME_RENAME = "rename_rename"  # Both sides renamed differently
    RENAME_MODIFY = "rename_modify"  # One renamed, other modified
    BINARY = "binary"  # Binary file conflict
    SUBMODULE = "submodule"  # Submodule conflict


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""

    OURS = "ours"  # Take our version
    THEIRS = "theirs"  # Take their version
    UNION = "union"  # Combine both (for additive changes)
    MANUAL = "manual"  # Requires manual intervention
    AI_SUGGESTED = "ai_suggested"  # AI-generated resolution


class ConflictComplexity(str, Enum):
    """Complexity level of a conflict."""

    TRIVIAL = "trivial"  # Whitespace, formatting only
    SIMPLE = "simple"  # Non-overlapping changes
    MODERATE = "moderate"  # Some semantic overlap
    COMPLEX = "complex"  # Significant logic conflicts


@dataclass
class ConflictHunk:
    """A single conflict hunk within a file."""

    start_line: int
    end_line: int
    ours: str  # Our version of the conflicting code
    theirs: str  # Their version of the conflicting code
    base: str | None = None  # Common ancestor (if available)
    context_before: str = ""  # Lines before the conflict
    context_after: str = ""  # Lines after the conflict

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "ours": self.ours,
            "theirs": self.theirs,
            "base": self.base,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


@dataclass
class FileConflict:
    """Conflicts within a single file."""

    file_path: Path
    conflict_type: ConflictType
    hunks: list[ConflictHunk] = field(default_factory=list)
    complexity: ConflictComplexity = ConflictComplexity.SIMPLE
    ours_branch: str = "HEAD"
    theirs_branch: str = ""
    ours_commit: str | None = None
    theirs_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "conflict_type": self.conflict_type.value,
            "hunks": [h.to_dict() for h in self.hunks],
            "complexity": self.complexity.value,
            "ours_branch": self.ours_branch,
            "theirs_branch": self.theirs_branch,
            "ours_commit": self.ours_commit,
            "theirs_commit": self.theirs_commit,
        }


@dataclass
class Resolution:
    """A proposed resolution for a conflict."""

    hunk_index: int  # Which hunk this resolves
    strategy: ResolutionStrategy
    resolved_content: str
    confidence: float = 0.0  # 0-1, how confident we are in this resolution
    explanation: str = ""
    requires_review: bool = True
    semantic_changes: list[str] = field(default_factory=list)  # Description of changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hunk_index": self.hunk_index,
            "strategy": self.strategy.value,
            "resolved_content": self.resolved_content,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "requires_review": self.requires_review,
            "semantic_changes": self.semantic_changes,
        }


@dataclass
class FileResolution:
    """Complete resolution for a file."""

    file_path: Path
    resolutions: list[Resolution]
    final_content: str
    fully_resolved: bool = False
    needs_manual_review: bool = True
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "resolutions": [r.to_dict() for r in self.resolutions],
            "final_content": self.final_content,
            "fully_resolved": self.fully_resolved,
            "needs_manual_review": self.needs_manual_review,
            "applied": self.applied,
        }


@dataclass
class MergeContext:
    """Context information for a merge operation."""

    source_branch: str
    target_branch: str
    merge_type: str = "merge"  # merge, rebase, cherry-pick
    source_commits: list[str] = field(default_factory=list)
    conflict_files: list[FileConflict] = field(default_factory=list)
    resolutions: list[FileResolution] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "merge_type": self.merge_type,
            "source_commits": self.source_commits,
            "conflict_files": [f.to_dict() for f in self.conflict_files],
            "resolutions": [r.to_dict() for r in self.resolutions],
            "started_at": self.started_at.isoformat(),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
        }


@dataclass
class ConflictAnalysis:
    """Analysis result for conflicts."""

    total_files: int
    total_hunks: int
    by_type: dict[str, int] = field(default_factory=dict)
    by_complexity: dict[str, int] = field(default_factory=dict)
    auto_resolvable: int = 0
    needs_manual: int = 0
    estimated_effort: str = "low"  # low, medium, high

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_hunks": self.total_hunks,
            "by_type": self.by_type,
            "by_complexity": self.by_complexity,
            "auto_resolvable": self.auto_resolvable,
            "needs_manual": self.needs_manual,
            "estimated_effort": self.estimated_effort,
        }


class ConflictResolverProtocol(ABC):
    """Abstract protocol for conflict resolution.

    Implementations provide different resolution strategies
    for merge conflicts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the resolver name."""
        ...

    @abstractmethod
    async def can_resolve(self, conflict: FileConflict, hunk_index: int) -> bool:
        """Check if this resolver can handle the conflict.

        Args:
            conflict: The file conflict
            hunk_index: Which hunk to check

        Returns:
            True if this resolver can handle it
        """
        ...

    @abstractmethod
    async def resolve(self, conflict: FileConflict, hunk_index: int) -> Resolution | None:
        """Attempt to resolve a conflict hunk.

        Args:
            conflict: The file conflict
            hunk_index: Which hunk to resolve

        Returns:
            Resolution if successful, None otherwise
        """
        ...


class MergeAnalyzerProtocol(ABC):
    """Abstract protocol for merge analysis.

    Implementations analyze conflicts and provide insights
    about resolution strategies.
    """

    @abstractmethod
    async def analyze_conflicts(self, conflicts: list[FileConflict]) -> ConflictAnalysis:
        """Analyze a set of conflicts.

        Args:
            conflicts: List of file conflicts

        Returns:
            Analysis result
        """
        ...

    @abstractmethod
    async def suggest_order(self, conflicts: list[FileConflict]) -> list[FileConflict]:
        """Suggest optimal resolution order.

        Args:
            conflicts: List of file conflicts

        Returns:
            Conflicts in suggested order
        """
        ...

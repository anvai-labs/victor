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

"""Conflict Resolvers - Strategies for resolving merge conflicts.

This module provides concrete implementations of resolution strategies
for different types of merge conflicts.
"""

import logging
import re

from .protocol import (
    ConflictAnalysis,
    ConflictComplexity,
    ConflictResolverProtocol,
    ConflictType,
    FileConflict,
    MergeAnalyzerProtocol,
    Resolution,
    ResolutionStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Trivial Conflict Resolver
# =============================================================================


class TrivialResolver(ConflictResolverProtocol):
    """Resolver for trivial conflicts (whitespace, formatting)."""

    @property
    def name(self) -> str:
        return "trivial"

    async def can_resolve(self, conflict: FileConflict, hunk_index: int) -> bool:
        """Check if the conflict is trivial."""
        if hunk_index >= len(conflict.hunks):
            return False

        hunk = conflict.hunks[hunk_index]

        # Normalize whitespace and compare
        ours_normalized = self._normalize(hunk.ours)
        theirs_normalized = self._normalize(hunk.theirs)

        return ours_normalized == theirs_normalized

    async def resolve(self, conflict: FileConflict, hunk_index: int) -> Resolution | None:
        """Resolve trivial conflicts by taking the cleaner version."""
        if not await self.can_resolve(conflict, hunk_index):
            return None

        hunk = conflict.hunks[hunk_index]

        # Prefer version with consistent indentation
        ours_indent_score = self._indent_consistency(hunk.ours)
        theirs_indent_score = self._indent_consistency(hunk.theirs)

        if ours_indent_score >= theirs_indent_score:
            content = hunk.ours
            strategy = ResolutionStrategy.OURS
        else:
            content = hunk.theirs
            strategy = ResolutionStrategy.THEIRS

        return Resolution(
            hunk_index=hunk_index,
            strategy=strategy,
            resolved_content=content,
            confidence=0.95,
            explanation="Trivial difference (whitespace/formatting only)",
            requires_review=False,
        )

    def _normalize(self, text: str) -> str:
        """Normalize text by removing whitespace variations."""
        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split("\n")]
        # Normalize indentation to spaces
        lines = [line.replace("\t", "    ") for line in lines]
        # Remove empty lines at start/end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def _indent_consistency(self, text: str) -> float:
        """Score indentation consistency (0-1)."""
        lines = text.split("\n")
        if not lines:
            return 1.0

        # Check for mixed tabs/spaces
        has_tabs = any("\t" in line for line in lines)
        has_leading_spaces = any(line.startswith(" ") for line in lines)

        if has_tabs and has_leading_spaces:
            return 0.5

        return 1.0


# =============================================================================
# Union Resolver
# =============================================================================


class UnionResolver(ConflictResolverProtocol):
    """Resolver for additive conflicts that can be combined."""

    @property
    def name(self) -> str:
        return "union"

    async def can_resolve(self, conflict: FileConflict, hunk_index: int) -> bool:
        """Check if the conflict can be resolved by union."""
        if hunk_index >= len(conflict.hunks):
            return False

        hunk = conflict.hunks[hunk_index]

        # Union works for ADD_ADD conflicts with non-overlapping content
        if conflict.conflict_type == ConflictType.ADD_ADD:
            return True

        # Check if changes are purely additive (no common modified lines)
        if hunk.base:
            ours_diff = self._get_additions(hunk.base, hunk.ours)
            theirs_diff = self._get_additions(hunk.base, hunk.theirs)

            # Check for overlap
            return not (ours_diff & theirs_diff)

        return False

    async def resolve(self, conflict: FileConflict, hunk_index: int) -> Resolution | None:
        """Resolve by combining both changes."""
        if not await self.can_resolve(conflict, hunk_index):
            return None

        hunk = conflict.hunks[hunk_index]

        # Simple union: combine both with separator
        if conflict.conflict_type == ConflictType.ADD_ADD:
            # For ADD_ADD, combine in order
            combined = f"{hunk.ours.rstrip()}\n{hunk.theirs.rstrip()}"
            return Resolution(
                hunk_index=hunk_index,
                strategy=ResolutionStrategy.UNION,
                resolved_content=combined,
                confidence=0.8,
                explanation="Combined both additions",
                requires_review=True,
                semantic_changes=["Combined additions from both branches"],
            )

        # For other cases with base, merge additions
        if hunk.base:
            merged = self._merge_additions(hunk.base, hunk.ours, hunk.theirs)
            return Resolution(
                hunk_index=hunk_index,
                strategy=ResolutionStrategy.UNION,
                resolved_content=merged,
                confidence=0.75,
                explanation="Merged non-overlapping additions from both sides",
                requires_review=True,
                semantic_changes=["Merged additions from both branches"],
            )

        return None

    def _get_additions(self, base: str, modified: str) -> set[str]:
        """Get lines added in modified relative to base."""
        base_lines = set(base.split("\n"))
        modified_lines = set(modified.split("\n"))
        return modified_lines - base_lines

    def _merge_additions(self, base: str, ours: str, theirs: str) -> str:
        """Merge additions from both sides onto base."""
        base_lines = base.split("\n")
        ours_lines = ours.split("\n")
        theirs_lines = theirs.split("\n")

        # Use difflib to merge
        merged = []

        # Simple approach: add all unique lines
        all_lines = set(ours_lines) | set(theirs_lines)
        base_set = set(base_lines)

        for line in base_lines:
            merged.append(line)
            # Add any new lines that appear after this in either version
            for new_line in all_lines - base_set:
                if new_line not in merged:
                    merged.append(new_line)

        return "\n".join(merged)


# =============================================================================
# Import Statement Resolver
# =============================================================================


class ImportResolver(ConflictResolverProtocol):
    """Resolver specialized for import statement conflicts."""

    @property
    def name(self) -> str:
        return "import"

    async def can_resolve(self, conflict: FileConflict, hunk_index: int) -> bool:
        """Check if the conflict is in import statements."""
        if hunk_index >= len(conflict.hunks):
            return False

        hunk = conflict.hunks[hunk_index]

        # Check if both sides are import statements
        import_patterns = [
            r"^\s*(import|from)\s+",  # Python
            r"^\s*import\s+",  # Java, TypeScript
            r"^\s*#include\s+",  # C/C++
            r"^\s*use\s+",  # Rust
            r"^\s*require\s*\(",  # Node.js require
            r"^\s*const\s+\w+\s*=\s*require\s*\(",  # Node.js const require
        ]

        combined_pattern = "|".join(import_patterns)
        ours_lines = hunk.ours.strip().split("\n")
        theirs_lines = hunk.theirs.strip().split("\n")

        ours_all_imports = all(
            re.match(combined_pattern, line) or not line.strip() for line in ours_lines
        )
        theirs_all_imports = all(
            re.match(combined_pattern, line) or not line.strip() for line in theirs_lines
        )

        return ours_all_imports and theirs_all_imports

    async def resolve(self, conflict: FileConflict, hunk_index: int) -> Resolution | None:
        """Resolve import conflicts by combining and sorting."""
        if not await self.can_resolve(conflict, hunk_index):
            return None

        hunk = conflict.hunks[hunk_index]

        # Combine imports
        ours_imports = {line.strip() for line in hunk.ours.split("\n") if line.strip()}
        theirs_imports = {line.strip() for line in hunk.theirs.split("\n") if line.strip()}

        combined = ours_imports | theirs_imports

        # Sort imports
        sorted_imports = sorted(combined, key=self._import_sort_key)

        return Resolution(
            hunk_index=hunk_index,
            strategy=ResolutionStrategy.UNION,
            resolved_content="\n".join(sorted_imports),
            confidence=0.9,
            explanation="Combined and sorted import statements from both branches",
            requires_review=False,
            semantic_changes=["Merged import statements"],
        )

    def _import_sort_key(self, import_line: str) -> tuple[int, str]:
        """Sort key for imports (stdlib first, then third-party, then local)."""
        # Python-style sorting
        if import_line.startswith("from __future__"):
            return (0, import_line)
        elif import_line.startswith("import ") or import_line.startswith("from "):
            # Check for common stdlib modules
            stdlib_prefixes = [
                "os",
                "sys",
                "re",
                "json",
                "typing",
                "datetime",
                "pathlib",
                "collections",
                "itertools",
                "functools",
                "logging",
                "abc",
            ]
            module = import_line.split()[1].split(".")[0]
            if module in stdlib_prefixes:
                return (1, import_line)
            elif module.startswith("."):
                return (3, import_line)  # Relative imports last
            else:
                return (2, import_line)  # Third-party
        return (4, import_line)


# =============================================================================
# Merge Analyzer
# =============================================================================


class DefaultMergeAnalyzer(MergeAnalyzerProtocol):
    """Default implementation of merge analysis."""

    async def analyze_conflicts(self, conflicts: list[FileConflict]) -> ConflictAnalysis:
        """Analyze a set of conflicts."""
        total_hunks = sum(len(c.hunks) for c in conflicts)

        # Count by type
        by_type: dict[str, int] = {}
        for conflict in conflicts:
            type_key = conflict.conflict_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        # Count by complexity
        by_complexity: dict[str, int] = {}
        for conflict in conflicts:
            complexity_key = conflict.complexity.value
            by_complexity[complexity_key] = by_complexity.get(complexity_key, 0) + 1

        # Estimate auto-resolvable
        auto_resolvable = sum(
            1
            for c in conflicts
            if c.complexity in [ConflictComplexity.TRIVIAL, ConflictComplexity.SIMPLE]
        )
        needs_manual = len(conflicts) - auto_resolvable

        # Estimate effort
        if needs_manual == 0:
            effort = "low"
        elif needs_manual <= 3:
            effort = "medium"
        else:
            effort = "high"

        return ConflictAnalysis(
            total_files=len(conflicts),
            total_hunks=total_hunks,
            by_type=by_type,
            by_complexity=by_complexity,
            auto_resolvable=auto_resolvable,
            needs_manual=needs_manual,
            estimated_effort=effort,
        )

    async def suggest_order(self, conflicts: list[FileConflict]) -> list[FileConflict]:
        """Suggest optimal resolution order."""
        # Sort by complexity (trivial first) then by file type
        complexity_order = {
            ConflictComplexity.TRIVIAL: 0,
            ConflictComplexity.SIMPLE: 1,
            ConflictComplexity.MODERATE: 2,
            ConflictComplexity.COMPLEX: 3,
        }

        def sort_key(conflict: FileConflict) -> tuple[int, int, str]:
            complexity = complexity_order.get(conflict.complexity, 99)
            # Prioritize config files (often have cascading effects)
            file_name = conflict.file_path.name.lower()
            if file_name in ["package.json", "pyproject.toml", "cargo.toml"]:
                priority = 0
            elif file_name.endswith((".json", ".yaml", ".yml", ".toml")):
                priority = 1
            else:
                priority = 2
            return (priority, complexity, str(conflict.file_path))

        return sorted(conflicts, key=sort_key)


# =============================================================================
# Resolver Registry
# =============================================================================


# Ordered list of resolvers (tried in order)
CONFLICT_RESOLVERS: list[ConflictResolverProtocol] = [
    TrivialResolver(),
    ImportResolver(),
    UnionResolver(),
]


def get_resolvers() -> list[ConflictResolverProtocol]:
    """Get all registered conflict resolvers."""
    return CONFLICT_RESOLVERS


def get_analyzer() -> MergeAnalyzerProtocol:
    """Get the merge analyzer."""
    return DefaultMergeAnalyzer()

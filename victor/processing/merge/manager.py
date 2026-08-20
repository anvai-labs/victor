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

"""Merge Conflict Manager - Orchestrates conflict detection and resolution.

This module provides the MergeManager class that coordinates conflict
detection, analysis, and resolution.
"""

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from victor.config.settings import get_project_paths, load_settings

from .protocol import (
    ConflictAnalysis,
    ConflictComplexity,
    ConflictHunk,
    ConflictType,
    FileConflict,
    FileResolution,
    MergeContext,
    Resolution,
    ResolutionStrategy,
)
from .resolvers import get_analyzer, get_resolvers

logger = logging.getLogger(__name__)


class MergeManager:
    """Manager for merge conflict resolution.

    This class orchestrates:
    - Conflict detection from git status
    - Conflict parsing from files
    - Automatic resolution attempts
    - Resolution application

    Configuration is driven by settings.py for consistency with Victor.
    """

    def __init__(self, root_path: str | Path | None = None):
        """Initialize the merge manager.

        Args:
            root_path: Root directory of the git repository. Defaults to current directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._settings = load_settings()
        self._paths = get_project_paths(self.root_path)
        self._history_file = self._paths.project_victor_dir / "merge_history.json"
        self._current_context: MergeContext | None = None
        self._resolvers = get_resolvers()
        self._analyzer = get_analyzer()

    async def detect_conflicts(self) -> list[FileConflict]:
        """Detect merge conflicts in the current repository.

        Returns:
            List of file conflicts
        """
        conflicts = []

        # Get list of unmerged files from git
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                check=True,
            )
            unmerged_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except subprocess.CalledProcessError:
            logger.warning("Failed to get unmerged files from git")
            return []

        # Parse each conflict file
        for file_path in unmerged_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                conflict = await self._parse_conflict_file(full_path)
                if conflict:
                    conflicts.append(conflict)

        # Get branch info
        source_branch = self._get_merging_branch()
        target_branch = self._get_current_branch()

        for conflict in conflicts:
            conflict.ours_branch = target_branch
            conflict.theirs_branch = source_branch

        logger.info(f"Detected {len(conflicts)} conflict files")
        return conflicts

    async def _parse_conflict_file(self, file_path: Path) -> FileConflict | None:
        """Parse conflict markers from a file.

        Args:
            file_path: Path to the conflict file

        Returns:
            FileConflict or None if no conflicts found
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to read conflict file {file_path}: {e}")
            return None

        # Detect conflict markers
        # <<<<<<< HEAD (or branch name)
        # our changes
        # ||||||| base (optional, if diff3 style)
        # base content
        # =======
        # their changes
        # >>>>>>> branch-name

        conflict_pattern = re.compile(
            r"^<<<<<<<\s*(.+?)$\n"  # Start marker with branch
            r"(.*?)"  # Our content
            r"(?:\|\|\|\|\|\|\|\s*.*?$\n(.*?))?"  # Optional base content
            r"^=======\s*$\n"  # Separator
            r"(.*?)"  # Their content
            r"^>>>>>>>\s*(.+?)$",  # End marker with branch
            re.MULTILINE | re.DOTALL,
        )

        hunks = []
        for match in conflict_pattern.finditer(content):
            start_line = content[: match.start()].count("\n") + 1
            end_line = content[: match.end()].count("\n") + 1

            ours = match.group(2)
            base = match.group(3) if match.group(3) else None
            theirs = match.group(4)

            # Get context
            lines = content.split("\n")
            context_start = max(0, start_line - 4)
            context_end = min(len(lines), end_line + 3)

            context_before = "\n".join(lines[context_start : start_line - 1])
            context_after = "\n".join(lines[end_line:context_end])

            hunks.append(
                ConflictHunk(
                    start_line=start_line,
                    end_line=end_line,
                    ours=ours.strip(),
                    theirs=theirs.strip(),
                    base=base.strip() if base else None,
                    context_before=context_before,
                    context_after=context_after,
                )
            )

        if not hunks:
            return None

        # Determine conflict type and complexity
        conflict_type = self._detect_conflict_type(hunks)
        complexity = self._assess_complexity(hunks)

        return FileConflict(
            file_path=file_path,
            conflict_type=conflict_type,
            hunks=hunks,
            complexity=complexity,
        )

    def _detect_conflict_type(self, hunks: list[ConflictHunk]) -> ConflictType:
        """Detect the type of conflict from hunks."""
        for hunk in hunks:
            if not hunk.ours.strip():
                return ConflictType.DELETE_MODIFY
            if not hunk.theirs.strip():
                return ConflictType.DELETE_MODIFY
            if hunk.base is None or not hunk.base.strip():
                return ConflictType.ADD_ADD

        return ConflictType.CONTENT

    def _assess_complexity(self, hunks: list[ConflictHunk]) -> ConflictComplexity:
        """Assess the complexity of conflicts."""
        total_lines = sum(len(h.ours.split("\n")) + len(h.theirs.split("\n")) for h in hunks)

        # Check for trivial (whitespace only)
        all_trivial = all(
            self._normalize_whitespace(h.ours) == self._normalize_whitespace(h.theirs)
            for h in hunks
        )
        if all_trivial:
            return ConflictComplexity.TRIVIAL

        # Simple: few lines, no complex logic
        if total_lines < 10 and len(hunks) <= 2:
            return ConflictComplexity.SIMPLE

        # Complex: many lines or semantic differences
        if total_lines > 50 or len(hunks) > 5:
            return ConflictComplexity.COMPLEX

        return ConflictComplexity.MODERATE

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for comparison."""
        return re.sub(r"\s+", " ", text.strip())

    def _get_current_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "HEAD"

    def _get_merging_branch(self) -> str:
        """Get the branch being merged in."""
        try:
            # Check for MERGE_HEAD
            merge_head = self.root_path / ".git" / "MERGE_HEAD"
            if merge_head.exists():
                commit = merge_head.read_text().strip()[:8]
                return f"MERGE_HEAD ({commit})"

            # Check REBASE_HEAD
            rebase_head = self.root_path / ".git" / "REBASE_HEAD"
            if rebase_head.exists():
                commit = rebase_head.read_text().strip()[:8]
                return f"REBASE_HEAD ({commit})"

            return "unknown"
        except Exception:
            return "unknown"

    async def analyze_conflicts(
        self, conflicts: list[FileConflict] | None = None
    ) -> ConflictAnalysis:
        """Analyze conflicts and provide insights.

        Args:
            conflicts: Conflicts to analyze. If None, detects from git.

        Returns:
            Analysis result
        """
        if conflicts is None:
            conflicts = await self.detect_conflicts()

        return await self._analyzer.analyze_conflicts(conflicts)

    async def resolve_conflicts(
        self,
        conflicts: list[FileConflict] | None = None,
        auto_apply: bool = False,
    ) -> list[FileResolution]:
        """Attempt to resolve conflicts automatically.

        Args:
            conflicts: Conflicts to resolve. If None, detects from git.
            auto_apply: If True, applies resolutions directly to files.

        Returns:
            List of file resolutions
        """
        if conflicts is None:
            conflicts = await self.detect_conflicts()

        # Sort for optimal resolution order
        conflicts = await self._analyzer.suggest_order(conflicts)

        resolutions = []
        for conflict in conflicts:
            file_resolution = await self._resolve_file(conflict)
            if file_resolution:
                resolutions.append(file_resolution)

                if auto_apply and file_resolution.fully_resolved:
                    await self._apply_resolution(file_resolution)

        # Update context
        self._current_context = MergeContext(
            source_branch=conflicts[0].theirs_branch if conflicts else "unknown",
            target_branch=conflicts[0].ours_branch if conflicts else "unknown",
            conflict_files=conflicts,
            resolutions=resolutions,
        )

        return resolutions

    async def _resolve_file(self, conflict: FileConflict) -> FileResolution | None:
        """Attempt to resolve all hunks in a file.

        Args:
            conflict: File conflict to resolve

        Returns:
            File resolution or None
        """
        hunk_resolutions = []
        all_resolved = True
        needs_review = False

        for i, _hunk in enumerate(conflict.hunks):
            resolution = await self._resolve_hunk(conflict, i)
            if resolution:
                hunk_resolutions.append(resolution)
                if resolution.requires_review:
                    needs_review = True
            else:
                all_resolved = False
                # Create manual resolution placeholder
                hunk_resolutions.append(
                    Resolution(
                        hunk_index=i,
                        strategy=ResolutionStrategy.MANUAL,
                        resolved_content="",
                        confidence=0.0,
                        explanation="Requires manual resolution",
                        requires_review=True,
                    )
                )

        # Build final content
        final_content = await self._build_resolved_content(conflict, hunk_resolutions)

        return FileResolution(
            file_path=conflict.file_path,
            resolutions=hunk_resolutions,
            final_content=final_content,
            fully_resolved=all_resolved,
            needs_manual_review=needs_review or not all_resolved,
        )

    async def _resolve_hunk(self, conflict: FileConflict, hunk_index: int) -> Resolution | None:
        """Attempt to resolve a single hunk using available resolvers.

        Args:
            conflict: File conflict
            hunk_index: Index of hunk to resolve

        Returns:
            Resolution or None
        """
        for resolver in self._resolvers:
            try:
                if await resolver.can_resolve(conflict, hunk_index):
                    resolution = await resolver.resolve(conflict, hunk_index)
                    if resolution:
                        logger.debug(
                            f"Resolver '{resolver.name}' resolved hunk {hunk_index} "
                            f"in {conflict.file_path.name}"
                        )
                        return resolution
            except Exception as e:
                logger.warning(f"Resolver '{resolver.name}' failed on {conflict.file_path}: {e}")

        return None

    async def _build_resolved_content(
        self, conflict: FileConflict, resolutions: list[Resolution]
    ) -> str:
        """Build the resolved file content.

        Args:
            conflict: Original conflict
            resolutions: Resolutions for each hunk

        Returns:
            Resolved file content
        """
        try:
            content = conflict.file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

        # Replace conflict markers with resolved content
        # Process in reverse order to maintain line positions
        hunks_and_resolutions = list(zip(conflict.hunks, resolutions, strict=True))
        hunks_and_resolutions.sort(key=lambda x: x[0].start_line, reverse=True)

        lines = content.split("\n")

        for hunk, resolution in hunks_and_resolutions:
            if resolution.strategy == ResolutionStrategy.MANUAL:
                continue  # Leave conflict markers for manual resolution

            # Find and replace the conflict block
            start = hunk.start_line - 1
            end = hunk.end_line

            # The conflict markers span from start to end
            # Replace with resolved content
            resolved_lines = resolution.resolved_content.split("\n")
            lines = lines[:start] + resolved_lines + lines[end:]

        return "\n".join(lines)

    async def _apply_resolution(self, resolution: FileResolution) -> bool:
        """Apply a resolution to the file.

        Args:
            resolution: Resolution to apply

        Returns:
            True if successful
        """
        try:
            resolution.file_path.write_text(resolution.final_content, encoding="utf-8")
            resolution.applied = True

            # Stage the file
            subprocess.run(
                ["git", "add", str(resolution.file_path)],
                cwd=self.root_path,
                check=True,
            )

            logger.info(f"Applied resolution to {resolution.file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply resolution: {e}")
            return False

    async def apply_resolution(self, file_path: str | Path, strategy: ResolutionStrategy) -> bool:
        """Apply a specific resolution strategy to a file.

        Args:
            file_path: File to resolve
            strategy: Strategy to use (OURS or THEIRS)

        Returns:
            True if successful
        """
        if strategy not in [ResolutionStrategy.OURS, ResolutionStrategy.THEIRS]:
            logger.warning(f"Invalid strategy for apply_resolution: {strategy}")
            return False

        try:
            # Use git checkout --ours/--theirs
            flag = "--ours" if strategy == ResolutionStrategy.OURS else "--theirs"
            subprocess.run(
                ["git", "checkout", flag, "--", str(file_path)],
                cwd=self.root_path,
                check=True,
            )

            # Stage the file
            subprocess.run(
                ["git", "add", str(file_path)],
                cwd=self.root_path,
                check=True,
            )

            logger.info(f"Applied {strategy.value} to {file_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply resolution: {e}")
            return False

    async def get_conflict_summary(self) -> dict[str, Any]:
        """Get a summary of current conflicts.

        Returns:
            Summary dictionary
        """
        conflicts = await self.detect_conflicts()
        analysis = await self.analyze_conflicts(conflicts)

        return {
            "has_conflicts": len(conflicts) > 0,
            "total_files": analysis.total_files,
            "total_hunks": analysis.total_hunks,
            "auto_resolvable": analysis.auto_resolvable,
            "needs_manual": analysis.needs_manual,
            "estimated_effort": analysis.estimated_effort,
            "by_type": analysis.by_type,
            "by_complexity": analysis.by_complexity,
            "files": [
                {
                    "path": str(c.file_path),
                    "hunks": len(c.hunks),
                    "complexity": c.complexity.value,
                    "type": c.conflict_type.value,
                }
                for c in conflicts
            ],
        }

    async def abort_merge(self) -> bool:
        """Abort the current merge operation.

        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ["git", "merge", "--abort"],
                cwd=self.root_path,
                check=True,
            )
            logger.info("Aborted merge")
            return True
        except subprocess.CalledProcessError:
            # Try rebase abort
            try:
                subprocess.run(
                    ["git", "rebase", "--abort"],
                    cwd=self.root_path,
                    check=True,
                )
                logger.info("Aborted rebase")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to abort: {e}")
                return False

    async def save_history(self) -> None:
        """Save merge context to history."""
        if not self._current_context:
            return

        self._current_context.completed_at = datetime.now()

        # Load existing history
        history = []
        if self._history_file.exists():
            try:
                with open(self._history_file, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass

        # Add current context
        history.append(self._current_context.to_dict())

        # Keep last 20 merges
        history = history[-20:]

        # Save
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

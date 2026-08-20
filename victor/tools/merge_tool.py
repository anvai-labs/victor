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

"""Merge Conflict Tool - Intelligent conflict resolution for Victor.

This tool provides integration with Victor's tool system for
merge conflict detection and resolution.
"""

import logging
from typing import Any

from victor.processing.merge import (
    FileResolution,
    MergeManager,
    ResolutionStrategy,
)
from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    Priority,
    ToolMetadata,
    ToolResult,
)
from victor.tools.tool_names import ToolNames

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icons
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter (lazy initialization)."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


class MergeConflictTool(BaseTool):
    """Tool for detecting and resolving merge conflicts."""

    name = ToolNames.MERGE
    description = """Detect and resolve git merge conflicts.

    Actions: detect, analyze, resolve (auto), apply (ours/theirs), abort.
    Smart strategies: trivial (whitespace), import (sort/combine), union."""

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["detect", "analyze", "resolve", "apply", "abort"],
                "description": "Action to perform",
            },
            "file_path": {
                "type": "string",
                "description": "File path for apply action",
            },
            "strategy": {
                "type": "string",
                "enum": ["ours", "theirs"],
                "description": "Resolution strategy for apply action",
            },
            "auto_apply": {
                "type": "boolean",
                "description": "Auto-apply successful resolutions (default: false)",
            },
        },
        "required": ["action"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.FREE

    @property
    def priority(self) -> Priority:
        """Tool priority for selection availability."""
        return Priority.MEDIUM  # Task-specific conflict resolution

    @property
    def access_mode(self) -> AccessMode:
        """Tool access mode for approval tracking."""
        return AccessMode.MIXED  # Reads repo state and can modify files

    @property
    def danger_level(self) -> DangerLevel:
        """Danger level for warning/confirmation logic."""
        return DangerLevel.MEDIUM  # Modifies conflicted files

    @property
    def metadata(self) -> ToolMetadata:
        """Inline semantic metadata for dynamic tool selection."""
        return ToolMetadata(
            category="merge",
            keywords=[
                "merge conflict",
                "conflict",
                "resolve conflict",
                "git conflict",
                "rebase conflict",
                "merge resolution",
                "conflict markers",
                "git merge",
            ],
            use_cases=[
                "detecting merge conflicts",
                "resolving git conflicts",
                "conflict analysis",
                "automated conflict resolution",
                "merge assistance",
            ],
            examples=[
                "detecting conflicts in current repository",
                "analyzing conflict complexity",
                "automatically resolving trivial conflicts",
                "getting conflict resolution suggestions",
            ],
            priority_hints=[
                "Use when git merge or rebase has conflicts",
                "Automatically resolves trivial and whitespace conflicts",
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute merge conflict action."""
        action = kwargs.get("action", "detect")
        file_path = kwargs.get("file_path")
        strategy_str = kwargs.get("strategy")
        auto_apply = kwargs.get("auto_apply", False)

        try:
            manager = MergeManager()

            if action == "detect":
                conflicts = await manager.detect_conflicts()
                if not conflicts:
                    return ToolResult(
                        success=True,
                        output=f"{_get_icon('success')} No merge conflicts detected",
                        metadata={"conflicts": []},
                    )
                return ToolResult(
                    success=True,
                    output=self._format_conflicts(conflicts),
                    metadata={
                        "conflict_count": len(conflicts),
                        "files": [str(c.file_path) for c in conflicts],
                    },
                )

            elif action == "analyze":
                summary = await manager.get_conflict_summary()
                return ToolResult(
                    success=True,
                    output=self._format_analysis(summary),
                    metadata=summary,
                )

            elif action == "resolve":
                resolutions = await manager.resolve_conflicts(auto_apply=auto_apply)
                return ToolResult(
                    success=True,
                    output=self._format_resolutions(resolutions),
                    metadata={
                        "resolved_count": sum(1 for r in resolutions if r.fully_resolved),
                        "needs_manual": sum(1 for r in resolutions if r.needs_manual_review),
                    },
                )

            elif action == "apply":
                if not file_path:
                    return ToolResult(
                        success=False,
                        output="file_path is required for apply action",
                        error="Missing parameter",
                    )
                if not strategy_str:
                    return ToolResult(
                        success=False,
                        output="strategy is required for apply action",
                        error="Missing parameter",
                    )

                strategy = ResolutionStrategy(strategy_str)
                success = await manager.apply_resolution(file_path, strategy)
                if success:
                    return ToolResult(
                        success=True,
                        output=f"{_get_icon('success')} Applied {strategy_str} strategy to {file_path}",
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=f"Failed to apply resolution to {file_path}",
                        error="Resolution failed",
                    )

            elif action == "abort":
                success = await manager.abort_merge()
                if success:
                    return ToolResult(
                        success=True,
                        output=f"{_get_icon('success')} Merge/rebase aborted successfully",
                    )
                else:
                    return ToolResult(
                        success=False,
                        output="Failed to abort merge/rebase",
                        error="Abort failed",
                    )

            else:
                return ToolResult(
                    success=False,
                    output=f"Unknown action: {action}",
                    error="Invalid action",
                )

        except Exception as e:
            logger.exception(f"Merge conflict operation failed: {e}")
            return ToolResult(
                success=False,
                output=f"Operation failed: {e}",
                error=str(e),
            )

    def _format_conflicts(self, conflicts: list) -> str:
        """Format detected conflicts."""
        lines = ["**Merge Conflicts Detected**", ""]
        lines.append(f"**Total Files:** {len(conflicts)}")
        lines.append("")

        for conflict in conflicts:
            complexity_icon = {
                "trivial": _get_icon("level_low"),
                "simple": _get_icon("level_medium"),
                "moderate": _get_icon("level_high"),
                "complex": _get_icon("level_critical"),
            }.get(conflict.complexity.value, _get_icon("level_unknown"))

            lines.append(f"{complexity_icon} **{conflict.file_path.name}**")
            lines.append(f"   Type: {conflict.conflict_type.value}")
            lines.append(f"   Hunks: {len(conflict.hunks)}")
            lines.append(f"   Complexity: {conflict.complexity.value}")
            lines.append("")

        return "\n".join(lines)

    def _format_analysis(self, summary: dict[str, Any]) -> str:
        """Format conflict analysis."""
        if not summary["has_conflicts"]:
            return f"{_get_icon('success')} No merge conflicts to analyze"

        lines = ["**Conflict Analysis**", ""]

        # Overview
        effort_icon = {
            "low": _get_icon("level_low"),
            "medium": _get_icon("level_medium"),
            "high": _get_icon("level_critical"),
        }.get(summary["estimated_effort"], _get_icon("level_unknown"))

        lines.append(f"**Effort Required:** {effort_icon} {summary['estimated_effort']}")
        lines.append(f"**Total Files:** {summary['total_files']}")
        lines.append(f"**Total Hunks:** {summary['total_hunks']}")
        lines.append("")

        lines.append("**Resolution Outlook:**")
        lines.append(f"- {_get_icon('success')} Auto-resolvable: {summary['auto_resolvable']}")
        lines.append(f"- {_get_icon('person')} Needs Manual: {summary['needs_manual']}")
        lines.append("")

        # By complexity
        if summary["by_complexity"]:
            lines.append("**By Complexity:**")
            for complexity, count in summary["by_complexity"].items():
                lines.append(f"- {complexity}: {count}")
            lines.append("")

        # File list
        if summary["files"]:
            lines.append("**Files:**")
            for f in summary["files"]:
                lines.append(f"- {f['path']} ({f['complexity']}, {f['hunks']} hunks)")

        return "\n".join(lines)

    def _format_resolutions(self, resolutions: list[FileResolution]) -> str:
        """Format resolution results."""
        if not resolutions:
            return "No conflicts to resolve"

        lines = ["**Resolution Results**", ""]

        resolved = sum(1 for r in resolutions if r.fully_resolved)
        partial = sum(1 for r in resolutions if not r.fully_resolved)

        lines.append(f"**Fully Resolved:** {resolved}")
        lines.append(f"**Needs Manual:** {partial}")
        lines.append("")

        for resolution in resolutions:
            if resolution.fully_resolved:
                icon = _get_icon("success")
                status = "Resolved"
            else:
                icon = _get_icon("warning")
                status = "Partial"

            applied = " (applied)" if resolution.applied else ""
            lines.append(f"{icon} **{resolution.file_path.name}** - {status}{applied}")

            for r in resolution.resolutions:
                if r.strategy.value != "manual":
                    lines.append(
                        f"   Hunk {r.hunk_index}: {r.strategy.value} "
                        f"(confidence: {r.confidence:.0%})"
                    )
                    if r.explanation:
                        lines.append(f"   → {r.explanation}")
                else:
                    lines.append(f"   Hunk {r.hunk_index}: requires manual resolution")

            lines.append("")

        return "\n".join(lines)

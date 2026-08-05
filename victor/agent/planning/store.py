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

"""Plan file storage for persistent plan management.

This module provides plan persistence similar to Claude Code's ~/.claude/plans/,
enabling plans to be saved, loaded, and resumed across sessions.

Features:
- Save plans as markdown files with YAML frontmatter
- Load plans from disk for resumption
- List available plans
- Auto-save plans during creation
- Plan versioning and history

Usage:
    store = PlanStore()

    # Save a plan
    path = store.save(plan)

    # Load a plan
    plan = store.load("plan_abc123")

    # List available plans
    plans = store.list_plans()
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType

logger = logging.getLogger(__name__)


class PlanStore:
    """Persistent storage for execution plans.

    Stores plans in .victor/plans/ directory as markdown files with
    JSON frontmatter for machine parsing.

    Attributes:
        plans_dir: Directory where plans are stored
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the plan store.

        Args:
            project_root: Project root directory (defaults to cwd)
        """
        if project_root is None:
            project_root = Path.cwd()

        self.project_root = Path(project_root)
        self.plans_dir = self.project_root / ".victor" / "plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"PlanStore initialized at {self.plans_dir}")

    def save(self, plan: ExecutionPlan, filename: Optional[str] = None) -> Path:
        """Save a plan to disk.

        Args:
            plan: The execution plan to save
            filename: Optional custom filename (without extension)

        Returns:
            Path to the saved plan file
        """
        if filename is None:
            # Generate filename from plan ID and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_goal = self._sanitize_filename(plan.goal[:30])
            filename = f"{timestamp}_{plan.id}_{safe_goal}"

        filepath = self.plans_dir / f"{filename}.md"

        # Build markdown content
        content = self._plan_to_markdown(plan)

        # Write to file
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Plan saved to {filepath}")

        return filepath

    def load(self, plan_id_or_filename: str) -> Optional[ExecutionPlan]:
        """Load a plan from disk.

        Args:
            plan_id_or_filename: Plan ID or filename to load

        Returns:
            ExecutionPlan if found, None otherwise
        """
        # Try exact filename match first
        for ext in [".md", ""]:
            filepath = self.plans_dir / f"{plan_id_or_filename}{ext}"
            if filepath.exists():
                return self._load_from_file(filepath)

        # Search by plan ID
        for filepath in self.plans_dir.glob("*.md"):
            plan = self._load_from_file(filepath)
            if plan and plan.id == plan_id_or_filename:
                return plan

        logger.warning(f"Plan not found: {plan_id_or_filename}")
        return None

    def list_plans(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List available plans.

        Args:
            limit: Maximum number of plans to return

        Returns:
            List of plan summaries (id, goal, created_at, status)
        """
        plans = []

        for filepath in sorted(self.plans_dir.glob("*.md"), reverse=True):
            try:
                plan = self._load_from_file(filepath)
                if plan:
                    completed = len(plan.get_completed_steps())
                    total = len(plan.steps)
                    status = "completed" if plan.is_complete() else f"{completed}/{total}"

                    plans.append(
                        {
                            "id": plan.id,
                            "goal": plan.goal[:50] + ("..." if len(plan.goal) > 50 else ""),
                            "created_at": datetime.fromtimestamp(plan.created_at).isoformat(),
                            "status": status,
                            "filepath": str(filepath),
                        }
                    )

                    if len(plans) >= limit:
                        break
            except Exception as e:
                logger.debug(f"Failed to parse {filepath}: {e}")
                continue

        return plans

    def get_latest(self) -> Optional[ExecutionPlan]:
        """Get the most recently created plan.

        Returns:
            Most recent ExecutionPlan if any, None otherwise
        """
        files = sorted(self.plans_dir.glob("*.md"), reverse=True)
        for filepath in files:
            plan = self._load_from_file(filepath)
            if plan:
                return plan
        return None

    def delete(self, plan_id: str) -> bool:
        """Delete a plan from disk.

        Args:
            plan_id: Plan ID to delete

        Returns:
            True if deleted, False if not found
        """
        for filepath in self.plans_dir.glob("*.md"):
            plan = self._load_from_file(filepath)
            if plan and plan.id == plan_id:
                filepath.unlink()
                logger.info(f"Plan {plan_id} deleted")
                return True
        return False

    def _plan_to_markdown(self, plan: ExecutionPlan) -> str:
        """Convert plan to markdown with JSON frontmatter."""
        lines = []

        # JSON frontmatter for machine parsing
        frontmatter = {
            "id": plan.id,
            "goal": plan.goal,
            "created_at": plan.created_at,
            "approved": plan.approved,
            "metadata": plan.metadata,
            "steps": [s.to_dict() for s in plan.steps],
        }

        lines.append("---")
        lines.append(json_dumps(frontmatter, indent=2))
        lines.append("---")
        lines.append("")

        # Human-readable markdown
        lines.append(plan.to_markdown())

        return "\n".join(lines)

    def _load_from_file(self, filepath: Path) -> Optional[ExecutionPlan]:
        """Load plan from a markdown file with frontmatter."""
        try:
            content = filepath.read_text(encoding="utf-8")

            # Parse frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_json = parts[1].strip()
                    data = json_loads(frontmatter_json)
                    return ExecutionPlan.from_dict(data)

            return None
        except Exception as e:
            logger.debug(f"Failed to load plan from {filepath}: {e}")
            return None

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as filename."""
        # Replace non-alphanumeric chars with underscore
        safe = re.sub(r"[^a-zA-Z0-9]", "_", name)
        # Remove consecutive underscores
        safe = re.sub(r"_+", "_", safe)
        # Remove leading/trailing underscores
        return safe.strip("_").lower()


# Global singleton
_plan_store: Optional[PlanStore] = None


def get_plan_store(project_root: Optional[Path] = None) -> PlanStore:
    """Get or create the plan store singleton.

    Args:
        project_root: Optional project root (only used on first call)

    Returns:
        PlanStore instance
    """
    global _plan_store
    if _plan_store is None:
        _plan_store = PlanStore(project_root)
    return _plan_store


def reset_plan_store() -> None:
    """Reset the plan store singleton (for testing)."""
    global _plan_store
    _plan_store = None


__all__ = [
    "PlanStore",
    "get_plan_store",
    "reset_plan_store",
]

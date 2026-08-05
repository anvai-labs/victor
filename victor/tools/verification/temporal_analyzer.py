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

"""Temporal context analysis for codebase issues.

This module analyzes the temporal nature of issues - distinguishing
between temporary problems (migration shims, WIP code) and permanent
issues requiring intentional fixes.

P3 Priority: Lower impact, higher effort. Provides useful context for prioritization.

Design Patterns:
- Git History Analysis: Uses git to determine file age and change frequency
- Pattern Recognition: Identifies temporal markers in comments
- Heuristic Classification: Combines multiple signals for classification

Usage:
    analyzer = TemporalContextAnalyzer(project_root=Path("."))
    nature = analyzer.estimate_temporal_nature("src/lib.rs")
    # Returns: "temporary", "permanent", or "unknown"
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.tools.verification.protocols import (
    ClaimIssue,
    TemporalNature,
)

logger = logging.getLogger(__name__)


@dataclass
class FileHistory:
    """Git history information for a file.

    Attributes:
        file_path: Path to the file
        exists: Whether file exists
        age_days: Age in days since first commit
        last_modified_days: Days since last modification
        modification_count: Number of commits touching this file
        recent_changes: Number of changes in last 30 days
        is_wip: Whether file appears to be work-in-progress
    """

    file_path: str
    exists: bool
    age_days: int = 0
    last_modified_days: int = 0
    modification_count: int = 0
    recent_changes: int = 0
    is_wip: bool = False


@dataclass
class TemporalMarkers:
    """Temporal markers found in code comments.

    Attributes:
        has_todo: TODO comments found
        has_fixme: FIXME comments found
        has_temporary: Explicit temporary markers found
        has_removal_plan: Documented removal plan found
        target_dates: Any target dates mentioned
    """

    has_todo: bool = False
    has_fixme: bool = False
    has_temporary: bool = False
    has_removal_plan: bool = False
    target_dates: List[str] = None

    def __post_init__(self):
        if self.target_dates is None:
            self.target_dates = []


class GitHistoryAnalyzer:
    """Analyzes git history for temporal context."""

    # Patterns for temporal markers in comments
    TEMPORARY_PATTERNS = [
        r"(?i)temporary",
        r"(?i)temp\s*hack",
        r"(?i)quick\s*fix",
        r"(?i)work\s*in\s*progress",
        r"(?i)wip",
        r"(?i)to\s*do\s*later",
        r"(?i)refactor\s*soon",
        r"(?i)cleanup\s*needed",
    ]

    REMOVAL_PLAN_PATTERNS = [
        r"(?i)remove\s+after",
        r"(?i)deprecate\s+after",
        r"(?i)delete\s+when",
        r"(?i)legacy\s+code",
        r"(?i)compatibility\s+shim",
        r"(?i)migration\s+aid",
        r"(?i)will\s+be\s+removed",
    ]

    TODO_PATTERN = re.compile(r"TODO[:\s]*(.+)", re.IGNORECASE)
    FIXME_PATTERN = re.compile(r"FIXME[:\s]*(.+)", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}|v\d+\.\d+)")

    def __init__(self, project_root: Path):
        """Initialize git history analyzer.

        Args:
            project_root: Root directory of the git repository
        """
        self._project_root = Path(project_root)
        self._is_git_repo = self._check_git_repo()
        self._cache: Dict[str, FileHistory] = {}

    def _check_git_repo(self) -> bool:
        """Check if project root is in a git repository.

        Returns:
            True if in a git repository
        """
        git_dir = self._project_root / ".git"
        return git_dir.exists()

    def _run_git(self, args: List[str]) -> str:
        """Run a git command and return output.

        Args:
            args: Git command arguments

        Returns:
            Command output as string
        """
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("Git command failed: %s", e)
            return ""

    def get_file_history(self, file_path: str) -> FileHistory:
        """Get git history for a file.

        Args:
            file_path: Relative path to file

        Returns:
            FileHistory object with temporal information
        """
        if not self._is_git_repo:
            return FileHistory(file_path=file_path, exists=False)

        if file_path in self._cache:
            return self._cache[file_path]

        full_path = self._project_root / file_path
        exists = full_path.exists()

        if not exists:
            history = FileHistory(file_path=file_path, exists=False)
            self._cache[file_path] = history
            return history

        # Get first commit date (file age)
        first_date_str = self._run_git(["log", "--diff-filter=A", "--format=%ct", "--", file_path])
        age_days = 0
        if first_date_str:
            try:
                first_timestamp = int(first_date_str.split()[0])
                age_days = (datetime.now().timestamp() - first_timestamp) // 86400
            except (ValueError, IndexError):
                pass

        # Get last modification date
        last_date_str = self._run_git(["log", "-1", "--format=%ct", "--", file_path])
        last_modified_days = 0
        if last_date_str:
            try:
                last_timestamp = int(last_date_str)
                last_modified_days = (datetime.now().timestamp() - last_timestamp) // 86400
            except ValueError:
                pass

        # Get modification count
        count_str = self._run_git(["rev-list", "--count", "HEAD", "--", file_path])
        modification_count = int(count_str) if count_str.isdigit() else 0

        # Get recent changes (last 30 days)
        since_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
        recent_str = self._run_git(
            [
                "rev-list",
                "--count",
                f"--since={since_timestamp}",
                "HEAD",
                "--",
                file_path,
            ]
        )
        recent_changes = int(recent_str) if recent_str.isdigit() else 0

        # Check if WIP (recently modified with many changes)
        is_wip = (last_modified_days < 7 and recent_changes >= 3) or modification_count == 1

        history = FileHistory(
            file_path=file_path,
            exists=True,
            age_days=age_days,
            last_modified_days=last_modified_days,
            modification_count=modification_count,
            recent_changes=recent_changes,
            is_wip=is_wip,
        )

        self._cache[file_path] = history
        return history

    def find_temporal_markers(self, file_path: str) -> TemporalMarkers:
        """Find temporal markers in file comments.

        Args:
            file_path: Relative path to file

        Returns:
            TemporalMarkers object with found markers
        """
        full_path = self._project_root / file_path

        if not full_path.exists():
            return TemporalMarkers()

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug("Error reading file %s: %s", file_path, e)
            return TemporalMarkers()

        markers = TemporalMarkers()

        # Check for TODO/FIXME
        if self.TODO_PATTERN.search(content):
            markers.has_todo = True

        if self.FIXME_PATTERN.search(content):
            markers.has_fixme = True

        # Check for temporary patterns
        for pattern in self.TEMPORARY_PATTERNS:
            if re.search(pattern, content):
                markers.has_temporary = True
                break

        # Check for removal plan patterns
        for pattern in self.REMOVAL_PLAN_PATTERNS:
            if re.search(pattern, content):
                markers.has_removal_plan = True
                break

        # Extract target dates from TODO/FIXME comments
        for match in self.TODO_PATTERN.finditer(content):
            text = match.group(1)
            date_match = self.DATE_PATTERN.search(text)
            if date_match:
                markers.target_dates.append(date_match.group(1))

        return markers


class TemporalContextAnalyzer:
    """Analyzes temporal context of codebase issues.

    Determines whether issues are temporary (likely to resolve soon)
    or permanent (require intentional fixes).
    """

    def __init__(self, project_root: Path):
        """Initialize temporal context analyzer.

        Args:
            project_root: Root directory of the project
        """
        self._project_root = Path(project_root)
        self._git_analyzer = GitHistoryAnalyzer(self._project_root)

    def estimate_temporal_nature(
        self,
        file_path: str,
    ) -> TemporalNature:
        """Estimate if issues in a file are temporary or permanent.

        Args:
            file_path: Path to the file to analyze

        Returns:
            TemporalNature enum value
        """
        if not self._git_analyzer._is_git_repo:
            return TemporalNature.UNKNOWN

        history = self._git_analyzer.get_file_history(file_path)
        markers = self._git_analyzer.find_temporal_markers(file_path)

        # Calculate temporary score
        temporary_score = 0.0

        # Strong indicators of temporary code
        if markers.has_removal_plan:
            temporary_score += 0.4
        if markers.has_temporary:
            temporary_score += 0.3
        if history.is_wip:
            temporary_score += 0.2

        # Weaker indicators
        if history.last_modified_days < 30 and history.recent_changes >= 2:
            temporary_score += 0.1
        if history.modification_count == 1:
            temporary_score += 0.1

        # Strong indicators of permanent code
        permanent_score = 0.0

        if history.age_days > 365:
            permanent_score += 0.3
        if history.modification_count > 10:
            permanent_score += 0.2
        if not markers.has_todo and not markers.has_fixme:
            permanent_score += 0.1

        # Classify based on scores
        if temporary_score >= 0.5:
            return TemporalNature.TEMPORARY
        elif permanent_score >= 0.4:
            return TemporalNature.PERMANENT
        else:
            return TemporalNature.UNKNOWN

    def check_for_removal_plan(self, file_path: str) -> bool:
        """Check if file has a documented removal plan.

        Args:
            file_path: Path to the file

        Returns:
            True if removal plan is documented
        """
        markers = self._git_analyzer.find_temporal_markers(file_path)
        return markers.has_removal_plan

    def get_file_age_days(self, file_path: str) -> int:
        """Get file age in days from git history.

        Args:
            file_path: Path to the file

        Returns:
            Age in days (0 if not available)
        """
        history = self._git_analyzer.get_file_history(file_path)
        return history.age_days

    def is_recently_modified(self, file_path: str, days: int = 30) -> bool:
        """Check if file was recently modified.

        Args:
            file_path: Path to the file
            days: Threshold for "recent" in days

        Returns:
            True if modified within threshold
        """
        history = self._git_analyzer.get_file_history(file_path)
        return history.last_modified_days <= days

    def analyze_issue_temporal_context(
        self,
        issue: Dict[str, Any] | ClaimIssue,
    ) -> Dict[str, Any]:
        """Analyze temporal context for an issue.

        Args:
            issue: Issue to analyze

        Returns:
            Dictionary with temporal context information
        """
        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = issue.model_dump()

        file_path = issue_dict.get("file_path", "")
        if not file_path:
            return {
                "temporal_nature": TemporalNature.UNKNOWN,
                "confidence": 0.0,
                "reason": "No file path provided",
            }

        nature = self.estimate_temporal_nature(file_path)
        removal_plan = self.check_for_removal_plan(file_path)
        age_days = self.get_file_age_days(file_path)
        is_recent = self.is_recently_modified(file_path)

        # Calculate confidence based on available signals
        confidence = 0.5  # Base confidence
        if removal_plan:
            confidence += 0.3
        if age_days > 0:
            confidence += 0.1
        if is_recent:
            confidence += 0.1

        confidence = min(confidence, 1.0)

        return {
            "temporal_nature": nature,
            "confidence": confidence,
            "file_age_days": age_days,
            "has_removal_plan": removal_plan,
            "is_recently_modified": is_recent,
            "reason": self._generate_temporal_reason(nature, age_days, removal_plan, is_recent),
        }

    def _generate_temporal_reason(
        self,
        nature: TemporalNature,
        age_days: int,
        removal_plan: bool,
        is_recent: bool,
    ) -> str:
        """Generate human-readable reason for temporal classification.

        Args:
            nature: Determined temporal nature
            age_days: File age in days
            removal_plan: Whether removal plan exists
            is_recent: Whether recently modified

        Returns:
            Human-readable explanation
        """
        if nature == TemporalNature.TEMPORARY:
            parts = ["Issue appears temporary"]
            if removal_plan:
                parts.append("(removal plan documented)")
            elif is_recent:
                parts.append("(recently modified)")
            return " ".join(parts)

        elif nature == TemporalNature.PERMANENT:
            if age_days > 365:
                return f"Issue appears permanent (file is {age_days // 365}+ years old)"
            return "Issue appears permanent (stable file with no temporary markers)"

        else:
            return "Unable to determine temporal nature (insufficient signals)"

    def batch_analyze_files(
        self,
        file_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze temporal context for multiple files.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary mapping file path to temporal context
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = self.analyze_issue_temporal_context({"file_path": file_path})
        return results

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

"""Documentation cross-reference for codebase analysis results.

This module cross-references codebase analysis findings with project
documentation to identify tracked technical debt and planned fixes.

P2 Priority: Medium impact, medium effort. Provides semantic context.

Design Patterns:
- Index Pattern: Parse and index documentation for efficient lookup
- Fuzzy Matching: Flexible matching of issue descriptions to doc entries
- Cache-Aside: Cache parsed documentation for performance

Usage:
    crossref = DocumentationCrossReference(project_root=Path("."))
    is_tracked = crossref.is_tracked_debt({
        "issue_type": "cross_layer_dependency",
        "description": "Storage module depends on Index module",
    })
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from functools import lru_cache

from victor.tools.verification.protocols import (
    ClaimIssue,
    DocumentationReference,
    VerificationContext,
)

logger = logging.getLogger(__name__)


@dataclass
class TechDebtEntry:
    """An entry in the technical debt documentation.

    Attributes:
        id: Technical debt identifier (e.g., TD-CROSS-LAYER)
        title: Title or heading
        description: Full description
        severity: Debt severity (critical/high/medium/low)
        status: Debt status (open/in_progress/closed)
        file: Source file path
        line_number: Line number in source file
        keywords: Associated keywords for matching
    """

    id: str
    title: str
    description: str
    severity: str = "medium"
    status: str = "open"
    file: str = ""
    line_number: int = 0
    keywords: Set[str] = field(default_factory=set)

    def matches_issue(self, issue: Dict[str, Any]) -> float:
        """Calculate match score between this entry and an issue.

        Args:
            issue: Issue dictionary

        Returns:
            Match score between 0.0 and 1.0
        """
        score = 0.0
        issue_text = " ".join(
            [
                issue.get("issue_type", ""),
                issue.get("description", ""),
            ]
        ).lower()

        # Exact ID match
        if self.id.lower() in issue_text:
            return 1.0

        # Keyword matches
        for keyword in self.keywords:
            if keyword.lower() in issue_text:
                score += 0.2

        # Title word matches
        title_words = set(self.title.lower().split())
        issue_words = set(issue_text.split())
        overlap = title_words & issue_words
        if overlap:
            score += len(overlap) / max(len(title_words), 1) * 0.3

        return min(score, 1.0)


@dataclass
class RoadmapEntry:
    """An entry in the roadmap documentation.

    Attributes:
        title: Feature or fix title
        description: Description
        status: Status (planned/in_progress/done)
        priority: Priority level (p0/p1/p2/p3)
        related_issues: List of related issue IDs
        file: Source file path
    """

    title: str
    description: str
    status: str = "planned"
    priority: str = "p2"
    related_issues: List[str] = field(default_factory=list)
    file: str = ""


class DocumentationParser:
    """Parser for project documentation files.

    Supports common documentation formats:
    - AsciiDoc (.adoc, .asciidoc)
    - Markdown (.md)
    - ReStructuredText (.rst)
    """

    # Patterns for technical debt markers
    TD_MARKER_PATTERN = re.compile(r"TD-[A-Z0-9-]+")
    TD_SECTION_PATTERN = re.compile(r"^==+\s*(.+?)\s*$", re.MULTILINE)
    TD_SEVERITY_PATTERN = re.compile(
        r"\[.*severity.*?(critical|high|medium|low).*?\]", re.IGNORECASE
    )
    TD_STATUS_PATTERN = re.compile(
        r"\[.*status.*?(open|in_progress|closed|done).*?\]", re.IGNORECASE
    )

    # Patterns for roadmap entries
    ROADMAP_HEADING = re.compile(r"^#+\s*(.+?)\s*$", re.MULTILINE)
    ROADMAP_PRIORITY = re.compile(r"\[.*priority.*?(p[0-3]).*?\]", re.IGNORECASE)

    def __init__(self, project_root: Path):
        """Initialize documentation parser.

        Args:
            project_root: Root directory for documentation resolution
        """
        self._project_root = project_root
        self._cache: Dict[str, List[Any]] = {}

    def parse_tech_debt_document(self, doc_path: Path) -> List[TechDebtEntry]:
        """Parse technical debt documentation.

        Args:
            doc_path: Path to the technical debt document

        Returns:
            List of technical debt entries
        """
        cache_key = f"tech_debt:{doc_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not doc_path.exists():
            logger.debug("Tech debt document not found: %s", doc_path)
            return []

        try:
            content = doc_path.read_text(encoding="utf-8")
            entries = self._extract_tech_debt_entries(content, doc_path)
            self._cache[cache_key] = entries
            return entries
        except Exception as e:
            logger.warning("Error parsing tech debt document %s: %s", doc_path, e)
            return []

    def _extract_tech_debt_entries(self, content: str, source_file: Path) -> List[TechDebtEntry]:
        """Extract technical debt entries from content.

        Args:
            content: Document content
            source_file: Source file for reference

        Returns:
            List of tech debt entries
        """
        entries = []
        lines = content.splitlines()
        current_entry: Optional[Dict[str, Any]] = None

        for i, line in enumerate(lines, 1):
            # Check for TD marker
            td_match = self.TD_MARKER_PATTERN.search(line)
            if td_match:
                if current_entry:
                    entries.append(self._finalize_entry(current_entry, lines, i))

                current_entry = {
                    "id": td_match.group(0),
                    "line": i,
                    "description": "",
                    "keywords": set(),
                }

            if current_entry:
                # Extract severity
                severity_match = self.TD_SEVERITY_PATTERN.search(line)
                if severity_match:
                    current_entry["severity"] = severity_match.group(1)

                # Extract status
                status_match = self.TD_STATUS_PATTERN.search(line)
                if status_match:
                    status = status_match.group(1)
                    current_entry["status"] = "in_progress" if status == "in_progress" else status

                # Accumulate description
                if not line.strip().startswith("//"):
                    current_entry["description"] += line + "\n"

        # Finalize last entry
        if current_entry:
            entries.append(self._finalize_entry(current_entry, lines, len(lines)))

        # Convert to TechDebtEntry objects
        return [
            TechDebtEntry(
                id=e["id"],
                title=e["id"],
                description=e["description"][:500],  # Truncate long descriptions
                severity=e.get("severity", "medium"),
                status=e.get("status", "open"),
                file=str(source_file),
                line_number=e["line"],
                keywords=e.get("keywords", set()),
            )
            for e in entries
        ]

    def _finalize_entry(
        self, entry: Dict[str, Any], lines: List[str], end_line: int
    ) -> Dict[str, Any]:
        """Finalize a tech debt entry.

        Args:
            entry: Partial entry dict
            lines: All document lines
            end_line: Line where entry ends

        Returns:
            Completed entry dict
        """
        description = entry.get("description", "").strip()

        # Extract keywords from description
        words = re.findall(r"\b[a-z]{4,}\b", description.lower())
        entry["keywords"] = set(words[:20])  # Top 20 keywords

        entry["description"] = description
        return entry

    def parse_roadmap_document(self, doc_path: Path) -> List[RoadmapEntry]:
        """Parse roadmap documentation.

        Args:
            doc_path: Path to the roadmap document

        Returns:
            List of roadmap entries
        """
        cache_key = f"roadmap:{doc_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not doc_path.exists():
            logger.debug("Roadmap document not found: %s", doc_path)
            return []

        try:
            content = doc_path.read_text(encoding="utf-8")
            entries = self._extract_roadmap_entries(content, doc_path)
            self._cache[cache_key] = entries
            return entries
        except Exception as e:
            logger.warning("Error parsing roadmap document %s: %s", doc_path, e)
            return []

    def _extract_roadmap_entries(self, content: str, source_file: Path) -> List[RoadmapEntry]:
        """Extract roadmap entries from content.

        Args:
            content: Document content
            source_file: Source file for reference

        Returns:
            List of roadmap entries
        """
        entries = []

        for match in self.ROADMAP_HEADING.finditer(content):
            title = match.group(1).strip()

            # Skip very short headings (likely not features)
            if len(title) < 5:
                continue

            # Extract priority from title first
            priority_match = self.ROADMAP_PRIORITY.search(title)
            priority = priority_match.group(1).lower() if priority_match else "p2"

            # Also check for priority in text immediately after heading
            section_end = match.end()
            next_heading = self.ROADMAP_HEADING.search(content[section_end:])
            section_text = content[
                section_end : section_end + (next_heading.start() if next_heading else 500)
            ]

            # Priority after heading overrides title priority
            after_priority = self.ROADMAP_PRIORITY.search(section_text[:200])
            if after_priority:
                priority = after_priority.group(1).lower()

            # Extract description
            description = section_text.strip()[:500]

            entries.append(
                RoadmapEntry(
                    title=title,
                    description=description,
                    priority=priority,
                    file=str(source_file),
                )
            )

        return entries


class DocumentationCrossReference:
    """Cross-references analysis results with project documentation.

    Checks if issues are tracked as technical debt or addressed in roadmap.

    Attributes:
        project_root: Root directory for documentation resolution
        tech_debt_doc: Path to technical debt documentation
        roadmap_doc: Path to roadmap documentation
    """

    DEFAULT_TECH_DEBT_PATHS = [
        "docs/10-quality/TECHNICAL_DEBT.adoc",
        "docs/TECHNICAL_DEBT.adoc",
        "TECHNICAL_DEBT.adoc",
        "docs/technical_debt.md",
        "TECHNICAL_DEBT.md",
    ]

    DEFAULT_ROADMAP_PATHS = [
        "docs/_internal/roadmap.md",
        "docs/roadmap.md",
        "ROADMAP.md",
    ]

    def __init__(
        self,
        project_root: Path,
        tech_debt_doc: Optional[Path] = None,
        roadmap_doc: Optional[Path] = None,
    ):
        """Initialize documentation cross-reference.

        Args:
            project_root: Root directory for documentation resolution
            tech_debt_doc: Optional path to tech debt documentation
            roadmap_doc: Optional path to roadmap documentation
        """
        self._project_root = Path(project_root)
        self._parser = DocumentationParser(self._project_root)

        # Resolve tech debt document path
        if tech_debt_doc:
            self._tech_debt_doc = self._project_root / tech_debt_doc
        else:
            self._tech_debt_doc = self._find_document(self.DEFAULT_TECH_DEBT_PATHS)

        # Resolve roadmap document path
        if roadmap_doc:
            self._roadmap_doc = self._project_root / roadmap_doc
        else:
            self._roadmap_doc = self._find_document(self.DEFAULT_ROADMAP_PATHS)

        # Parse documentation
        self._tech_debt_entries: List[TechDebtEntry] = []
        self._roadmap_entries: List[RoadmapEntry] = []

        self._load_documentation()

    def _find_document(self, candidate_paths: List[str]) -> Optional[Path]:
        """Find first existing document from candidate paths.

        Args:
            candidate_paths: List of relative paths to try

        Returns:
            Resolved path or None
        """
        for rel_path in candidate_paths:
            full_path = self._project_root / rel_path
            if full_path.exists():
                return full_path
        return None

    def _load_documentation(self) -> None:
        """Load and parse all documentation files."""
        if self._tech_debt_doc:
            self._tech_debt_entries = self._parser.parse_tech_debt_document(self._tech_debt_doc)
            logger.debug(
                "Loaded %d tech debt entries from %s",
                len(self._tech_debt_entries),
                self._tech_debt_doc,
            )

        if self._roadmap_doc:
            self._roadmap_entries = self._parser.parse_roadmap_document(self._roadmap_doc)
            logger.debug(
                "Loaded %d roadmap entries from %s",
                len(self._roadmap_entries),
                self._roadmap_doc,
            )

    def is_tracked_debt(self, issue: Dict[str, Any] | ClaimIssue) -> bool:
        """Check if issue is tracked as technical debt.

        Args:
            issue: Issue to check

        Returns:
            True if issue matches a tech debt entry
        """
        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = issue.model_dump()

        for entry in self._tech_debt_entries:
            if entry.matches_issue(issue_dict) > 0.5:
                return True

        return False

    def check_roadmap_alignment(self, issue: Dict[str, Any] | ClaimIssue) -> bool:
        """Check if issue is addressed in roadmap.

        Args:
            issue: Issue to check

        Returns:
            True if issue matches a roadmap entry
        """
        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = issue.model_dump()

        issue_text = " ".join(
            [
                issue_dict.get("issue_type", ""),
                issue_dict.get("description", ""),
            ]
        ).lower()

        for entry in self._roadmap_entries:
            entry_text = f"{entry.title} {entry.description}".lower()

            # Check for keyword overlap
            issue_words = set(issue_text.split())
            entry_words = set(entry_text.split())
            overlap = issue_words & entry_words

            if len(overlap) >= 3:  # Threshold for match
                return True

        return False

    def get_doc_references(
        self, issue: Dict[str, Any] | ClaimIssue
    ) -> List[DocumentationReference]:
        """Get all documentation references for an issue.

        Args:
            issue: Issue to look up

        Returns:
            List of documentation references
        """
        references = []

        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = issue.model_dump()

        # Check tech debt entries
        for entry in self._tech_debt_entries:
            score = entry.matches_issue(issue_dict)
            if score > 0.3:
                references.append(
                    DocumentationReference(
                        doc_type="TECHNICAL_DEBT",
                        reference_id=entry.id,
                        title=entry.title,
                        relevant_section=entry.description[:200],
                        file_path=entry.file,
                    )
                )

        # Check roadmap entries
        issue_text = " ".join(
            [
                issue_dict.get("issue_type", ""),
                issue_dict.get("description", ""),
            ]
        ).lower()

        for entry in self._roadmap_entries:
            entry_text = f"{entry.title} {entry.description}".lower()
            if any(word in entry_text for word in issue_text.split() if len(word) > 4):
                references.append(
                    DocumentationReference(
                        doc_type="ROADMAP",
                        reference_id=None,
                        title=entry.title,
                        relevant_section=entry.description[:200],
                        file_path=entry.file,
                    )
                )

        return references

    def get_tech_debt_markers(self) -> List[str]:
        """Get all technical debt marker IDs.

        Returns:
            List of TD-* marker IDs
        """
        return [entry.id for entry in self._tech_debt_entries]

    def get_roadmap_priorities(self) -> Dict[str, int]:
        """Get count of roadmap items by priority.

        Returns:
            Dictionary mapping priority to count
        """
        priorities: Dict[str, int] = {}
        for entry in self._roadmap_entries:
            priorities[entry.priority] = priorities.get(entry.priority, 0) + 1
        return priorities

    def reload_documentation(self) -> None:
        """Reload documentation from files.

        Call after documentation changes.
        """
        self._parser._cache.clear()
        self._load_documentation()

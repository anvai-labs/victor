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

"""FEP manager for parsing, validating, and managing Framework Enhancement Proposals.

This module provides high-level management functions for working with FEPs:
- Parsing FEP markdown files
- Validating FEP format and content
- Tracking FEP status
- Generating FEP listings
- Managing FEP lifecycle
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from victor.feps.schema import (
    FEPMetadata,
    FEPSection,
    FEPStatus,
    FEPType,
    FEPValidator,
    FEPValidationResult,
    parse_fep_metadata,
)


class FEPManager:
    """Manager for Framework Enhancement Proposals.

    Provides high-level operations for managing FEPs including:
    - Discovering and listing FEPs
    - Validating FEP structure and content
    - Updating FEP status
    - Generating FEP statistics and reports
    """

    def __init__(self, feps_dir: Optional[Path] = None):
        """Initialize FEP manager.

        Args:
            feps_dir: Path to FEPs directory (default: feps/ in repo root)
        """
        self.validator = FEPValidator(feps_dir=feps_dir)
        self.feps_dir = self.validator.feps_dir

    def list_feps(
        self,
        status_filter: Optional[FEPStatus] = None,
        type_filter: Optional[FEPType] = None,
        sort_by: str = "number",
    ) -> List[FEPMetadata]:
        """List all FEPs with optional filtering and sorting.

        Args:
            status_filter: Optional status filter (Draft, Review, Accepted, etc.)
            type_filter: Optional type filter (Standards Track, Informational, Process)
            sort_by: Sort field - "number", "created", "modified", "title"

        Returns:
            List of FEP metadata objects
        """
        feps = self.validator.list_feps(
            status_filter=status_filter,
            type_filter=type_filter,
        )

        # Apply sorting
        if sort_by == "number":
            feps.sort(key=lambda f: f.fep)
        elif sort_by == "created":
            feps.sort(key=lambda f: f.created, reverse=True)
        elif sort_by == "modified":
            feps.sort(key=lambda f: f.modified, reverse=True)
        elif sort_by == "title":
            feps.sort(key=lambda f: f.title.lower())

        return feps

    def get_fep(self, fep_number: int) -> Optional[FEPMetadata]:
        """Get FEP by number.

        Args:
            fep_number: FEP number

        Returns:
            FEPMetadata or None if not found
        """
        return self.validator.get_fep(fep_number)

    def get_fep_content(self, fep_number: int) -> Optional[str]:
        """Get full FEP markdown content.

        Args:
            fep_number: FEP number

        Returns:
            Markdown content or None if not found
        """
        if not self.feps_dir:
            return None

        # Find matching file
        matches = list(self.feps_dir.glob(f"fep-{fep_number:04d}-*.md"))
        if not matches:
            return None

        try:
            return matches[0].read_text(encoding="utf-8")
        except Exception:
            return None

    def get_fep_sections(self, fep_number: int) -> Optional[Dict[str, FEPSection]]:
        """Extract and return FEP sections.

        Args:
            fep_number: FEP number

        Returns:
            Dictionary of section name to FEPSection, or None if not found
        """
        content = self.get_fep_content(fep_number)
        if not content:
            return None

        # Remove frontmatter
        content_without_frontmatter = self.validator._remove_frontmatter(content)

        # Extract sections
        sections: Dict[str, FEPSection] = {}
        self.validator._extract_and_validate_sections(content_without_frontmatter, sections)

        return sections

    def validate_fep(self, fep_number: int) -> Optional[FEPValidationResult]:
        """Validate a FEP by number.

        Args:
            fep_number: FEP number

        Returns:
            FEPValidationResult or None if FEP not found
        """
        if not self.feps_dir:
            return None

        # Find matching file
        matches = list(self.feps_dir.glob(f"fep-{fep_number:04d}-*.md"))
        if not matches:
            return None

        return self.validator.validate_file(matches[0])

    def update_fep_status(self, fep_number: int, new_status: FEPStatus) -> Tuple[bool, str]:
        """Update FEP status.

        Args:
            fep_number: FEP number
            new_status: New status to set

        Returns:
            Tuple of (success, message)
        """
        if not self.feps_dir:
            return False, "FEPs directory not configured"

        # Find matching file
        matches = list(self.feps_dir.glob(f"fep-{fep_number:04d}-*.md"))
        if not matches:
            return False, f"FEP-{fep_number:04d} not found"

        fep_file = matches[0]

        try:
            # Read content
            content = fep_file.read_text(encoding="utf-8")

            # Update status in frontmatter
            # Pattern: status: CurrentStatus
            status_pattern = r"^status:\s*\S+"
            new_status_line = f"status: {new_status.value}"

            new_content = re.sub(
                status_pattern, new_status_line, content, count=1, flags=re.MULTILINE
            )

            # Update modified date
            today = datetime.now().strftime("%Y-%m-%d")
            modified_pattern = r"^modified:\s*\d{4}-\d{2}-\d{2}"
            new_modified_line = f"modified: {today}"
            new_content = re.sub(
                modified_pattern,
                new_modified_line,
                new_content,
                count=1,
                flags=re.MULTILINE,
            )

            # Write back
            fep_file.write_text(new_content, encoding="utf-8")

            return True, f"FEP-{fep_number:04d} status updated to {new_status.value}"

        except Exception as e:
            return False, f"Failed to update status: {e}"

    def get_statistics(self) -> Dict[str, any]:
        """Get FEP statistics.

        Returns:
            Dictionary with statistics:
            - total: Total number of FEPs
            - by_status: Count by status
            - by_type: Count by type
            - recent: Recently modified FEPs
        """
        all_feps = self.list_feps()

        # Count by status
        by_status: Dict[str, int] = {}
        for status in FEPStatus:
            by_status[status.value] = 0

        # Count by type
        by_type: Dict[str, int] = {}
        for fep_type in FEPType:
            by_type[fep_type.value] = 0

        for fep in all_feps:
            by_status[fep.status.value] += 1
            by_type[fep.type.value] += 1

        # Get recently modified (last 5)
        recent = sorted(all_feps, key=lambda f: f.modified, reverse=True)[:5]

        return {
            "total": len(all_feps),
            "by_status": by_status,
            "by_type": by_type,
            "recent": recent,
        }

    def find_next_number(self) -> int:
        """Find the next available FEP number.

        Returns:
            Next available FEP number
        """
        all_feps = self.list_feps()

        if not all_feps:
            return 1

        # Find highest number
        max_number = max(fep.fep for fep in all_feps)
        return max_number + 1

    def search_feps(
        self,
        query: str,
        search_in: Optional[List[str]] = None,
    ) -> List[FEPMetadata]:
        """Search FEPs by query string.

        Args:
            query: Search query
            search_in: List of fields to search - default: ["title", "authors"]

        Returns:
            List of matching FEPs
        """
        if search_in is None:
            search_in = ["title", "authors"]

        all_feps = self.list_feps()
        query_lower = query.lower()
        results = []

        for fep in all_feps:
            # Search title
            if "title" in search_in and query_lower in fep.title.lower():
                results.append(fep)
                continue

            # Search authors
            if "authors" in search_in:
                for author in fep.authors:
                    if query_lower in author.get("name", "").lower():
                        results.append(fep)
                        break

        return results

    def get_fep_summary(self, fep_number: int) -> Optional[str]:
        """Get a brief summary of a FEP.

        Args:
            fep_number: FEP number

        Returns:
            Summary string or None if not found
        """
        metadata = self.get_fep(fep_number)
        if not metadata:
            return None

        sections = self.get_fep_sections(fep_number)
        summary_section = sections.get("Summary") if sections else None

        summary = f"FEP-{metadata.fep:04d}: {metadata.title}\n"
        summary += f"Type: {metadata.type.value}\n"
        summary += f"Status: {metadata.status.value}\n"
        summary += f"Created: {metadata.created}\n"

        if summary_section:
            # Get first paragraph of summary
            first_para = summary_section.content.split("\n\n")[0]
            summary += f"\n{first_para}"

        return summary


def create_fep_manager(feps_dir: Optional[Path] = None) -> FEPManager:
    """Create an FEP manager instance.

    Convenience function for creating an FEPManager.

    Args:
        feps_dir: Optional path to FEPs directory

    Returns:
        FEPManager instance
    """
    return FEPManager(feps_dir=feps_dir)


__all__ = [
    "FEPManager",
    "create_fep_manager",
]

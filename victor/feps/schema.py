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

"""FEP schema and validation.

Defines the schema for Framework Enhancement Proposals (FEPs) and
provides validation logic.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class FEPType(str, Enum):
    """FEP types."""

    STANDARDS = "Standards Track"
    INFORMATIONAL = "Informational"
    PROCESS = "Process"


class FEPStatus(str, Enum):
    """FEP status states."""

    DRAFT = "Draft"
    REVIEW = "Review"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"
    DEFERRED = "Deferred"
    WITHDRAWN = "Withdrawn"
    IMPLEMENTED = "Implemented"


@dataclass
class FEPMetadata:
    """FEP metadata from YAML frontmatter."""

    fep: int
    title: str
    type: FEPType
    status: FEPStatus
    created: str
    modified: str
    authors: List[Dict[str, str]] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    discussion: Optional[str] = None
    implementation: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        return f"FEP-{self.fep:04d}: {self.title}"


@dataclass
class FEPSection:
    """FEP section information."""

    name: str
    content: str
    word_count: int
    required: bool


@dataclass
class FEPValidationError:
    """FEP validation error or warning."""

    severity: str  # "error" or "warning"
    section: str
    message: str

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.severity.upper()}] {self.section}: {self.message}"


@dataclass
class FEPValidationResult:
    """FEP validation result."""

    is_valid: bool
    metadata: Optional[FEPMetadata]
    sections: Dict[str, FEPSection]
    errors: List[FEPValidationError]
    warnings: List[FEPValidationError]

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0

    def print_summary(self) -> str:
        """Print validation summary."""
        lines = [
            "FEP Validation Summary",
            "=" * 50,
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]

        if self.metadata:
            lines.extend(
                [
                    "",
                    "Metadata:",
                    f"  FEP: {self.metadata.fep:04d}",
                    f"  Title: {self.metadata.title}",
                    f"  Type: {self.metadata.type.value}",
                    f"  Status: {self.metadata.status.value}",
                ]
            )

        if self.errors:
            lines.extend(["", "Errors:"])
            for error in self.errors:
                lines.append(f"  {error}")

        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings:
                lines.append(f"  {warning}")

        return "\n".join(lines)


class FEPValidator:
    """FEP validator.

    Validates FEP structure, metadata, and section content.
    """

    # Required sections for all FEPs
    REQUIRED_SECTIONS = {
        "Summary",
        "Motivation",
        "Proposed Change",
        "Benefits",
        "Drawbacks and Alternatives",
        "Unresolved Questions",
        "Implementation Plan",
        "Migration Path",
        "Compatibility",
        "References",
    }

    # Minimum word counts for sections
    MIN_WORD_COUNTS = {
        "Summary": 150,
        "Motivation": 100,
        "Proposed Change": 200,
        "Benefits": 100,
        "Drawbacks and Alternatives": 100,
        "Implementation Plan": 100,
    }

    def __init__(self, feps_dir: Optional[Path] = None):
        """Initialize FEP validator.

        Args:
            feps_dir: Path to FEPs directory (default: feps/ in repo root)
        """
        if feps_dir is None:
            # Try to find feps directory
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                candidate = parent / "feps"
                if candidate.exists() and candidate.is_dir():
                    feps_dir = candidate
                    break

        self.feps_dir = Path(feps_dir) if feps_dir else None

    def validate_file(self, fep_path: Path) -> FEPValidationResult:
        """Validate a FEP file.

        Args:
            fep_path: Path to FEP markdown file

        Returns:
            FEPValidationResult with validation details
        """
        errors: List[FEPValidationError] = []
        warnings: List[FEPValidationError] = []
        sections: Dict[str, FEPSection] = {}

        # Check file exists
        if not fep_path.exists():
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="file",
                    message=f"FEP file not found: {fep_path}",
                )
            )
            return FEPValidationResult(
                is_valid=False,
                metadata=None,
                sections=sections,
                errors=errors,
                warnings=warnings,
            )

        # Read file content
        try:
            content = fep_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="file",
                    message=f"Failed to read file: {e}",
                )
            )
            return FEPValidationResult(
                is_valid=False,
                metadata=None,
                sections=sections,
                errors=errors,
                warnings=warnings,
            )

        # Parse YAML frontmatter
        try:
            metadata = parse_fep_metadata(content)
        except ValueError as e:
            errors.append(FEPValidationError(severity="error", section="metadata", message=str(e)))
            return FEPValidationResult(
                is_valid=False,
                metadata=None,
                sections=sections,
                errors=errors,
                warnings=warnings,
            )

        # Validate metadata
        metadata_errors = self._validate_metadata(metadata)
        errors.extend(metadata_errors)

        # Extract and validate sections
        content_without_frontmatter = self._remove_frontmatter(content)
        section_errors, section_warnings = self._extract_and_validate_sections(
            content_without_frontmatter, sections
        )
        errors.extend(section_errors)
        warnings.extend(section_warnings)

        # Check for required sections
        present_sections = set(sections.keys())
        missing_sections = self.REQUIRED_SECTIONS - present_sections
        for missing in missing_sections:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="structure",
                    message=f"Missing required section: {missing}",
                )
            )

        is_valid = len(errors) == 0

        return FEPValidationResult(
            is_valid=is_valid,
            metadata=metadata,
            sections=sections,
            errors=errors,
            warnings=warnings,
        )

    def _validate_metadata(self, metadata: FEPMetadata) -> List[FEPValidationError]:
        """Validate FEP metadata.

        Args:
            metadata: Parsed FEPMetadata

        Returns:
            List of validation errors
        """
        errors: List[FEPValidationError] = []

        # Validate FEP number (allow 0 for placeholder/draft FEPs)
        if metadata.fep < 0:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message="FEP number must be >= 0",
                )
            )

        # Validate title
        if not metadata.title or len(metadata.title.strip()) == 0:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message="Title cannot be empty",
                )
            )

        # Validate type
        if metadata.type not in FEPType:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message=f"Invalid type: {metadata.type}. Must be one of: {[t.value for t in FEPType]}",
                )
            )

        # Validate status
        if metadata.status not in FEPStatus:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message=f"Invalid status: {metadata.status}. Must be one of: {[s.value for s in FEPStatus]}",
                )
            )

        # Validate dates
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(date_pattern, metadata.created):
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message=f"Invalid created date format: {metadata.created}. Use YYYY-MM-DD",
                )
            )

        if not re.match(date_pattern, metadata.modified):
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message=f"Invalid modified date format: {metadata.modified}. Use YYYY-MM-DD",
                )
            )

        # Validate authors
        if not metadata.authors:
            errors.append(
                FEPValidationError(
                    severity="error",
                    section="metadata",
                    message="At least one author required",
                )
            )

        return errors

    def _extract_and_validate_sections(
        self,
        content: str,
        sections: Dict[str, FEPSection],
    ) -> tuple[List[FEPValidationError], List[FEPValidationError]]:
        """Extract and validate FEP sections.

        Args:
            content: Markdown content (without frontmatter)
            sections: Dictionary to populate with sections

        Returns:
            Tuple of (errors, warnings)
        """
        errors: List[FEPValidationError] = []
        warnings: List[FEPValidationError] = []

        # Split by markdown headers (##)
        # Pattern: ## Section Name
        section_pattern = re.compile(r"^##\s+(.+?)$", re.MULTILINE)

        # Find all section headers
        matches = list(section_pattern.finditer(content))

        for i, match in enumerate(matches):
            section_name = match.group(1).strip()

            # Get section content
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            # Count words
            word_count = len(section_content.split())

            # Create section
            sections[section_name] = FEPSection(
                name=section_name,
                content=section_content,
                word_count=word_count,
                required=section_name in self.REQUIRED_SECTIONS,
            )

            # Check minimum word count
            if section_name in self.MIN_WORD_COUNTS:
                min_words = self.MIN_WORD_COUNTS[section_name]
                if word_count < min_words:
                    warnings.append(
                        FEPValidationError(
                            severity="warning",
                            section=section_name,
                            message=f"Section too short: {word_count} words (minimum {min_words} recommended)",
                        )
                    )

        return errors, warnings

    def _remove_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from content.

        Args:
            content: Full markdown content

        Returns:
            Content without frontmatter
        """
        # Check for YAML frontmatter markers
        if content.startswith("---"):
            # Find the closing ---
            parts = content.split("---", 2)
            if len(parts) >= 3:
                # Return content after second ---
                return parts[2].lstrip()
            else:
                # Only one --- found, return everything after it
                return parts[1].lstrip()

        return content

    def list_feps(
        self,
        status_filter: Optional[FEPStatus] = None,
        type_filter: Optional[FEPType] = None,
    ) -> List[FEPMetadata]:
        """List all FEPs in the feps directory.

        Args:
            status_filter: Optional status filter
            type_filter: Optional type filter

        Returns:
            List of FEP metadata
        """
        if not self.feps_dir or not self.feps_dir.exists():
            return []

        feps = []

        for fep_file in sorted(self.feps_dir.glob("fep-*.md")):
            try:
                metadata = parse_fep_metadata(fep_file.read_text(encoding="utf-8"))

                # Apply filters
                if status_filter and metadata.status != status_filter:
                    continue
                if type_filter and metadata.type != type_filter:
                    continue

                feps.append(metadata)
            except Exception:
                # Skip files that can't be parsed (e.g., old format, template, etc.)
                continue

        return sorted(feps, key=lambda f: f.fep)

    def get_fep(self, fep_number: int) -> Optional[FEPMetadata]:
        """Get FEP by number.

        Args:
            fep_number: FEP number

        Returns:
            FEPMetadata or None if not found
        """
        if not self.feps_dir or not self.feps_dir.exists():
            return None

        # Find matching file
        matches = list(self.feps_dir.glob(f"fep-{fep_number:04d}-*.md"))
        if not matches:
            return None

        try:
            return parse_fep_metadata(matches[0].read_text(encoding="utf-8"))
        except Exception:
            return None


def parse_fep_metadata(content: str) -> FEPMetadata:
    """Parse FEP metadata from YAML frontmatter.

    Args:
        content: FEP markdown content

    Returns:
        FEPMetadata

    Raises:
        ValueError: If frontmatter is missing or invalid
    """
    if not content.startswith("---"):
        raise ValueError("Missing YAML frontmatter (must start with ---)")

    # Find the closing ---
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Invalid YAML frontmatter (missing closing ---)")

    frontmatter_text = parts[1].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}")

    if not frontmatter or not isinstance(frontmatter, dict):
        raise ValueError("Frontmatter is empty or not a dictionary")

    # Extract required fields
    try:
        fep_value = frontmatter.get("fep", 0)
        # Allow placeholder values (XXXX, 0000, etc.)
        if isinstance(fep_value, str):
            if fep_value.isdigit():
                fep = int(fep_value)
            else:
                # Placeholder value - use 0 as temporary
                fep = 0
        else:
            fep = int(fep_value)

        title = frontmatter.get("title", "")
        type_str = frontmatter.get("type", "Standards Track")
        status_str = frontmatter.get("status", "Draft")
        created = frontmatter.get("created", "")
        modified = frontmatter.get("modified", "")

        # Parse type
        try:
            fep_type = FEPType(type_str)
        except ValueError:
            raise ValueError(f"Invalid FEP type: {type_str}")

        # Parse status
        try:
            fep_status = FEPStatus(status_str)
        except ValueError:
            raise ValueError(f"Invalid FEP status: {status_str}")

        # Parse authors
        authors_list = frontmatter.get("authors", [])
        if isinstance(authors_list, str):
            authors_list = [{"name": authors_list}]
        elif not isinstance(authors_list, list):
            authors_list = []

        # Parse reviewers
        reviewers_list = frontmatter.get("reviewers", [])
        if isinstance(reviewers_list, str):
            reviewers_list = [reviewers_list]
        elif not isinstance(reviewers_list, list):
            reviewers_list = []

        # Convert dates to strings if they're date objects
        from datetime import date

        if isinstance(created, date):
            created = created.isoformat()
        if isinstance(modified, date):
            modified = modified.isoformat()

        return FEPMetadata(
            fep=fep,
            title=title,
            type=fep_type,
            status=fep_status,
            created=created,
            modified=modified,
            authors=authors_list,
            reviewers=reviewers_list,
            discussion=frontmatter.get("discussion"),
            implementation=frontmatter.get("implementation"),
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in frontmatter: {e}")


def validate_fep(fep_path: Path, feps_dir: Optional[Path] = None) -> FEPValidationResult:
    """Validate a FEP file.

    Convenience function that creates a validator and validates.

    Args:
        fep_path: Path to FEP markdown file
        feps_dir: Optional path to FEPs directory

    Returns:
        FEPValidationResult
    """
    validator = FEPValidator(feps_dir=feps_dir)
    return validator.validate_file(fep_path)

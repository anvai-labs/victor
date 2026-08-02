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

"""Verification protocols and data models for enhanced codebase analysis.

This module provides the data structures and protocols used across
the verification module for semantic validation, false positive detection,
documentation cross-referencing, temporal analysis, and severity weighting.

Design Patterns:
- Result Objects: Immutable dataclasses for verification results
- Protocol-based interfaces: Extensible verification strategies
- Fluent Builder: EnhancedClaimResult.Builder for complex result construction

Usage:
    result = EnhancedClaimResult(
        is_grounded=True,
        confidence=0.9,
        evidence={"file": "src/lib.rs", "line": 42},
        severity="high",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Severity levels for verification issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def weight(self) -> int:
        """Return weight for sorting (higher = more severe)."""
        weights = {
            SeverityLevel.CRITICAL: 5,
            SeverityLevel.HIGH: 4,
            SeverityLevel.MEDIUM: 3,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1,
        }
        return weights[self]


class TemporalNature(str, Enum):
    """Temporal nature of verification issues."""

    TEMPORARY = "temporary"  # Likely to be resolved soon (migration shim, WIP)
    PERMANENT = "permanent"  # Long-standing issue requiring intentional fix
    UNKNOWN = "unknown"  # Cannot determine temporal nature


class IssueCategory(str, Enum):
    """Categories of verification issues."""

    ARCHITECTURAL = "architectural"  # Cross-layer dependencies, coupling
    CODE_QUALITY = "code_quality"  # Style, complexity, duplication
    PERFORMANCE = "performance"  # Compilation time, runtime overhead
    SECURITY = "security"  # Vulnerabilities, unsafe patterns
    MAINTAINABILITY = "maintainability"  # Documentation, modularity
    COMPATIBILITY = "compatibility"  # Shims, re-exports, migration aids
    TESTING = "testing"  # Test-specific patterns
    DOCUMENTATION = "documentation"  # Missing or outdated docs


@dataclass(frozen=True)
class Evidence:
    """Structured evidence for a verification claim.

    Attributes:
        source_file: Path to the source file containing evidence
        line_number: Line number where evidence was found
        snippet: Code snippet or text excerpt
        context: Additional context around the evidence
        hash_value: Optional hash for fingerprinting
    """

    source_file: Optional[str] = None
    line_number: Optional[int] = None
    snippet: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    hash_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_file": self.source_file,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "context": self.context,
            "hash_value": self.hash_value,
        }


@dataclass(frozen=True)
class FalsePositiveResult:
    """Result of false positive analysis.

    Attributes:
        is_likely_false_positive: Whether the issue is likely a false positive
        confidence: Confidence in the false positive determination (0.0-1.0)
        reason: Human-readable explanation
        pattern_match: The pattern that matched (if any)
    """

    is_likely_false_positive: bool
    confidence: float
    reason: str
    pattern_match: Optional[str] = None


@dataclass(frozen=True)
class DocumentationReference:
    """Reference to project documentation.

    Attributes:
        doc_type: Type of documentation (TECHNICAL_DEBT, ROADMAP, etc.)
        reference_id: Identifier (e.g., TD-CROSS-LAYER)
        title: Title or heading
        relevant_section: Relevant section text
        file_path: Path to the documentation file
    """

    doc_type: str
    reference_id: Optional[str]
    title: str
    relevant_section: Optional[str]
    file_path: str


class EnhancedClaimResult(BaseModel):
    """Enhanced claim verification result with all context.

    Extends the basic ClaimVerificationResult with semantic validation,
    false positive detection, documentation cross-reference, temporal analysis,
    and severity weighting.

    Attributes:
        is_grounded: Whether the claim is verified as true
        confidence: Confidence in the verification (0.0-1.0)
        evidence: Evidence supporting or refuting the claim
        reason: Human-readable explanation
        false_positive_risk: Risk that this is a false positive (0.0-1.0)
        severity: Severity level (critical/high/medium/low/info)
        temporal_nature: Whether issue is temporary/permanent/unknown
        doc_references: References to project documentation
        category: Issue category for classification
    """

    is_grounded: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    false_positive_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    severity: Optional[SeverityLevel] = None
    temporal_nature: Optional[TemporalNature] = None
    doc_references: List[str] = Field(default_factory=list)
    category: Optional[IssueCategory] = None

    def add_evidence(self, key: str, value: Any) -> "EnhancedClaimResult":
        """Add evidence and return a new result (immutable)."""
        new_evidence = dict(self.evidence)
        new_evidence[key] = value
        return self.model_copy(update={"evidence": new_evidence})

    def with_severity(self, severity: SeverityLevel) -> "EnhancedClaimResult":
        """Return a new result with specified severity."""
        return self.model_copy(update={"severity": severity})

    def with_temporal_nature(self, nature: TemporalNature) -> "EnhancedClaimResult":
        """Return a new result with specified temporal nature."""
        return self.model_copy(update={"temporal_nature": nature})

    def adjusted_confidence(self, fp_penalty: float = 0.3) -> float:
        """Calculate confidence adjusted by false positive risk.

        Args:
            fp_penalty: Penalty multiplier for false positive risk (0.0-1.0)

        Returns:
            Adjusted confidence score accounting for false positive risk.
        """
        return self.confidence * (1.0 - (self.false_positive_risk * fp_penalty))

    class Config:
        """Pydantic config."""

        use_enum_values = False  # Keep enum objects, not values


class VerificationContext(BaseModel):
    """Context provided to verification modules.

    Attributes:
        project_root: Root directory of the project
        file_path: Current file being analyzed (if applicable)
        line_number: Current line being analyzed (if applicable)
        git_available: Whether git history is available
        documentation_paths: Paths to documentation files
        settings: Optional verification settings
    """

    project_root: Path
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    git_available: bool = True
    documentation_paths: Dict[str, Path] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)

    def get_doc_path(self, doc_type: str) -> Optional[Path]:
        """Get documentation path by type."""
        return self.documentation_paths.get(doc_type)

    def has_documentation(self, doc_type: str) -> bool:
        """Check if documentation of a type exists."""
        path = self.get_doc_path(doc_type)
        return path is not None and path.exists()


@runtime_checkable
class IVerificationStrategy(Protocol):
    """Protocol for verification strategies.

    Each strategy implements a specific type of verification analysis.
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        ...

    async def analyze(
        self,
        claim: Dict[str, Any],
        context: VerificationContext,
    ) -> Dict[str, Any]:
        """Analyze a claim and return findings.

        Args:
            claim: The claim to analyze (structured dict)
            context: Verification context

        Returns:
            Analysis results as a dictionary
        """
        ...

    def is_applicable(self, claim: Dict[str, Any]) -> bool:
        """Check if this strategy applies to the claim."""
        ...


class ClaimIssue(BaseModel):
    """A claim or issue found during codebase analysis.

    This is the input format for verification modules.

    Attributes:
        issue_type: Type of issue (e.g., "cross_layer_dependency", "global_mutable_state")
        description: Human-readable description
        file_path: Path to file where issue was found
        line_number: Line number where issue was found
        snippet: Code snippet showing the issue
        severity: Initial severity (may be adjusted)
        category: Issue category
    """

    issue_type: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    snippet: Optional[str] = None
    severity: Optional[SeverityLevel] = None
    category: Optional[IssueCategory] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

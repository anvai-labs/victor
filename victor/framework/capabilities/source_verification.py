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

"""Source verification capability for framework-level citation and reference checking.

This module provides source verification for citations and references used
in research, RAG, and data analysis verticals.

Design Pattern: Capability Provider + Strategy Pattern
- Citation validation
- Reference checking
- Source reliability scoring
- URL verification

Integration Point:
    Use in research, RAG, and dataanalysis verticals

Example:
    capability = SourceVerificationCapabilityProvider()
    result = capability.verify_citation("https://example.com/article", "Claim text")

Phase 1: Promote Generic Capabilities to Framework
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from urllib.parse import urlparse


class SourceType(Enum):
    """Types of sources."""

    WEB_PAGE = "web_page"
    ACADEMIC_PAPER = "academic_paper"
    DOCUMENTATION = "documentation"
    CODE_REPOSITORY = "code_repository"
    NEWS_ARTICLE = "news_article"
    BOOK = "book"
    OTHER = "other"


class ReliabilityLevel(Enum):
    """Reliability levels for sources."""

    HIGH = "high"  # Authoritative, peer-reviewed, official documentation
    MEDIUM = "medium"  # Generally reliable, community-maintained
    LOW = "low"  # Unverified, user-generated, potentially biased
    UNKNOWN = "unknown"  # Cannot determine reliability


@dataclass
class Citation:
    """A citation reference.

    Attributes:
        source: Source identifier (URL, DOI, etc.)
        claim: The claim or statement being cited
        context: Additional context about the citation
        source_type: Type of source
    """

    source: str
    claim: str
    context: str = ""
    source_type: SourceType = SourceType.OTHER

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "claim": self.claim,
            "context": self.context,
            "source_type": self.source_type.value,
        }


@dataclass
class VerificationResult:
    """Result of source verification.

    Attributes:
        is_valid: Whether the citation is valid
        reliability: Reliability level of the source
        confidence: Confidence score (0-1)
        issues: List of issues found
        warnings: List of warnings
        suggestions: List of suggestions for improvement
    """

    is_valid: bool
    reliability: ReliabilityLevel
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "reliability": self.reliability.value,
            "confidence": self.confidence,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


class SourceVerifier:
    """Abstract base for source verifiers."""

    def verify(self, citation: Citation) -> VerificationResult:
        """Verify a citation.

        Args:
            citation: Citation to verify

        Returns:
            VerificationResult
        """
        raise NotImplementedError


class URLVerifier(SourceVerifier):
    """Verifier for web page URLs."""

    # Reliable domain patterns
    RELIABLE_DOMAINS = {
        # Academic
        "edu",
        "ac.uk",
        "ac.jp",
        # Government
        "gov",
        "gov.uk",
        "go.jp",
        "eu.eu",
        # Official documentation
        "docs.",
        "developer.",
        "api.",
        # Reputable organizations
        "w3.org",
        "ietf.org",
        "iso.org",
    }

    # Low reliability patterns
    LOW_RELIABILITY_PATTERNS = [
        r"wiki\.",  # Wikis (except Wikipedia which is handled separately)
        r"blog\.",
        r"forum\.",
        r".*\.blogspot\.",
        r".*\.wordpress\.",
    ]

    def __init__(
        self,
        check_accessibility: bool = False,
    ):
        """Initialize the URL verifier.

        Args:
            check_accessibility: Whether to check if URL is accessible
        """
        self._check_accessibility = check_accessibility

    def verify(self, citation: Citation) -> VerificationResult:
        """Verify a URL citation.

        Args:
            citation: Citation to verify

        Returns:
            VerificationResult
        """
        issues = []
        warnings = []
        suggestions = []
        confidence = 0.5
        reliability = ReliabilityLevel.UNKNOWN

        # Parse URL
        try:
            parsed = urlparse(citation.source)
            if not parsed.scheme or not parsed.netloc:
                issues.append("Invalid URL format")
                return VerificationResult(
                    is_valid=False,
                    reliability=ReliabilityLevel.UNKNOWN,
                    confidence=0.0,
                    issues=issues,
                )
        except Exception:
            issues.append("Failed to parse URL")
            return VerificationResult(
                is_valid=False,
                reliability=ReliabilityLevel.UNKNOWN,
                confidence=0.0,
                issues=issues,
            )

        domain = parsed.netloc.lower()

        # Check for reliable domains
        if any(pattern in domain for pattern in self.RELIABLE_DOMAINS):
            reliability = ReliabilityLevel.HIGH
            confidence = 0.8
        elif "wikipedia.org" in domain:
            reliability = ReliabilityLevel.MEDIUM
            confidence = 0.7
            warnings.append("Wikipedia is a tertiary source - verify with primary sources")
        elif any(re.search(pattern, domain) for pattern in self.LOW_RELIABILITY_PATTERNS):
            reliability = ReliabilityLevel.LOW
            confidence = 0.4
            warnings.append(
                "Source may not be reliable - consider verifying with authoritative sources"
            )
        else:
            reliability = ReliabilityLevel.MEDIUM
            confidence = 0.6

        # Check for HTTPS
        if parsed.scheme != "https":
            warnings.append("URL does not use HTTPS")
            confidence -= 0.1

        # Check claim is not empty
        if not citation.claim.strip():
            issues.append("Citation claim is empty")

        # Suggestions
        if not citation.context:
            suggestions.append("Add context about what information this source provides")

        is_valid = len(issues) == 0
        confidence = max(0.0, min(1.0, confidence))

        return VerificationResult(
            is_valid=is_valid,
            reliability=reliability,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )


class DOIVerifier(SourceVerifier):
    """Verifier for academic paper DOIs."""

    def verify(self, citation: Citation) -> VerificationResult:
        """Verify a DOI citation.

        Args:
            citation: Citation to verify

        Returns:
            VerificationResult
        """
        issues = []
        warnings = []
        suggestions = []
        confidence = 0.9
        reliability = ReliabilityLevel.HIGH

        # Check DOI format
        doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        if not re.match(doi_pattern, citation.source, re.IGNORECASE):
            issues.append("Invalid DOI format")

        # Check claim
        if not citation.claim.strip():
            issues.append("Citation claim is empty")

        # Suggestions
        if not citation.context:
            suggestions.append("Add context about which parts of the paper support the claim")

        is_valid = len(issues) == 0

        return VerificationResult(
            is_valid=is_valid,
            reliability=reliability,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )


class DocumentationVerifier(SourceVerifier):
    """Verifier for documentation sources."""

    OFFICIAL_DOC_PATTERNS = [
        r"docs\.",
        r"developer\.",
        r"readthedocs\.io",
        r"\.org/documentation",
        r"\.org/docs",
    ]

    def verify(self, citation: Citation) -> VerificationResult:
        """Verify a documentation citation.

        Args:
            citation: Citation to verify

        Returns:
            VerificationResult
        """
        issues = []
        warnings = []
        suggestions = []
        confidence = 0.7
        reliability = ReliabilityLevel.MEDIUM

        # Check if it's official documentation
        is_official = any(
            re.search(pattern, citation.source.lower()) for pattern in self.OFFICIAL_DOC_PATTERNS
        )

        if is_official:
            reliability = ReliabilityLevel.HIGH
            confidence = 0.85

        # Check claim
        if not citation.claim.strip():
            issues.append("Citation claim is empty")

        # Check for version information
        if not any(v in citation.source.lower() for v in ["version", "v.", "/v", "/stable"]):
            suggestions.append("Consider including version information for documentation")

        is_valid = len(issues) == 0

        return VerificationResult(
            is_valid=is_valid,
            reliability=reliability,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )


class SourceVerificationCapabilityProvider:
    """Generic source verification capability provider.

    Provides source verification for:
    - Web search results (URL verification)
    - Citation validation
    - Reference checking
    - Source reliability scoring

    This helps ensure that research and RAG operations use reliable
    sources and provide proper citations.

    Attributes:
        custom_verifiers: Optional custom verifiers to add
    """

    def __init__(self, custom_verifiers: Optional[Dict[SourceType, SourceVerifier]] = None):
        """Initialize the source verification capability provider.

        Args:
            custom_verifiers: Optional custom verifiers
        """
        self._custom_verifiers = custom_verifiers or {}
        self._default_verifiers = {
            SourceType.WEB_PAGE: URLVerifier(),
            SourceType.ACADEMIC_PAPER: DOIVerifier(),
            SourceType.DOCUMENTATION: DocumentationVerifier(),
        }

    def verify_citation(
        self,
        source: str,
        claim: str,
        context: str = "",
        source_type: Optional[SourceType] = None,
    ) -> VerificationResult:
        """Verify a citation.

        Args:
            source: Source identifier (URL, DOI, etc.)
            claim: The claim being cited
            context: Additional context
            source_type: Type of source (auto-detected if None)

        Returns:
            VerificationResult
        """
        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self._detect_source_type(source)

        citation = Citation(
            source=source,
            claim=claim,
            context=context,
            source_type=source_type,
        )

        # Get appropriate verifier
        verifier = self._get_verifier(source_type)
        return verifier.verify(citation)

    def verify_citations(self, citations: List[Citation]) -> List[VerificationResult]:
        """Verify multiple citations.

        Args:
            citations: List of citations to verify

        Returns:
            List of VerificationResults
        """
        results = []
        for citation in citations:
            verifier = self._get_verifier(citation.source_type)
            result = verifier.verify(citation)
            results.append(result)
        return results

    def get_reliability_score(self, source: str) -> Tuple[ReliabilityLevel, float]:
        """Get reliability score for a source.

        Args:
            source: Source identifier

        Returns:
            Tuple of (reliability_level, confidence_score)
        """
        result = self.verify_citation(source, "test claim")
        return (result.reliability, result.confidence)

    def _detect_source_type(self, source: str) -> SourceType:
        """Detect source type from source string.

        Args:
            source: Source identifier

        Returns:
            Detected SourceType
        """
        source_lower = source.lower()

        # Check for DOI
        if re.match(r"10\.\d{4,9}/", source_lower):
            return SourceType.ACADEMIC_PAPER

        # Check for URL
        if source_lower.startswith(("http://", "https://")):
            # Check for documentation
            if any(
                pattern in source_lower
                for pattern in [
                    "docs.",
                    "developer.",
                    "readthedocs",
                    "/docs",
                    "/documentation",
                ]
            ):
                return SourceType.DOCUMENTATION

            # Check for code repository
            if "github.com" in source_lower or "gitlab.com" in source_lower:
                return SourceType.CODE_REPOSITORY

            return SourceType.WEB_PAGE

        return SourceType.OTHER

    def _get_verifier(self, source_type: SourceType) -> SourceVerifier:
        """Get verifier for a source type.

        Args:
            source_type: Type of source

        Returns:
            SourceVerifier instance
        """
        # Check custom verifiers first
        if source_type in self._custom_verifiers:
            return self._custom_verifiers[source_type]

        # Use default verifier
        if source_type in self._default_verifiers:
            return self._default_verifiers[source_type]

        # Fall back to URL verifier for unknown types
        return URLVerifier()

    def add_verifier(self, source_type: SourceType, verifier: SourceVerifier) -> None:
        """Add a custom verifier.

        Args:
            source_type: Type this verifier handles
            verifier: Verifier instance
        """
        self._custom_verifiers[source_type] = verifier

    def list_source_types(self) -> List[SourceType]:
        """List all available source types.

        Returns:
            List of SourceType enum values
        """
        return list(SourceType)

    def get_citation_suggestions(self, source: str, claim: str) -> List[str]:
        """Get suggestions for improving a citation.

        Args:
            source: Source identifier
            claim: The claim being cited

        Returns:
            List of suggestions
        """
        result = self.verify_citation(source, claim)
        return result.suggestions


# Pre-configured source verification for common vertical types
class SourceVerificationPresets:
    """Pre-configured source verification for common verticals."""

    @staticmethod
    def research() -> SourceVerificationCapabilityProvider:
        """Get source verification optimized for research vertical."""
        # Research uses strict URL verification
        return SourceVerificationCapabilityProvider(
            custom_verifiers={
                SourceType.WEB_PAGE: URLVerifier(check_accessibility=True),
            }
        )

    @staticmethod
    def rag() -> SourceVerificationCapabilityProvider:
        """Get source verification optimized for RAG vertical."""
        # RAG focuses on document verification
        return SourceVerificationCapabilityProvider()

    @staticmethod
    def data_analysis() -> SourceVerificationCapabilityProvider:
        """Get source verification optimized for data analysis vertical."""
        # Data analysis may cite documentation and papers
        return SourceVerificationCapabilityProvider()


__all__ = [
    "SourceVerificationCapabilityProvider",
    "SourceVerificationPresets",
    "SourceType",
    "ReliabilityLevel",
    "Citation",
    "VerificationResult",
    "URLVerifier",
    "DOIVerifier",
    "DocumentationVerifier",
]

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

"""Core claim verification module with evidence capture.

This module provides the foundation for semantic claim verification,
building on Victor's existing grounding protocols. It adds evidence
capture, confidence scoring, and structured verification results.

P0 Priority: Core verification infrastructure that other modules depend on.

Design Patterns:
- Builder Pattern: Fluent result construction
- Strategy Pattern: Pluggable verification strategies
- Composite Pattern: Multiple evidence sources

Usage:
    verifier = ClaimVerifier(project_root=Path("/path/to/project"))
    result = await verifier.verify_claim({
        "issue_type": "cross_layer_dependency",
        "file_path": "src/storage/lib.rs",
        "line_number": 42,
    })
    if result.is_grounded:
        print(f"Verified with confidence: {result.confidence}")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.tools.verification.protocols import (
    ClaimIssue,
    DocumentationReference,
    EnhancedClaimResult,
    Evidence,
    SeverityLevel,
    TemporalNature,
    VerificationContext,
)

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """Collects and structures evidence for claim verification.

    Evidence sources include:
    - File existence and content
    - Line-level code snippets
    - AST parsing results
    - Import/dependency analysis
    - Git history
    """

    def __init__(self, project_root: Path):
        """Initialize evidence collector.

        Args:
            project_root: Root directory for file resolution
        """
        self._project_root = project_root
        self._cache: Dict[str, Any] = {}

    def collect_file_evidence(self, file_path: str) -> Evidence:
        """Collect evidence from a file.

        Args:
            file_path: Relative or absolute path to file

        Returns:
            Evidence object with file information
        """
        full_path = self._project_root / file_path

        if not full_path.exists():
            return Evidence(
                source_file=file_path,
                line_number=None,
                snippet=None,
                context={"exists": False, "reason": "File not found"},
            )

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            return Evidence(
                source_file=file_path,
                line_number=None,
                snippet=None,
                context={
                    "exists": True,
                    "size_bytes": len(content),
                    "line_count": len(content.splitlines()),
                    "hash": self._hash_content(content),
                },
            )
        except Exception as e:
            return Evidence(
                source_file=file_path,
                line_number=None,
                snippet=None,
                context={"exists": True, "error": str(e)},
            )

    def collect_line_evidence(
        self, file_path: str, line_number: int, context_lines: int = 3
    ) -> Evidence:
        """Collect evidence from a specific line.

        Args:
            file_path: Path to the file
            line_number: Line number (1-indexed)
            context_lines: Number of context lines before/after

        Returns:
            Evidence with line snippet and context
        """
        file_evidence = self.collect_file_evidence(file_path)

        if not file_evidence.context.get("exists"):
            return file_evidence

        full_path = self._project_root / file_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            if 1 <= line_number <= len(lines):
                start = max(0, line_number - context_lines - 1)
                end = min(len(lines), line_number + context_lines)
                snippet = lines[line_number - 1]  # The actual line

                return Evidence(
                    source_file=file_path,
                    line_number=line_number,
                    snippet=snippet,
                    context={
                        **file_evidence.context,
                        "context_lines": lines[start:end],
                        "line_index": line_number - 1,
                    },
                )
        except Exception as e:
            logger.debug("Error collecting line evidence: %s", e)

        return file_evidence

    def collect_import_evidence(self, file_path: str) -> Evidence:
        """Collect evidence about imports in a file.

        Args:
            file_path: Path to the file

        Returns:
            Evidence with import information
        """
        file_evidence = self.collect_file_evidence(file_path)

        if not file_evidence.context.get("exists"):
            return file_evidence

        full_path = self._project_root / file_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            imports = self._extract_imports(content, full_path.suffix)

            return Evidence(
                source_file=file_path,
                line_number=None,
                snippet=None,
                context={
                    **file_evidence.context,
                    "imports": imports,
                    "import_count": len(imports),
                },
            )
        except Exception as e:
            logger.debug("Error collecting import evidence: %s", e)

        return file_evidence

    def _extract_imports(self, content: str, file_ext: str) -> List[Dict[str, Any]]:
        """Extract import statements from content.

        Args:
            content: File content
            file_ext: File extension for language detection

        Returns:
            List of import dictionaries
        """
        imports = []

        if file_ext in {".py", ".pyi"}:
            imports.extend(self._extract_python_imports(content))
        elif file_ext in {".rs", ".rust"}:
            imports.extend(self._extract_rust_imports(content))
        elif file_ext in {".ts", ".tsx", ".js", ".jsx"}:
            imports.extend(self._extract_js_imports(content))
        elif file_ext in {".go"}:
            imports.extend(self._extract_go_imports(content))

        return imports

    def _extract_python_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract Python imports."""
        imports = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(("import ", "from ")):
                imports.append({"language": "python", "statement": line})
        return imports

    def _extract_rust_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract Rust use statements."""
        imports = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("use ") and line.endswith(";"):
                imports.append({"language": "rust", "statement": line})
        return imports

    def _extract_js_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript imports."""
        imports = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(("import ", "require(")):
                imports.append({"language": "javascript", "statement": line})
        return imports

    def _extract_go_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract Go imports."""
        imports = []
        in_import_block = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ("):
                in_import_block = True
                continue
            if in_import_block:
                if stripped == ")":
                    in_import_block = False
                elif stripped and not stripped.startswith("//"):
                    imports.append({"language": "go", "statement": stripped})
            elif stripped.startswith("import "):
                imports.append({"language": "go", "statement": stripped})
        return imports

    def _hash_content(self, content: str) -> str:
        """Generate hash for content fingerprinting."""
        return hashlib.md5(content.encode()).hexdigest()[:16]


class ConfidenceScorer:
    """Calculates confidence scores for verification results.

    Confidence factors:
    - Direct evidence (file exists, line found)
    - Corroborating evidence (multiple sources agree)
    - Specificity (exact match vs. pattern match)
    - Recency (recent changes reduce confidence)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize confidence scorer.

        Args:
            weights: Custom weights for confidence factors
        """
        self._weights = weights or {
            "direct_evidence": 0.4,
            "corroboration": 0.3,
            "specificity": 0.2,
            "recency_penalty": 0.1,
        }

    def score_claim(
        self,
        claim: Dict[str, Any],
        evidence: Evidence,
        corroborating_sources: int = 1,
    ) -> float:
        """Calculate confidence score for a claim.

        Args:
            claim: The claim being verified
            evidence: Collected evidence
            corroborating_sources: Number of independent sources

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Direct evidence: Does the evidence directly support the claim?
        if evidence.context.get("exists"):
            score += self._weights["direct_evidence"]

        # Corroboration: Multiple sources increase confidence
        corroboration_bonus = min(corroborating_sources - 1, 3) / 3
        score += self._weights["corroboration"] * corroboration_bonus

        # Specificity: Exact matches are better than patterns
        if claim.get("exact_match"):
            score += self._weights["specificity"]
        elif claim.get("pattern_match"):
            score += self._weights["specificity"] * 0.5

        # Recency penalty: Recent file changes reduce confidence
        # (content may have changed since analysis)
        if claim.get("is_recent"):
            score *= 1.0 - self._weights["recency_penalty"]

        return min(max(score, 0.0), 1.0)


class ClaimVerifier:
    """Verifies claims with evidence capture and confidence scoring.

    This is the P0 core module that provides the foundation for all
    other verification enhancements.

    Usage:
        verifier = ClaimVerifier(project_root=Path("."))
        result = await verifier.verify_claim({
            "issue_type": "cross_layer_dependency",
            "file_path": "src/storage/lib.rs",
            "line_number": 42,
        })
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize claim verifier.

        Args:
            project_root: Root directory for file resolution
        """
        self._project_root = project_root or Path.cwd()
        self._evidence_collector = EvidenceCollector(self._project_root)
        self._confidence_scorer = ConfidenceScorer()

    async def verify_claim(
        self,
        claim: Dict[str, Any],
        context: Optional[VerificationContext] = None,
    ) -> EnhancedClaimResult:
        """Verify a claim with evidence collection.

        Args:
            claim: The claim to verify (dict with issue_type, file_path, etc.)
            context: Optional verification context

        Returns:
            Enhanced claim result with evidence and confidence
        """
        ctx = context or VerificationContext(project_root=self._project_root)

        # Convert claim to ClaimIssue if it's a dict
        if isinstance(claim, dict):
            claim_issue = ClaimIssue(**claim)
        else:
            claim_issue = claim

        # Collect evidence
        evidence_list = await self._collect_evidence(claim_issue, ctx)

        # Calculate confidence
        confidence = self._calculate_confidence(claim_issue, evidence_list, ctx)

        # Build result
        return EnhancedClaimResult(
            is_grounded=confidence > 0.5,
            confidence=confidence,
            evidence={"sources": [e.to_dict() for e in evidence_list]},
            reason=self._generate_reason(claim_issue, evidence_list, confidence),
            severity=claim_issue.severity,
            category=claim_issue.category,
        )

    async def _collect_evidence(
        self,
        claim: ClaimIssue,
        context: VerificationContext,
    ) -> List[Evidence]:
        """Collect all relevant evidence for a claim.

        Args:
            claim: The claim to verify
            context: Verification context

        Returns:
            List of evidence objects
        """
        evidence_list = []

        # File-level evidence
        if claim.file_path:
            file_evidence = self._evidence_collector.collect_file_evidence(claim.file_path)
            evidence_list.append(file_evidence)

            # Line-level evidence if line number specified
            if claim.line_number:
                line_evidence = self._evidence_collector.collect_line_evidence(
                    claim.file_path, claim.line_number
                )
                evidence_list.append(line_evidence)

            # Import evidence for dependency-related claims
            if "dependency" in claim.issue_type.lower():
                import_evidence = self._evidence_collector.collect_import_evidence(claim.file_path)
                evidence_list.append(import_evidence)

        return evidence_list

    def _calculate_confidence(
        self,
        claim: ClaimIssue,
        evidence_list: List[Evidence],
        context: VerificationContext,
    ) -> float:
        """Calculate overall confidence score.

        Args:
            claim: The claim being verified
            evidence_list: Collected evidence
            context: Verification context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not evidence_list:
            return 0.0

        # Use the best evidence (highest confidence)
        scores = [
            self._confidence_scorer.score_claim(claim.model_dump(), e, len(evidence_list))
            for e in evidence_list
        ]

        return max(scores) if scores else 0.0

    def _generate_reason(
        self,
        claim: ClaimIssue,
        evidence_list: List[Evidence],
        confidence: float,
    ) -> str:
        """Generate human-readable reason for verification result.

        Args:
            claim: The claim being verified
            evidence_list: Collected evidence
            confidence: Calculated confidence

        Returns:
            Human-readable explanation
        """
        if not evidence_list:
            return f"No evidence found for {claim.issue_type}"

        primary_evidence = evidence_list[0]

        if not primary_evidence.context.get("exists"):
            return f"File not found: {claim.file_path}"

        parts = [
            f"Issue '{claim.issue_type}' verified",
            f"in {claim.file_path}",
        ]

        if claim.line_number:
            parts.append(f"at line {claim.line_number}")

        parts.append(f"with {confidence:.0%} confidence")

        return " ".join(parts)


class SelfVerifyingAnalyzer:
    """Analyzer that verifies its own claims in one pass.

    This combines analysis and verification for efficiency.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        enable_fp_detection: bool = True,
    ):
        """Initialize self-verifying analyzer.

        Args:
            project_root: Root directory for analysis
            enable_fp_detection: Whether to enable false positive detection
        """
        self._project_root = project_root or Path.cwd()
        self._verifier = ClaimVerifier(self._project_root)
        self._enable_fp_detection = enable_fp_detection

        # Import FP detector if enabled (will be implemented in P1)
        self._fp_detector = None
        if enable_fp_detection:
            try:
                from victor.tools.verification.false_positive_detector import (
                    FalsePositiveDetector,
                )

                self._fp_detector = FalsePositiveDetector()
            except ImportError:
                logger.debug("FalsePositiveDetector not yet available")

    async def analyze_with_verification(
        self,
        codebase: Path,
        analysis_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze codebase and verify claims in one pass.

        Args:
            codebase: Path to codebase
            analysis_params: Optional analysis parameters

        Returns:
            Analysis results with verification metadata
        """
        # This would delegate to the actual analysis tools
        # (graph_tool, code_search_tool, etc.)
        # For now, return a placeholder structure
        return {
            "verified_claims": [],
            "unverified_claims": [],
            "false_positives": [],
            "summary": {
                "total": 0,
                "verified": 0,
                "confidence_avg": 0.0,
            },
        }

    def filter_by_confidence(
        self,
        claims: List[EnhancedClaimResult],
        min_confidence: float = 0.7,
    ) -> List[EnhancedClaimResult]:
        """Filter claims by minimum confidence threshold.

        Args:
            claims: List of claim results
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of claims
        """
        return [c for c in claims if c.adjusted_confidence() >= min_confidence]

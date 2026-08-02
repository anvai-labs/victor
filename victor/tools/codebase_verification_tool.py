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

"""Codebase verification tool integrating all verification modules.

This tool provides comprehensive codebase verification with semantic validation,
false positive detection, documentation cross-referencing, temporal analysis,
and severity weighting.

Usage:
    result = await codebase_verify(
        query="cross_layer_dependency in src/storage/lib.rs",
        enable_fp_detection=True,
        enable_doc_crossref=True,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.verification.protocols import (
    ClaimIssue,
    EnhancedClaimResult,
    SeverityLevel,
    TemporalNature,
    VerificationContext,
)

logger = logging.getLogger(__name__)


@tool(
    name="codebase_verify",
    category="verification",
    description="Verify codebase claims with semantic validation and context awareness",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READ,
    priority=Priority.LOW,
)
async def codebase_verify(
    query: str,
    file_path: Optional[str] = None,
    line_number: Optional[int] = None,
    enable_fp_detection: bool = True,
    enable_doc_crossref: bool = True,
    enable_temporal_analysis: bool = False,
    enable_severity_weighting: bool = True,
    min_confidence: float = 0.5,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify codebase claims with semantic validation.

    Integrates all verification modules to provide comprehensive analysis
    of codebase issues with semantic context, false positive filtering,
    documentation cross-referencing, and severity weighting.

    Args:
        query: The claim or issue to verify
        file_path: Optional file path for targeted verification
        line_number: Optional line number for targeted verification
        enable_fp_detection: Enable false positive detection
        enable_doc_crossref: Enable documentation cross-reference
        enable_temporal_analysis: Enable temporal context analysis
        enable_severity_weighting: Enable severity weighting
        min_confidence: Minimum confidence threshold for reporting
        exec_ctx: Optional execution context

    Returns:
        Dictionary with verification results including:
        - verified: Whether the claim is verified
        - confidence: Confidence score (0.0-1.0)
        - false_positive_risk: Risk of false positive (0.0-1.0)
        - severity: Severity level (if weighting enabled)
        - temporal_nature: Temporal classification (if analysis enabled)
        - doc_references: Documentation references (if enabled)
        - evidence: Supporting evidence
        - reason: Human-readable explanation
    """
    # Get project root from execution context
    project_root = Path.cwd()
    if exec_ctx:
        root_from_ctx = exec_ctx.get("workspace_root") or exec_ctx.get("project_root")
        if root_from_ctx:
            project_root = Path(root_from_ctx)

    # Import verification modules
    try:
        from victor.tools.verification import (
            ClaimVerifier,
            FalsePositiveDetector,
            DocumentationCrossReference,
            TemporalContextAnalyzer,
            SeverityWeighting,
        )
    except ImportError as e:
        logger.error("Failed to import verification modules: %s", e)
        return {
            "success": False,
            "error": f"Verification module not available: {e}",
            "verified": False,
        }

    # Create verification context
    context = VerificationContext(project_root=project_root)

    # Create issue from parameters
    issue = ClaimIssue(
        issue_type=_extract_issue_type(query),
        description=query,
        file_path=file_path,
        line_number=line_number,
    )

    # Step 1: Core verification
    verifier = ClaimVerifier(project_root=project_root)
    result = await verifier.verify_claim(issue, context)

    # Step 2: False positive detection
    if enable_fp_detection:
        fp_detector = FalsePositiveDetector()
        is_fp, fp_reason, fp_confidence = fp_detector.is_likely_false_positive(issue)
        result.false_positive_risk = fp_confidence

        if is_fp and fp_confidence > 0.7:
            result.reason = f"{result.reason} [Note: {fp_reason}]"

    # Step 3: Documentation cross-reference
    if enable_doc_crossref:
        crossref = DocumentationCrossReference(project_root=project_root)
        doc_refs = crossref.get_doc_references(issue.model_dump())
        result.doc_references = [str(ref) for ref in doc_refs]

    # Step 4: Temporal analysis
    if enable_temporal_analysis and file_path:
        temporal_analyzer = TemporalContextAnalyzer(project_root=project_root)
        temporal_ctx = temporal_analyzer.analyze_issue_temporal_context(issue)
        result.temporal_nature = temporal_ctx.get("temporal_nature")

    # Step 5: Severity weighting
    if enable_severity_weighting:
        severity_weighting = SeverityWeighting()
        score, severity = severity_weighting.score_and_classify(issue.model_dump())
        result.severity = severity

    # Filter by confidence
    if result.adjusted_confidence() < min_confidence:
        result.is_grounded = False

    return {
        "success": True,
        "verified": result.is_grounded,
        "confidence": result.confidence,
        "adjusted_confidence": result.adjusted_confidence(),
        "false_positive_risk": result.false_positive_risk,
        "severity": result.severity.value if result.severity else None,
        "temporal_nature": (result.temporal_nature.value if result.temporal_nature else None),
        "doc_references": result.doc_references,
        "evidence": result.evidence,
        "reason": result.reason,
    }


@tool(
    name="codebase_verify_batch",
    category="verification",
    description="Verify multiple codebase issues with semantic validation",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READ,
    priority=Priority.LOW,
)
async def codebase_verify_batch(
    issues: List[Dict[str, Any]],
    enable_fp_detection: bool = True,
    enable_doc_crossref: bool = True,
    enable_temporal_analysis: bool = False,
    enable_severity_weighting: bool = True,
    min_confidence: float = 0.5,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify multiple codebase issues with semantic validation.

    Args:
        issues: List of issue dictionaries to verify
        enable_fp_detection: Enable false positive detection
        enable_doc_crossref: Enable documentation cross-reference
        enable_temporal_analysis: Enable temporal context analysis
        enable_severity_weighting: Enable severity weighting
        min_confidence: Minimum confidence threshold for reporting
        exec_ctx: Optional execution context

    Returns:
        Dictionary with batch verification results including:
        - total: Total number of issues
        - verified: Number of verified issues
        - false_positives: Number of likely false positives
        - issues: List of processed issue results
        - summary: Severity distribution and other statistics
    """
    results = []
    false_positive_count = 0
    verified_count = 0

    for issue in issues:
        result = await codebase_verify(
            query=issue.get("description", issue.get("issue_type", "")),
            file_path=issue.get("file_path"),
            line_number=issue.get("line_number"),
            enable_fp_detection=enable_fp_detection,
            enable_doc_crossref=enable_doc_crossref,
            enable_temporal_analysis=enable_temporal_analysis,
            enable_severity_weighting=enable_severity_weighting,
            min_confidence=min_confidence,
            exec_ctx=exec_ctx,
        )

        results.append({**issue, **result})

        if result.get("false_positive_risk", 0) > 0.7:
            false_positive_count += 1
        if result.get("verified", False):
            verified_count += 1

    # Calculate summary
    severity_distribution: Dict[str, int] = {}
    for result in results:
        severity = result.get("severity", "unknown")
        severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

    return {
        "success": True,
        "total": len(issues),
        "verified": verified_count,
        "false_positives": false_positive_count,
        "issues": results,
        "summary": {
            "severity_distribution": severity_distribution,
            "verification_rate": verified_count / len(issues) if issues else 0.0,
            "false_positive_rate": (false_positive_count / len(issues) if issues else 0.0),
        },
    }


def _extract_issue_type(query: str) -> str:
    """Extract issue type from query text.

    Args:
        query: Query string

    Returns:
        Normalized issue type
    """
    query_lower = query.lower()

    # Common issue type patterns
    type_keywords = {
        "cross_layer_dependency": ["cross", "layer", "dependency"],
        "global_mutable_state": ["global", "mutable", "state"],
        "code_duplication": ["duplicat", "duplicate"],
        "performance": ["slow", "performance", "inefficient"],
        "security": ["security", "vulnerab", "unsafe"],
        "compilation_time": ["compile", "build", "time"],
        "maintainability": ["complex", "maintain", "coupl"],
    }

    for issue_type, keywords in type_keywords.items():
        if all(kw in query_lower for kw in keywords):
            return issue_type

    return "general"

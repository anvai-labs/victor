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

"""Victor codebase verification module.

Provides semantic validation, false positive detection, documentation
cross-referencing, temporal analysis, and severity weighting for codebase
analysis results.

This module enhances Victor's codebase analysis capabilities beyond
structural accuracy to include semantic context and awareness.

Usage:
    from victor.tools.verification import (
        ClaimVerifier,
        FalsePositiveDetector,
        DocumentationCrossReference,
        TemporalContextAnalyzer,
        SeverityWeighting,
        EnhancedClaimResult,
        VerificationContext,
    )

    verifier = ClaimVerifier()
    result = await verifier.verify_claim(issue_dict, context)
"""

from victor.tools.verification.protocols import (
    DocumentationReference,
    EnhancedClaimResult,
    Evidence,
    FalsePositiveResult,
    IVerificationStrategy,
    IssueCategory,
    ClaimIssue,
    SeverityLevel,
    TemporalNature,
    VerificationContext,
)

# Core verification components
from .claim_verifier import ClaimVerifier, SelfVerifyingAnalyzer
from .false_positive_detector import FalsePositiveDetector
from .documentation_crossref import DocumentationCrossReference
from .temporal_analyzer import TemporalContextAnalyzer
from .severity_weighting import SeverityWeighting

__all__ = [
    # Protocols and models
    "EnhancedClaimResult",
    "VerificationContext",
    "ClaimIssue",
    "Evidence",
    "FalsePositiveResult",
    "DocumentationReference",
    "SeverityLevel",
    "TemporalNature",
    "IssueCategory",
    "IVerificationStrategy",
    # Verification components
    "ClaimVerifier",
    "SelfVerifyingAnalyzer",
    "FalsePositiveDetector",
    "DocumentationCrossReference",
    "TemporalContextAnalyzer",
    "SeverityWeighting",
]

__version__ = "0.1.0"

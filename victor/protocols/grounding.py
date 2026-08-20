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

"""Grounding verification protocols and strategies.

This module defines the interface for grounding verification using the
Strategy pattern. Each strategy handles a specific type of claim verification:

- FileExistenceStrategy: Verifies file path references
- SymbolReferenceStrategy: Verifies code symbol references
- ContentMatchStrategy: Verifies quoted content matches source

Design Patterns:
- Strategy Pattern: Different verification strategies for different claim types
- Composite Pattern: CompositeGroundingVerifier aggregates multiple strategies
- Dependency Inversion: Core code depends on IGroundingStrategy interface

Usage:
    verifier = CompositeGroundingVerifier([
        FileExistenceStrategy(project_root),
        SymbolReferenceStrategy(symbol_table),
    ])

    result = await verifier.verify(response, context)
    if result.is_grounded:
        # Response is verified
        pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import re


class GroundingClaimType(str, Enum):
    """Types of claims that can be grounded."""

    FILE_EXISTS = "file_exists"
    FILE_NOT_EXISTS = "file_not_exists"
    SYMBOL_EXISTS = "symbol_exists"
    CONTENT_MATCH = "content_match"
    LINE_NUMBER = "line_number"
    DIRECTORY_EXISTS = "directory_exists"
    UNKNOWN = "unknown"


@dataclass
class GroundingClaim:
    """A claim extracted from a response that needs verification.

    Attributes:
        claim_type: Type of the claim
        value: The claimed value (e.g., file path, symbol name)
        context: Additional context for verification
        source_text: Original text containing the claim
        confidence: Confidence that this is actually a claim (0.0-1.0)
    """

    claim_type: GroundingClaimType
    value: str
    context: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single grounding claim.

    Renamed from VerificationResult to be semantically distinct:
    - ClaimVerificationResult (here): Protocol-level single claim verification
    - GroundingVerificationResult (victor.agent.grounding_verifier): Agent-level full verification

    Attributes:
        is_grounded: Whether the claim is verified as true
        confidence: Confidence in the verification (0.0-1.0)
        claim: The claim that was verified
        evidence: Evidence supporting or refuting the claim
        reason: Human-readable explanation
    """

    is_grounded: bool
    confidence: float = 0.0
    claim: Optional[GroundingClaim] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# Backward compatibility alias
VerificationResult = ClaimVerificationResult


@dataclass
class AggregatedVerificationResult:
    """Aggregated result from multiple verification strategies.

    Attributes:
        is_grounded: Overall grounding status
        confidence: Overall confidence score
        total_claims: Number of claims verified
        verified_claims: Number of claims that passed verification
        failed_claims: Number of claims that failed verification
        results: Individual verification results
        strategy_scores: Scores from each strategy
    """

    is_grounded: bool
    confidence: float = 0.0
    total_claims: int = 0
    verified_claims: int = 0
    failed_claims: int = 0
    results: List[VerificationResult] = field(default_factory=list)
    strategy_scores: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class IGroundingStrategy(Protocol):
    """Strategy interface for grounding verification.

    Each strategy handles a specific type of claim verification.
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        ...

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        """Return claim types this strategy can verify."""
        ...

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify a claim against context.

        Args:
            claim: The claim to verify
            context: Additional context for verification

        Returns:
            Verification result with grounding status
        """
        ...

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        """Extract claims of this type from a response.

        Args:
            response: The response text to analyze
            context: Additional context

        Returns:
            List of claims found in the response
        """
        ...


class FileExistenceStrategy(IGroundingStrategy):
    """Verify file path references in responses."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize with project root directory.

        Args:
            project_root: Root directory for resolving paths
        """
        self._project_root = project_root or Path.cwd()

    @property
    def name(self) -> str:
        return "file_existence"

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        return [GroundingClaimType.FILE_EXISTS, GroundingClaimType.FILE_NOT_EXISTS]

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify file existence claims."""
        project_root = context.get("project_root", self._project_root)
        if isinstance(project_root, str):
            project_root = Path(project_root)

        file_path = project_root / claim.value

        # Check actual file existence
        exists = file_path.exists()

        if claim.claim_type == GroundingClaimType.FILE_EXISTS:
            # Claim says file exists
            is_grounded = exists
            reason = f"File {'exists' if exists else 'does not exist'}: {claim.value}"
        else:
            # Claim says file does not exist
            is_grounded = not exists
            reason = f"File {'does not exist' if not exists else 'exists'}: {claim.value}"

        return VerificationResult(
            is_grounded=is_grounded,
            confidence=0.95 if is_grounded else 0.0,
            claim=claim,
            evidence={"path": str(file_path), "exists": exists},
            reason=reason,
        )

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        """Extract file path claims from response."""
        claims = []

        # Pattern for file paths
        path_patterns = [
            # Explicit file references
            r"(?:file|reading|read|found|exists?|looking at|in)\s+[`'\"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`'\"]?",
            # Backtick-quoted paths
            r"`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`",
            # Code blocks with file paths
            r"```[a-z]*\s*#\s*([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)",
        ]

        # Pattern for non-existence claims
        not_exists_patterns = [
            r"(?:not found|doesn't exist|does not exist|missing|no such file)[:\s]+[`'\"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`'\"]?",
            r"[`'\"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`'\"]?\s+(?:not found|doesn't exist|does not exist|is missing|was not found)",
        ]

        # Check for non-existence claims first
        for pattern in not_exists_patterns:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                path = match.group(1)
                if self._is_valid_path(path):
                    claims.append(
                        GroundingClaim(
                            claim_type=GroundingClaimType.FILE_NOT_EXISTS,
                            value=path,
                            source_text=match.group(0),
                            confidence=0.9,
                        )
                    )

        # Check for existence claims
        for pattern in path_patterns:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                path = match.group(1)
                if self._is_valid_path(path):
                    # Check it's not already a non-existence claim
                    if not any(c.value == path for c in claims):
                        claims.append(
                            GroundingClaim(
                                claim_type=GroundingClaimType.FILE_EXISTS,
                                value=path,
                                source_text=match.group(0),
                                confidence=0.8,
                            )
                        )

        return claims

    def _is_valid_path(self, path: str) -> bool:
        """Check if path looks valid."""
        if not path or len(path) < 3:
            return False
        # Must have an extension
        if "." not in path:
            return False
        # Filter out URLs
        if path.startswith(("http://", "https://", "ftp://")):
            return False
        # Filter out common false positives
        if path in ("e.g.", "i.e.", "etc.", "a.k.a."):
            return False
        return True


class SymbolReferenceStrategy(IGroundingStrategy):
    """Verify code symbol references in responses."""

    def __init__(self, symbol_table: Optional[Dict[str, Any]] = None):
        """Initialize with optional symbol table.

        Args:
            symbol_table: Pre-computed symbol table for quick lookups
        """
        self._symbol_table = symbol_table or {}

    @property
    def name(self) -> str:
        return "symbol_reference"

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        return [GroundingClaimType.SYMBOL_EXISTS]

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify symbol existence claims."""
        symbol_table = context.get("symbol_table", self._symbol_table)
        symbol_name = claim.value

        # Look up symbol
        found = symbol_name in symbol_table

        return VerificationResult(
            is_grounded=found,
            confidence=0.9 if found else 0.3,
            claim=claim,
            evidence={"symbol": symbol_name, "found": found},
            reason=f"Symbol '{symbol_name}' {'found' if found else 'not found'} in symbol table",
        )

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        """Extract symbol reference claims from response."""
        claims = []

        # Patterns for symbol references
        patterns = [
            # Function/method calls
            r"(?:function|method|def|class)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?",
            # Backtick-quoted identifiers
            r"`([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)`",
            # Import statements
            r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, response):
                symbol = match.group(1)
                if len(symbol) > 1 and not symbol.isupper():  # Skip constants
                    claims.append(
                        GroundingClaim(
                            claim_type=GroundingClaimType.SYMBOL_EXISTS,
                            value=symbol,
                            source_text=match.group(0),
                            confidence=0.7,
                        )
                    )

        return claims


class ContentMatchStrategy(IGroundingStrategy):
    """Verify quoted content matches source files."""

    @property
    def name(self) -> str:
        return "content_match"

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        return [GroundingClaimType.CONTENT_MATCH]

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify content match claims."""
        expected_content = claim.value
        file_path = claim.context.get("file_path")

        if not file_path:
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
                claim=claim,
                reason="No file path provided for content verification",
            )

        project_root = context.get("project_root", Path.cwd())
        if isinstance(project_root, str):
            project_root = Path(project_root)

        full_path = project_root / file_path

        if not full_path.exists():
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
                claim=claim,
                reason=f"File not found: {file_path}",
            )

        try:
            content = full_path.read_text(encoding="utf-8")
            # Normalize whitespace for comparison
            normalized_expected = " ".join(expected_content.split())
            normalized_content = " ".join(content.split())

            is_match = normalized_expected in normalized_content

            return VerificationResult(
                is_grounded=is_match,
                confidence=0.95 if is_match else 0.1,
                claim=claim,
                evidence={"file_path": str(full_path), "found": is_match},
                reason=f"Content {'matches' if is_match else 'does not match'} source file",
            )
        except Exception as e:
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
                claim=claim,
                reason=f"Error reading file: {e}",
            )

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        """Extract content match claims from response."""
        claims = []

        # Pattern for code blocks with file paths
        pattern = r"```[a-z]*\s*(?:#\s*([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)\n)?(.*?)```"

        for match in re.finditer(pattern, response, re.DOTALL):
            file_path = match.group(1)
            content = match.group(2).strip()

            if file_path and content and len(content) > 10:
                claims.append(
                    GroundingClaim(
                        claim_type=GroundingClaimType.CONTENT_MATCH,
                        value=content[:500],  # Limit content size
                        context={"file_path": file_path},
                        source_text=match.group(0)[:100],
                        confidence=0.8,
                    )
                )

        return claims


class CompositeGroundingVerifier:
    """Composite verifier applying multiple grounding strategies.

    Aggregates results from multiple strategies to produce an overall
    grounding score.
    """

    def __init__(
        self,
        strategies: Optional[List[IGroundingStrategy]] = None,
        threshold: float = 0.7,
        require_all: bool = False,
    ):
        """Initialize with verification strategies.

        Args:
            strategies: List of grounding strategies to apply
            threshold: Minimum confidence threshold for grounded status
            require_all: If True, all claims must be verified
        """
        self._strategies = strategies or [
            FileExistenceStrategy(),
            SymbolReferenceStrategy(),
        ]
        self._threshold = threshold
        self._require_all = require_all

    def add_strategy(self, strategy: IGroundingStrategy) -> None:
        """Add a verification strategy."""
        self._strategies.append(strategy)

    def remove_strategy(self, name: str) -> None:
        """Remove a strategy by name."""
        self._strategies = [s for s in self._strategies if s.name != name]

    async def verify(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> AggregatedVerificationResult:
        """Verify all claims in a response.

        Args:
            response: The response text to verify
            context: Verification context (project_root, symbol_table, etc.)

        Returns:
            Aggregated verification result
        """
        all_claims = []
        results = []
        strategy_scores = {}

        # Extract claims from each strategy
        for strategy in self._strategies:
            claims = strategy.extract_claims(response, context)
            all_claims.extend(claims)

        # Verify each claim
        for claim in all_claims:
            # Find appropriate strategy for claim type
            for strategy in self._strategies:
                if claim.claim_type in strategy.claim_types:
                    result = await strategy.verify(claim, context)
                    results.append(result)

                    # Track per-strategy scores
                    if strategy.name not in strategy_scores:
                        strategy_scores[strategy.name] = []
                    strategy_scores[strategy.name].append(
                        result.confidence if result.is_grounded else 0.0
                    )
                    break

        # Calculate aggregated result
        if not results:
            # No claims to verify - assume grounded
            return AggregatedVerificationResult(
                is_grounded=True,
                confidence=1.0,
                total_claims=0,
                verified_claims=0,
                failed_claims=0,
                results=[],
                strategy_scores={},
            )

        verified = sum(1 for r in results if r.is_grounded)
        failed = len(results) - verified

        # Calculate overall confidence
        if self._require_all:
            overall_confidence = min(r.confidence for r in results)
            is_grounded = all(r.is_grounded for r in results)
        else:
            overall_confidence = sum(r.confidence for r in results) / len(results)
            is_grounded = overall_confidence >= self._threshold

        # Aggregate strategy scores
        avg_strategy_scores = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in strategy_scores.items()
        }

        return AggregatedVerificationResult(
            is_grounded=is_grounded,
            confidence=overall_confidence,
            total_claims=len(results),
            verified_claims=verified,
            failed_claims=failed,
            results=results,
            strategy_scores=avg_strategy_scores,
        )

    async def verify_claim(
        self,
        claim: str,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify a single claim string.

        Args:
            claim: The claim text to verify
            context: Verification context

        Returns:
            Verification result
        """
        result = await self.verify(claim, context)
        if result.results:
            return result.results[0]
        return VerificationResult(
            is_grounded=True,
            confidence=1.0,
            reason="No verifiable claims found",
        )

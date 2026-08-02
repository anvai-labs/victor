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

"""Tests for ClaimVerifier module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from victor.tools.verification import (
    ClaimVerifier,
    ClaimIssue,
    EnhancedClaimResult,
    VerificationContext,
)


@pytest.fixture
def temp_project_root(tmp_path: Path) -> Path:
    """Create a temporary project root with test files."""
    # Create a test file
    test_file = tmp_path / "src" / "test.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("""
def example_function():
    # Test function
    pass
""")

    return tmp_path


@pytest.fixture
def claim_verifier(temp_project_root: Path) -> ClaimVerifier:
    """Create a ClaimVerifier instance."""
    return ClaimVerifier(project_root=temp_project_root)


class TestClaimVerifier:
    """Tests for ClaimVerifier class."""

    @pytest.mark.asyncio
    async def test_verify_claim_with_file(
        self, claim_verifier: ClaimVerifier, temp_project_root: Path
    ):
        """Test verification of a claim with existing file."""
        claim = ClaimIssue(
            issue_type="test_issue",
            description="Test issue",
            file_path="src/test.py",
        )

        result = await claim_verifier.verify_claim(claim)

        assert isinstance(result, EnhancedClaimResult)
        assert result.confidence > 0.0
        assert "evidence" in result.model_dump()
        assert len(result.evidence.get("sources", [])) > 0

    @pytest.mark.asyncio
    async def test_verify_claim_with_nonexistent_file(self, claim_verifier: ClaimVerifier):
        """Test verification of a claim with non-existent file."""
        claim = ClaimIssue(
            issue_type="test_issue",
            description="Test issue",
            file_path="nonexistent/file.py",
        )

        result = await claim_verifier.verify_claim(claim)

        assert isinstance(result, EnhancedClaimResult)
        assert result.confidence < 0.5  # Low confidence for missing file

    @pytest.mark.asyncio
    async def test_verify_claim_with_line_number(
        self, claim_verifier: ClaimVerifier, temp_project_root: Path
    ):
        """Test verification with specific line number."""
        claim = ClaimIssue(
            issue_type="test_issue",
            description="Test issue",
            file_path="src/test.py",
            line_number=2,
        )

        result = await claim_verifier.verify_claim(claim)

        assert isinstance(result, EnhancedClaimResult)
        # Check that line evidence was collected
        sources = result.evidence.get("sources", [])
        assert any(s.get("line_number") == 2 for s in sources)

    def test_confidence_adjustment(self, claim_verifier: ClaimVerifier):
        """Test confidence adjustment with false positive risk."""
        result = EnhancedClaimResult(
            is_grounded=True,
            confidence=0.9,
            false_positive_risk=0.2,
        )

        adjusted = result.adjusted_confidence(fp_penalty=0.5)

        assert adjusted == 0.81  # 0.9 * (1 - 0.2 * 0.5)


class TestEnhancedClaimResult:
    """Tests for EnhancedClaimResult class."""

    def test_add_evidence(self):
        """Test adding evidence creates new result."""
        result = EnhancedClaimResult(
            is_grounded=True,
            confidence=0.8,
        )

        updated = result.add_evidence("file", "test.py")

        assert updated is not result  # Immutable
        assert "file" in updated.evidence
        assert updated.evidence["file"] == "test.py"

    def test_with_severity(self):
        """Test setting severity creates new result."""
        from victor.tools.verification.protocols import SeverityLevel

        result = EnhancedClaimResult(is_grounded=True, confidence=0.8)
        updated = result.with_severity(SeverityLevel.HIGH)

        assert updated.severity == SeverityLevel.HIGH
        assert result.severity is None  # Original unchanged


class TestVerificationContext:
    """Tests for VerificationContext class."""

    def test_get_doc_path(self, tmp_path: Path):
        """Test getting documentation path."""
        doc_file = tmp_path / "TECHNICAL_DEBT.adoc"
        doc_file.write_text("# Test")

        context = VerificationContext(
            project_root=tmp_path,
            documentation_paths={"TECHNICAL_DEBT": doc_file},
        )

        assert context.get_doc_path("TECHNICAL_DEBT") == doc_file

    def test_has_documentation(self, tmp_path: Path):
        """Test checking for documentation existence."""
        doc_file = tmp_path / "TECHNICAL_DEBT.adoc"
        doc_file.write_text("# Test")

        context = VerificationContext(
            project_root=tmp_path,
            documentation_paths={"TECHNICAL_DEBT": doc_file},
        )

        assert context.has_documentation("TECHNICAL_DEBT")
        assert not context.has_documentation("NONEXISTENT")

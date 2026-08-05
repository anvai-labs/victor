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

"""Tests for grounding verification protocols and strategies."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from victor.protocols.grounding import (
    GroundingClaimType,
    GroundingClaim,
    VerificationResult,
    AggregatedVerificationResult,
    IGroundingStrategy,
    FileExistenceStrategy,
    SymbolReferenceStrategy,
    ContentMatchStrategy,
    CompositeGroundingVerifier,
)

# =============================================================================
# GROUNDING CLAIM TYPE TESTS
# =============================================================================


class TestGroundingClaimType:
    """Tests for GroundingClaimType enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert GroundingClaimType.FILE_EXISTS.value == "file_exists"
        assert GroundingClaimType.FILE_NOT_EXISTS.value == "file_not_exists"
        assert GroundingClaimType.SYMBOL_EXISTS.value == "symbol_exists"
        assert GroundingClaimType.CONTENT_MATCH.value == "content_match"
        assert GroundingClaimType.LINE_NUMBER.value == "line_number"
        assert GroundingClaimType.DIRECTORY_EXISTS.value == "directory_exists"
        assert GroundingClaimType.UNKNOWN.value == "unknown"


# =============================================================================
# GROUNDING CLAIM TESTS
# =============================================================================


class TestGroundingClaim:
    """Tests for GroundingClaim dataclass."""

    def test_creation(self):
        """Test basic claim creation."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="test.py",
        )
        assert claim.claim_type == GroundingClaimType.FILE_EXISTS
        assert claim.value == "test.py"
        assert claim.context == {}
        assert claim.source_text == ""
        assert claim.confidence == 1.0

    def test_creation_with_all_fields(self):
        """Test claim creation with all fields."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.SYMBOL_EXISTS,
            value="MyClass.method",
            context={"file": "module.py"},
            source_text="function `MyClass.method`",
            confidence=0.85,
        )
        assert claim.context == {"file": "module.py"}
        assert claim.source_text == "function `MyClass.method`"
        assert claim.confidence == 0.85


# =============================================================================
# VERIFICATION RESULT TESTS
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_creation_grounded(self):
        """Test grounded verification result."""
        result = VerificationResult(
            is_grounded=True,
            confidence=0.95,
        )
        assert result.is_grounded is True
        assert result.confidence == 0.95
        assert result.claim is None
        assert result.evidence == {}
        assert result.reason == ""

    def test_creation_not_grounded(self):
        """Test not grounded verification result."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="missing.py",
        )
        result = VerificationResult(
            is_grounded=False,
            confidence=0.0,
            claim=claim,
            evidence={"path": "/project/missing.py", "exists": False},
            reason="File does not exist: missing.py",
        )
        assert result.is_grounded is False
        assert result.claim == claim
        assert result.evidence["exists"] is False


# =============================================================================
# AGGREGATED VERIFICATION RESULT TESTS
# =============================================================================


class TestAggregatedVerificationResult:
    """Tests for AggregatedVerificationResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = AggregatedVerificationResult(is_grounded=True)
        assert result.is_grounded is True
        assert result.confidence == 0.0
        assert result.total_claims == 0
        assert result.verified_claims == 0
        assert result.failed_claims == 0
        assert result.results == []
        assert result.strategy_scores == {}

    def test_with_results(self):
        """Test with verification results."""
        results = [
            VerificationResult(is_grounded=True, confidence=0.9),
            VerificationResult(is_grounded=False, confidence=0.1),
        ]
        agg = AggregatedVerificationResult(
            is_grounded=True,
            confidence=0.5,
            total_claims=2,
            verified_claims=1,
            failed_claims=1,
            results=results,
            strategy_scores={"file_existence": 0.9},
        )
        assert agg.total_claims == 2
        assert agg.verified_claims == 1
        assert len(agg.results) == 2


# =============================================================================
# FILE EXISTENCE STRATEGY TESTS
# =============================================================================


class TestFileExistenceStrategy:
    """Tests for FileExistenceStrategy."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        with TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            # Create some test files
            (project / "existing.py").write_text("# test file")
            (project / "src").mkdir()
            (project / "src" / "module.py").write_text("def hello(): pass")
            yield project

    @pytest.fixture
    def strategy(self, temp_project):
        """Create strategy with temp project."""
        return FileExistenceStrategy(project_root=temp_project)

    def test_name_property(self, strategy):
        """Test strategy name."""
        assert strategy.name == "file_existence"

    def test_claim_types(self, strategy):
        """Test claim types property."""
        types = strategy.claim_types
        assert GroundingClaimType.FILE_EXISTS in types
        assert GroundingClaimType.FILE_NOT_EXISTS in types

    @pytest.mark.asyncio
    async def test_verify_file_exists_true(self, strategy, temp_project):
        """Test verifying existing file."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="existing.py",
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is True
        assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_verify_file_exists_false(self, strategy, temp_project):
        """Test verifying non-existing file."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="missing.py",
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_verify_file_not_exists_true(self, strategy, temp_project):
        """Test verifying file does not exist (true)."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_NOT_EXISTS,
            value="missing.py",
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_verify_file_not_exists_false(self, strategy, temp_project):
        """Test verifying file does not exist (false - file exists)."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_NOT_EXISTS,
            value="existing.py",
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False

    def test_extract_claims_file_reference(self, strategy):
        """Test extracting file references."""
        response = "I found the issue in `src/module.py` at line 10."
        claims = strategy.extract_claims(response, {})
        assert len(claims) >= 1
        assert any(c.value == "src/module.py" for c in claims)

    def test_extract_claims_not_found(self, strategy):
        """Test extracting not-found claims."""
        response = "File not found: missing.py"
        claims = strategy.extract_claims(response, {})
        assert len(claims) >= 1
        not_exists_claims = [
            c for c in claims if c.claim_type == GroundingClaimType.FILE_NOT_EXISTS
        ]
        assert len(not_exists_claims) >= 1

    def test_is_valid_path_filters_urls(self, strategy):
        """Test path validation filters URLs."""
        assert strategy._is_valid_path("src/test.py") is True
        assert strategy._is_valid_path("https://example.com/file.py") is False
        assert strategy._is_valid_path("http://localhost/test.py") is False

    def test_is_valid_path_filters_abbreviations(self, strategy):
        """Test path validation filters abbreviations."""
        assert strategy._is_valid_path("e.g.") is False
        assert strategy._is_valid_path("i.e.") is False
        assert strategy._is_valid_path("etc.") is False

    def test_is_valid_path_requires_extension(self, strategy):
        """Test path validation requires extension."""
        assert strategy._is_valid_path("README") is False
        assert strategy._is_valid_path("README.md") is True


# =============================================================================
# SYMBOL REFERENCE STRATEGY TESTS
# =============================================================================


class TestSymbolReferenceStrategy:
    """Tests for SymbolReferenceStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with symbol table."""
        symbol_table = {
            "MyClass": {"type": "class", "file": "module.py"},
            "my_function": {"type": "function", "file": "utils.py"},
            "CONSTANT": {"type": "constant", "file": "config.py"},
        }
        return SymbolReferenceStrategy(symbol_table=symbol_table)

    def test_name_property(self, strategy):
        """Test strategy name."""
        assert strategy.name == "symbol_reference"

    def test_claim_types(self, strategy):
        """Test claim types property."""
        types = strategy.claim_types
        assert GroundingClaimType.SYMBOL_EXISTS in types

    @pytest.mark.asyncio
    async def test_verify_symbol_found(self, strategy):
        """Test verifying existing symbol."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.SYMBOL_EXISTS,
            value="MyClass",
        )
        result = await strategy.verify(claim, {"symbol_table": strategy._symbol_table})
        assert result.is_grounded is True
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_verify_symbol_not_found(self, strategy):
        """Test verifying non-existing symbol."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.SYMBOL_EXISTS,
            value="NonExistentClass",
        )
        result = await strategy.verify(claim, {"symbol_table": strategy._symbol_table})
        assert result.is_grounded is False

    def test_extract_claims_function(self, strategy):
        """Test extracting function references."""
        response = "The function `my_function` does the processing."
        claims = strategy.extract_claims(response, {})
        assert len(claims) >= 1
        assert any(c.value == "my_function" for c in claims)

    def test_extract_claims_class(self, strategy):
        """Test extracting class references."""
        response = "class MyClass handles the logic."
        claims = strategy.extract_claims(response, {})
        assert any(c.value == "MyClass" for c in claims)

    def test_extract_claims_import(self, strategy):
        """Test extracting import references."""
        response = "You need to import module.utils"
        claims = strategy.extract_claims(response, {})
        assert any("module" in c.value or "utils" in c.value for c in claims)


# =============================================================================
# CONTENT MATCH STRATEGY TESTS
# =============================================================================


class TestContentMatchStrategy:
    """Tests for ContentMatchStrategy."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        with TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / "test.py").write_text("def hello():\n    print('Hello World')")
            yield project

    @pytest.fixture
    def strategy(self):
        """Create content match strategy."""
        return ContentMatchStrategy()

    def test_name_property(self, strategy):
        """Test strategy name."""
        assert strategy.name == "content_match"

    def test_claim_types(self, strategy):
        """Test claim types property."""
        types = strategy.claim_types
        assert GroundingClaimType.CONTENT_MATCH in types

    @pytest.mark.asyncio
    async def test_verify_content_match(self, strategy, temp_project):
        """Test verifying matching content."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="def hello():",
            context={"file_path": "test.py"},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_verify_content_no_match(self, strategy, temp_project):
        """Test verifying non-matching content."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="def goodbye():",
            context={"file_path": "test.py"},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False

    @pytest.mark.asyncio
    async def test_verify_no_file_path(self, strategy, temp_project):
        """Test verifying without file path."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="some content",
            context={},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False
        assert "No file path" in result.reason

    @pytest.mark.asyncio
    async def test_verify_file_not_found(self, strategy, temp_project):
        """Test verifying with missing file."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="some content",
            context={"file_path": "missing.py"},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False
        assert "not found" in result.reason.lower()

    def test_extract_claims_code_block(self, strategy):
        """Test extracting claims from code blocks."""
        response = """Here's the code:

```python
# test.py
def example():
    return 42
```
"""
        claims = strategy.extract_claims(response, {})
        # Should extract claims from code blocks with file paths
        assert isinstance(claims, list)


# =============================================================================
# COMPOSITE GROUNDING VERIFIER TESTS
# =============================================================================


class TestCompositeGroundingVerifier:
    """Tests for CompositeGroundingVerifier."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        with TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / "src").mkdir()
            (project / "src" / "module.py").write_text("class Handler: pass")
            yield project

    @pytest.fixture
    def verifier(self, temp_project):
        """Create composite verifier."""
        return CompositeGroundingVerifier(
            strategies=[
                FileExistenceStrategy(project_root=temp_project),
                SymbolReferenceStrategy(symbol_table={"Handler": {}}),
            ],
            threshold=0.7,
        )

    def test_init_default_strategies(self):
        """Test initialization with default strategies."""
        verifier = CompositeGroundingVerifier()
        assert len(verifier._strategies) >= 2

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        verifier = CompositeGroundingVerifier(threshold=0.5)
        assert verifier._threshold == 0.5

    def test_add_strategy(self, verifier):
        """Test adding a strategy."""
        initial_count = len(verifier._strategies)
        verifier.add_strategy(ContentMatchStrategy())
        assert len(verifier._strategies) == initial_count + 1

    def test_remove_strategy(self, verifier):
        """Test removing a strategy by name."""
        initial_count = len(verifier._strategies)
        verifier.remove_strategy("file_existence")
        assert len(verifier._strategies) == initial_count - 1

    @pytest.mark.asyncio
    async def test_verify_no_claims(self, verifier, temp_project):
        """Test verifying response with no claims."""
        response = "This is a general response without file references."
        result = await verifier.verify(response, {"project_root": temp_project})
        assert result.is_grounded is True
        assert result.total_claims == 0

    @pytest.mark.asyncio
    async def test_verify_with_file_claims(self, verifier, temp_project):
        """Test verifying response with file claims."""
        response = "Looking at `src/module.py`, I found the Handler class."
        result = await verifier.verify(response, {"project_root": temp_project})
        assert result.total_claims >= 1

    @pytest.mark.asyncio
    async def test_verify_require_all(self, temp_project):
        """Test require_all mode."""
        verifier = CompositeGroundingVerifier(
            strategies=[FileExistenceStrategy(project_root=temp_project)],
            require_all=True,
        )
        response = "Files `src/module.py` and `missing.py` are relevant."
        result = await verifier.verify(response, {"project_root": temp_project})
        # With require_all, if any claim fails, overall should fail
        if result.failed_claims > 0:
            assert result.is_grounded is False

    @pytest.mark.asyncio
    async def test_verify_claim_single(self, verifier, temp_project):
        """Test verify_claim for single claim."""
        claim = "The file `src/module.py` contains the Handler class."
        result = await verifier.verify_claim(claim, {"project_root": temp_project})
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_claim_no_claims_found(self, verifier, temp_project):
        """Test verify_claim when no claims found."""
        claim = "General statement without file references."
        result = await verifier.verify_claim(claim, {"project_root": temp_project})
        assert result.is_grounded is True
        assert "No verifiable claims" in result.reason


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Tests for IGroundingStrategy protocol compliance."""

    def test_file_strategy_is_protocol(self):
        """Test FileExistenceStrategy implements protocol."""
        strategy = FileExistenceStrategy()
        assert isinstance(strategy, IGroundingStrategy)

    def test_symbol_strategy_is_protocol(self):
        """Test SymbolReferenceStrategy implements protocol."""
        strategy = SymbolReferenceStrategy()
        assert isinstance(strategy, IGroundingStrategy)

    def test_content_strategy_is_protocol(self):
        """Test ContentMatchStrategy implements protocol."""
        strategy = ContentMatchStrategy()
        assert isinstance(strategy, IGroundingStrategy)

    def test_custom_strategy_protocol(self):
        """Test custom strategy can implement protocol."""

        class CustomStrategy:
            @property
            def name(self) -> str:
                return "custom"

            @property
            def claim_types(self):
                return [GroundingClaimType.UNKNOWN]

            async def verify(self, claim, context):
                return VerificationResult(is_grounded=True, confidence=1.0)

            def extract_claims(self, response, context):
                return []

        strategy = CustomStrategy()
        assert isinstance(strategy, IGroundingStrategy)

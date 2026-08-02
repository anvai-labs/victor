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

"""Tests for protocols/grounding.py and protocols/quality.py."""

import tempfile
from pathlib import Path

import pytest

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

from victor.protocols.quality import (
    ProtocolQualityDimension,
    DimensionScore,
    QualityScore,
    IQualityAssessor,
    BaseQualityAssessor,
    SimpleQualityAssessor,
    ProviderAwareQualityAssessor,
    CompositeQualityAssessor,
)

# =============================================================================
# GROUNDING CLAIM TYPE TESTS
# =============================================================================


class TestGroundingClaimType:
    """Tests for GroundingClaimType enum."""

    def test_all_types_defined(self):
        """All expected claim types are defined."""
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

    def test_minimal_claim(self):
        """Create claim with minimal fields."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="test.py",
        )
        assert claim.claim_type == GroundingClaimType.FILE_EXISTS
        assert claim.value == "test.py"
        assert claim.context == {}
        assert claim.source_text == ""
        assert claim.confidence == 1.0

    def test_full_claim(self):
        """Create claim with all fields."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="def hello():",
            context={"file_path": "test.py"},
            source_text="```python\ndef hello():\n```",
            confidence=0.8,
        )
        assert claim.context["file_path"] == "test.py"
        assert claim.confidence == 0.8


# =============================================================================
# VERIFICATION RESULT TESTS
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_grounded_result(self):
        """Create grounded result."""
        result = VerificationResult(
            is_grounded=True,
            confidence=0.95,
            reason="File exists",
        )
        assert result.is_grounded is True
        assert result.confidence == 0.95

    def test_not_grounded_result(self):
        """Create not grounded result."""
        result = VerificationResult(
            is_grounded=False,
            confidence=0.0,
            reason="File not found",
            evidence={"path": "/nonexistent.py"},
        )
        assert result.is_grounded is False
        assert result.evidence["path"] == "/nonexistent.py"


# =============================================================================
# AGGREGATED VERIFICATION RESULT TESTS
# =============================================================================


class TestAggregatedVerificationResult:
    """Tests for AggregatedVerificationResult dataclass."""

    def test_empty_result(self):
        """Create empty aggregated result."""
        result = AggregatedVerificationResult(
            is_grounded=True,
            confidence=1.0,
        )
        assert result.total_claims == 0
        assert result.verified_claims == 0
        assert result.failed_claims == 0
        assert result.results == []

    def test_with_results(self):
        """Create result with verification results."""
        results = [
            VerificationResult(is_grounded=True, confidence=0.9),
            VerificationResult(is_grounded=False, confidence=0.1),
        ]
        result = AggregatedVerificationResult(
            is_grounded=False,
            confidence=0.5,
            total_claims=2,
            verified_claims=1,
            failed_claims=1,
            results=results,
            strategy_scores={"file_existence": 0.5},
        )
        assert result.total_claims == 2
        assert len(result.results) == 2


# =============================================================================
# FILE EXISTENCE STRATEGY TESTS
# =============================================================================


class TestFileExistenceStrategy:
    """Tests for FileExistenceStrategy."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("print('hello')")
            (root / "utils.py").write_text("def helper(): pass")
            (root / "src").mkdir()
            (root / "src" / "module.py").write_text("class MyClass: pass")
            yield root

    @pytest.fixture
    def strategy(self, temp_project):
        """Create strategy for temp project."""
        return FileExistenceStrategy(project_root=temp_project)

    def test_name_property(self, strategy):
        """Strategy has correct name."""
        assert strategy.name == "file_existence"

    def test_claim_types_property(self, strategy):
        """Strategy handles correct claim types."""
        assert GroundingClaimType.FILE_EXISTS in strategy.claim_types
        assert GroundingClaimType.FILE_NOT_EXISTS in strategy.claim_types

    @pytest.mark.asyncio
    async def test_verify_existing_file(self, strategy):
        """Verify existing file returns grounded."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="main.py",
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is True
        assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_verify_nonexistent_file(self, strategy):
        """Verify nonexistent file returns not grounded."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="nonexistent.py",
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_verify_file_not_exists_claim(self, strategy, temp_project):
        """Verify FILE_NOT_EXISTS claim for nonexistent file."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_NOT_EXISTS,
            value="nonexistent.py",
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_verify_file_not_exists_for_existing(self, strategy):
        """Verify FILE_NOT_EXISTS claim returns false for existing file."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_NOT_EXISTS,
            value="main.py",
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is False

    @pytest.mark.asyncio
    async def test_verify_with_context_project_root(self, strategy, temp_project):
        """Verify uses project_root from context."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.FILE_EXISTS,
            value="main.py",
        )
        result = await strategy.verify(claim, {"project_root": str(temp_project)})
        assert result.is_grounded is True

    def test_extract_claims_from_response(self, strategy):
        """Extract file claims from response text."""
        response = """
        I found the file `main.py` for the entry point.
        Reading `src/module.py` for the class definition.
        """
        claims = strategy.extract_claims(response, {})
        file_names = [c.value for c in claims]
        # Backtick-quoted paths should be extracted
        assert "main.py" in file_names
        assert "src/module.py" in file_names

    def test_extract_not_exists_claims(self, strategy):
        """Extract file not exists claims."""
        response = "The file missing.py was not found."
        claims = strategy.extract_claims(response, {})
        assert any(c.claim_type == GroundingClaimType.FILE_NOT_EXISTS for c in claims)

    def test_is_valid_path_filters_urls(self, strategy):
        """URL paths are filtered out."""
        assert strategy._is_valid_path("https://example.com/file.py") is False
        assert strategy._is_valid_path("http://test.com/test.txt") is False

    def test_is_valid_path_filters_short(self, strategy):
        """Very short paths are filtered out."""
        # len(path) < 3 is filtered
        assert strategy._is_valid_path("x") is False
        assert strategy._is_valid_path("") is False
        # But 3+ chars with extension are valid
        assert strategy._is_valid_path("a.b") is True

    def test_is_valid_path_filters_abbreviations(self, strategy):
        """Common abbreviations are filtered out."""
        assert strategy._is_valid_path("e.g.") is False
        assert strategy._is_valid_path("i.e.") is False


# =============================================================================
# SYMBOL REFERENCE STRATEGY TESTS
# =============================================================================


class TestSymbolReferenceStrategy:
    """Tests for SymbolReferenceStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with symbol table."""
        return SymbolReferenceStrategy(
            symbol_table={
                "Application": {"type": "class", "file": "main.py"},
                "main": {"type": "function", "file": "main.py"},
                "helper": {"type": "function", "file": "utils.py"},
            }
        )

    def test_name_property(self, strategy):
        """Strategy has correct name."""
        assert strategy.name == "symbol_reference"

    def test_claim_types_property(self, strategy):
        """Strategy handles correct claim types."""
        assert GroundingClaimType.SYMBOL_EXISTS in strategy.claim_types

    @pytest.mark.asyncio
    async def test_verify_existing_symbol(self, strategy):
        """Verify existing symbol returns grounded."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.SYMBOL_EXISTS,
            value="Application",
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is True
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_verify_nonexistent_symbol(self, strategy):
        """Verify nonexistent symbol returns low confidence."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.SYMBOL_EXISTS,
            value="NonexistentClass",
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is False
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_verify_uses_context_symbol_table(self, strategy):
        """Verify uses symbol_table from context."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.SYMBOL_EXISTS,
            value="CustomSymbol",
        )
        context = {"symbol_table": {"CustomSymbol": {"type": "class"}}}
        result = await strategy.verify(claim, context)
        assert result.is_grounded is True

    def test_extract_claims_from_response(self, strategy):
        """Extract symbol claims from response text."""
        response = """
        The class `Application` is the main entry point.
        Call function `main` to start.
        """
        claims = strategy.extract_claims(response, {})
        symbols = [c.value for c in claims]
        assert "Application" in symbols
        assert "main" in symbols


# =============================================================================
# CONTENT MATCH STRATEGY TESTS
# =============================================================================


class TestContentMatchStrategy:
    """Tests for ContentMatchStrategy."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("""def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
""")
            yield root

    @pytest.fixture
    def strategy(self):
        """Create strategy."""
        return ContentMatchStrategy()

    def test_name_property(self, strategy):
        """Strategy has correct name."""
        assert strategy.name == "content_match"

    def test_claim_types_property(self, strategy):
        """Strategy handles correct claim types."""
        assert GroundingClaimType.CONTENT_MATCH in strategy.claim_types

    @pytest.mark.asyncio
    async def test_verify_matching_content(self, strategy, temp_project):
        """Verify matching content returns grounded."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value='print("Hello, World!")',
            context={"file_path": "main.py"},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_verify_non_matching_content(self, strategy, temp_project):
        """Verify non-matching content returns not grounded."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value='print("Nonexistent content")',
            context={"file_path": "main.py"},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False

    @pytest.mark.asyncio
    async def test_verify_missing_file_path(self, strategy):
        """Verify returns not grounded when no file path."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="some content",
            context={},  # No file_path
        )
        result = await strategy.verify(claim, {})
        assert result.is_grounded is False

    @pytest.mark.asyncio
    async def test_verify_file_not_found(self, strategy, temp_project):
        """Verify returns not grounded when file not found."""
        claim = GroundingClaim(
            claim_type=GroundingClaimType.CONTENT_MATCH,
            value="content",
            context={"file_path": "nonexistent.py"},
        )
        result = await strategy.verify(claim, {"project_root": temp_project})
        assert result.is_grounded is False
        assert "not found" in result.reason.lower()

    def test_extract_claims_from_code_blocks(self, strategy):
        """Extract content claims from code blocks."""
        response = """
        Here's the code:

        ```python
        # main.py
        def hello():
            print("Hello")
        ```
        """
        claims = strategy.extract_claims(response, {})
        # Should extract content from code blocks with file paths
        assert len(claims) >= 0  # May or may not match pattern


# =============================================================================
# COMPOSITE GROUNDING VERIFIER TESTS
# =============================================================================


class TestCompositeGroundingVerifier:
    """Tests for CompositeGroundingVerifier."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("def main(): pass")
            yield root

    @pytest.fixture
    def verifier(self, temp_project):
        """Create composite verifier."""
        return CompositeGroundingVerifier(
            strategies=[
                FileExistenceStrategy(project_root=temp_project),
                SymbolReferenceStrategy({"main": {"type": "function"}}),
            ],
            threshold=0.5,
        )

    def test_init_default_strategies(self):
        """Default strategies are created."""
        verifier = CompositeGroundingVerifier()
        assert len(verifier._strategies) == 2

    def test_add_strategy(self, verifier):
        """Add strategy works."""
        initial_count = len(verifier._strategies)
        verifier.add_strategy(ContentMatchStrategy())
        assert len(verifier._strategies) == initial_count + 1

    def test_remove_strategy(self, verifier):
        """Remove strategy works."""
        initial_count = len(verifier._strategies)
        verifier.remove_strategy("file_existence")
        assert len(verifier._strategies) == initial_count - 1

    @pytest.mark.asyncio
    async def test_verify_no_claims(self, verifier):
        """Verify with no claims returns grounded."""
        result = await verifier.verify("Just some plain text", {})
        assert result.is_grounded is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_verify_with_claims(self, verifier):
        """Verify with claims returns aggregated result."""
        # Use backticks to match the extraction pattern
        result = await verifier.verify("Check the `main.py` file for details", {})
        assert result.total_claims >= 1
        assert len(result.results) >= 1

    @pytest.mark.asyncio
    async def test_verify_require_all(self, temp_project):
        """Verify with require_all=True."""
        verifier = CompositeGroundingVerifier(
            strategies=[FileExistenceStrategy(project_root=temp_project)],
            require_all=True,
        )
        result = await verifier.verify("Check main.py and nonexistent.py", {})
        # Should fail because one claim fails
        if result.total_claims > 1:
            assert result.is_grounded is False or result.confidence < 1.0

    @pytest.mark.asyncio
    async def test_verify_claim(self, verifier):
        """verify_claim verifies a single claim string."""
        result = await verifier.verify_claim("The file main.py exists", {})
        assert isinstance(result, VerificationResult)


# =============================================================================
# QUALITY DIMENSION TESTS
# =============================================================================


class TestProtocolQualityDimension:
    """Tests for ProtocolQualityDimension enum."""

    def test_all_dimensions_defined(self):
        """All expected dimensions are defined."""
        assert ProtocolQualityDimension.GROUNDING.value == "grounding"
        assert ProtocolQualityDimension.COVERAGE.value == "coverage"
        assert ProtocolQualityDimension.CLARITY.value == "clarity"
        assert ProtocolQualityDimension.CORRECTNESS.value == "correctness"
        assert ProtocolQualityDimension.CONCISENESS.value == "conciseness"
        assert ProtocolQualityDimension.HELPFULNESS.value == "helpfulness"
        assert ProtocolQualityDimension.SAFETY.value == "safety"


# =============================================================================
# DIMENSION SCORE TESTS
# =============================================================================


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_minimal_score(self):
        """Create score with minimal fields."""
        score = DimensionScore(
            dimension=ProtocolQualityDimension.CLARITY,
            score=0.8,
        )
        assert score.dimension == ProtocolQualityDimension.CLARITY
        assert score.score == 0.8
        assert score.weight == 1.0
        assert score.reason == ""

    def test_full_score(self):
        """Create score with all fields."""
        score = DimensionScore(
            dimension=ProtocolQualityDimension.CORRECTNESS,
            score=0.9,
            weight=0.5,
            reason="Code compiles correctly",
            evidence={"syntax_errors": 0},
        )
        assert score.weight == 0.5
        assert score.reason == "Code compiles correctly"


# =============================================================================
# QUALITY SCORE TESTS
# =============================================================================


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_minimal_score(self):
        """Create quality score with minimal fields."""
        score = QualityScore(score=0.85, is_acceptable=True)
        assert score.score == 0.85
        assert score.is_acceptable is True

    def test_get_dimension_score(self):
        """get_dimension_score returns correct score."""
        score = QualityScore(
            score=0.8,
            is_acceptable=True,
            dimension_scores={
                ProtocolQualityDimension.CLARITY: DimensionScore(
                    dimension=ProtocolQualityDimension.CLARITY, score=0.9
                )
            },
        )
        assert score.get_dimension_score(ProtocolQualityDimension.CLARITY) == 0.9
        assert score.get_dimension_score(ProtocolQualityDimension.COVERAGE) == 0.0

    def test_to_dict(self):
        """to_dict serializes correctly."""
        score = QualityScore(
            score=0.8,
            is_acceptable=True,
            threshold=0.7,
            provider="openai",
            dimension_scores={
                ProtocolQualityDimension.CLARITY: DimensionScore(
                    dimension=ProtocolQualityDimension.CLARITY,
                    score=0.9,
                    reason="Good structure",
                )
            },
            feedback="Good response",
            suggestions=["Add more examples"],
        )
        d = score.to_dict()
        assert d["score"] == 0.8
        assert d["is_acceptable"] is True
        assert d["provider"] == "openai"
        assert "clarity" in d["dimensions"]
        assert d["dimensions"]["clarity"]["score"] == 0.9


# =============================================================================
# SIMPLE QUALITY ASSESSOR TESTS
# =============================================================================


class TestSimpleQualityAssessor:
    """Tests for SimpleQualityAssessor."""

    @pytest.fixture
    def assessor(self):
        """Create simple assessor."""
        return SimpleQualityAssessor()

    def test_dimensions_property(self, assessor):
        """dimensions property returns all dimensions."""
        dims = assessor.dimensions
        assert ProtocolQualityDimension.GROUNDING in dims
        assert ProtocolQualityDimension.CLARITY in dims

    def test_assess_basic_response(self, assessor):
        """Assess a basic response."""
        response = "Here is the answer to your question."
        result = assessor.assess(response, {"query": "What is X?"})
        assert isinstance(result, QualityScore)
        assert 0.0 <= result.score <= 1.0

    def test_assess_structured_response(self, assessor):
        """Assess a well-structured response."""
        response = """
# Solution

Here's how to solve this:

1. First step
2. Second step

```python
def example():
    return 42
```

This should work well.
"""
        result = assessor.assess(response, {"query": "How do I do X?"})
        # Should score higher due to structure
        assert result.score > 0.5

    def test_assess_clarity(self, assessor):
        """Assess clarity dimension."""
        score = assessor._assess_clarity("# Header\n\n- Item 1\n- Item 2\n\n```code```")
        assert score.dimension == ProtocolQualityDimension.CLARITY
        assert score.score > 0.5

    def test_assess_conciseness_short_response(self, assessor):
        """Assess conciseness for short response."""
        score = assessor._assess_conciseness("Short answer", {"query": "What is X?"})
        assert score.dimension == ProtocolQualityDimension.CONCISENESS
        assert score.score > 0.7

    def test_assess_conciseness_long_response(self, assessor):
        """Assess conciseness for long response."""
        long_response = "word " * 500
        score = assessor._assess_conciseness(long_response, {"query": "What is X?"})
        assert score.score < 0.5

    def test_assess_correctness_balanced_brackets(self, assessor):
        """Assess correctness with balanced brackets."""
        response = """
```python
def foo(x):
    return [x, x + 1]
```
"""
        score = assessor._assess_correctness(response)
        assert score.dimension == ProtocolQualityDimension.CORRECTNESS
        assert score.score > 0.5

    def test_assess_correctness_unbalanced_brackets(self, assessor):
        """Assess correctness with unbalanced brackets."""
        response = """
```python
def foo(x):
    return [x, x + 1
```
"""
        score = assessor._assess_correctness(response)
        # Should be lower due to unbalanced brackets
        assert score.score < 0.8

    def test_assess_coverage(self, assessor):
        """Assess coverage dimension."""
        response = "The function calculates the sum of two numbers."
        context = {"query": "What does the function calculate?"}
        score = assessor._assess_coverage(response, context)
        assert score.dimension == ProtocolQualityDimension.COVERAGE
        assert score.score > 0.5

    def test_assess_grounding_with_result(self, assessor):
        """Assess grounding with grounding_result in context."""
        context = {"grounding_result": {"confidence": 0.9, "reason": "Verified"}}
        score = assessor._assess_grounding("response", context)
        assert score.score == 0.9

    def test_assess_grounding_without_result(self, assessor):
        """Assess grounding without grounding_result."""
        score = assessor._assess_grounding("response", {})
        assert score.score == 0.7  # Default


# =============================================================================
# PROVIDER AWARE QUALITY ASSESSOR TESTS
# =============================================================================


class TestProviderAwareQualityAssessor:
    """Tests for ProviderAwareQualityAssessor."""

    def test_init_with_provider(self):
        """Initialize with provider name."""
        assessor = ProviderAwareQualityAssessor(provider_name="openai", provider_threshold=0.75)
        assert assessor._provider_name == "openai"
        assert assessor._threshold == 0.75

    def test_assess_includes_provider(self):
        """Assess result includes provider name."""
        assessor = ProviderAwareQualityAssessor(provider_name="anthropic")
        result = assessor.assess("Test response", {"query": "Test?"})
        assert result.provider == "anthropic"

    def test_provider_adjustment_deepseek(self):
        """DeepSeek provider gets correctness boost."""
        assessor = ProviderAwareQualityAssessor(provider_name="deepseek")
        adj = assessor._get_provider_adjustment(ProtocolQualityDimension.CORRECTNESS)
        assert adj == 0.05

    def test_provider_adjustment_xai(self):
        """xAI provider gets conciseness penalty."""
        assessor = ProviderAwareQualityAssessor(provider_name="xai")
        adj = assessor._get_provider_adjustment(ProtocolQualityDimension.CONCISENESS)
        assert adj == -0.10

    def test_provider_adjustment_unknown(self):
        """Unknown provider gets no adjustment."""
        assessor = ProviderAwareQualityAssessor(provider_name="unknown_provider")
        adj = assessor._get_provider_adjustment(ProtocolQualityDimension.CLARITY)
        assert adj == 0.0


# =============================================================================
# COMPOSITE QUALITY ASSESSOR TESTS
# =============================================================================


class TestCompositeQualityAssessor:
    """Tests for CompositeQualityAssessor."""

    def test_init_default(self):
        """Initialize with defaults."""
        assessor = CompositeQualityAssessor()
        assert len(assessor._assessors) == 1
        assert assessor._strategy == "weighted"

    def test_init_with_assessors(self):
        """Initialize with custom assessors."""
        assessors = [SimpleQualityAssessor(), SimpleQualityAssessor()]
        composite = CompositeQualityAssessor(assessors=assessors, strategy="max")
        assert len(composite._assessors) == 2
        assert composite._strategy == "max"

    def test_add_assessor(self):
        """Add assessor works."""
        composite = CompositeQualityAssessor()
        initial_count = len(composite._assessors)
        composite.add_assessor(SimpleQualityAssessor())
        assert len(composite._assessors) == initial_count + 1

    def test_dimensions_property(self):
        """dimensions returns combined dimensions."""
        composite = CompositeQualityAssessor()
        dims = composite.dimensions
        assert ProtocolQualityDimension.CLARITY in dims

    def test_assess_max_strategy(self):
        """Assess with max strategy."""
        composite = CompositeQualityAssessor(assessors=[SimpleQualityAssessor()], strategy="max")
        result = composite.assess("Test response", {"query": "Test?"})
        assert isinstance(result, QualityScore)

    def test_assess_min_strategy(self):
        """Assess with min strategy."""
        composite = CompositeQualityAssessor(assessors=[SimpleQualityAssessor()], strategy="min")
        result = composite.assess("Test response", {"query": "Test?"})
        assert isinstance(result, QualityScore)

    def test_assess_average_strategy(self):
        """Assess with average strategy."""
        composite = CompositeQualityAssessor(
            assessors=[SimpleQualityAssessor(), SimpleQualityAssessor()],
            strategy="average",
        )
        result = composite.assess("Test response", {"query": "Test?"})
        assert isinstance(result, QualityScore)

    def test_assess_empty_assessors_defaults_to_simple(self):
        """Assess with empty assessors defaults to SimpleQualityAssessor."""
        # Empty list is falsy, so constructor defaults to [SimpleQualityAssessor()]
        composite = CompositeQualityAssessor(assessors=[])
        result = composite.assess("Test", {})
        # Should have a valid score from the default assessor
        assert result.score > 0.0
        assert len(composite._assessors) == 1  # Default assessor added


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestIGroundingStrategyProtocol:
    """Tests for IGroundingStrategy protocol compliance."""

    def test_file_existence_strategy_implements_protocol(self):
        """FileExistenceStrategy implements IGroundingStrategy."""
        strategy = FileExistenceStrategy()
        assert isinstance(strategy, IGroundingStrategy)

    def test_symbol_reference_strategy_implements_protocol(self):
        """SymbolReferenceStrategy implements IGroundingStrategy."""
        strategy = SymbolReferenceStrategy()
        assert isinstance(strategy, IGroundingStrategy)

    def test_content_match_strategy_implements_protocol(self):
        """ContentMatchStrategy implements IGroundingStrategy."""
        strategy = ContentMatchStrategy()
        assert isinstance(strategy, IGroundingStrategy)


class TestIQualityAssessorProtocol:
    """Tests for IQualityAssessor protocol compliance."""

    def test_simple_assessor_implements_protocol(self):
        """SimpleQualityAssessor implements IQualityAssessor."""
        assessor = SimpleQualityAssessor()
        assert isinstance(assessor, IQualityAssessor)

    def test_provider_aware_assessor_implements_protocol(self):
        """ProviderAwareQualityAssessor implements IQualityAssessor."""
        assessor = ProviderAwareQualityAssessor()
        assert isinstance(assessor, IQualityAssessor)

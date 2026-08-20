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

"""Tests for quality assessment protocols and implementations."""

import pytest

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
# QUALITY DIMENSION TESTS
# =============================================================================


class TestProtocolQualityDimension:
    """Tests for ProtocolQualityDimension enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
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

    def test_creation(self):
        """Test basic dimension score creation."""
        score = DimensionScore(
            dimension=ProtocolQualityDimension.CLARITY,
            score=0.85,
        )
        assert score.dimension == ProtocolQualityDimension.CLARITY
        assert score.score == 0.85
        assert score.weight == 1.0
        assert score.reason == ""
        assert score.evidence == {}

    def test_creation_with_all_fields(self):
        """Test dimension score with all fields."""
        score = DimensionScore(
            dimension=ProtocolQualityDimension.CORRECTNESS,
            score=0.9,
            weight=0.25,
            reason="Code syntax is correct",
            evidence={"balanced_parens": True},
        )
        assert score.weight == 0.25
        assert score.reason == "Code syntax is correct"
        assert score.evidence["balanced_parens"] is True


# =============================================================================
# QUALITY SCORE TESTS
# =============================================================================


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_creation(self):
        """Test basic quality score creation."""
        score = QualityScore(
            score=0.82,
            is_acceptable=True,
        )
        assert score.score == 0.82
        assert score.is_acceptable is True
        assert score.threshold == 0.80
        assert score.provider == ""
        assert score.dimension_scores == {}
        assert score.feedback == ""
        assert score.suggestions == []

    def test_creation_with_dimensions(self):
        """Test quality score with dimension scores."""
        dim_scores = {
            ProtocolQualityDimension.CLARITY: DimensionScore(
                dimension=ProtocolQualityDimension.CLARITY,
                score=0.9,
            ),
            ProtocolQualityDimension.CORRECTNESS: DimensionScore(
                dimension=ProtocolQualityDimension.CORRECTNESS,
                score=0.8,
            ),
        }
        score = QualityScore(
            score=0.85,
            is_acceptable=True,
            dimension_scores=dim_scores,
        )
        assert len(score.dimension_scores) == 2

    def test_get_dimension_score(self):
        """Test get_dimension_score method."""
        dim_scores = {
            ProtocolQualityDimension.CLARITY: DimensionScore(
                dimension=ProtocolQualityDimension.CLARITY,
                score=0.9,
            ),
        }
        score = QualityScore(
            score=0.85,
            is_acceptable=True,
            dimension_scores=dim_scores,
        )
        assert score.get_dimension_score(ProtocolQualityDimension.CLARITY) == 0.9
        assert score.get_dimension_score(ProtocolQualityDimension.CORRECTNESS) == 0.0

    def test_to_dict(self):
        """Test to_dict method."""
        dim_scores = {
            ProtocolQualityDimension.CLARITY: DimensionScore(
                dimension=ProtocolQualityDimension.CLARITY,
                score=0.9,
                weight=0.15,
                reason="Well structured",
            ),
        }
        score = QualityScore(
            score=0.85,
            is_acceptable=True,
            threshold=0.80,
            provider="openai",
            dimension_scores=dim_scores,
            feedback="Good response",
            suggestions=["Consider more examples"],
        )

        result = score.to_dict()

        assert result["score"] == 0.85
        assert result["is_acceptable"] is True
        assert result["threshold"] == 0.80
        assert result["provider"] == "openai"
        assert result["feedback"] == "Good response"
        assert len(result["suggestions"]) == 1
        assert "clarity" in result["dimensions"]


# =============================================================================
# SIMPLE QUALITY ASSESSOR TESTS
# =============================================================================


class TestSimpleQualityAssessor:
    """Tests for SimpleQualityAssessor."""

    @pytest.fixture
    def assessor(self):
        """Create a simple assessor."""
        return SimpleQualityAssessor()

    def test_dimensions_property(self, assessor):
        """Test dimensions property."""
        dims = assessor.dimensions
        assert ProtocolQualityDimension.CLARITY in dims
        assert ProtocolQualityDimension.CORRECTNESS in dims
        assert ProtocolQualityDimension.COVERAGE in dims
        assert ProtocolQualityDimension.CONCISENESS in dims
        assert ProtocolQualityDimension.GROUNDING in dims

    def test_assess_basic_response(self, assessor):
        """Test assessing a basic response."""
        response = "Here is a simple answer to your question."
        context = {"query": "What is the answer?"}

        score = assessor.assess(response, context)

        assert isinstance(score, QualityScore)
        assert 0.0 <= score.score <= 1.0
        assert isinstance(score.is_acceptable, bool)

    def test_assess_structured_response(self, assessor):
        """Test assessing a well-structured response."""
        response = """# Solution

Here's the implementation:

```python
def solve():
    return 42
```

## Key Points

- Point 1
- Point 2

1. Step one
2. Step two
"""
        context = {"query": "How do I implement this?"}

        score = assessor.assess(response, context)

        # Structured responses should score well on clarity
        clarity = score.get_dimension_score(ProtocolQualityDimension.CLARITY)
        assert clarity >= 0.5

    def test_assess_code_correctness(self, assessor):
        """Test assessing code correctness."""
        # Good code
        good_response = """
```python
def add(a, b):
    return a + b
```
"""
        score = assessor.assess(good_response, {})
        good_correctness = score.get_dimension_score(ProtocolQualityDimension.CORRECTNESS)

        # Bad code with unbalanced brackets
        bad_response = """
```python
def add(a, b:
    return a + b
```
"""
        score = assessor.assess(bad_response, {})
        bad_correctness = score.get_dimension_score(ProtocolQualityDimension.CORRECTNESS)

        assert good_correctness >= bad_correctness

    def test_assess_conciseness(self, assessor):
        """Test assessing conciseness."""
        query = "What is 2+2?"

        # Concise response
        concise = "The answer is 4."
        concise_score = assessor.assess(concise, {"query": query})

        # Verbose response
        verbose = """
Well, let me think about this mathematical problem in great detail.
When we consider the addition operation, we take two numbers and
combine them together. In this case, we have the number 2 and we
are adding another 2 to it. Using the rules of arithmetic, we can
determine that 2 + 2 equals 4. Therefore, the final answer to your
question about what 2+2 equals is simply 4.
"""
        verbose_score = assessor.assess(verbose, {"query": query})

        concise_val = concise_score.get_dimension_score(ProtocolQualityDimension.CONCISENESS)
        verbose_val = verbose_score.get_dimension_score(ProtocolQualityDimension.CONCISENESS)

        assert concise_val >= verbose_val

    def test_assess_coverage(self, assessor):
        """Test assessing query coverage."""
        query = "Explain authentication and authorization in Python"

        # Good coverage
        good_response = "Authentication verifies identity. Authorization controls access. In Python, use libraries like Flask-Login for authentication and role-based authorization."
        good_score = assessor.assess(good_response, {"query": query})

        # Poor coverage
        poor_response = "Here's how to use Python."
        poor_score = assessor.assess(poor_response, {"query": query})

        good_coverage = good_score.get_dimension_score(ProtocolQualityDimension.COVERAGE)
        poor_coverage = poor_score.get_dimension_score(ProtocolQualityDimension.COVERAGE)

        assert good_coverage > poor_coverage

    def test_assess_grounding_with_result(self, assessor):
        """Test assessing grounding with provided result."""
        response = "The function is in module.py"
        context = {
            "grounding_result": {
                "confidence": 0.95,
                "reason": "File verified to exist",
            }
        }

        score = assessor.assess(response, context)
        grounding = score.get_dimension_score(ProtocolQualityDimension.GROUNDING)

        assert grounding == 0.95

    def test_assess_clarity_with_headers(self, assessor):
        """Test clarity assessment rewards headers."""
        with_headers = "# Title\n\nContent here\n\n## Section\n\nMore content"
        without_headers = "Title Content here Section More content"

        with_score = assessor.assess(with_headers, {})
        without_score = assessor.assess(without_headers, {})

        with_clarity = with_score.get_dimension_score(ProtocolQualityDimension.CLARITY)
        without_clarity = without_score.get_dimension_score(ProtocolQualityDimension.CLARITY)

        assert with_clarity >= without_clarity


# =============================================================================
# PROVIDER AWARE QUALITY ASSESSOR TESTS
# =============================================================================


class TestProviderAwareQualityAssessor:
    """Tests for ProviderAwareQualityAssessor."""

    def test_init_with_provider(self):
        """Test initialization with provider."""
        assessor = ProviderAwareQualityAssessor(
            provider_name="anthropic",
            provider_threshold=0.85,
        )
        assert assessor._provider_name == "anthropic"
        assert assessor._threshold == 0.85

    def test_assess_includes_provider(self):
        """Test assessment includes provider name."""
        assessor = ProviderAwareQualityAssessor(provider_name="openai")
        response = "A response"
        score = assessor.assess(response, {})

        assert score.provider == "openai"

    def test_provider_adjustment_anthropic(self):
        """Test Anthropic gets positive adjustments."""
        assessor = ProviderAwareQualityAssessor(provider_name="anthropic")

        clarity_adj = assessor._get_provider_adjustment(ProtocolQualityDimension.CLARITY)
        grounding_adj = assessor._get_provider_adjustment(ProtocolQualityDimension.GROUNDING)

        assert clarity_adj > 0
        assert grounding_adj > 0

    def test_provider_adjustment_xai(self):
        """Test xAI gets negative conciseness adjustment."""
        assessor = ProviderAwareQualityAssessor(provider_name="xai")

        conciseness_adj = assessor._get_provider_adjustment(ProtocolQualityDimension.CONCISENESS)

        assert conciseness_adj < 0

    def test_provider_adjustment_ollama(self):
        """Test Ollama gets appropriate adjustments."""
        assessor = ProviderAwareQualityAssessor(provider_name="ollama")

        correctness_adj = assessor._get_provider_adjustment(ProtocolQualityDimension.CORRECTNESS)
        grounding_adj = assessor._get_provider_adjustment(ProtocolQualityDimension.GROUNDING)

        assert correctness_adj < 0
        assert grounding_adj < 0

    def test_provider_adjustment_unknown(self):
        """Test unknown provider gets zero adjustment."""
        assessor = ProviderAwareQualityAssessor(provider_name="unknown_provider")

        adjustment = assessor._get_provider_adjustment(ProtocolQualityDimension.CLARITY)

        assert adjustment == 0.0

    def test_assess_dimension_with_adjustment(self):
        """Test dimension assessment includes provider adjustment."""
        assessor = ProviderAwareQualityAssessor(provider_name="deepseek")
        response = """
```python
def example():
    return 42
```
"""
        dim_score = assessor._assess_dimension(ProtocolQualityDimension.CORRECTNESS, response, {})

        assert "provider adjustment" in dim_score.reason


# =============================================================================
# COMPOSITE QUALITY ASSESSOR TESTS
# =============================================================================


class TestCompositeQualityAssessor:
    """Tests for CompositeQualityAssessor."""

    def test_init_default(self):
        """Test default initialization."""
        assessor = CompositeQualityAssessor()
        assert len(assessor._assessors) >= 1
        assert assessor._strategy == "weighted"

    def test_init_custom_strategy(self):
        """Test initialization with custom strategy."""
        assessor = CompositeQualityAssessor(strategy="max")
        assert assessor._strategy == "max"

    def test_add_assessor(self):
        """Test adding an assessor."""
        assessor = CompositeQualityAssessor()
        initial_count = len(assessor._assessors)

        assessor.add_assessor(SimpleQualityAssessor())

        assert len(assessor._assessors) == initial_count + 1

    def test_dimensions_property(self):
        """Test dimensions property combines all assessors."""
        assessor = CompositeQualityAssessor(
            assessors=[
                SimpleQualityAssessor(),
                ProviderAwareQualityAssessor(provider_name="openai"),
            ]
        )

        dims = assessor.dimensions
        assert ProtocolQualityDimension.CLARITY in dims
        assert ProtocolQualityDimension.CORRECTNESS in dims

    def test_assess_weighted_strategy(self):
        """Test assessment with weighted strategy."""
        assessor = CompositeQualityAssessor(
            assessors=[SimpleQualityAssessor()],
            strategy="weighted",
        )

        response = "A test response"
        score = assessor.assess(response, {})

        assert isinstance(score, QualityScore)
        assert 0.0 <= score.score <= 1.0

    def test_assess_max_strategy(self):
        """Test assessment with max strategy."""
        assessor = CompositeQualityAssessor(
            assessors=[
                SimpleQualityAssessor(),
                ProviderAwareQualityAssessor(provider_name="anthropic"),
            ],
            strategy="max",
        )

        response = "A test response"
        score = assessor.assess(response, {})

        # Max strategy takes highest score
        assert isinstance(score, QualityScore)

    def test_assess_min_strategy(self):
        """Test assessment with min strategy."""
        assessor = CompositeQualityAssessor(
            assessors=[
                SimpleQualityAssessor(),
                ProviderAwareQualityAssessor(provider_name="anthropic"),
            ],
            strategy="min",
        )

        response = "A test response"
        score = assessor.assess(response, {})

        # Min strategy takes lowest score
        assert isinstance(score, QualityScore)

    def test_assess_empty_assessors(self):
        """Test assessment with empty assessors list falls back to default."""
        # Note: Empty list is falsy in Python, so `assessors or [default]` uses default
        assessor = CompositeQualityAssessor(assessors=[])

        response = "A test response"
        score = assessor.assess(response, {"query": "test"})

        # With default fallback, we should get a valid score
        assert isinstance(score, QualityScore)
        assert 0.0 <= score.score <= 1.0

    def test_assess_combines_dimensions(self):
        """Test assessment combines dimension scores."""
        assessor = CompositeQualityAssessor(
            assessors=[
                SimpleQualityAssessor(),
                ProviderAwareQualityAssessor(provider_name="openai"),
            ],
        )

        response = """# Title
Some content here.
```python
print("hello")
```
"""
        score = assessor.assess(response, {"query": "test"})

        # Should have merged dimensions
        assert len(score.dimension_scores) >= 1


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Tests for IQualityAssessor protocol compliance."""

    def test_simple_assessor_is_protocol(self):
        """Test SimpleQualityAssessor implements protocol."""
        assessor = SimpleQualityAssessor()
        assert isinstance(assessor, IQualityAssessor)

    def test_provider_aware_assessor_is_protocol(self):
        """Test ProviderAwareQualityAssessor implements protocol."""
        assessor = ProviderAwareQualityAssessor()
        assert isinstance(assessor, IQualityAssessor)

    def test_composite_assessor_is_protocol(self):
        """Test CompositeQualityAssessor implements protocol."""
        assessor = CompositeQualityAssessor()
        assert isinstance(assessor, IQualityAssessor)

    def test_custom_assessor_protocol(self):
        """Test custom assessor can implement protocol."""

        class CustomAssessor:
            def assess(self, response, context):
                return QualityScore(score=1.0, is_acceptable=True)

            @property
            def dimensions(self):
                return [ProtocolQualityDimension.CLARITY]

        assessor = CustomAssessor()
        assert isinstance(assessor, IQualityAssessor)


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_assess_empty_response(self):
        """Test assessing empty response."""
        assessor = SimpleQualityAssessor()
        score = assessor.assess("", {})

        assert isinstance(score, QualityScore)
        # Empty response should still produce a score

    def test_assess_very_long_response(self):
        """Test assessing very long response."""
        assessor = SimpleQualityAssessor()
        response = "word " * 10000

        score = assessor.assess(response, {"query": "test"})

        assert isinstance(score, QualityScore)
        # Very long response may have lower conciseness

    def test_assess_unicode_content(self):
        """Test assessing unicode content."""
        assessor = SimpleQualityAssessor()
        response = "这是中文内容。日本語もあります。"

        score = assessor.assess(response, {})

        assert isinstance(score, QualityScore)

    def test_assess_only_code_blocks(self):
        """Test assessing response with only code."""
        assessor = SimpleQualityAssessor()
        response = """```python
def add(a, b):
    return a + b
```"""

        score = assessor.assess(response, {})
        correctness = score.get_dimension_score(ProtocolQualityDimension.CORRECTNESS)

        assert correctness >= 0.5

    def test_custom_weights(self):
        """Test assessor with custom weights."""
        custom_weights = {
            ProtocolQualityDimension.CORRECTNESS: 0.5,
            ProtocolQualityDimension.CLARITY: 0.5,
        }
        assessor = SimpleQualityAssessor(weights=custom_weights)

        assert len(assessor.dimensions) == 2

    def test_custom_threshold(self):
        """Test assessor with custom threshold."""
        assessor = SimpleQualityAssessor(threshold=0.95)

        response = "A decent response"
        score = assessor.assess(response, {})

        # With high threshold, many responses won't be acceptable
        assert score.threshold == 0.95

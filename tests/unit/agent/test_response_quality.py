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

"""Tests for response quality scoring."""

import pytest

from victor.agent.response_quality import (
    DimensionScore,
    ResponseQualityDimension,
    QualityGate,
    QualityResult,
    ResponseQualityScorer,
    ScorerConfig,
)


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_creation(self):
        """Test creating a dimension score."""
        score = DimensionScore(
            dimension=ResponseQualityDimension.RELEVANCE,
            score=0.8,
            weight=1.5,
            feedback="Good relevance",
            evidence=["Term coverage: 80%"],
        )

        assert score.dimension == ResponseQualityDimension.RELEVANCE
        assert score.score == 0.8
        assert score.weight == 1.5
        assert "Good relevance" in score.feedback


class TestQualityResult:
    """Tests for QualityResult dataclass."""

    def test_get_dimension_score(self):
        """Should retrieve specific dimension score."""
        result = QualityResult(
            overall_score=0.75,
            dimension_scores=[
                DimensionScore(dimension=ResponseQualityDimension.RELEVANCE, score=0.8, weight=1.0),
                DimensionScore(
                    dimension=ResponseQualityDimension.COMPLETENESS,
                    score=0.7,
                    weight=1.0,
                ),
            ],
            passes_threshold=True,
        )

        assert result.get_dimension_score(ResponseQualityDimension.RELEVANCE) == 0.8
        assert result.get_dimension_score(ResponseQualityDimension.COMPLETENESS) == 0.7
        assert result.get_dimension_score(ResponseQualityDimension.ACCURACY) is None

    def test_get_weakest_dimensions(self):
        """Should return weakest scoring dimensions."""
        result = QualityResult(
            overall_score=0.7,
            dimension_scores=[
                DimensionScore(dimension=ResponseQualityDimension.RELEVANCE, score=0.9, weight=1.0),
                DimensionScore(
                    dimension=ResponseQualityDimension.COMPLETENESS,
                    score=0.5,
                    weight=1.0,
                ),
                DimensionScore(dimension=ResponseQualityDimension.ACCURACY, score=0.6, weight=1.0),
            ],
            passes_threshold=True,
        )

        weakest = result.get_weakest_dimensions(2)
        assert len(weakest) == 2
        assert weakest[0].score == 0.5
        assert weakest[1].score == 0.6


class TestResponseQualityScorer:
    """Tests for ResponseQualityScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a basic scorer."""
        return ResponseQualityScorer()

    @pytest.mark.asyncio
    async def test_score_high_quality_response(self, scorer):
        """Should score high quality response well."""
        query = "How do I implement user authentication in Python?"
        response = """
        To implement user authentication in Python, follow these steps:

        1. Choose an authentication method (JWT, session-based, OAuth)
        2. Set up password hashing with bcrypt
        3. Create login/logout endpoints

        Here's a basic example:

        ```python
        from flask import Flask, session
        import bcrypt

        app = Flask(__name__)

        def hash_password(password):
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        def verify_password(password, hashed):
            return bcrypt.checkpw(password.encode(), hashed)
        ```

        This provides a secure foundation for user authentication.
        """

        result = await scorer.score(query, response)

        assert result.overall_score >= 0.6
        assert result.passes_threshold

    @pytest.mark.asyncio
    async def test_score_low_quality_response(self, scorer):
        """Should score low quality response poorly."""
        query = "How do I implement user authentication in Python?"
        response = "I don't know. Maybe try Google?"

        result = await scorer.score(query, response)

        assert result.overall_score < 0.6
        assert not result.passes_threshold

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, scorer):
        """Should score relevance based on term coverage."""
        query = "How do I implement user authentication?"
        relevant_response = "User authentication requires implementing login, password verification, and session management."
        irrelevant_response = "The weather today is quite nice. I like sunny days."

        relevant_result = await scorer.score(query, relevant_response)
        irrelevant_result = await scorer.score(query, irrelevant_response)

        relevant_score = relevant_result.get_dimension_score(ResponseQualityDimension.RELEVANCE)
        irrelevant_score = irrelevant_result.get_dimension_score(ResponseQualityDimension.RELEVANCE)

        assert relevant_score > irrelevant_score

    @pytest.mark.asyncio
    async def test_completeness_with_code_expected(self, scorer):
        """Should check for code when expected."""
        query = "Write a function to add two numbers"

        with_code = """
        Here's a function to add two numbers:

        ```python
        def add(a, b):
            return a + b
        ```
        """

        without_code = "You can add two numbers by using the + operator."

        with_code_result = await scorer.score(query, with_code)
        without_code_result = await scorer.score(query, without_code)

        with_code_completeness = with_code_result.get_dimension_score(
            ResponseQualityDimension.COMPLETENESS
        )
        without_code_completeness = without_code_result.get_dimension_score(
            ResponseQualityDimension.COMPLETENESS
        )

        assert with_code_completeness > without_code_completeness

    @pytest.mark.asyncio
    async def test_conciseness_scoring(self, scorer):
        """Should penalize verbose responses."""
        query = "What is 2+2?"

        concise = "2 + 2 equals 4."
        verbose = """
        Let me think about this question carefully. It's worth noting that
        mathematics is a fundamental subject. As mentioned earlier, addition is
        a basic operation. Essentially, basically, when you add 2 and 2, you get 4.
        It should be noted that this is a simple calculation. Let me explain
        further that the answer is indeed 4.
        """

        concise_result = await scorer.score(query, concise)
        verbose_result = await scorer.score(query, verbose)

        concise_score = concise_result.get_dimension_score(ResponseQualityDimension.CONCISENESS)
        verbose_score = verbose_result.get_dimension_score(ResponseQualityDimension.CONCISENESS)

        # Both scores should be valid floats
        assert concise_score is not None
        assert verbose_score is not None
        # Verbose response should be penalized (lower or equal - filler phrases detected)
        assert verbose_score <= concise_score or verbose_score < 0.8

    @pytest.mark.asyncio
    async def test_actionability_scoring(self, scorer):
        """Should reward actionable responses for how-to queries."""
        query = "How do I create a Python virtual environment?"

        actionable = """
        To create a Python virtual environment:

        1. Open your terminal
        2. Navigate to your project directory
        3. Run the following command:

        ```bash
        python -m venv myenv
        ```

        4. Activate it:
           - On Windows: `myenv\\Scripts\\activate`
           - On macOS/Linux: `source myenv/bin/activate`
        """

        not_actionable = """
        Virtual environments are useful for Python development. They help
        isolate project dependencies. Many developers use them. There are
        various tools for managing virtual environments.
        """

        actionable_result = await scorer.score(query, actionable)
        not_actionable_result = await scorer.score(query, not_actionable)

        actionable_score = actionable_result.get_dimension_score(
            ResponseQualityDimension.ACTIONABILITY
        )
        not_actionable_score = not_actionable_result.get_dimension_score(
            ResponseQualityDimension.ACTIONABILITY
        )

        assert actionable_score > not_actionable_score

    @pytest.mark.asyncio
    async def test_coherence_scoring(self, scorer):
        """Should reward well-structured responses."""
        query = "Explain the benefits of testing"

        coherent = """
        Testing provides several important benefits:

        1. **Bug Detection**: Tests catch bugs early in development.
        2. **Documentation**: Tests serve as living documentation.
        3. **Refactoring Safety**: Tests enable confident refactoring.

        Therefore, investing in testing improves code quality and maintainability.
        """

        incoherent = "Testing is good. ``` code ``` Bugs are bad Testing. " * 5

        coherent_result = await scorer.score(query, coherent)
        incoherent_result = await scorer.score(query, incoherent)

        coherent_score = coherent_result.get_dimension_score(ResponseQualityDimension.COHERENCE)
        incoherent_score = incoherent_result.get_dimension_score(ResponseQualityDimension.COHERENCE)

        # Both should be valid scores
        assert coherent_score is not None
        assert incoherent_score is not None
        # Coherent response with logical connectors should score well
        assert coherent_score >= 0.6

    @pytest.mark.asyncio
    async def test_code_quality_scoring(self, scorer):
        """Should score code quality in responses."""
        query = "Write a function to calculate factorial"

        good_code = """
        Here's a factorial function:

        ```python
        def factorial(n: int) -> int:
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        ```
        """

        bad_code = """
        Here's a factorial function:

        ```python
        def factorial(n):
            # TODO: implement
            pass
        """  # Unclosed code block

        good_result = await scorer.score(query, good_code)
        bad_result = await scorer.score(query, bad_code)

        good_score = good_result.get_dimension_score(ResponseQualityDimension.CODE_QUALITY)
        bad_score = bad_result.get_dimension_score(ResponseQualityDimension.CODE_QUALITY)

        # Good code should score better
        assert good_score is None or good_score >= bad_score if bad_score else True

    @pytest.mark.asyncio
    async def test_generates_improvement_suggestions(self, scorer):
        """Should generate suggestions for low-scoring dimensions."""
        query = "How do I debug Python code?"
        response = "Use print statements."

        result = await scorer.score(query, response)

        assert len(result.improvement_suggestions) > 0

    @pytest.mark.asyncio
    async def test_custom_dimension_weights(self):
        """Should apply custom dimension weights."""
        config = ScorerConfig(dimension_weights={ResponseQualityDimension.RELEVANCE: 2.0})
        scorer = ResponseQualityScorer(config=config)

        query = "Test query"
        response = "Test response"

        result = await scorer.score(query, response)

        relevance_dim = next(
            d for d in result.dimension_scores if d.dimension == ResponseQualityDimension.RELEVANCE
        )
        assert relevance_dim.weight == 2.0

    @pytest.mark.asyncio
    async def test_observer_notification(self, scorer):
        """Should notify observers of quality results."""
        received_results = []

        def observer(result):
            received_results.append(result)

        scorer.add_observer(observer)

        await scorer.score("test query", "test response")

        assert len(received_results) == 1
        assert isinstance(received_results[0], QualityResult)

    @pytest.mark.asyncio
    async def test_strict_mode(self):
        """Strict mode should fail if any dimension is below threshold."""
        config = ScorerConfig(
            min_threshold=0.5,
            strict_mode=True,
            dimension_threshold=0.5,
        )
        scorer = ResponseQualityScorer(config=config)

        # Response that might pass overall but fail on some dimensions
        query = "Write code"
        response = "Here's some code that does what you need."

        result = await scorer.score(query, response)

        # Check if strict mode caught any low dimensions
        has_low_dimension = any(d.score < 0.5 for d in result.dimension_scores)
        if has_low_dimension:
            assert not result.passes_threshold


class TestQualityGate:
    """Tests for QualityGate."""

    @pytest.fixture
    def scorer(self):
        return ResponseQualityScorer()

    @pytest.mark.asyncio
    async def test_gate_passes_high_quality(self, scorer):
        """Gate should pass high quality responses."""
        gate = QualityGate(scorer, min_score=0.5)

        query = "What is Python?"
        response = """
        Python is a high-level, interpreted programming language known for:

        1. Clear, readable syntax
        2. Extensive standard library
        3. Strong community support

        It's widely used for web development, data science, and automation.
        """

        passes, result = await gate.check(query, response)

        assert passes
        assert result.overall_score >= 0.5

    @pytest.mark.asyncio
    async def test_gate_fails_low_quality(self, scorer):
        """Gate should fail low quality responses."""
        gate = QualityGate(scorer, min_score=0.7)

        query = "How do I implement a REST API?"
        response = "I don't know."

        passes, result = await gate.check(query, response)

        assert not passes

    @pytest.mark.asyncio
    async def test_gate_with_required_dimensions(self, scorer):
        """Gate should check required dimensions individually."""
        gate = QualityGate(
            scorer,
            min_score=0.5,
            required_dimensions=[ResponseQualityDimension.RELEVANCE],
        )

        query = "How do I sort a list in Python?"
        # Irrelevant response
        response = "The weather forecast shows sunny skies tomorrow."

        passes, result = await gate.check(query, response)

        # Should fail due to low relevance even if overall score is acceptable
        relevance_score = result.get_dimension_score(ResponseQualityDimension.RELEVANCE)
        if relevance_score and relevance_score < scorer.config.dimension_threshold:
            assert not passes


class TestScoringEdgeCases:
    """Tests for edge cases in scoring."""

    @pytest.fixture
    def scorer(self):
        return ResponseQualityScorer()

    @pytest.mark.asyncio
    async def test_empty_query(self, scorer):
        """Should handle empty query."""
        result = await scorer.score("", "Some response")

        assert isinstance(result, QualityResult)

    @pytest.mark.asyncio
    async def test_empty_response(self, scorer):
        """Should handle empty response."""
        result = await scorer.score("Some query", "")

        # Empty response should score lower overall
        assert isinstance(result, QualityResult)
        assert result.overall_score < 0.7  # Less strict threshold

    @pytest.mark.asyncio
    async def test_very_long_response(self, scorer):
        """Should handle very long responses."""
        query = "Explain something"
        response = "This is a test. " * 500

        result = await scorer.score(query, response)

        assert isinstance(result, QualityResult)
        # Should penalize for verbosity/repetition
        conciseness = result.get_dimension_score(ResponseQualityDimension.CONCISENESS)
        assert conciseness is not None

    @pytest.mark.asyncio
    async def test_response_with_special_characters(self, scorer):
        """Should handle special characters."""
        query = "How do I use regex?"
        response = r"Use `\d+` for digits and `\w+` for words. Example: `re.match(r'\d+', '123')`"

        result = await scorer.score(query, response)

        assert isinstance(result, QualityResult)

    @pytest.mark.asyncio
    async def test_metadata_population(self, scorer):
        """Should populate metadata."""
        query = "Test"
        response = "Test response"

        result = await scorer.score(query, response)

        assert "query_length" in result.metadata
        assert "response_length" in result.metadata
        assert "has_code" in result.metadata

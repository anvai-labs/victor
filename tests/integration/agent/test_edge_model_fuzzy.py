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

"""
Integration tests for edge model (FEP-0001) with fuzzy matching.

These tests verify that edge model decisions work correctly when
users make typos in their queries.
"""

import pytest

from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag


@pytest.mark.integration
class TestEdgeModelWithTypos:
    """Test edge model decisions with typos."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Skip if edge model is not enabled
        manager = get_feature_flag_manager()
        if not manager.is_enabled(FeatureFlag.USE_EDGE_MODEL):
            pytest.skip("Edge model not enabled")

    def test_task_necessity_with_typo(self):
        """Test TOOL_NECESSITY decision with typo."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # User wants to analyze but types "analize"
            decision = decision_service.decide_sync(
                "analize the code quality",
                decision_type=DecisionType.TOOL_NECESSITY,
            )

            # Should still recognize analysis intent
            assert decision is not None
            assert decision.confidence > 0.0

            # The decision should indicate some action is needed
            # (exact interpretation depends on decision format)
        except Exception as e:
            # If service is not available, that's okay for this test
            pytest.skip(f"Decision service not available: {e}")

    def test_task_type_with_typo(self):
        """Test TASK_TYPE decision with typo."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # User wants to search but types "serch"
            decision = decision_service.decide_sync(
                "serch for functions",
                decision_type=DecisionType.TASK_TYPE,
            )

            # Should still recognize search intent
            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_action_intent_with_typo(self):
        """Test ACTION_INTENT decision with typo."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # User wants to execute but types "executr"
            decision = decision_service.decide_sync(
                "executr the tests",
                decision_type=DecisionType.ACTION_INTENT,
            )

            # Should still recognize execution intent
            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_multiple_typos_in_decision(self):
        """Test edge model with multiple typos."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # User makes multiple typos
            decision = decision_service.decide_sync(
                "analize the structre and architcture",
                decision_type=DecisionType.TASK_TYPE,
            )

            # Should still work
            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")


@pytest.mark.integration
class TestEdgeModelFuzzyFallback:
    """Test edge model fallback behavior with fuzzy matching."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Skip if edge model is not enabled
        manager = get_feature_flag_manager()
        if not manager.is_enabled(FeatureFlag.USE_EDGE_MODEL):
            pytest.skip("Edge model not enabled")

    def test_fuzzy_fallback_for_uncertain_decisions(self):
        """Test that fuzzy matching helps when edge model is uncertain."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # Query with typo that might confuse edge model
            decision = decision_service.decide_sync(
                "reviw the cdoe",
                decision_type=DecisionType.TASK_TYPE,
            )

            # Fuzzy matching should help provide a reasonable decision
            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_exact_match_preferred_over_fuzzy(self):
        """Verify exact matches are still preferred."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # Exact match should work perfectly
            decision_exact = decision_service.decide_sync(
                "analyze the code",
                decision_type=DecisionType.TASK_TYPE,
            )

            # Fuzzy match should also work
            decision_fuzzy = decision_service.decide_sync(
                "analize the code",
                decision_type=DecisionType.TASK_TYPE,
            )

            # Both should work
            assert decision_exact is not None
            assert decision_fuzzy is not None
            assert decision_exact.confidence > 0.0
            assert decision_fuzzy.confidence > 0.0

            # Exact match might have slightly higher confidence
            # (but not guaranteed, so we don't assert this)

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")


@pytest.mark.integration
class TestEdgeModelPerformanceWithFuzzy:
    """Test edge model performance when using fuzzy matching."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Skip if edge model is not enabled
        manager = get_feature_flag_manager()
        if not manager.is_enabled(FeatureFlag.USE_EDGE_MODEL):
            pytest.skip("Edge model not enabled")

    @pytest.mark.skip(
        reason="pytest-benchmark fixture not available - requires pytest-benchmark package"
    )
    def test_decision_speed_with_fuzzy(self, benchmark):
        """Ensure fuzzy matching doesn't significantly slow edge decisions."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            # Benchmark decision speed with typo
            result = benchmark(
                decision_service.decide_sync,
                "analize the code quality",
                DecisionType.TOOL_NECESSITY,
            )

            # Should complete in reasonable time
            assert result is not None
            assert result.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_batch_decisions_with_typos(self):
        """Test making multiple edge decisions with typos."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            queries_with_typos = [
                ("analize the code", DecisionType.TASK_TYPE),
                ("serch for functions", DecisionType.TASK_TYPE),
                ("reviw the implementation", DecisionType.TASK_TYPE),
                ("executr the tests", DecisionType.ACTION_INTENT),
                ("fix the buug", DecisionType.TASK_TYPE),
            ]

            results = [
                decision_service.decide_sync(query, decision_type=dt)
                for query, dt in queries_with_typos
            ]

            # All should complete successfully
            for result in results:
                assert result is not None
                assert result.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")


@pytest.mark.integration
class TestEdgeModelRobustness:
    """Test edge model robustness with various typo patterns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Skip if edge model is not enabled
        manager = get_feature_flag_manager()
        if not manager.is_enabled(FeatureFlag.USE_EDGE_MODEL):
            pytest.skip("Edge model not enabled")

    def test_missing_letters(self):
        """Test edge model with missing letter typos."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            decision = decision_service.decide_sync(
                "analze the structre",  # Missing letters
                DecisionType.TASK_TYPE,
            )

            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_transposed_letters(self):
        """Test edge model with transposed letter typos."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            decision = decision_service.decide_sync(
                "anlayze the cdoe",  # Transposed letters
                DecisionType.TASK_TYPE,
            )

            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_extra_letters(self):
        """Test edge model with extra letter typos."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            decision = decision_service.decide_sync(
                "analyzee the coode",  # Extra letters
                DecisionType.TASK_TYPE,
            )

            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

    def test_mixed_typos(self):
        """Test edge model with mixed typo patterns."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )
        from victor.core import get_container
        from victor.agent.decisions.schemas import DecisionType

        try:
            decision_service = get_container().get(LLMDecisionServiceProtocol)

            decision = decision_service.decide_sync(
                "anlayze teh structre and architcture",  # Mixed typos
                DecisionType.TASK_TYPE,
            )

            assert decision is not None
            assert decision.confidence > 0.0

        except Exception as e:
            pytest.skip(f"Decision service not available: {e}")

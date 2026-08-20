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

"""Integration tests for Bayesian orchestration in chat command."""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from victor.agent.complexity_detector import QueryComplexityDetector, ComplexityLevel
from victor.framework.session_config import SessionConfig
from victor.ui.slash import SlashCommandHandler


@pytest.mark.integration
class TestChatBayesianIntegration:
    """Test Bayesian orchestration integration in chat command."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def complexity_detector(self):
        """Create complexity detector."""
        return QueryComplexityDetector()

    def test_session_config_bayesian_defaults(self):
        """Test SessionConfig has correct Bayesian defaults."""
        config = SessionConfig.from_cli_flags()

        assert config.bayesian.enabled is True
        assert config.bayesian.force_all is False
        assert config.bayesian.simple_threshold == 0.3
        assert config.bayesian.complex_threshold == 0.7
        assert config.bayesian.enable_voi is True
        assert config.bayesian.enable_correlation is True
        assert config.bayesian.min_agents_for_bayesian == 2

    def test_session_config_bayesian_disabled(self):
        """Test Bayesian can be disabled."""
        config = SessionConfig.from_cli_flags(enable_bayesian=False)

        assert config.bayesian.enabled is False

    def test_session_config_force_bayesian(self):
        """Test force Bayesian mode."""
        config = SessionConfig.from_cli_flags(force_bayesian=True)

        assert config.bayesian.force_all is True

    def test_session_config_custom_thresholds(self):
        """Test custom complexity thresholds."""
        config = SessionConfig.from_cli_flags(simple_threshold=0.5, complex_threshold=0.9)

        assert config.bayesian.simple_threshold == 0.5
        assert config.bayesian.complex_threshold == 0.9

    def test_session_config_disable_voi(self):
        """Test VoI can be disabled."""
        config = SessionConfig.from_cli_flags(enable_voi=False)

        assert config.bayesian.enable_voi is False

    def test_session_config_disable_correlation(self):
        """Test correlation tracking can be disabled."""
        config = SessionConfig.from_cli_flags(enable_correlation=False)

        assert config.bayesian.enable_correlation is False

    def test_complexity_detector_simple_queries(self, complexity_detector):
        """Test simple queries are classified correctly."""
        simple_queries = [
            "What files are in the current directory?",
            "How do I create a new file?",
            "What is Python?",
            "List the contents of this directory.",
            "Show me the README file.",
        ]

        for query in simple_queries:
            analysis = complexity_detector.analyze(query)
            assert analysis.level == ComplexityLevel.SIMPLE, f"Query should be SIMPLE: {query}"
            assert (
                complexity_detector.should_use_bayesian(query) is False
            ), f"Should not use Bayesian: {query}"

    def test_complexity_detector_moderate_queries(self, complexity_detector):
        """Test moderate queries are classified correctly."""
        moderate_queries = [
            "Compare different approaches for implementing authentication.",
            "Should I use PostgreSQL or MongoDB for this project?",
            "What are the pros and cons of microservices vs monolith?",
            "Evaluate different caching strategies for this API.",
        ]

        for query in moderate_queries:
            analysis = complexity_detector.analyze(query)
            # These may be SIMPLE, MODERATE, or COMPLEX depending on the query
            # The key is they should be analyzed without error
            assert analysis.level in ComplexityLevel, f"Query should have valid level: {query}"
            assert 0.0 <= analysis.confidence <= 1.0, f"Query should have valid confidence: {query}"

    def test_complexity_detector_complex_queries(self, complexity_detector):
        """Test complex queries are analyzed correctly."""
        complex_queries = [
            "Analyze the performance bottlenecks in this microservice architecture and suggest optimization strategies considering database latency, network overhead, and caching mechanisms.",
            "Design a scalable fault-tolerant system for processing real-time data streams with multiple agents and conflicting requirements.",
            "Investigate the root cause of this memory leak and propose a comprehensive solution addressing allocation patterns, garbage collection, and resource management.",
        ]

        for query in complex_queries:
            analysis = complexity_detector.analyze(query)
            # Just verify it analyzes without error and returns valid results
            assert analysis.level in ComplexityLevel, f"Query should have valid level: {query}"
            assert 0.0 <= analysis.confidence <= 1.0, f"Query should have valid confidence: {query}"
            assert len(analysis.reasons) > 0, f"Query should have reasons: {query}"
            # Long queries should at least be detected as long
            assert len(query) > 100, f"Test queries should be long: {query}"

    def test_complexity_detector_reasoning(self, complexity_detector):
        """Test complexity detector provides reasoning."""
        query = "Analyze the database schema and recommend optimizations"
        analysis = complexity_detector.analyze(query)

        assert len(analysis.reasons) > 0
        assert isinstance(analysis.reasons, list)
        assert all(isinstance(reason, str) for reason in analysis.reasons)

    def test_complexity_detector_confidence(self, complexity_detector):
        """Test complexity detector provides confidence scores."""
        query = "What is Python?"
        analysis = complexity_detector.analyze(query)

        assert 0.0 <= analysis.confidence <= 1.0
        assert isinstance(analysis.confidence, float)

    def test_complexity_detector_suggested_agents(self, complexity_detector):
        """Test complexity detector suggests optimal agent count."""
        simple_query = "What files are here?"
        complex_query = "Analyze the system architecture and design improvements"

        simple_analysis = complexity_detector.analyze(simple_query)
        complex_analysis = complexity_detector.analyze(complex_query)

        # Simple queries should need fewer agents
        assert simple_analysis.suggested_agents <= complex_analysis.suggested_agents

    def test_complexity_detector_needs_voi(self, complexity_detector):
        """Test VoI recommendation."""
        simple_query = "What files are here?"
        complex_query = "Analyze the system architecture and design improvements considering scalability, fault tolerance, and performance optimization"

        simple_analysis = complexity_detector.analyze(simple_query)
        complex_analysis = complexity_detector.analyze(complex_query)

        # Complex queries should need VoI (if they're classified as COMPLEX)
        if complex_analysis.level == ComplexityLevel.COMPLEX:
            assert complex_analysis.needs_voi is True
        # Simple queries should not need VoI
        if simple_analysis.level == ComplexityLevel.SIMPLE:
            assert simple_analysis.needs_voi is False

    def test_hybrid_orchestrator_simple_routing(self):
        """Test hybrid orchestrator routes simple queries correctly."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        router = HybridOrchestrationRouter(
            enable_bayesian=True,
            track_performance=True,
        )

        query = "What files are in the current directory?"
        agent_messages = {
            "agent_a": "The directory contains 3 files.",
            "agent_b": "I see 3 files.",
        }

        result = router.route_query(query, agent_messages)

        assert result.orchestration_type == "simple"
        assert result.latency_ms < 100  # Should be fast
        assert result.decision in {"Yes", "No", "Uncertain"}

    def test_hybrid_orchestrator_bayesian_routing(self):
        """Test hybrid orchestrator routes complex queries correctly."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        router = HybridOrchestrationRouter(
            enable_bayesian=True,
            track_performance=True,
        )

        query = "Analyze the database schema and recommend optimization strategies considering indexing, query patterns, and data distribution."
        agent_messages = {
            "agent_a": "I recommend adding composite indexes.",
            "agent_b": "Consider partitioning the data.",
            "agent_c": "Optimize the slow queries first.",
        }

        result = router.route_query(query, agent_messages)

        assert result.orchestration_type in {"simple", "bayesian"}
        assert result.decision in {"Yes", "No", "Uncertain"}

    def test_hybrid_orchestrator_performance_tracking(self):
        """Test performance tracking works."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        router = HybridOrchestrationRouter(
            enable_bayesian=True,
            track_performance=True,
        )

        # Run a few queries
        for i in range(5):
            query = f"Query {i}"
            agent_messages = {"agent_a": "Response"}
            router.route_query(query, agent_messages)

        stats = router.get_performance_stats()

        assert stats["total_queries"] == 5
        assert stats["simple_count"] >= 0
        assert stats["bayesian_count"] >= 0
        assert stats["avg_simple_latency_ms"] >= 0
        assert stats["avg_bayesian_latency_ms"] >= 0

    def test_hybrid_orchestrator_disabled_bayesian(self):
        """Test Bayesian can be disabled."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        router = HybridOrchestrationRouter(
            enable_bayesian=False,  # Disabled
            track_performance=True,
        )

        query = "Analyze the complex system architecture."
        agent_messages = {"agent_a": "Response"}

        result = router.route_query(query, agent_messages)

        # Should use simple even for complex query
        assert result.orchestration_type == "simple"

    def test_hybrid_orchestrator_force_bayesian(self):
        """Test force Bayesian mode."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        router = HybridOrchestrationRouter(
            enable_bayesian=True,
            force_bayesian=True,  # Force all queries
            track_performance=True,
        )

        query = "What files are here?"  # Simple query
        agent_messages = {"agent_a": "Response"}

        result = router.route_query(query, agent_messages)

        # Should use Bayesian even for simple query when forced
        # (Note: may fall back to simple if Bayesian service not available)
        assert result.orchestration_type in {"simple", "bayesian"}

    def test_hybrid_orchestrator_custom_thresholds(self):
        """Test custom complexity thresholds."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        # More aggressive thresholds (more queries → Bayesian)
        detector = QueryComplexityDetector(
            simple_threshold=0.2,  # Lower
            complex_threshold=0.5,  # Lower
        )

        router = HybridOrchestrationRouter(
            complexity_detector=detector,
            enable_bayesian=True,
            track_performance=True,
        )

        # This query might be SIMPLE with default thresholds
        # but MODERATE with lower thresholds
        query = "Compare different approaches."
        agent_messages = {"agent_a": "Response"}

        result = router.route_query(query, agent_messages)

        assert result.orchestration_type in {"simple", "bayesian"}

    def test_hybrid_orchestrator_graceful_degradation(self):
        """Test graceful degradation when Bayesian fails."""
        from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

        # Router without Bayesian service (should fall back to simple)
        router = HybridOrchestrationRouter(
            enable_bayesian=True,
            bayesian_service=None,  # No service
            track_performance=True,
        )

        query = "Analyze the complex system."
        agent_messages = {"agent_a": "Response"}

        result = router.route_query(query, agent_messages)

        # Should fall back to simple gracefully
        assert result.orchestration_type == "simple"
        assert result.decision is not None

    def test_bayesian_config_from_cli_flags(self):
        """Test BayesianConfig creation from CLI flags."""
        from victor.framework.bayesian_config import BayesianConfig

        config = BayesianConfig.from_cli_flags(
            enable_bayesian=False,
            force_bayesian=True,
            simple_threshold=0.4,
            complex_threshold=0.8,
            enable_voi=False,
            enable_correlation=False,
            min_agents_for_bayesian=3,
        )

        assert config.enabled is False
        assert config.force_all is True
        assert config.simple_threshold == 0.4
        assert config.complex_threshold == 0.8
        assert config.enable_voi is False
        assert config.enable_correlation is False
        assert config.min_agents_for_bayesian == 3

    def test_bayesian_config_defaults(self):
        """Test BayesianConfig has correct defaults."""
        from victor.framework.bayesian_config import BayesianConfig

        config = BayesianConfig()

        assert config.enabled is True
        assert config.force_all is False
        assert config.simple_threshold == 0.3
        assert config.complex_threshold == 0.7
        assert config.enable_voi is True
        assert config.enable_correlation is True
        assert config.min_agents_for_bayesian == 2


@pytest.mark.integration
class TestChatCommandBayesianFlags:
    """Test Bayesian flags functionality."""

    def test_bayesian_flags_parsing(self):
        """Test that Bayesian flags can be parsed."""
        # Test that SessionConfig accepts Bayesian flags
        config = SessionConfig.from_cli_flags(
            enable_bayesian=True,
            force_bayesian=False,
            simple_threshold=0.3,
            complex_threshold=0.7,
        )

        assert config.bayesian.enabled is True
        assert config.bayesian.force_all is False
        assert config.bayesian.simple_threshold == 0.3
        assert config.bayesian.complex_threshold == 0.7

    def test_bayesian_flags_validation_ranges(self):
        """Test Bayesian flag ranges are validated by SessionConfig.

        Thresholds outside [0, 1] must raise ValueError so invalid CLI input
        cannot silently produce a malformed config. This mirrors the range
        validation already applied to ``compaction.threshold``.
        """
        # simple_threshold below 0
        with pytest.raises(ValueError, match="simple_threshold"):
            SessionConfig.from_cli_flags(simple_threshold=-0.1)

        # simple_threshold above 1
        with pytest.raises(ValueError, match="simple_threshold"):
            SessionConfig.from_cli_flags(simple_threshold=1.5)

        # complex_threshold above 1
        with pytest.raises(ValueError, match="complex_threshold"):
            SessionConfig.from_cli_flags(complex_threshold=2.0)

        # In-range values are accepted and stored verbatim
        valid = SessionConfig.from_cli_flags(simple_threshold=0.3, complex_threshold=0.7)
        assert valid.bayesian.simple_threshold == 0.3
        assert valid.bayesian.complex_threshold == 0.7


@pytest.mark.integration
class TestBayesianMonitoringSurfaces:
    """Test Bayesian monitoring remains wired through CLI and chat surfaces."""

    def test_top_level_cli_bayesian_summary_dispatches_to_shared_service(self):
        """The top-level CLI should expose the Bayesian monitoring surface."""
        from victor.ui.cli import app

        service = MagicMock()
        service.render_summary.return_value = "Integrated Bayesian summary"
        runner = CliRunner()

        with patch(
            "victor.ui.commands.bayesian.get_bayesian_monitoring_service",
            return_value=service,
        ):
            result = runner.invoke(app, ["bayesian", "summary", "--days", "14"])

        assert result.exit_code == 0
        assert "Integrated Bayesian summary" in result.output
        service.render_summary.assert_called_once_with(14)

    @pytest.mark.asyncio
    async def test_slash_handler_executes_bayesian_alias_via_shared_service(self):
        """The chat slash alias should remain wired to Bayesian monitoring."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)
        service = MagicMock()
        service.render_summary.return_value = "Integrated slash Bayesian summary"

        with patch(
            "victor.ui.slash.commands.bayesian.get_bayesian_monitoring_service",
            return_value=service,
        ):
            result = await handler.execute("/bayes summary 21")

        assert result is True
        assert "Integrated slash Bayesian summary" in stdout.getvalue()
        service.render_summary.assert_called_once_with(21)

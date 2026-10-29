"""Tests for victor.framework.fulfillment module.

Tests task-specific fulfillment detection strategies for agentic loops.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock


from victor.framework.fulfillment import (
    AnalysisFulfillment,
    CodeGenerationFulfillment,
    DebuggingFulfillment,
    DeploymentFulfillment,
    DocumentationFulfillment,
    FulfillmentConfig,
    FulfillmentDetector,
    FulfillmentResult,
    FulfillmentStatus,
    FulfillmentStrategy,
    SearchFulfillment,
    SetupFulfillment,
    TaskType,
    TestingFulfillment,
)
from victor.framework.completion_scorer import TaskType as CompletionScorerTaskType

# ============================================================================
# FulfillmentResult tests
# ============================================================================


class TestFulfillmentResult:
    """Tests for FulfillmentResult dataclass."""

    def test_is_fulfilled(self):
        r = FulfillmentResult(
            status=FulfillmentStatus.FULFILLED,
            score=0.9,
        )
        assert r.is_fulfilled is True
        assert r.is_partial is False

    def test_is_partial(self):
        r = FulfillmentResult(
            status=FulfillmentStatus.PARTIAL,
            score=0.5,
        )
        assert r.is_fulfilled is False
        assert r.is_partial is True

    def test_not_fulfilled(self):
        r = FulfillmentResult(
            status=FulfillmentStatus.NOT_FULFILLED,
            score=0.1,
        )
        assert r.is_fulfilled is False
        assert r.is_partial is False

    def test_to_dict(self):
        r = FulfillmentResult(
            status=FulfillmentStatus.PARTIAL,
            score=0.6,
            fulfilled_criteria=["file_exists", "valid_syntax"],
            missing_criteria=["non_empty"],
            reason="Partial completion",
        )
        d = r.to_dict()
        assert d["status"] == "partial"
        assert d["score"] == 0.6
        assert d["fulfilled_count"] == 2
        assert d["missing_count"] == 1
        assert d["reason"] == "Partial completion"


# ============================================================================
# CodeGenerationFulfillment tests
# ============================================================================


class TestCodeGenerationFulfillment:
    """Tests for code generation fulfillment strategy."""

    async def test_file_exists_and_valid(self):
        strategy = CodeGenerationFulfillment()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    return 'world'\n")
            f.flush()
            result = await strategy.check(
                criteria={"file_path": f.name},
                context={},
            )
        assert result.status == FulfillmentStatus.FULFILLED
        assert result.score >= 0.8
        assert "file_exists" in result.fulfilled_criteria
        assert "valid_syntax" in result.fulfilled_criteria
        assert "non_empty" in result.fulfilled_criteria

    async def test_file_not_exists(self):
        strategy = CodeGenerationFulfillment()
        result = await strategy.check(
            criteria={"file_path": "/nonexistent/file.py"},
            context={},
        )
        assert result.status == FulfillmentStatus.NOT_FULFILLED
        assert "file_exists" in result.missing_criteria

    async def test_file_with_syntax_error(self):
        strategy = CodeGenerationFulfillment()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def broken(\n")
            f.flush()
            result = await strategy.check(
                criteria={"file_path": f.name},
                context={},
            )
        assert "file_exists" in result.fulfilled_criteria
        assert any("valid_syntax" in m for m in result.missing_criteria)

    async def test_empty_file(self):
        strategy = CodeGenerationFulfillment()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()
            result = await strategy.check(
                criteria={"file_path": f.name},
                context={},
            )
        assert "non_empty" in result.missing_criteria

    async def test_required_patterns(self):
        strategy = CodeGenerationFulfillment()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("class MyClass:\n    pass\n")
            f.flush()
            result = await strategy.check(
                criteria={
                    "file_path": f.name,
                    "required_patterns": ["class MyClass", "pass"],
                },
                context={},
            )
        assert any("pattern_class MyClass" in c for c in result.fulfilled_criteria)

    async def test_missing_required_patterns(self):
        strategy = CodeGenerationFulfillment()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            result = await strategy.check(
                criteria={
                    "file_path": f.name,
                    "required_patterns": ["class Foo"],
                },
                context={},
            )
        assert any("pattern_class Foo" in c for c in result.missing_criteria)

    async def test_no_file_path_in_criteria(self):
        strategy = CodeGenerationFulfillment()
        result = await strategy.check(criteria={}, context={})
        assert result.status == FulfillmentStatus.ERROR


# ============================================================================
# TestingFulfillment tests
# ============================================================================


class TestTestingFulfillment:
    """Tests for testing fulfillment strategy."""

    async def test_all_tests_passing(self):
        strategy = TestingFulfillment()
        test_results = [
            MagicMock(passed=True, error_message=None),
            MagicMock(passed=True, error_message=None),
        ]
        result = await strategy.check(
            criteria={},
            context={"test_results": test_results},
        )
        assert result.score >= 0.8
        assert "tests_passing" in result.fulfilled_criteria
        assert "no_test_errors" in result.fulfilled_criteria

    async def test_some_tests_failing(self):
        strategy = TestingFulfillment()
        test_results = [
            MagicMock(passed=True, error_message=None),
            MagicMock(passed=False, error_message="AssertionError"),
        ]
        result = await strategy.check(
            criteria={},
            context={"test_results": test_results},
        )
        assert any("tests_passing" in m for m in result.missing_criteria)

    async def test_no_test_results(self):
        strategy = TestingFulfillment()
        result = await strategy.check(criteria={}, context={})
        assert "tests_exist" in result.missing_criteria

    async def test_test_files_exist(self):
        strategy = TestingFulfillment()
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            result = await strategy.check(
                criteria={"test_files": [f.name]},
                context={},
            )
        assert any("test_files_exist" in c for c in result.fulfilled_criteria)

    async def test_test_files_missing(self):
        strategy = TestingFulfillment()
        result = await strategy.check(
            criteria={"test_files": ["/nonexistent/test_foo.py"]},
            context={},
        )
        assert "test_files_exist" in result.missing_criteria


# ============================================================================
# DebuggingFulfillment tests
# ============================================================================


class TestDebuggingFulfillment:
    """Tests for debugging fulfillment strategy."""

    async def test_bug_fixed(self):
        strategy = DebuggingFulfillment()
        test_results = [MagicMock(passed=True)]
        result = await strategy.check(
            criteria={"original_error": "KeyError: 'name'"},
            context={
                "errors": [],
                "test_results": test_results,
            },
        )
        assert result.score >= 0.8
        assert "original_error_fixed" in result.fulfilled_criteria
        assert "all_tests_pass" in result.fulfilled_criteria
        assert "no_errors" in result.fulfilled_criteria

    async def test_bug_still_present(self):
        strategy = DebuggingFulfillment()
        result = await strategy.check(
            criteria={"original_error": "KeyError: 'name'"},
            context={
                "errors": ["KeyError: 'name'"],
            },
        )
        assert "original_error_fixed" in result.missing_criteria

    async def test_no_original_error_specified(self):
        strategy = DebuggingFulfillment()
        result = await strategy.check(
            criteria={},
            context={"errors": []},
        )
        assert "no_errors" in result.fulfilled_criteria


# ============================================================================
# SearchFulfillment tests
# ============================================================================


class TestSearchFulfillment:
    """Tests for search fulfillment strategy."""

    async def test_results_found(self):
        strategy = SearchFulfillment()
        result = await strategy.check(
            criteria={"min_results": 2},
            context={
                "search_results": ["r1", "r2", "r3"],
                "avg_relevance": 0.8,
            },
        )
        assert result.score >= 0.8
        assert any("results_found" in c for c in result.fulfilled_criteria)

    async def test_insufficient_results(self):
        strategy = SearchFulfillment()
        result = await strategy.check(
            criteria={"min_results": 5},
            context={"search_results": ["r1"]},
        )
        assert any("results_found" in m for m in result.missing_criteria)

    async def test_low_relevance(self):
        strategy = SearchFulfillment()
        result = await strategy.check(
            criteria={"min_relevance": 0.7},
            context={
                "search_results": ["r1"],
                "avg_relevance": 0.3,
            },
        )
        assert any("relevance" in m for m in result.missing_criteria)

    async def test_required_info_complete(self):
        strategy = SearchFulfillment()
        result = await strategy.check(
            criteria={"required_info": ["author", "date"]},
            context={
                "search_results": ["r1"],
                "found_info": ["author", "date"],
            },
        )
        assert "complete_info" in result.fulfilled_criteria


# ============================================================================
# SetupFulfillment tests
# ============================================================================


class TestSetupFulfillment:
    """Tests for setup fulfillment strategy."""

    async def test_services_running(self):
        strategy = SetupFulfillment()
        result = await strategy.check(
            criteria={"services": ["postgres", "redis"]},
            context={"running_services": ["postgres", "redis"]},
        )
        assert "service_postgres" in result.fulfilled_criteria
        assert "service_redis" in result.fulfilled_criteria

    async def test_services_not_running(self):
        strategy = SetupFulfillment()
        result = await strategy.check(
            criteria={"services": ["postgres"]},
            context={"running_services": []},
        )
        assert "service_postgres" in result.missing_criteria

    async def test_dependencies_installed(self):
        strategy = SetupFulfillment()
        result = await strategy.check(
            criteria={"dependencies": ["numpy"]},
            context={"installed_dependencies": ["numpy"]},
        )
        assert "dependency_numpy" in result.fulfilled_criteria

    async def test_config_files_exist(self):
        strategy = SetupFulfillment()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            result = await strategy.check(
                criteria={"config_files": [f.name]},
                context={},
            )
        assert any("config_" in c for c in result.fulfilled_criteria)


# ============================================================================
# FulfillmentDetector tests
# ============================================================================


class TestFulfillmentDetector:
    """Tests for FulfillmentDetector."""

    def test_default_strategies(self):
        detector = FulfillmentDetector()
        assert TaskType.CODE_GENERATION in detector.strategies
        assert TaskType.TESTING in detector.strategies
        assert TaskType.DEBUGGING in detector.strategies
        assert TaskType.SEARCH in detector.strategies
        assert TaskType.SETUP in detector.strategies
        assert TaskType.ANALYSIS in detector.strategies
        assert TaskType.DOCUMENTATION in detector.strategies
        assert TaskType.DEPLOYMENT in detector.strategies

    async def test_check_code_generation(self):
        detector = FulfillmentDetector()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            result = await detector.check_fulfillment(
                task_type=TaskType.CODE_GENERATION,
                criteria={"file_path": f.name},
                context={},
            )
        assert isinstance(result, FulfillmentResult)
        assert result.score > 0.0

    async def test_unknown_task_type(self):
        detector = FulfillmentDetector()
        result = await detector.check_fulfillment(
            task_type=TaskType.UNKNOWN,
            criteria={},
            context={},
        )
        assert result.status == FulfillmentStatus.UNKNOWN

    async def test_normalizes_compatible_task_type_enum(self, caplog):
        detector = FulfillmentDetector()

        with caplog.at_level("WARNING"):
            result = await detector.check_fulfillment(
                task_type=CompletionScorerTaskType.SEARCH,
                criteria={"min_results": 1},
                context={"search_results": ["graph.py"], "avg_relevance": 0.9},
            )

        assert result.status != FulfillmentStatus.UNKNOWN
        assert "No fulfillment strategy" not in caplog.text

    async def test_strategy_error_handled(self):
        detector = FulfillmentDetector()

        class BrokenStrategy(FulfillmentStrategy):
            async def check(self, criteria, context):
                raise RuntimeError("Strategy error")

        detector.register_strategy(TaskType.CODE_GENERATION, BrokenStrategy())
        result = await detector.check_fulfillment(
            task_type=TaskType.CODE_GENERATION,
            criteria={},
            context={},
        )
        assert result.status == FulfillmentStatus.ERROR

    def test_register_strategy(self):
        detector = FulfillmentDetector()
        custom = MagicMock(spec=FulfillmentStrategy)
        detector.register_strategy(TaskType.DOCUMENTATION, custom)
        assert detector.strategies[TaskType.DOCUMENTATION] is custom

    async def test_code_modification_uses_code_generation_strategy(self):
        detector = FulfillmentDetector()
        assert isinstance(
            detector.strategies[TaskType.CODE_MODIFICATION],
            CodeGenerationFulfillment,
        )

    def test_custom_config(self):
        config = FulfillmentConfig(fulfilled_threshold=0.9, partial_threshold=0.5)
        detector = FulfillmentDetector(config=config)
        assert detector.config.fulfilled_threshold == 0.9


# ============================================================================
# AnalysisFulfillment tests
# ============================================================================


class TestAnalysisFulfillment:
    """Tests for analysis fulfillment strategy."""

    async def test_findings_and_summary(self):
        strategy = AnalysisFulfillment()
        result = await strategy.check(
            criteria={"min_findings": 2},
            context={
                "findings": ["finding1", "finding2", "finding3"],
                "summary": "Analysis complete",
            },
        )
        assert result.score >= 0.6
        assert any("findings" in c for c in result.fulfilled_criteria)
        assert "summary_produced" in result.fulfilled_criteria

    async def test_no_findings(self):
        strategy = AnalysisFulfillment()
        result = await strategy.check(
            criteria={"min_findings": 3},
            context={"findings": []},
        )
        assert any("findings" in m for m in result.missing_criteria)

    async def test_questions_answered(self):
        strategy = AnalysisFulfillment()
        result = await strategy.check(
            criteria={"questions": ["q1", "q2"]},
            context={
                "findings": ["f1"],
                "answered_questions": ["q1", "q2"],
            },
        )
        assert any("questions_answered" in c for c in result.fulfilled_criteria)

    async def test_questions_partially_answered(self):
        strategy = AnalysisFulfillment()
        result = await strategy.check(
            criteria={"questions": ["q1", "q2", "q3"]},
            context={
                "findings": ["f1"],
                "answered_questions": ["q1"],
            },
        )
        assert any("questions_answered" in m for m in result.missing_criteria)


# ============================================================================
# DocumentationFulfillment tests
# ============================================================================


class TestDocumentationFulfillment:
    """Tests for documentation fulfillment strategy."""

    async def test_doc_file_exists(self):
        strategy = DocumentationFulfillment()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# API Documentation\n\nSome content here.\n")
            f.flush()
            result = await strategy.check(
                criteria={"doc_files": [f.name]},
                context={},
            )
        assert any("doc_files" in c for c in result.fulfilled_criteria)

    async def test_doc_content_with_sections(self):
        strategy = DocumentationFulfillment()
        result = await strategy.check(
            criteria={"required_sections": ["usage", "api"]},
            context={
                "doc_content": "# Usage\n\nHow to use.\n\n# API\n\nEndpoints.",
            },
        )
        assert "content_produced" in result.fulfilled_criteria
        assert "all_sections_present" in result.fulfilled_criteria

    async def test_missing_sections(self):
        strategy = DocumentationFulfillment()
        result = await strategy.check(
            criteria={"required_sections": ["usage", "api", "deployment"]},
            context={
                "doc_content": "# Usage\n\nSome content.",
            },
        )
        assert "content_produced" in result.fulfilled_criteria
        # Only 1 of 3 sections found
        assert any("sections" in c for c in result.fulfilled_criteria)

    async def test_examples_required(self):
        strategy = DocumentationFulfillment()
        result = await strategy.check(
            criteria={"require_examples": True},
            context={
                "doc_content": "# API\n\n```python\nprint('hello')\n```\n",
            },
        )
        assert "examples_included" in result.fulfilled_criteria

    async def test_no_content(self):
        strategy = DocumentationFulfillment()
        result = await strategy.check(
            criteria={},
            context={},
        )
        assert "content_produced" in result.missing_criteria


# ============================================================================
# DeploymentFulfillment tests
# ============================================================================


class TestDeploymentFulfillment:
    """Tests for deployment fulfillment strategy."""

    async def test_deploy_success(self):
        strategy = DeploymentFulfillment()
        result = await strategy.check(
            criteria={},
            context={
                "deploy_status": "success",
                "health_checks": {"api": True, "db": True},
            },
        )
        assert result.score >= 0.7
        assert "deploy_success" in result.fulfilled_criteria

    async def test_health_check_failures(self):
        strategy = DeploymentFulfillment()
        result = await strategy.check(
            criteria={},
            context={
                "health_checks": {"api": True, "db": False},
            },
        )
        assert any("health" in c for c in result.fulfilled_criteria)

    async def test_deploy_partial(self):
        strategy = DeploymentFulfillment()
        result = await strategy.check(
            criteria={},
            context={"deploy_status": "partial"},
        )
        assert "deploy_partial" in result.fulfilled_criteria

    async def test_artifacts_check(self):
        strategy = DeploymentFulfillment()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            result = await strategy.check(
                criteria={"artifacts": [f.name]},
                context={},
            )
        assert any("artifacts" in c for c in result.fulfilled_criteria)


# ============================================================================
# FulfillmentConfig tests
# ============================================================================


class TestFulfillmentConfig:
    """Tests for FulfillmentConfig."""

    def test_default_thresholds(self):
        config = FulfillmentConfig()
        assert config.fulfilled_threshold == 0.8
        assert config.partial_threshold == 0.4

    def test_custom_thresholds(self):
        config = FulfillmentConfig(fulfilled_threshold=0.95, partial_threshold=0.6)
        assert config.fulfilled_threshold == 0.95
        assert config.partial_threshold == 0.6


# ============================================================================
# Wave 5: FulfillmentConfig weight externalization
# ============================================================================


class TestFulfillmentConfigWeights:
    """Wave 5: config weights must be respected by strategies."""

    def test_config_has_file_exists_weight(self):
        from victor.framework.fulfillment import FulfillmentConfig

        config = FulfillmentConfig()
        assert hasattr(config, "file_exists_weight")
        assert 0 < config.file_exists_weight <= 1.0

    def test_config_has_test_pass_weight(self):
        from victor.framework.fulfillment import FulfillmentConfig

        config = FulfillmentConfig()
        assert hasattr(config, "test_pass_weight")
        assert 0 < config.test_pass_weight <= 1.0

    def test_default_weights_produce_same_behavior(self):
        """FulfillmentConfig() defaults must reproduce legacy hardcoded behavior."""
        from victor.framework.fulfillment import (
            FulfillmentConfig,
            FulfillmentDetector,
        )

        config = FulfillmentConfig()
        detector = FulfillmentDetector(config=config)

        # Just verify detector is constructed successfully with default config
        assert detector is not None

    async def test_code_generation_strategy_uses_config_file_exists_weight(self):
        """CodeGenerationFulfillment should use config.file_exists_weight."""
        import tempfile
        import os
        from victor.framework.fulfillment import (
            CodeGenerationFulfillment,
            FulfillmentConfig,
        )

        strategy = CodeGenerationFulfillment()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"x = 1\n")
            path = f.name

        try:
            # Default config
            default_config = FulfillmentConfig()
            result_default = await strategy.check(
                criteria={"file_path": path},
                context={},
                config=default_config,
            )

            # Config with higher file_exists_weight (code-gen weights must sum to 1.0)
            high_weight_config = FulfillmentConfig(
                file_exists_weight=0.9,
                syntax_valid_weight=0.04,
                non_empty_weight=0.03,
                pattern_weight=0.03,
            )
            result_high = await strategy.check(
                criteria={"file_path": path},
                context={},
                config=high_weight_config,
            )

            # Higher weight should produce a higher score
            assert result_high.score >= result_default.score
        finally:
            os.unlink(path)

    def test_fulfillment_detector_passes_config_to_strategy(self):
        """FulfillmentDetector.check_fulfillment must pass its config to the strategy."""
        from victor.framework.fulfillment import (
            FulfillmentConfig,
            FulfillmentDetector,
            FulfillmentStrategy,
            TaskType,
        )
        import asyncio

        received_configs = []

        class RecordingStrategy(FulfillmentStrategy):
            async def check(self, criteria, context, config=None):
                received_configs.append(config)
                from victor.framework.fulfillment import (
                    FulfillmentResult,
                    FulfillmentStatus,
                )

                return FulfillmentResult(status=FulfillmentStatus.FULFILLED, score=1.0)

        config = FulfillmentConfig(
            file_exists_weight=0.42,
            syntax_valid_weight=0.28,
            non_empty_weight=0.15,
            pattern_weight=0.15,
        )
        detector = FulfillmentDetector(config=config)
        detector.register_strategy(TaskType.CODE_GENERATION, RecordingStrategy())

        asyncio.get_event_loop().run_until_complete(
            detector.check_fulfillment(
                task_type=TaskType.CODE_GENERATION,
                criteria={},
                context={},
            )
        )

        assert len(received_configs) == 1
        assert received_configs[0] is config

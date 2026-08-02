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

"""Tests for Pipeline Analyzer Tool."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from datetime import datetime

from victor.tools.pipeline_tool import PipelineAnalyzerTool
from victor.tools.base import CostTier
from victor.observability.pipeline import (
    CoverageMetrics,
    PipelineAnalysisResult,
    PipelineConfig,
    PipelineIssue,
    PipelinePlatform,
    PipelineStep,
    StepType,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tool():
    """Create pipeline analyzer tool instance."""
    return PipelineAnalyzerTool()


@pytest.fixture
def sample_coverage():
    """Create sample coverage metrics."""
    return CoverageMetrics(
        total_lines=1000,
        covered_lines=750,
        total_branches=200,
        covered_branches=150,
        total_functions=100,
        covered_functions=80,
        line_coverage=75.0,
        branch_coverage=75.0,
        function_coverage=80.0,
        uncovered_files=["src/module1.py", "src/module2.py"],
        coverage_by_file={"src/main.py": 90.0, "src/utils.py": 60.0},
        report_path=Path("coverage.xml"),
        report_format="cobertura",
    )


@pytest.fixture
def sample_config():
    """Create sample pipeline config."""
    return PipelineConfig(
        platform=PipelinePlatform.GITHUB_ACTIONS,
        file_path=Path(".github/workflows/ci.yml"),
        name="CI Pipeline",
        triggers=["push", "pull_request"],
        branches=["main", "develop"],
        steps=[
            PipelineStep(
                name="test",
                step_type=StepType.TEST,
                command="pytest",
            ),
            PipelineStep(
                name="lint",
                step_type=StepType.LINT,
                command="flake8",
            ),
        ],
    )


@pytest.fixture
def sample_analysis_result(sample_config):
    """Create sample analysis result."""
    return PipelineAnalysisResult(
        configs=[sample_config],
        issues=[
            PipelineIssue(
                severity="critical",
                category="security",
                message="Secrets exposed in workflow",
                recommendation="Use GitHub secrets",
            ),
            PipelineIssue(
                severity="warning",
                category="performance",
                message="No caching configured",
            ),
            PipelineIssue(
                severity="info",
                category="maintainability",
                message="Consider adding matrix builds",
            ),
        ],
        recommendations=["Enable dependency caching", "Add parallel jobs"],
        analyzed_at=datetime(2025, 1, 1, 12, 0, 0),
    )


# =============================================================================
# TOOL PROPERTIES TESTS
# =============================================================================


class TestPipelineAnalyzerToolProperties:
    """Tests for tool properties and metadata."""

    def test_tool_name(self, tool):
        """Test tool name uses canonical name from ToolNames."""
        from victor.tools.tool_names import ToolNames

        assert tool.name == ToolNames.PIPELINE
        assert tool.name == "pipeline"  # Canonical short name

    def test_tool_description_contains_platforms(self, tool):
        """Test description mentions supported platforms."""
        assert "GitHub Actions" in tool.description
        assert "GitLab" in tool.description

    def test_tool_description_contains_formats(self, tool):
        """Test description mentions coverage formats."""
        assert "Cobertura" in tool.description
        assert "LCOV" in tool.description
        assert "JaCoCo" in tool.description

    def test_cost_tier(self, tool):
        """Test cost tier is LOW."""
        assert tool.cost_tier == CostTier.LOW

    def test_parameters_schema(self, tool):
        """Test parameters schema structure."""
        assert tool.parameters["type"] == "object"
        assert "action" in tool.parameters["properties"]
        assert "platform" in tool.parameters["properties"]
        assert "coverage_format" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["action"]

    def test_action_enum_values(self, tool):
        """Test action enum has all expected values."""
        actions = tool.parameters["properties"]["action"]["enum"]
        assert "analyze" in actions
        assert "coverage" in actions
        assert "compare_coverage" in actions
        assert "summary" in actions
        assert "detect" in actions

    def test_platform_enum_values(self, tool):
        """Test platform enum has expected values."""
        platforms = tool.parameters["properties"]["platform"]["enum"]
        assert "github_actions" in platforms
        assert "gitlab_ci" in platforms
        assert "all" in platforms

    def test_metadata_category(self, tool):
        """Test metadata category."""
        assert tool.metadata.category == "pipeline"

    def test_metadata_keywords(self, tool):
        """Test metadata keywords."""
        keywords = tool.metadata.keywords
        assert "pipeline" in keywords
        assert "ci/cd" in keywords
        assert "coverage" in keywords

    def test_metadata_use_cases(self, tool):
        """Test metadata use cases."""
        use_cases = tool.metadata.use_cases
        assert any("CI/CD" in uc for uc in use_cases)


# =============================================================================
# DETECT ACTION TESTS
# =============================================================================


class TestDetectAction:
    """Tests for the detect action."""

    @pytest.mark.asyncio
    async def test_detect_platforms_found(self, tool):
        """Test detecting platforms when found."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.return_value = [
                PipelinePlatform.GITHUB_ACTIONS,
                PipelinePlatform.GITLAB_CI,
            ]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert "github_actions" in result.output
            assert "gitlab_ci" in result.output
            assert result.metadata["platforms"] == ["github_actions", "gitlab_ci"]

    @pytest.mark.asyncio
    async def test_detect_no_platforms(self, tool):
        """Test detecting when no platforms found."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.return_value = []
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert "No CI/CD platforms detected" in result.output


# =============================================================================
# SUMMARY ACTION TESTS
# =============================================================================


class TestSummaryAction:
    """Tests for the summary action."""

    @pytest.mark.asyncio
    async def test_summary_basic(self, tool):
        """Test basic summary generation."""
        summary_data = {
            "platforms_detected": 2,
            "pipeline_configs": 3,
            "critical_issues": 1,
            "warning_issues": 5,
            "total_issues": 8,
            "success_rate": 95.5,
            "avg_duration_seconds": 120,
            "coverage": {
                "line_coverage": 75.0,
                "branch_coverage": 70.0,
                "uncovered_files": 5,
            },
            "recommendations_count": 3,
        }

        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_pipeline_summary.return_value = summary_data
            MockManager.return_value = mock_manager

            result = await tool.execute(action="summary")

            assert result.success is True
            assert "Pipeline Health Summary" in result.output
            assert "Critical: 1" in result.output
            assert "Warnings: 5" in result.output
            assert "95.5%" in result.output
            assert "75.0%" in result.output

    @pytest.mark.asyncio
    async def test_summary_without_coverage(self, tool):
        """Test summary without coverage data."""
        summary_data = {
            "platforms_detected": 1,
            "pipeline_configs": 1,
            "critical_issues": 0,
            "warning_issues": 0,
            "total_issues": 0,
            "success_rate": 0,
            "avg_duration_seconds": None,
            "coverage": None,
            "recommendations_count": 0,
        }

        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_pipeline_summary.return_value = summary_data
            MockManager.return_value = mock_manager

            result = await tool.execute(action="summary")

            assert result.success is True
            # Should not crash when coverage is None


# =============================================================================
# ANALYZE ACTION TESTS
# =============================================================================


class TestAnalyzeAction:
    """Tests for the analyze action."""

    @pytest.mark.asyncio
    async def test_analyze_all_platforms(self, tool, sample_analysis_result):
        """Test analyzing all platforms."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.analyze_pipelines.return_value = sample_analysis_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="analyze")

            assert result.success is True
            assert "Pipeline Analysis Report" in result.output
            mock_manager.analyze_pipelines.assert_called_once_with(
                platforms=None,
                include_coverage=True,
            )

    @pytest.mark.asyncio
    async def test_analyze_specific_platform(self, tool, sample_analysis_result):
        """Test analyzing specific platform."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.analyze_pipelines.return_value = sample_analysis_result
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="analyze",
                platform="github_actions",
            )

            assert result.success is True
            mock_manager.analyze_pipelines.assert_called_once()
            call_args = mock_manager.analyze_pipelines.call_args
            assert call_args[1]["platforms"] == [PipelinePlatform.GITHUB_ACTIONS]

    @pytest.mark.asyncio
    async def test_analyze_without_coverage(self, tool, sample_analysis_result):
        """Test analyzing without coverage."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.analyze_pipelines.return_value = sample_analysis_result
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="analyze",
                include_coverage=False,
            )

            assert result.success is True
            call_args = mock_manager.analyze_pipelines.call_args
            assert call_args[1]["include_coverage"] is False

    @pytest.mark.asyncio
    async def test_analyze_includes_issues_in_output(self, tool, sample_analysis_result):
        """Test analysis output includes issues."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.analyze_pipelines.return_value = sample_analysis_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="analyze")

            assert "Critical" in result.output
            assert "Warnings" in result.output
            assert "security" in result.output

    @pytest.mark.asyncio
    async def test_analyze_includes_recommendations(self, tool, sample_analysis_result):
        """Test analysis output includes recommendations."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.analyze_pipelines.return_value = sample_analysis_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="analyze")

            assert "Recommendations" in result.output
            assert "Enable dependency caching" in result.output


# =============================================================================
# COVERAGE ACTION TESTS
# =============================================================================


class TestCoverageAction:
    """Tests for the coverage action."""

    @pytest.mark.asyncio
    async def test_coverage_found(self, tool, sample_coverage):
        """Test coverage when report found."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_coverage.return_value = sample_coverage
            MockManager.return_value = mock_manager

            result = await tool.execute(action="coverage")

            assert result.success is True
            assert "Code Coverage Report" in result.output
            assert "75.0%" in result.output
            assert "cobertura" in result.output

    @pytest.mark.asyncio
    async def test_coverage_not_found(self, tool):
        """Test coverage when no report found."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_coverage.return_value = None
            MockManager.return_value = mock_manager

            result = await tool.execute(action="coverage")

            assert result.success is False
            assert "No coverage reports found" in result.output

    @pytest.mark.asyncio
    async def test_coverage_with_format_hint(self, tool, sample_coverage):
        """Test coverage with specific format hint."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_coverage.return_value = sample_coverage
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="coverage",
                coverage_format="lcov",
            )

            assert result.success is True
            call_args = mock_manager.get_coverage.call_args
            assert call_args[1]["format_hint"] == "lcov"

    @pytest.mark.asyncio
    async def test_coverage_auto_format(self, tool, sample_coverage):
        """Test coverage with auto format detection."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_coverage.return_value = sample_coverage
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="coverage",
                coverage_format="auto",
            )

            assert result.success is True
            call_args = mock_manager.get_coverage.call_args
            assert call_args[1]["format_hint"] is None


# =============================================================================
# COMPARE COVERAGE ACTION TESTS
# =============================================================================


class TestCompareCoverageAction:
    """Tests for the compare_coverage action."""

    @pytest.mark.asyncio
    async def test_compare_coverage_improved(self, tool):
        """Test coverage comparison with improvement."""
        comparison_data = {
            "line_coverage_delta": 5.0,
            "branch_coverage_delta": 3.0,
            "function_coverage_delta": 2.0,
            "improved": True,
            "improved_files": [
                {"file": "src/main.py", "delta": 10.0},
            ],
            "regressed_files": [],
            "new_uncovered_files": [],
        }

        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.compare_coverage.return_value = comparison_data
            MockManager.return_value = mock_manager

            result = await tool.execute(action="compare_coverage")

            assert result.success is True
            assert "Improved" in result.output
            assert "+5.0%" in result.output

    @pytest.mark.asyncio
    async def test_compare_coverage_regressed(self, tool):
        """Test coverage comparison with regression."""
        comparison_data = {
            "line_coverage_delta": -5.0,
            "branch_coverage_delta": -3.0,
            "improved": False,
            "improved_files": [],
            "regressed_files": [
                {"file": "src/utils.py", "delta": -15.0},
            ],
            "new_uncovered_files": ["src/new_module.py"],
        }

        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.compare_coverage.return_value = comparison_data
            MockManager.return_value = mock_manager

            result = await tool.execute(action="compare_coverage")

            assert result.success is True
            assert "Regressed" in result.output
            assert "-5.0%" in result.output

    @pytest.mark.asyncio
    async def test_compare_coverage_with_baseline_path(self, tool):
        """Test coverage comparison with baseline path."""
        comparison_data = {
            "line_coverage_delta": 0.0,
            "branch_coverage_delta": 0.0,
            "improved": False,
            "improved_files": [],
            "regressed_files": [],
            "new_uncovered_files": [],
        }

        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.compare_coverage.return_value = comparison_data
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="compare_coverage",
                baseline_path="/path/to/baseline.xml",
            )

            assert result.success is True
            call_args = mock_manager.compare_coverage.call_args
            assert call_args[1]["baseline_path"] == Path("/path/to/baseline.xml")

    @pytest.mark.asyncio
    async def test_compare_coverage_error(self, tool):
        """Test coverage comparison with error."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.compare_coverage.return_value = {
                "error": "No baseline coverage found",
            }
            MockManager.return_value = mock_manager

            result = await tool.execute(action="compare_coverage")

            assert result.success is False
            assert "No baseline coverage found" in result.output


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """Test handling unknown action."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            MockManager.return_value = AsyncMock()

            result = await tool.execute(action="invalid_action")

            assert result.success is False
            assert "Unknown action" in result.output

    @pytest.mark.asyncio
    async def test_exception_handling(self, tool):
        """Test exception handling in execute."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.side_effect = Exception("Connection failed")
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is False
            assert "Pipeline analysis failed" in result.output
            assert "Connection failed" in result.error


# =============================================================================
# FORMATTING TESTS
# =============================================================================


class TestFormatMethods:
    """Tests for formatting helper methods."""

    def test_format_platforms_multiple(self, tool):
        """Test formatting multiple platforms."""
        platforms = [PipelinePlatform.GITHUB_ACTIONS, PipelinePlatform.GITLAB_CI]
        output = tool._format_platforms(platforms)

        assert "Detected CI/CD Platforms" in output
        assert "github_actions" in output
        assert "gitlab_ci" in output

    def test_format_platforms_empty(self, tool):
        """Test formatting empty platforms list."""
        output = tool._format_platforms([])
        assert "No CI/CD platforms detected" in output

    def test_format_coverage_with_functions(self, tool, sample_coverage):
        """Test formatting coverage with function coverage."""
        output = tool._format_coverage(sample_coverage)

        assert "Code Coverage Report" in output
        assert "Line Coverage: 75.0%" in output
        assert "Branch Coverage: 75.0%" in output
        assert "Function Coverage: 80.0%" in output

    def test_format_coverage_without_functions(self, tool):
        """Test formatting coverage without function data."""
        coverage = CoverageMetrics(
            total_lines=100,
            covered_lines=80,
            total_branches=20,
            covered_branches=16,
            total_functions=0,
            covered_functions=0,
            line_coverage=80.0,
            branch_coverage=80.0,
            report_format="lcov",
        )
        output = tool._format_coverage(coverage)

        assert "Line Coverage: 80.0%" in output
        assert "Function Coverage" not in output

    def test_format_coverage_uncovered_files(self, tool, sample_coverage):
        """Test formatting coverage with uncovered files."""
        output = tool._format_coverage(sample_coverage)
        assert "Uncovered Files" in output
        assert "module1.py" in output

    def test_format_coverage_many_uncovered(self, tool):
        """Test formatting coverage with many uncovered files."""
        coverage = CoverageMetrics(
            total_lines=100,
            covered_lines=50,
            line_coverage=50.0,
            uncovered_files=[f"file{i}.py" for i in range(15)],
        )
        output = tool._format_coverage(coverage)
        assert "... and 5 more" in output

    def test_format_analysis_with_all_severities(self, tool, sample_analysis_result):
        """Test formatting analysis with all severity levels."""
        output = tool._format_analysis(sample_analysis_result)

        assert "Critical" in output
        assert "Warnings" in output
        assert "Info" in output

    def test_format_analysis_shows_steps(self, tool, sample_analysis_result):
        """Test formatting analysis shows pipeline steps."""
        output = tool._format_analysis(sample_analysis_result)
        assert "Steps:" in output
        assert "Triggers:" in output

    def test_format_summary_with_coverage(self, tool):
        """Test formatting summary with coverage."""
        summary = {
            "platforms_detected": 2,
            "pipeline_configs": 1,
            "critical_issues": 0,
            "warning_issues": 2,
            "total_issues": 2,
            "success_rate": 98.5,
            "avg_duration_seconds": 180,
            "coverage": {
                "line_coverage": 85.0,
                "branch_coverage": 80.0,
                "uncovered_files": 3,
            },
            "recommendations_count": 2,
        }
        output = tool._format_summary(summary)

        assert "Pipeline Health Summary" in output
        assert "Lines: 85.0%" in output
        assert "98.5%" in output
        assert "180s" in output

    def test_format_comparison_improved_files(self, tool):
        """Test formatting comparison with improved files."""
        comparison = {
            "line_coverage_delta": 5.0,
            "branch_coverage_delta": 3.0,
            "improved": True,
            "improved_files": [
                {"file": "src/main.py", "delta": 10.0},
                {"file": "src/utils.py", "delta": 5.0},
            ],
            "regressed_files": [],
            "new_uncovered_files": [],
        }
        output = tool._format_comparison(comparison)

        assert "Improved Files" in output
        assert "src/main.py" in output
        assert "+10.0%" in output

    def test_format_comparison_regressed_files(self, tool):
        """Test formatting comparison with regressed files."""
        comparison = {
            "line_coverage_delta": -3.0,
            "branch_coverage_delta": -2.0,
            "improved": False,
            "improved_files": [],
            "regressed_files": [
                {"file": "src/broken.py", "delta": -20.0},
            ],
            "new_uncovered_files": ["src/new.py"],
        }
        output = tool._format_comparison(comparison)

        assert "Regressed Files" in output
        assert "src/broken.py" in output
        assert "New Uncovered Files" in output


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests combining multiple aspects."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, tool, sample_analysis_result, sample_coverage):
        """Test a full analysis workflow."""
        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.return_value = [PipelinePlatform.GITHUB_ACTIONS]
            mock_manager.analyze_pipelines.return_value = sample_analysis_result
            mock_manager.get_coverage.return_value = sample_coverage
            MockManager.return_value = mock_manager

            # Step 1: Detect
            detect_result = await tool.execute(action="detect")
            assert detect_result.success is True

            # Step 2: Analyze
            analyze_result = await tool.execute(action="analyze")
            assert analyze_result.success is True
            assert analyze_result.metadata is not None

            # Step 3: Get coverage
            coverage_result = await tool.execute(action="coverage")
            assert coverage_result.success is True

    @pytest.mark.asyncio
    async def test_gitlab_platform_analysis(self, tool):
        """Test GitLab CI specific analysis."""
        gitlab_config = PipelineConfig(
            platform=PipelinePlatform.GITLAB_CI,
            file_path=Path(".gitlab-ci.yml"),
            name="GitLab CI",
            triggers=["push"],
            steps=[],
        )
        result = PipelineAnalysisResult(
            configs=[gitlab_config],
            issues=[],
            recommendations=[],
        )

        with patch("victor.tools.pipeline_tool.PipelineManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.analyze_pipelines.return_value = result
            MockManager.return_value = mock_manager

            analyze_result = await tool.execute(
                action="analyze",
                platform="gitlab_ci",
            )

            assert analyze_result.success is True
            call_args = mock_manager.analyze_pipelines.call_args
            assert call_args[1]["platforms"] == [PipelinePlatform.GITLAB_CI]

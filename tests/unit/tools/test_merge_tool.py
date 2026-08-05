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

"""Tests for Merge Conflict Tool."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import victor.tools.merge_tool as merge_tool_module
from victor.tools.merge_tool import MergeConflictTool
from victor.agent.presentation import NullPresentationAdapter
from victor.tools.base import CostTier
from victor.processing.merge import (
    ConflictComplexity,
    ConflictHunk,
    ConflictType,
    FileConflict,
    FileResolution,
    Resolution,
    ResolutionStrategy,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def use_null_presentation_adapter():
    """Use NullPresentationAdapter for predictable test output."""
    merge_tool_module._presentation = NullPresentationAdapter()
    yield
    merge_tool_module._presentation = None


@pytest.fixture
def tool():
    """Create merge conflict tool instance."""
    return MergeConflictTool()


@pytest.fixture
def sample_hunk():
    """Create sample conflict hunk."""
    return ConflictHunk(
        start_line=10,
        end_line=15,
        ours="def foo():\n    return 1",
        theirs="def foo():\n    return 2",
        context_before="# Module code",
        context_after="# End",
    )


@pytest.fixture
def sample_conflict(sample_hunk):
    """Create sample file conflict."""
    return FileConflict(
        file_path=Path("src/main.py"),
        conflict_type=ConflictType.CONTENT,
        hunks=[sample_hunk],
        complexity=ConflictComplexity.SIMPLE,
        ours_branch="HEAD",
        theirs_branch="feature-branch",
    )


@pytest.fixture
def sample_resolution():
    """Create sample resolution."""
    return Resolution(
        hunk_index=0,
        strategy=ResolutionStrategy.UNION,
        resolved_content="def foo():\n    return 3",
        confidence=0.85,
        explanation="Combined both changes",
    )


@pytest.fixture
def sample_file_resolution(sample_resolution):
    """Create sample file resolution."""
    return FileResolution(
        file_path=Path("src/main.py"),
        resolutions=[sample_resolution],
        final_content="# Complete resolved file",
        fully_resolved=True,
        needs_manual_review=False,
        applied=False,
    )


@pytest.fixture
def sample_summary():
    """Create sample conflict summary."""
    return {
        "has_conflicts": True,
        "total_files": 3,
        "total_hunks": 5,
        "estimated_effort": "medium",
        "auto_resolvable": 2,
        "needs_manual": 1,
        "by_complexity": {
            "trivial": 1,
            "simple": 1,
            "complex": 1,
        },
        "files": [
            {"path": "src/main.py", "complexity": "simple", "hunks": 2},
            {"path": "src/utils.py", "complexity": "trivial", "hunks": 1},
            {"path": "src/config.py", "complexity": "complex", "hunks": 2},
        ],
    }


# =============================================================================
# TOOL PROPERTIES TESTS
# =============================================================================


class TestMergeConflictToolProperties:
    """Tests for tool properties and metadata."""

    def test_tool_name(self, tool):
        """Test tool name uses canonical name from ToolNames."""
        from victor.tools.tool_names import ToolNames

        assert tool.name == ToolNames.MERGE
        assert tool.name == "merge"  # Canonical short name

    def test_tool_description_contains_strategies(self, tool):
        """Test description mentions resolution strategies."""
        assert "trivial" in tool.description
        assert "import" in tool.description
        assert "union" in tool.description

    def test_cost_tier(self, tool):
        """Test cost tier is FREE."""
        assert tool.cost_tier == CostTier.FREE

    def test_parameters_schema(self, tool):
        """Test parameters schema structure."""
        assert tool.parameters["type"] == "object"
        assert "action" in tool.parameters["properties"]
        assert "file_path" in tool.parameters["properties"]
        assert "strategy" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["action"]

    def test_action_enum_values(self, tool):
        """Test action enum has all expected values."""
        actions = tool.parameters["properties"]["action"]["enum"]
        assert "detect" in actions
        assert "analyze" in actions
        assert "resolve" in actions
        assert "apply" in actions
        assert "abort" in actions

    def test_strategy_enum_values(self, tool):
        """Test strategy enum has expected values."""
        strategies = tool.parameters["properties"]["strategy"]["enum"]
        assert "ours" in strategies
        assert "theirs" in strategies

    def test_metadata_category(self, tool):
        """Test metadata category."""
        assert tool.metadata.category == "merge"

    def test_metadata_keywords(self, tool):
        """Test metadata keywords."""
        keywords = tool.metadata.keywords
        assert "merge conflict" in keywords
        assert "conflict" in keywords
        assert "git merge" in keywords


# =============================================================================
# DETECT ACTION TESTS
# =============================================================================


class TestDetectAction:
    """Tests for the detect action."""

    @pytest.mark.asyncio
    async def test_detect_no_conflicts(self, tool):
        """Test detection when no conflicts."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_conflicts.return_value = []
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert "No merge conflicts detected" in result.output
            assert result.metadata["conflicts"] == []

    @pytest.mark.asyncio
    async def test_detect_with_conflicts(self, tool, sample_conflict):
        """Test detection with conflicts."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_conflicts.return_value = [sample_conflict]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert "Merge Conflicts Detected" in result.output
            assert result.metadata["conflict_count"] == 1
            assert "src/main.py" in result.metadata["files"][0]

    @pytest.mark.asyncio
    async def test_detect_multiple_conflicts(self, tool):
        """Test detection with multiple conflicts."""
        conflicts = [
            FileConflict(
                file_path=Path(f"src/file{i}.py"),
                conflict_type=ConflictType.CONTENT,
                hunks=[ConflictHunk(1, 5, "a", "b")],
                complexity=ConflictComplexity.SIMPLE,
            )
            for i in range(3)
        ]

        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_conflicts.return_value = conflicts
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert result.metadata["conflict_count"] == 3


# =============================================================================
# ANALYZE ACTION TESTS
# =============================================================================


class TestAnalyzeAction:
    """Tests for the analyze action."""

    @pytest.mark.asyncio
    async def test_analyze_basic(self, tool, sample_summary):
        """Test basic conflict analysis."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_conflict_summary.return_value = sample_summary
            MockManager.return_value = mock_manager

            result = await tool.execute(action="analyze")

            assert result.success is True
            assert "Conflict Analysis" in result.output
            assert "medium" in result.output

    @pytest.mark.asyncio
    async def test_analyze_no_conflicts(self, tool):
        """Test analysis with no conflicts."""
        summary = {"has_conflicts": False}

        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_conflict_summary.return_value = summary
            MockManager.return_value = mock_manager

            result = await tool.execute(action="analyze")

            assert result.success is True
            assert "No merge conflicts to analyze" in result.output

    @pytest.mark.asyncio
    async def test_analyze_shows_auto_resolvable(self, tool, sample_summary):
        """Test analysis shows auto-resolvable count."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_conflict_summary.return_value = sample_summary
            MockManager.return_value = mock_manager

            result = await tool.execute(action="analyze")

            assert "Auto-resolvable: 2" in result.output
            assert "Needs Manual: 1" in result.output


# =============================================================================
# RESOLVE ACTION TESTS
# =============================================================================


class TestResolveAction:
    """Tests for the resolve action."""

    @pytest.mark.asyncio
    async def test_resolve_basic(self, tool, sample_file_resolution):
        """Test basic conflict resolution."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.resolve_conflicts.return_value = [sample_file_resolution]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="resolve")

            assert result.success is True
            assert "Resolution Results" in result.output
            mock_manager.resolve_conflicts.assert_called_once_with(auto_apply=False)

    @pytest.mark.asyncio
    async def test_resolve_with_auto_apply(self, tool, sample_file_resolution):
        """Test resolution with auto-apply."""
        sample_file_resolution.applied = True

        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.resolve_conflicts.return_value = [sample_file_resolution]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="resolve", auto_apply=True)

            assert result.success is True
            mock_manager.resolve_conflicts.assert_called_once_with(auto_apply=True)

    @pytest.mark.asyncio
    async def test_resolve_no_conflicts(self, tool):
        """Test resolution with no conflicts."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.resolve_conflicts.return_value = []
            MockManager.return_value = mock_manager

            result = await tool.execute(action="resolve")

            assert result.success is True
            assert "No conflicts to resolve" in result.output

    @pytest.mark.asyncio
    async def test_resolve_partial(self, tool):
        """Test partial resolution."""
        partial_resolution = FileResolution(
            file_path=Path("src/complex.py"),
            resolutions=[
                Resolution(
                    hunk_index=0,
                    strategy=ResolutionStrategy.MANUAL,
                    resolved_content="",
                    confidence=0.0,
                    explanation="",
                )
            ],
            final_content="",
            fully_resolved=False,
            needs_manual_review=True,
        )

        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.resolve_conflicts.return_value = [partial_resolution]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="resolve")

            assert result.success is True
            assert result.metadata["needs_manual"] == 1


# =============================================================================
# APPLY ACTION TESTS
# =============================================================================


class TestApplyAction:
    """Tests for the apply action."""

    @pytest.mark.asyncio
    async def test_apply_success(self, tool):
        """Test successful resolution application."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.apply_resolution.return_value = True
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="apply",
                file_path="src/main.py",
                strategy="ours",
            )

            assert result.success is True
            assert "Applied ours strategy" in result.output

    @pytest.mark.asyncio
    async def test_apply_theirs_strategy(self, tool):
        """Test apply with theirs strategy."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.apply_resolution.return_value = True
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="apply",
                file_path="src/main.py",
                strategy="theirs",
            )

            assert result.success is True
            assert "theirs" in result.output

    @pytest.mark.asyncio
    async def test_apply_failure(self, tool):
        """Test failed resolution application."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.apply_resolution.return_value = False
            MockManager.return_value = mock_manager

            result = await tool.execute(
                action="apply",
                file_path="src/main.py",
                strategy="ours",
            )

            assert result.success is False
            assert "Failed to apply" in result.output

    @pytest.mark.asyncio
    async def test_apply_missing_file_path(self, tool):
        """Test apply without file path."""
        result = await tool.execute(action="apply", strategy="ours")

        assert result.success is False
        assert "file_path is required" in result.output

    @pytest.mark.asyncio
    async def test_apply_missing_strategy(self, tool):
        """Test apply without strategy."""
        result = await tool.execute(action="apply", file_path="src/main.py")

        assert result.success is False
        assert "strategy is required" in result.output


# =============================================================================
# ABORT ACTION TESTS
# =============================================================================


class TestAbortAction:
    """Tests for the abort action."""

    @pytest.mark.asyncio
    async def test_abort_success(self, tool):
        """Test successful merge abort."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.abort_merge.return_value = True
            MockManager.return_value = mock_manager

            result = await tool.execute(action="abort")

            assert result.success is True
            assert "aborted successfully" in result.output

    @pytest.mark.asyncio
    async def test_abort_failure(self, tool):
        """Test failed merge abort."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.abort_merge.return_value = False
            MockManager.return_value = mock_manager

            result = await tool.execute(action="abort")

            assert result.success is False
            assert "Failed to abort" in result.output


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """Test handling unknown action."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            MockManager.return_value = AsyncMock()

            result = await tool.execute(action="invalid_action")

            assert result.success is False
            assert "Unknown action" in result.output

    @pytest.mark.asyncio
    async def test_exception_handling(self, tool):
        """Test exception handling in execute."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_conflicts.side_effect = Exception("Git error")
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is False
            assert "Operation failed" in result.output
            assert "Git error" in result.error


# =============================================================================
# FORMATTING TESTS
# =============================================================================


class TestFormatMethods:
    """Tests for formatting helper methods."""

    def test_format_conflicts_basic(self, tool, sample_conflict):
        """Test formatting conflicts."""
        output = tool._format_conflicts([sample_conflict])

        assert "Merge Conflicts Detected" in output
        assert "**Total Files:** 1" in output
        assert "main.py" in output
        assert "content" in output

    def test_format_conflicts_complexity_icons(self, tool):
        """Test conflict formatting with different complexity levels."""
        conflicts = [
            FileConflict(
                file_path=Path(f"file_{complexity.value}.py"),
                conflict_type=ConflictType.CONTENT,
                hunks=[ConflictHunk(1, 5, "a", "b")],
                complexity=complexity,
            )
            for complexity in [
                ConflictComplexity.TRIVIAL,
                ConflictComplexity.SIMPLE,
                ConflictComplexity.MODERATE,
                ConflictComplexity.COMPLEX,
            ]
        ]

        output = tool._format_conflicts(conflicts)

        assert "[L]" in output  # trivial (level_low)
        assert "[M]" in output  # simple (level_medium)
        assert "[H]" in output  # moderate (level_high)
        assert "[!]" in output  # complex (level_critical)

    def test_format_analysis_no_conflicts(self, tool):
        """Test formatting analysis with no conflicts."""
        summary = {"has_conflicts": False}
        output = tool._format_analysis(summary)

        assert "No merge conflicts to analyze" in output

    def test_format_analysis_with_effort(self, tool, sample_summary):
        """Test formatting analysis with effort levels."""
        output = tool._format_analysis(sample_summary)

        assert "Effort Required" in output
        assert "[M]" in output  # medium effort (level_medium)

    def test_format_analysis_low_effort(self, tool):
        """Test formatting analysis with low effort."""
        summary = {
            "has_conflicts": True,
            "estimated_effort": "low",
            "total_files": 1,
            "total_hunks": 1,
            "auto_resolvable": 1,
            "needs_manual": 0,
            "by_complexity": {"trivial": 1},
            "files": [],
        }
        output = tool._format_analysis(summary)

        assert "[L]" in output  # low effort (level_low)

    def test_format_analysis_high_effort(self, tool):
        """Test formatting analysis with high effort."""
        summary = {
            "has_conflicts": True,
            "estimated_effort": "high",
            "total_files": 10,
            "total_hunks": 50,
            "auto_resolvable": 2,
            "needs_manual": 8,
            "by_complexity": {"complex": 8},
            "files": [],
        }
        output = tool._format_analysis(summary)

        assert "[!]" in output  # high effort (level_critical)

    def test_format_resolutions_empty(self, tool):
        """Test formatting empty resolutions."""
        output = tool._format_resolutions([])
        assert "No conflicts to resolve" in output

    def test_format_resolutions_with_data(self, tool, sample_file_resolution):
        """Test formatting resolutions with data."""
        output = tool._format_resolutions([sample_file_resolution])

        assert "Resolution Results" in output
        assert "**Fully Resolved:** 1" in output
        assert "+" in output  # success icon
        assert "85%" in output  # confidence

    def test_format_resolutions_partial(self, tool):
        """Test formatting partial resolutions."""
        partial = FileResolution(
            file_path=Path("src/partial.py"),
            resolutions=[
                Resolution(
                    hunk_index=0,
                    strategy=ResolutionStrategy.MANUAL,
                    resolved_content="",
                    confidence=0.0,
                )
            ],
            final_content="",
            fully_resolved=False,
            needs_manual_review=True,
        )
        output = tool._format_resolutions([partial])

        assert "!" in output  # warning icon
        assert "Partial" in output
        assert "requires manual resolution" in output

    def test_format_resolutions_applied(self, tool, sample_file_resolution):
        """Test formatting applied resolutions."""
        sample_file_resolution.applied = True
        output = tool._format_resolutions([sample_file_resolution])

        assert "(applied)" in output


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests combining multiple aspects."""

    @pytest.mark.asyncio
    async def test_full_conflict_workflow(
        self, tool, sample_conflict, sample_summary, sample_file_resolution
    ):
        """Test a full conflict resolution workflow."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_conflicts.return_value = [sample_conflict]
            mock_manager.get_conflict_summary.return_value = sample_summary
            mock_manager.resolve_conflicts.return_value = [sample_file_resolution]
            MockManager.return_value = mock_manager

            # Step 1: Detect
            detect_result = await tool.execute(action="detect")
            assert detect_result.success is True
            assert detect_result.metadata["conflict_count"] == 1

            # Step 2: Analyze
            analyze_result = await tool.execute(action="analyze")
            assert analyze_result.success is True

            # Step 3: Resolve
            resolve_result = await tool.execute(action="resolve", auto_apply=True)
            assert resolve_result.success is True

    @pytest.mark.asyncio
    async def test_apply_specific_strategy_workflow(self, tool):
        """Test applying specific strategy to a file."""
        with patch("victor.tools.merge_tool.MergeManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.apply_resolution.return_value = True
            MockManager.return_value = mock_manager

            # Apply ours strategy
            result = await tool.execute(
                action="apply",
                file_path="src/conflicted.py",
                strategy="ours",
            )

            assert result.success is True
            mock_manager.apply_resolution.assert_called_once()
            call_args = mock_manager.apply_resolution.call_args
            assert call_args[0][0] == "src/conflicted.py"
            assert call_args[0][1] == ResolutionStrategy.OURS

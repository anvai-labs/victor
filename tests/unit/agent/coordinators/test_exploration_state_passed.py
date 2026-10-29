"""Tests for ExplorationStatePassedCoordinator (SPA-2).

Validates the state-passed migration pattern:
1. Reads from ContextSnapshot (no orchestrator reference)
2. Returns CoordinatorResult with transitions
3. Reuses ExplorationCoordinator logic unchanged
4. Transitions correctly encode exploration findings
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.services.exploration_runtime import ExplorationResult
from victor.agent.coordinators.exploration_state_passed import (
    ExplorationStatePassedCoordinator,
)
from victor.agent.coordinators.state_context import (
    CoordinatorResult,
    ContextSnapshot,
    TransitionType,
)


def _make_snapshot(**overrides: Any) -> ContextSnapshot:
    """Create a minimal ContextSnapshot for testing."""
    defaults = {
        "messages": (),
        "session_id": "test-session",
        "conversation_stage": "initial",
        "settings": MagicMock(),
        "model": "test-model",
        "provider": "test-provider",
        "max_tokens": 4096,
        "temperature": 0.7,
        "conversation_state": {},
        "session_state": {},
        "observed_files": (),
        "capabilities": {},
    }
    defaults.update(overrides)
    return ContextSnapshot(**defaults)


class TestExplorationStatePassedInit:
    """Test coordinator initialization."""

    def test_creates_with_defaults(self):
        coord = ExplorationStatePassedCoordinator()
        assert coord._inner is not None
        assert coord._max_results == 5

    def test_creates_with_custom_project_root(self):
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp/test"))
        assert coord._project_root == Path("/tmp/test")


class TestExplorationStatePassedExplore:
    """Test the explore() method — pure function pattern."""

    @pytest.mark.asyncio
    async def test_reads_provider_from_snapshot(self):
        """Provider/model should come from snapshot, not orchestrator."""
        snapshot = _make_snapshot(provider="anthropic", model="claude-3")
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp"))

        with patch.object(
            coord._inner,
            "explore_parallel",
            new_callable=AsyncMock,
            return_value=ExplorationResult(),
        ) as mock:
            await coord.explore(snapshot, "test task")
            mock.assert_called_once()
            call_kwargs = mock.call_args
            assert (
                call_kwargs.kwargs.get("provider") == "anthropic"
                or call_kwargs[1].get("provider") == "anthropic"
            )

    @pytest.mark.asyncio
    async def test_reads_complexity_from_capabilities(self):
        """Task complexity should come from snapshot capabilities."""
        snapshot = _make_snapshot(
            capabilities={"task_complexity": "planning"},
        )
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp"))

        with patch.object(
            coord._inner,
            "explore_parallel",
            new_callable=AsyncMock,
            return_value=ExplorationResult(),
        ) as mock:
            await coord.explore(snapshot, "complex task")
            call_kwargs = mock.call_args
            assert (
                call_kwargs.kwargs.get("complexity") == "planning"
                or call_kwargs[1].get("complexity") == "planning"
            )

    @pytest.mark.asyncio
    async def test_allows_runtime_project_root_and_max_results_override(self):
        """Callers should be able to override project root and max_results per turn."""
        snapshot = _make_snapshot()
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp/default"), max_results=2)

        with patch.object(
            coord._inner,
            "explore_parallel",
            new_callable=AsyncMock,
            return_value=ExplorationResult(),
        ) as mock:
            await coord.explore(
                snapshot,
                "override task",
                project_root=Path("/tmp/runtime"),
                max_results=7,
            )
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs.get("project_root") == Path("/tmp/runtime") or call_kwargs[
                1
            ].get("project_root") == Path("/tmp/runtime")
            assert (
                call_kwargs.kwargs.get("max_results") == 7 or call_kwargs[1].get("max_results") == 7
            )

    @pytest.mark.asyncio
    async def test_returns_no_op_when_no_results(self):
        """Empty exploration should return no-op result."""
        snapshot = _make_snapshot()
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp"))

        with patch.object(
            coord._inner,
            "explore_parallel",
            new_callable=AsyncMock,
            return_value=ExplorationResult(),
        ):
            result = await coord.explore(snapshot, "test")
            assert isinstance(result, CoordinatorResult)
            assert result.transitions.is_empty()

    @pytest.mark.asyncio
    async def test_returns_transitions_with_file_paths(self):
        """Found files should produce UPDATE_STATE transitions."""
        snapshot = _make_snapshot()
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp"))

        exploration = ExplorationResult(
            file_paths=["src/main.py", "src/utils.py"],
            summary="Found 2 files",
            duration_seconds=1.5,
            tool_calls=3,
        )
        with patch.object(
            coord._inner,
            "explore_parallel",
            new_callable=AsyncMock,
            return_value=exploration,
        ):
            result = await coord.explore(snapshot, "fix the bug")

            assert not result.transitions.is_empty()
            # Should have transitions for files, summary, and metrics
            types = [t.transition_type for t in result.transitions.transitions]
            assert TransitionType.UPDATE_STATE in types

    @pytest.mark.asyncio
    async def test_result_has_metadata(self):
        """Result should include file paths and tool call count in metadata."""
        snapshot = _make_snapshot()
        coord = ExplorationStatePassedCoordinator(project_root=Path("/tmp"))

        exploration = ExplorationResult(
            file_paths=["src/main.py"],
            summary="Found file",
            duration_seconds=0.5,
            tool_calls=2,
        )
        with patch.object(
            coord._inner,
            "explore_parallel",
            new_callable=AsyncMock,
            return_value=exploration,
        ):
            result = await coord.explore(snapshot, "test")
            assert result.metadata["file_paths"] == ["src/main.py"]
            assert result.metadata["summary"] == "Found file"
            assert result.metadata["tool_calls"] == 2
            assert result.metadata["duration_seconds"] == 0.5

    @pytest.mark.asyncio
    async def test_no_orchestrator_reference(self):
        """Coordinator must not hold any orchestrator reference."""
        coord = ExplorationStatePassedCoordinator()
        assert not hasattr(coord, "_orchestrator")
        assert not hasattr(coord, "orchestrator")


class TestExplorationStatePassedTransitions:
    """Test that transitions correctly encode exploration findings."""

    def test_to_coordinator_result_empty(self):
        """Empty result should produce no-op."""
        coord = ExplorationStatePassedCoordinator()
        result = coord._to_coordinator_result(ExplorationResult())
        assert result.transitions.is_empty()

    def test_to_coordinator_result_with_files(self):
        """Files should be stored as explored_files state transition."""
        coord = ExplorationStatePassedCoordinator()
        exploration = ExplorationResult(
            file_paths=["a.py", "b.py"],
            summary="",
            duration_seconds=1.0,
            tool_calls=2,
        )
        result = coord._to_coordinator_result(exploration)
        assert not result.transitions.is_empty()

        # Find the explored_files transition
        file_transitions = [
            t
            for t in result.transitions.transitions
            if t.transition_type == TransitionType.UPDATE_STATE
            and t.data.get("key") == "explored_files"
        ]
        assert len(file_transitions) == 1
        assert file_transitions[0].data["value"] == ["a.py", "b.py"]

    def test_confidence_scales_with_results(self):
        """Confidence should increase with more files found."""
        coord = ExplorationStatePassedCoordinator()

        # 0 files → no-op
        result0 = coord._to_coordinator_result(ExplorationResult())
        assert result0.transitions.is_empty()

        # 1 file → low confidence
        result1 = coord._to_coordinator_result(
            ExplorationResult(
                file_paths=["a.py"], summary="found", duration_seconds=0.1, tool_calls=1
            )
        )
        assert result1.confidence < 1.0

        # 3+ files → high confidence
        result3 = coord._to_coordinator_result(
            ExplorationResult(
                file_paths=["a.py", "b.py", "c.py"],
                summary="found",
                duration_seconds=0.1,
                tool_calls=3,
            )
        )
        assert result3.confidence >= 1.0

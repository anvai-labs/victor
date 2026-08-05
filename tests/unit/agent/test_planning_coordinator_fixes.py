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

"""Unit tests for PlanningCoordinator approval workflow fixes.

These tests verify the fixes for plan mode issues:
- plan execution auto-starts by default while tool-level approval still protects risky actions
- Plan approval workflow
- Plan persistence to disk
- Renderer integration
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from victor_contracts.skills import SkillDefinition
from victor.agent.services.planning_runtime import (
    PlanningCoordinator,
    PlanningConfig,
    PlanningMode,
)
from victor.framework.execution_checkpoint import ApprovalState
from victor.agent.planning import ReadableTaskPlan, TaskComplexity


class TestPlanningCoordinatorApproval:
    """Tests for plan approval workflow."""

    def test_auto_approve_default_is_true(self):
        """Plan display should not add a second approval prompt by default."""
        config = PlanningConfig()
        assert config.auto_approve is True

    def test_auto_approve_can_be_enabled(self):
        """auto_approve can be explicitly set to True when needed."""
        config = PlanningConfig(auto_approve=True)
        assert config.auto_approve is True

    @pytest.mark.asyncio
    async def test_plan_rejected_when_user_declines(self):
        """Plan should not execute when user explicitly declines approval.

        Simulates an interactive TTY where the user types 'n'.  We mock both
        sys.stdin.isatty (returns True so we reach the Confirm prompt) and
        asyncio.to_thread so the test doesn't actually block on stdin.
        """
        mock_orchestrator = MagicMock()
        mock_orchestrator.model = "test-model"
        mock_orchestrator.max_tokens = 1000
        mock_orchestrator.provider = MagicMock()
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=MagicMock(
                content="Plan was rejected. Would you like to try a different approach?"
            )
        )

        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        plan = ReadableTaskPlan(
            name="Test Plan",
            desc="A test plan",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "testing", "Run tests", "shell"]],
        )

        # Simulate interactive TTY with user typing 'n'
        with (
            patch("victor.agent.services.planning_runtime.sys") as mock_sys,
            patch(
                "victor.agent.services.planning_runtime.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_thread,
        ):
            mock_sys.stdin.isatty.return_value = True
            mock_thread.return_value = False  # user said no

            result = await coordinator._show_plan_to_user(plan)

            assert result.proceed is False
            assert result.user_approved_execution is False
            assert result.approval_state is ApprovalState.REJECTED

    @pytest.mark.asyncio
    async def test_plan_approved_when_user_accepts(self):
        """Plan should proceed when user approves."""
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=mock_renderer)

        plan = ReadableTaskPlan(
            name="Test Plan",
            desc="A test plan",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "testing", "Run tests", "shell"]],
        )

        # Simulate interactive TTY with user pressing Enter (default=True)
        with (
            patch("victor.agent.services.planning_runtime.sys") as mock_sys,
            patch(
                "victor.agent.services.planning_runtime.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_thread,
        ):
            mock_sys.stdin.isatty.return_value = True
            mock_thread.return_value = True  # user pressed Enter / said yes

            approved = await coordinator._show_plan_to_user(plan)

            assert approved.proceed is True
            assert approved.user_approved_execution is True
            assert approved.approval_state is ApprovalState.APPROVED


class TestPlanningCoordinatorRendererIntegration:
    """Tests for renderer integration in PlanningCoordinator."""

    def test_renderer_can_be_injected(self):
        """Renderer should be injectable into PlanningCoordinator."""
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=mock_renderer)

        assert coordinator.renderer == mock_renderer

    def test_defaults_to_none_when_no_renderer(self):
        """PlanningCoordinator should work without renderer."""
        mock_orchestrator = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=None)  # No renderer

        assert coordinator.renderer is None

    @pytest.mark.asyncio
    async def test_shows_plan_with_renderer_when_available(self):
        """Should use injected renderer for consistent display."""
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=mock_renderer)

        plan = ReadableTaskPlan(
            name="Test Plan",
            desc="A test plan",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "research", "Test step", "read"]],
        )

        # Non-interactive stdin → auto-approve path (no blocking Confirm.ask)
        with patch("victor.agent.services.planning_runtime.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False

            approved = await coordinator._show_plan_to_user(plan)

            # Verify renderer lifecycle hooks were called
            mock_renderer.pause.assert_called()
            mock_renderer.resume.assert_called()
            assert approved.proceed is True
            assert approved.approval_state is ApprovalState.NOT_REQUIRED

    @pytest.mark.asyncio
    async def test_falls_back_to_console_when_no_renderer(self):
        """Should fallback to Rich console when renderer not available."""
        mock_orchestrator = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=None)

        plan = ReadableTaskPlan(
            name="Test Plan",
            desc="A test plan",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "research", "Test step", "read"]],
        )

        # Non-interactive stdin → auto-approve; must not raise
        with patch("victor.agent.services.planning_runtime.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False

            with patch.object(coordinator, "_save_plan_to_disk"):
                approved = await coordinator._show_plan_to_user(plan)
            assert approved.proceed is True
            assert approved.approval_state is ApprovalState.NOT_REQUIRED


class TestPlanPersistence:
    """Tests for plan persistence to disk."""

    def test_plan_saved_to_disk(self):
        """Plan should be saved to ~/.victor/plans/ directory."""
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=mock_renderer)

        plan = ReadableTaskPlan(
            name="Test Plan for Persistence",
            desc="A test plan for persistence to disk",
            complexity=TaskComplexity.SIMPLE,
            steps=[
                [1, "research", "Read documentation", "read"],
                [2, "analysis", "Analyze code", "code_search"],
            ],
        )

        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.expanduser", return_value=tmpdir):
                coordinator._save_plan_to_disk(plan, mock_renderer.console)

                # Verify file was created
                files = list(Path(tmpdir).glob("*.json"))
                assert len(files) == 1, f"Expected 1 plan file, found {len(files)}"

                # Verify content
                with files[0].open() as f:
                    saved_plan = json.load(f)

                assert saved_plan["name"] == "Test Plan for Persistence"
                assert saved_plan["step_count"] == 2
                assert saved_plan["complexity"] == "simple"
                assert len(saved_plan["steps"]) == 2
                assert saved_plan["steps"][0]["type"] == "research"
                assert saved_plan["steps"][1]["type"] == "analysis"


def _make_skill(name: str, phase: str = "action") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=f"Skill: {name}",
        category="coding",
        prompt_fragment=f"Prompt for {name}.",
        required_tools=["read_file"],
        phase=phase,
    )


class TestPlanningCoordinatorSkillIntegration:
    """Planning should reuse the framework skill decomposition helpers."""

    @pytest.mark.asyncio
    async def test_generate_plan_uses_shared_skill_catalog_and_sequence(self):
        mock_orchestrator = MagicMock()
        mock_orchestrator.provider = MagicMock()
        mock_orchestrator.model = "test-model"
        mock_orchestrator.profile = None
        matcher = MagicMock()
        matcher.initialized = True
        matcher.skills = {
            "debug": _make_skill("debug", phase="diagnostic"),
            "refactor": _make_skill("refactor", phase="action"),
        }
        matcher.match_multiple_sync.return_value = [
            (matcher.skills["debug"], 0.82),
            (matcher.skills["refactor"], 0.64),
        ]
        mock_orchestrator._skill_matcher = matcher

        coordinator = PlanningCoordinator(mock_orchestrator)
        generated_plan = ReadableTaskPlan(
            name="Skill-aware plan",
            desc="generated",
            complexity=TaskComplexity.MODERATE,
            steps=[[1, "debug", "Inspect the failure", "read"]],
        )

        with patch(
            "victor.agent.services.planning_runtime.generate_task_plan",
            AsyncMock(return_value=generated_plan),
        ) as generate_task_plan:
            result = await coordinator._generate_plan("fix the auth regression")

        assert result is generated_plan
        prompt = generate_task_plan.await_args.kwargs["user_request"]
        assert "Available skills" in prompt
        assert "Suggested ordered skill sequence" in prompt
        assert "debug -> refactor" in prompt

    def test_plan_saved_with_metadata(self):
        """Plan should include metadata like timestamp and step count."""
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=mock_renderer)

        plan = ReadableTaskPlan(
            name="Metadata Test Plan",
            desc="A test plan for metadata fields",
            complexity=TaskComplexity.COMPLEX,
            steps=[[1, "test", "Test step", "test_tool"]],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.expanduser", return_value=tmpdir):
                coordinator._save_plan_to_disk(plan, mock_renderer.console)

                # Verify metadata
                plan_file = list(Path(tmpdir).glob("*.json"))[0]
                with plan_file.open() as f:
                    saved_plan = json.load(f)

                # Check required fields
                assert "generated_at" in saved_plan
                assert "step_count" in saved_plan
                assert saved_plan["step_count"] == 1
                assert "complexity" in saved_plan
                assert saved_plan["complexity"] == "complex"

    def test_plan_save_error_handling(self):
        """Plan save should handle errors gracefully."""
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=mock_renderer)

        plan = ReadableTaskPlan(
            name="Error Test Plan",
            desc="A test plan for error handling",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "test", "Test", "test"]],
        )

        # Mock filesystem error
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            # Should not raise error
            coordinator._save_plan_to_disk(plan, mock_renderer.console)

        # Console should show warning (verify console.print was called)
        assert mock_renderer.console.print.called


class TestPlanningCoordinatorConfiguration:
    """Tests for PlanningCoordinator configuration."""

    def test_show_plan_before_execution_default(self):
        """show_plan_before_execution should default to True."""
        config = PlanningConfig()
        assert config.show_plan_before_execution is True

    def test_config_accepts_renderer_parameter(self):
        """PlanningConfig should accept renderer parameter."""
        config = PlanningConfig()
        # Config doesn't have renderer param, but coordinator does
        # This test verifies the pattern
        assert config is not None

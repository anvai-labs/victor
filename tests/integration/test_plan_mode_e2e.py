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

"""Integration tests for plan mode end-to-end flows.

These tests verify the complete plan mode workflow including approval,
persistence, and rendering integration.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from victor.agent.services.planning_runtime import (
    PlanningCoordinator,
    PlanningConfig,
    PlanningMode,
)
from victor.framework.execution_checkpoint import ApprovalState
from victor.agent.planning import ReadableTaskPlan, TaskComplexity
from rich.console import Console


class TestPlanModeApprovalFlow:
    """Tests for plan approval workflow."""

    @pytest.mark.asyncio
    async def test_plan_requires_approval_before_execution(self):
        """Plan should require user approval before executing steps.

        This is a critical safety test - plans must not execute silently.
        """
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.model = "test-model"
        mock_orchestrator.max_tokens = 1000
        mock_orchestrator.provider = MagicMock()
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=MagicMock(content="I cannot execute this plan without your approval.")
        )

        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        # Create a test plan
        plan = ReadableTaskPlan(
            name="Test Plan",
            desc="A test plan for approval workflow",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "testing", "Run tests", "shell"]],
        )

        # Mock user declining approval
        with (
            patch("rich.prompt.Confirm") as mock_confirm,
            patch(
                "victor.agent.services.planning_runtime.sys.stdin.isatty",
                return_value=True,
            ),
        ):
            mock_confirm.ask.return_value = False

            decision = await coordinator._show_plan_to_user(plan)

            # Plan should NOT be approved
            assert decision.proceed is False
            assert decision.user_approved_execution is False
            assert decision.approval_state is ApprovalState.REJECTED

            # Verify confirmation was shown
            assert mock_confirm.ask.called

    @pytest.mark.asyncio
    async def test_auto_approve_flag_skips_confirmation(self):
        """auto_approve=True should skip confirmation prompt.

        This allows for automated workflows where user approval is
        pre-configured.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # Create coordinator with auto_approve enabled
        config = PlanningConfig(auto_approve=True)
        coordinator = PlanningCoordinator(mock_orchestrator, config=config, renderer=mock_renderer)

        plan = ReadableTaskPlan(
            name="Auto-Approve Plan",
            desc="A plan that should be auto-approved",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "test", "Test step", "test"]],
        )

        with patch("rich.prompt.Confirm") as mock_confirm:
            decision = await coordinator._show_plan_to_user(plan)

            # Should be auto-approved without prompting
            assert decision.proceed is True
            assert decision.user_approved_execution is True
            assert decision.approval_state is ApprovalState.APPROVED
            # Confirm should NOT have been called
            assert not mock_confirm.ask.called

    @pytest.mark.asyncio
    async def test_plan_rejection_handled_gracefully(self):
        """Plan rejection should generate helpful response.

        When user rejects a plan, the system should acknowledge and
        offer alternatives.
        """
        mock_orchestrator = MagicMock()
        mock_orchestrator.model = "test-model"
        mock_orchestrator.max_tokens = 1000
        mock_orchestrator.provider = MagicMock()
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=MagicMock(
                content="I understand you've rejected the plan. Would you like to try a different approach?"
            )
        )

        coordinator = PlanningCoordinator(mock_orchestrator)

        plan = ReadableTaskPlan(
            name="Rejected Plan",
            desc="A plan that will be rejected",
            complexity=TaskComplexity.COMPLEX,
            steps=[
                [1, "research", "Step 1", "read"],
                [2, "analysis", "Step 2", "analyze"],
            ],
        )

        # Generate rejection response
        response = await coordinator._generate_plan_rejected_response(plan)

        # Should return a helpful response
        assert response is not None
        assert "rejected" in response.content.lower() or "understand" in response.content.lower()
        assert len(response.content) > 0


class TestPlanPersistence:
    """Tests for plan persistence to disk."""

    def test_plan_saved_to_victor_plans_directory(self):
        """Plans should be saved to ~/.victor/plans/ directory.

        This provides an audit trail and allows users to review
        historical plans.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        plan = ReadableTaskPlan(
            name="Persistence Test Plan",
            desc="A plan to test persistence to disk",
            complexity=TaskComplexity.SIMPLE,
            steps=[
                [1, "research", "Read documentation", "read"],
                [2, "implementation", "Write code", "code_search"],
            ],
        )

        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.expanduser", return_value=tmpdir):
                coordinator._save_plan_to_disk(plan, mock_renderer.console)

                # Verify ~/.victor/plans/ directory was created
                plans_dir = Path(tmpdir)
                assert plans_dir.exists()

                # Verify plan file was created
                plan_files = list(plans_dir.glob("*.json"))
                assert len(plan_files) == 1

                # Verify file content
                with plan_files[0].open() as f:
                    saved_plan = json.load(f)

                assert saved_plan["name"] == "Persistence Test Plan"
                assert saved_plan["step_count"] == 2
                assert saved_plan["complexity"] == "simple"
                assert len(saved_plan["steps"]) == 2

    def test_plan_file_contains_all_required_fields(self):
        """Plan JSON should contain all required metadata.

        Each saved plan should include name, complexity, steps,
        timestamp, and step count for proper tracking.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        plan = ReadableTaskPlan(
            name="Metadata Test Plan",
            desc="A plan to test metadata fields",
            complexity=TaskComplexity.COMPLEX,
            steps=[
                [1, "research", "Research phase", "read"],
                [2, "analysis", "Analysis phase", "analyze"],
                [3, "implementation", "Implementation phase", "write"],
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.expanduser", return_value=tmpdir):
                coordinator._save_plan_to_disk(plan, mock_renderer.console)

                plan_file = list(Path(tmpdir).glob("*.json"))[0]
                with plan_file.open() as f:
                    saved_plan = json.load(f)

                # Verify all required fields
                assert "name" in saved_plan
                assert "complexity" in saved_plan
                assert "steps" in saved_plan
                assert "generated_at" in saved_plan
                assert "step_count" in saved_plan

                # Verify field values
                assert saved_plan["name"] == "Metadata Test Plan"
                assert saved_plan["complexity"] == "complex"
                assert saved_plan["step_count"] == 3
                assert len(saved_plan["steps"]) == 3

    def test_multiple_plans_saved_with_unique_filenames(self):
        """Multiple plans should have unique timestamp-based filenames.

        This prevents overwriting and maintains chronological history.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        plan1 = ReadableTaskPlan(
            name="Plan 1",
            desc="First plan",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "test", "Step 1", "test"]],
        )

        plan2 = ReadableTaskPlan(
            name="Plan 2",
            desc="Second plan",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "test", "Step 1", "test"]],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.expanduser", return_value=tmpdir):
                # Save both plans
                coordinator._save_plan_to_disk(plan1, mock_renderer.console)
                coordinator._save_plan_to_disk(plan2, mock_renderer.console)

                # Verify two separate files
                plan_files = list(Path(tmpdir).glob("*.json"))
                assert len(plan_files) == 2

                # Verify filenames are unique
                filenames = [f.name for f in plan_files]
                assert len(set(filenames)) == 2  # All unique


class TestPlanModeRenderingIntegration:
    """Tests for consistent rendering in plan mode."""

    @pytest.mark.asyncio
    async def test_uses_injected_renderer_when_available(self):
        """PlanningCoordinator should use injected renderer for display.

        This ensures consistent UI between normal chat and plan mode.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        plan = ReadableTaskPlan(
            name="Renderer Test Plan",
            desc="A plan to test renderer integration",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "test", "Test step", "test"]],
        )

        with patch("rich.prompt.Confirm") as mock_confirm:
            mock_confirm.ask.return_value = True

            decision = await coordinator._show_plan_to_user(plan)

            # Verify renderer was used
            assert mock_renderer.pause.called
            assert mock_renderer.resume.called
            assert decision.proceed is True
            assert decision.approval_state is ApprovalState.APPROVED

    @pytest.mark.asyncio
    async def test_falls_back_to_console_when_no_renderer(self):
        """PlanningCoordinator should fallback to Rich console when renderer not available.

        This ensures backward compatibility and graceful degradation.
        """
        mock_orchestrator = MagicMock()

        coordinator = PlanningCoordinator(mock_orchestrator, renderer=None)  # No renderer injected

        plan = ReadableTaskPlan(
            name="Fallback Test Plan",
            desc="A plan to test console fallback",
            complexity=TaskComplexity.SIMPLE,
            steps=[[1, "test", "Test step", "test"]],
        )

        with patch("rich.prompt.Confirm") as mock_confirm:
            mock_confirm.ask.return_value = True

            # Should not raise error
            decision = await coordinator._show_plan_to_user(plan)
            assert decision.proceed is True
            assert decision.approval_state is ApprovalState.APPROVED

    @pytest.mark.asyncio
    async def test_plan_table_displayed_correctly(self):
        """Plan should be displayed in a well-formatted Rich table.

        This tests the visual presentation of plans to users.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        plan = ReadableTaskPlan(
            name="Display Test Plan",
            desc="A plan to test table display",
            complexity=TaskComplexity.COMPLEX,
            steps=[
                [1, "research", "Research requirements", "read"],
                [2, "design", "Design architecture", "analyze"],
                [3, "implementation", "Implement solution", "write"],
            ],
        )

        with patch("rich.prompt.Confirm") as mock_confirm:
            mock_confirm.ask.return_value = True

            decision = await coordinator._show_plan_to_user(plan)

            # Console.print should have been called for table display
            assert mock_renderer.console.print.called
            assert decision.proceed is True
            assert decision.approval_state is ApprovalState.APPROVED


class TestPlanModeEndToEnd:
    """End-to-end tests for complete plan mode workflows."""

    @pytest.mark.asyncio
    async def test_full_plan_workflow_with_approval(self):
        """Complete workflow: generate plan → show to user → get approval → save to disk.

        This is the primary happy path for plan mode.
        """
        # Mock orchestrator with realistic behavior
        mock_orchestrator = MagicMock()
        mock_orchestrator.model = "test-model"
        mock_orchestrator.max_tokens = 1000
        mock_orchestrator.provider = MagicMock()

        # Mock chat responses for plan generation and execution
        mock_orchestrator.provider.chat = AsyncMock(
            side_effect=[
                # First call: plan generation
                MagicMock(
                    content="I'll help you with that. Here's my plan:",
                    metadata={
                        "plan": ReadableTaskPlan(
                            name="End-to-End Test Plan",
                            desc="A comprehensive end-to-end test plan",
                            complexity=TaskComplexity.SIMPLE,
                            steps=[[1, "test", "Test step", "test"]],
                        )
                    },
                ),
                # Second call: after approval
                MagicMock(content="Plan approved. Executing step 1...", metadata={}),
            ]
        )

        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        # Simulate user approving the plan
        with (
            patch("rich.prompt.Confirm") as mock_confirm,
            patch(
                "victor.agent.services.planning_runtime.sys.stdin.isatty",
                return_value=True,
            ),
        ):
            mock_confirm.ask.return_value = True

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("os.path.expanduser", return_value=tmpdir):
                    # Execute the workflow
                    plan = ReadableTaskPlan(
                        name="End-to-End Test Plan",
                        desc="A comprehensive end-to-end test plan",
                        complexity=TaskComplexity.SIMPLE,
                        steps=[[1, "test", "Test step", "test"]],
                    )

                    decision = await coordinator._show_plan_to_user(plan)

                    # Verify approval workflow
                    assert decision.proceed is True
                    assert decision.approval_state is ApprovalState.APPROVED
                    assert mock_confirm.ask.called

                    # Verify plan was saved
                    plan_files = list(Path(tmpdir).glob("*.json"))
                    assert len(plan_files) == 1

    @pytest.mark.asyncio
    async def test_full_plan_workflow_with_rejection(self):
        """Complete workflow: generate plan → show to user → rejection → helpful response.

        This tests the rejection path which should provide alternatives.
        """
        mock_orchestrator = MagicMock()
        mock_orchestrator.model = "test-model"
        mock_orchestrator.max_tokens = 1000
        mock_orchestrator.provider = MagicMock()
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=MagicMock(content="I understand. Let's try a different approach.")
        )

        coordinator = PlanningCoordinator(mock_orchestrator)

        plan = ReadableTaskPlan(
            name="Rejected E2E Plan",
            desc="A plan that will be rejected in e2e test",
            complexity=TaskComplexity.COMPLEX,
            steps=[[1, "test", "Test step", "test"]],
        )

        # Generate rejection response
        response = await coordinator._generate_plan_rejected_response(plan)

        # Verify helpful response
        assert response is not None
        assert len(response.content) > 0
        # Should offer alternatives
        assert any(
            term in response.content.lower()
            for term in ["modify", "different", "approach", "alternative"]
        )

    @pytest.mark.asyncio
    async def test_plan_mode_with_complex_multistep_plan(self):
        """Complex plans with multiple steps should display correctly.

        This tests scalability of plan display with realistic plan sizes.
        """
        mock_orchestrator = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()

        # PlanningConfig.auto_approve defaults to True (intended production
        # behavior). Opt into the interactive-approval path under test.
        coordinator = PlanningCoordinator(
            mock_orchestrator,
            config=PlanningConfig(auto_approve=False),
            renderer=mock_renderer,
        )

        # Create a complex multi-step plan
        plan = ReadableTaskPlan(
            name="Complex Multi-Step Plan",
            desc="A complex plan with multiple steps to test scalability",
            complexity=TaskComplexity.COMPLEX,
            steps=[
                [1, "research", "Analyze requirements", "read"],
                [2, "research", "Review existing code", "code_search"],
                [3, "design", "Design solution", "analyze"],
                [4, "implementation", "Implement feature", "write"],
                [5, "testing", "Write tests", "test"],
                [6, "testing", "Run integration tests", "test"],
                [7, "documentation", "Update docs", "write"],
                [8, "review", "Code review", "analyze"],
            ],
        )

        with patch("rich.prompt.Confirm") as mock_confirm:
            mock_confirm.ask.return_value = True

            decision = await coordinator._show_plan_to_user(plan)

            # Should handle large plans without issues
            assert decision.proceed is True
            assert decision.approval_state is ApprovalState.APPROVED
            assert mock_renderer.console.print.called

            # Verify all steps are preserved
            assert len(plan.steps) == 8

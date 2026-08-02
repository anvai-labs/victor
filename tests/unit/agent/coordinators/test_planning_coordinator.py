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

"""Unit tests for PlanningCoordinator and chat service/coordinator integration."""

import pytest

from victor.agent.services.planning_runtime import (
    PlanningCoordinator,
    PlanningConfig,
    PlanningMode,
    PlanningResult,
)
from victor.framework.task import TaskComplexity


class TestPlanningConfig:
    """Tests for PlanningConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PlanningConfig()

        assert config.min_planning_complexity == TaskComplexity.MEDIUM
        assert config.min_steps_threshold == 3
        assert len(config.complexity_keywords) > 0
        assert config.show_plan_before_execution is True
        assert config.fallback_on_planning_failure is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlanningConfig(
            min_planning_complexity=TaskComplexity.COMPLEX,
            min_steps_threshold=2,
            show_plan_before_execution=False,
        )

        assert config.min_planning_complexity == TaskComplexity.COMPLEX
        assert config.min_steps_threshold == 2
        assert config.show_plan_before_execution is False


class TestPlanningMode:
    """Tests for PlanningMode enum."""

    def test_planning_mode_values(self):
        """Test planning mode enum values."""
        assert PlanningMode.AUTO.value == "auto"
        assert PlanningMode.ALWAYS.value == "always"
        assert PlanningMode.NEVER.value == "never"


class TestPlanningResult:
    """Tests for PlanningResult."""

    def test_planned_result_success(self):
        """Test planned result with success."""
        from victor.agent.planning.readable_schema import (
            ReadableTaskPlan,
            TaskComplexity as PlanningTaskComplexity,
        )

        plan = ReadableTaskPlan(
            name="Test plan",
            complexity=PlanningTaskComplexity.MODERATE,
            desc="Test description",
            steps=[[1, "research", "Test step", "overview"]],
            duration="10min",
        )

        result = PlanningResult(
            mode="planned",
            plan=plan,
            steps_completed=1,
            steps_total=1,
        )

        assert result.mode == "planned"
        assert result.plan is not None
        assert result.steps_completed == 1
        assert result.steps_total == 1
        # Success requires execution_result
        assert result.success is False

    def test_direct_result_success(self):
        """Test direct chat result with success."""
        from victor.providers.base import CompletionResponse

        response = CompletionResponse(
            content="Test response",
            role="assistant",
            tool_calls=None,
        )

        result = PlanningResult(
            mode="direct",
            response=response,
        )

        assert result.mode == "direct"
        assert result.response is not None
        assert result.success is True


class TestPlanningCoordinator:
    """Tests for PlanningCoordinator (without orchestrator)."""

    def _create_mock_orchestrator(self):
        """Helper to create a mock orchestrator."""

        class MockOrchestrator:
            provider = None
            model = "test-model"
            max_tokens = 4096

        return MockOrchestrator()

    def _create_coordinator(self):
        """Helper to create a coordinator with default config."""
        mock_orch = self._create_mock_orchestrator()
        return PlanningCoordinator(mock_orch)

    def test_planning_coordinator_init(self):
        """Test coordinator initialization."""
        mock_orch = self._create_mock_orchestrator()
        coordinator = PlanningCoordinator(mock_orch)

        assert coordinator.orchestrator == mock_orch
        assert coordinator.active_plan is None
        assert coordinator.config is not None

    def test_planning_mode_setter(self):
        """Test setting planning mode."""
        coordinator = self._create_coordinator()

        coordinator.set_planning_mode(PlanningMode.ALWAYS)
        assert coordinator._planning_mode == PlanningMode.ALWAYS

        coordinator.set_planning_mode(PlanningMode.NEVER)
        assert coordinator._planning_mode == PlanningMode.NEVER

    def test_clear_active_plan(self):
        """Test clearing active plan."""
        coordinator = self._create_coordinator()

        # Set a mock plan
        from victor.agent.planning.readable_schema import (
            ReadableTaskPlan,
            TaskComplexity as PlanningTaskComplexity,
        )

        plan = ReadableTaskPlan(
            name="Test",
            complexity=PlanningTaskComplexity.SIMPLE,
            desc="Test",
            steps=[[1, "research", "Test", "overview"]],
            duration="5min",
        )
        coordinator.active_plan = plan

        assert coordinator.get_active_plan() is not None

        coordinator.clear_active_plan()

        assert coordinator.get_active_plan() is None

    def test_should_use_planning_keywords(self):
        """Test planning detection via keywords."""
        coordinator = self._create_coordinator()

        # Test with multi-step keywords
        message = "Analyze the architecture and evaluate scalability"
        should_plan = coordinator._should_use_planning(message)

        assert should_plan is True

    def test_should_not_use_planning_simple(self):
        """Test that simple messages don't trigger planning."""
        coordinator = self._create_coordinator()

        # Test simple message
        message = "What is the weather today?"
        should_plan = coordinator._should_use_planning(message)

        assert should_plan is False

    def test_should_use_planning_mode_never(self):
        """Test NEVER planning mode."""
        coordinator = self._create_coordinator()
        coordinator.set_planning_mode(PlanningMode.NEVER)

        # Even with keywords, should not plan
        message = "Analyze the architecture and evaluate scalability"
        should_plan = coordinator._should_use_planning(message)

        assert should_plan is False

    def test_should_use_planning_mode_always(self):
        """Test ALWAYS planning mode."""
        coordinator = self._create_coordinator()
        coordinator.set_planning_mode(PlanningMode.ALWAYS)

        # Even simple message should plan
        message = "Hello world"
        should_plan = coordinator._should_use_planning(message)

        assert should_plan is True

    def test_should_use_planning_for_checklist_first_request(self):
        """Explicit checklist-first requests should not fall through to direct chat."""
        coordinator = self._create_coordinator()

        message = (
            "Review this Rust workspace by creating a checklist first and showing me, "
            "then go module by module."
        )
        should_plan = coordinator._should_use_planning(message)

        assert should_plan is True

    def test_summary_prompt_includes_evidence_and_failure_state(self):
        """Final summary prompt should be grounded in step results, not generic completion."""
        from victor.agent.planning.base import PlanResult, StepResult
        from victor.agent.planning.readable_schema import (
            ReadableTaskPlan,
            TaskComplexity as PlanningTaskComplexity,
        )

        coordinator = self._create_coordinator()
        plan = ReadableTaskPlan(
            name="Rust Audit",
            complexity=PlanningTaskComplexity.COMPLEX,
            desc="Audit Rust performance",
            steps=[
                [1, "research", "Map workspaces", "read"],
                [2, "analyze", "Inspect Arc usage", "grep"],
            ],
            duration="1h",
        )
        result = PlanResult(
            plan_id="plan_1",
            success=False,
            total_steps=2,
            steps_completed=1,
            steps_failed=1,
            final_output="workspace evidence",
            step_results={
                "1": StepResult(success=True, output="Cargo.toml declares workspace members"),
                "2": StepResult(
                    success=False,
                    output="",
                    error="Insufficient progress",
                ),
            },
        )

        prompt = coordinator._build_summary_prompt(plan, result)

        assert "Overall success: False" in prompt
        assert "Evidence: Cargo.toml declares workspace members" in prompt
        assert "Error: Insufficient progress" in prompt
        assert "do not invent repository findings" in prompt

    def test_build_plan_generation_context_includes_repository_profile(self, tmp_path):
        """Plan generation should receive language-aware repository inventory guidance."""
        coordinator = self._create_coordinator()
        (tmp_path / "go.mod").write_text("module example.com/app\n")
        (tmp_path / "main.go").write_text("package main\n")

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "victor.config.settings.get_project_paths",
                lambda: type("Paths", (), {"project_root": str(tmp_path)})(),
            )
            context = coordinator._build_plan_generation_context()

        assert "Repository profile" in context
        assert "primary=go" in context
        assert "read go.mod first" in context


class TestPlanningCoordinatorIntegration:
    """Integration tests for PlanningCoordinator with chat entry points."""

    def test_legacy_coordinator_module_reexports_service_runtime(self):
        """Legacy planning coordinator import path should re-export the service runtime."""
        from victor.agent.services.planning_runtime import (
            PlanningCoordinator as legacy_planning_coordinator,
        )
        from victor.agent.services.planning_runtime import (
            PlanningCoordinator as service_planning_coordinator,
        )

        assert legacy_planning_coordinator is service_planning_coordinator

    def test_planning_entrypoints_exist(self):
        """Planning should be canonical on ChatService."""
        from victor.agent.services.chat_service import ChatService
        from victor.agent.services.protocols import ChatServiceProtocol

        assert hasattr(ChatServiceProtocol, "chat_with_planning")
        assert hasattr(ChatService, "chat_with_planning")

    def test_config_settings_exist(self):
        """Test that planning settings exist in config."""
        from victor.config.settings import Settings

        settings = Settings()

        # Check that planning settings are available in nested group
        assert hasattr(settings, "agent")
        assert hasattr(settings.agent, "enable_planning")
        assert hasattr(settings.agent, "planning_min_complexity")
        assert hasattr(settings.agent, "planning_show_plan")

        # Check defaults
        assert settings.agent.enable_planning is False
        assert settings.agent.planning_min_complexity == "moderate"
        assert settings.agent.planning_show_plan is True

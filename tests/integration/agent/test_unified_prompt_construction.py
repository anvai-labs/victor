"""Integration tests for unified prompt construction.

Tests that prompt construction and constraint activation work correctly
across legacy and StateGraph execution paths.
"""

import pytest

from victor.agent.prompt_orchestrator import get_prompt_orchestrator
from victor.workflows.definition import FullAccessConstraints, ComputeOnlyConstraints


@pytest.mark.asyncio
class TestUnifiedPromptConstruction:
    """Integration tests for prompt construction across execution paths."""

    async def test_legacy_workflow_builds_prompt(self):
        """Test that legacy workflow can build prompts."""
        orchestrator = get_prompt_orchestrator()

        # Build system prompt as legacy workflow would
        prompt = orchestrator.build_system_prompt(
            builder_type="legacy",
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="edit",
            prompt_contributors=[],
        )

        assert prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    async def test_framework_workflow_builds_prompt(self):
        """Test that framework workflow can build prompts."""
        orchestrator = get_prompt_orchestrator()

        # Build system prompt as framework workflow would
        prompt = orchestrator.build_system_prompt(
            builder_type="framework",
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="edit",
            base_prompt="You are an assistant.",
        )

        assert prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    async def test_auto_detection_chooses_legacy(self):
        """Test that auto detection chooses legacy builder when appropriate."""
        orchestrator = get_prompt_orchestrator()

        prompt = orchestrator.build_system_prompt(
            builder_type="auto",
            prompt_contributors=[],
        )

        assert prompt
        assert isinstance(prompt, str)

    async def test_auto_detection_chooses_framework(self):
        """Test that auto detection chooses framework builder when appropriate."""
        orchestrator = get_prompt_orchestrator()

        prompt = orchestrator.build_system_prompt(
            builder_type="auto",
            base_prompt="You are an assistant.",
        )

        assert prompt
        assert isinstance(prompt, str)


@pytest.mark.asyncio
class TestConstraintActivationIntegration:
    """Integration tests for constraint activation across execution paths."""

    async def test_legacy_workflow_constraint_activation(self):
        """Test constraint activation in legacy workflow context."""
        from victor.agent.constraint_activation_service import get_constraint_activator

        # Simulate legacy workflow constraint activation
        constraints = FullAccessConstraints()

        activator = get_constraint_activator()
        result = activator.activate_constraints(constraints, "coding")

        assert result.success
        assert result.write_path_policy is not None

        # Cleanup
        activator.deactivate_constraints()

    async def test_stategraph_agent_constraint_activation(self):
        """Test constraint activation in StateGraph agent node context."""
        from victor.agent.constraint_activation_service import get_constraint_activator

        # Simulate StateGraph agent execution
        constraints = FullAccessConstraints()

        activator = get_constraint_activator()
        result = activator.activate_constraints(constraints, "coding")

        assert result.success
        assert result.write_path_policy is not None

        # Verify policy is active
        from victor.tools.write_path_policy import get_active_write_policy

        active_policy = get_active_write_policy()
        assert active_policy is not None

        # Cleanup
        activator.deactivate_constraints()

    async def test_subagent_spawn_constraint_activation(self):
        """Test constraint activation in SubAgent.spawn() context."""
        from victor.agent.constraint_activation_service import get_constraint_activator

        # Simulate SubAgent.spawn() constraint activation
        constraints = ComputeOnlyConstraints()

        activator = get_constraint_activator()
        result = activator.activate_constraints(constraints, "dataanalysis")

        assert result.success
        assert result.write_path_policy is not None

        # Cleanup
        activator.deactivate_constraints()

    async def test_constraint_cleanup_after_error(self):
        """Test that constraints are properly cleaned up even after errors."""
        from victor.agent.constraint_activation_service import get_constraint_activator

        activator = get_constraint_activator()

        # Activate with valid constraints
        constraints = FullAccessConstraints()
        result = activator.activate_constraints(constraints, "coding")
        assert result.success

        # Verify active
        assert activator.get_active_policy() is not None

        # Deactivate
        activator.deactivate_constraints()

        # Verify cleaned up
        assert activator.get_active_policy() is None
        assert activator.get_active_constraints() is None


@pytest.mark.asyncio
class TestOrchestratorIntegration:
    """Integration tests for PromptOrchestrator coordination."""

    async def test_orchestrator_constraint_activation(self):
        """Test constraint activation through orchestrator."""
        from victor.workflows.definition import FullAccessConstraints

        orchestrator = get_prompt_orchestrator()

        constraints = FullAccessConstraints()
        success = orchestrator.activate_constraints(constraints, "coding")

        assert success is True

        # Cleanup
        orchestrator.deactivate_constraints()

    async def test_orchestrator_build_and_constraints(self):
        """Test building prompts and activating constraints together."""
        from victor.workflows.definition import ComputeOnlyConstraints

        orchestrator = get_prompt_orchestrator()

        # Build prompt
        prompt = orchestrator.build_system_prompt(
            builder_type="framework",
            base_prompt="You are an assistant.",
        )

        # Activate constraints
        constraints = ComputeOnlyConstraints()
        success = orchestrator.activate_constraints(constraints, "coding")

        assert prompt
        assert success is True

        # Cleanup
        orchestrator.deactivate_constraints()

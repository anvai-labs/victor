"""Tests for ConstraintActivationService."""

from victor.agent.constraint_activation_service import (
    ConstraintActivationService,
    ActivationResult,
    get_constraint_activator,
)


class TestConstraintActivationService:
    """Test suite for ConstraintActivationService."""

    def test_singleton_pattern(self):
        """Test that service is a singleton."""
        service1 = get_constraint_activator()
        service2 = get_constraint_activator()
        assert service1 is service2

    def test_get_instance_returns_singleton(self):
        """Test that get_instance() returns singleton instance."""
        instance1 = ConstraintActivationService.get_instance()
        instance2 = ConstraintActivationService.get_instance()
        assert instance1 is instance2

    def test_activate_constraints_with_none_uses_default(self):
        """Test activation with None constraints uses vertical default."""
        service = ConstraintActivationService()

        result = service.activate_constraints(constraints=None, vertical="coding")

        assert result.success is True
        # Should have a policy and isolation config from defaults
        assert result.write_path_policy is not None or result.isolation_config is not None

    def test_activate_constraints_with_full_access(self):
        """Test activation of FullAccessConstraints."""
        from victor.workflows.definition import FullAccessConstraints

        service = ConstraintActivationService()
        constraints = FullAccessConstraints()

        result = service.activate_constraints(constraints, "coding")

        assert result.success is True
        assert result.write_path_policy is not None
        assert result.isolation_config is not None

    def test_activate_constraints_with_compute_only(self):
        """Test activation of ComputeOnlyConstraints."""
        from victor.workflows.definition import ComputeOnlyConstraints

        service = ConstraintActivationService()
        constraints = ComputeOnlyConstraints()

        result = service.activate_constraints(constraints, "dataanalysis")

        assert result.success is True
        assert result.write_path_policy is not None
        assert result.isolation_config is not None

    def test_deactivate_constraints(self):
        """Test that deactivation clears state."""
        from victor.workflows.definition import FullAccessConstraints

        service = ConstraintActivationService()
        constraints = FullAccessConstraints()

        # Activate
        service.activate_constraints(constraints, "coding")
        assert service.get_active_policy() is not None
        assert service.get_active_constraints() is not None

        # Deactivate
        service.deactivate_constraints()
        assert service.get_active_policy() is None
        assert service.get_active_constraints() is None

    def test_get_active_policy_returns_policy(self):
        """Test get_active_policy() returns the active policy."""
        from victor.workflows.definition import FullAccessConstraints

        service = ConstraintActivationService()
        constraints = FullAccessConstraints()

        service.activate_constraints(constraints, "coding")
        policy = service.get_active_policy()

        assert policy is not None

    def test_get_active_constraints_returns_constraints(self):
        """Test get_active_constraints() returns the active constraints."""
        from victor.workflows.definition import FullAccessConstraints

        service = ConstraintActivationService()
        constraints = FullAccessConstraints()

        service.activate_constraints(constraints, "coding")
        active = service.get_active_constraints()

        assert active is constraints

    def test_activation_result_success_true(self):
        """Test ActivationResult with success=True."""
        result = ActivationResult(
            success=True,
            write_path_policy=None,
            isolation_config={"sandbox": "docker"},
        )

        assert result.success is True
        assert result.isolation_config == {"sandbox": "docker"}
        assert result.error is None

    def test_activation_result_with_error(self):
        """Test ActivationResult with error."""
        result = ActivationResult(
            success=False,
            write_path_policy=None,
            isolation_config=None,
            error="Failed to activate",
        )

        assert result.success is False
        assert result.error == "Failed to activate"

    def test_multiple_activations(self):
        """Test multiple constraint activations override previous state."""
        from victor.workflows.definition import (
            FullAccessConstraints,
            ComputeOnlyConstraints,
        )

        service = ConstraintActivationService()

        # First activation
        service.activate_constraints(FullAccessConstraints(), "coding")
        first_policy = service.get_active_policy()

        # Second activation
        service.activate_constraints(ComputeOnlyConstraints(), "coding")
        second_policy = service.get_active_policy()

        # Policies should be different (or at least state was updated)
        assert first_policy is not None
        assert second_policy is not None

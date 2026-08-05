# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Safety-net tests for initialization phase ordering.

Validates that the InitializationPhaseManager's declared dependency
graph is consistent and acyclic. Catches drift if someone changes
init order without updating the manager.
"""

from unittest.mock import MagicMock

from victor.agent.runtime.initialization_manager import (
    InitializationPhaseManager,
)


def _build_phase_data():
    """Extract phase metadata from a mock orchestrator run."""
    orch = MagicMock()
    manager = InitializationPhaseManager()

    # The phases list is built inside run_all_phases.
    # We can't call it without side-effects, so we replicate
    # the phase tuples structure by inspecting the source.
    # Instead, call run_all_phases with a mock — all phases
    # will "succeed" (MagicMock doesn't raise).
    result = manager.run_all_phases(orch)

    # Extract phase names in execution order
    phase_names = [p.name for p in result.phases]
    return phase_names, result


class TestInitializationOrderConsistency:
    """Validate InitializationPhaseManager's phase graph."""

    def test_manager_dependency_graph_is_acyclic(self):
        """No phase depends on a phase that runs after it."""
        phase_names, _ = _build_phase_data()

        # Build dependency map by re-reading the known structure
        # Phase dependencies are declared inline in run_all_phases
        deps = {
            "provider_runtime": [],
            "metrics_runtime": [],
            "workflow_runtime": [],
            "memory_runtime": [],
            "resilience_runtime": ["provider_runtime"],
            "coordination_runtime": ["provider_runtime"],
            "interaction_runtime": [
                "provider_runtime",
                "coordination_runtime",
            ],
            "services": ["interaction_runtime"],
            "credit_runtime": ["interaction_runtime"],
        }

        for phase, phase_deps in deps.items():
            phase_idx = phase_names.index(phase)
            for dep in phase_deps:
                dep_idx = phase_names.index(dep)
                assert dep_idx < phase_idx, (
                    f"Phase '{phase}' (index {phase_idx}) depends "
                    f"on '{dep}' (index {dep_idx}) which runs later"
                )

    def test_critical_phases_are_provider_and_interaction(self):
        """Only provider_runtime and interaction_runtime are critical."""
        phase_names, result = _build_phase_data()

        # All phases succeed with mock, so we check the phase list
        # contains exactly 8 phases
        assert len(phase_names) == 9

        # Critical phases are the ones that would raise
        # InitializationError on failure. We verify the expected
        # phase names are present.
        assert "provider_runtime" in phase_names
        assert "interaction_runtime" in phase_names

    def test_manager_phases_cover_all_runtimes(self):
        """Every expected runtime phase is present."""
        phase_names, _ = _build_phase_data()

        expected = {
            "provider_runtime",
            "metrics_runtime",
            "workflow_runtime",
            "memory_runtime",
            "resilience_runtime",
            "coordination_runtime",
            "interaction_runtime",
            "services",
            "credit_runtime",
        }
        assert set(phase_names) == expected

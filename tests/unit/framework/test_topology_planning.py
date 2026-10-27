"""TopologyPlanningMixin: extracted from AgenticLoop (god-class decomposition)."""

from __future__ import annotations

import pytest

from victor.framework.agentic_loop import AgenticLoop
from victor.framework.topology_planning import TopologyPlanningMixin


class TestTopologyPlanningMixin:
    def test_mixin_provides_three_methods(self):
        """The mixin carries the three topology integration methods."""
        for name in (
            "_initialize_topology_plan",
            "_prepare_topology_runtime",
            "_get_topology_provider_hints",
        ):
            assert hasattr(TopologyPlanningMixin, name), name

    def test_agentic_loop_inherits_mixin(self):
        """AgenticLoop's MRO includes TopologyPlanningMixin."""
        assert TopologyPlanningMixin in AgenticLoop.__mro__

    def test_module_logger_defined(self):
        """The mixin module has its own logger (extracted from agentic_loop)."""
        import victor.framework.topology_planning as module

        assert hasattr(module, "logger")
        assert module.logger.name == "victor.framework.topology_planning"

    def test_telemetry_functions_importable(self):
        """The mixin module carries the telemetry functions (tests patch here)."""
        import victor.framework.topology_planning as module

        assert hasattr(module, "emit_topology_telemetry_event")
        assert hasattr(module, "build_topology_telemetry_event")

"""Structural pins for the orchestrator facade decomposition.

Whole-program guards for the mixin split (Rec 1 Phase 1): MRO membership,
module residency of moved methods, no silent cross-base shadowing, the
install_properties override status quo, the cached tool-strategy locator,
and the historical module import surface.
"""

import inspect
from unittest.mock import MagicMock

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.orchestrator_context import OrchestratorContextMixin
from victor.agent.orchestrator_init_runtime import OrchestratorInitRuntimeMixin
from victor.agent.orchestrator_metrics import OrchestratorMetricsMixin
from victor.agent.orchestrator_prompting import OrchestratorPromptingMixin
from victor.agent.orchestrator_protocol_conformance import (
    OrchestratorProtocolConformanceMixin,
)
from victor.agent.orchestrator_runtime_intelligence import (
    OrchestratorRuntimeIntelligenceMixin,
)
from victor.agent.orchestrator_vertical import OrchestratorVerticalMixin

NEW_MIXINS = (
    OrchestratorInitRuntimeMixin,
    OrchestratorProtocolConformanceMixin,
    OrchestratorRuntimeIntelligenceMixin,
    OrchestratorContextMixin,
    OrchestratorMetricsMixin,
    OrchestratorVerticalMixin,
    OrchestratorPromptingMixin,
)

# One representative method per mixin (full per-method pins live in each
# module's mirror test file).
RESIDENT_METHODS = {
    "_initialize_provider_runtime": OrchestratorInitRuntimeMixin,
    "get_stage": OrchestratorProtocolConformanceMixin,
    "_should_continue_runtime_intelligence": OrchestratorRuntimeIntelligenceMixin,
    "_compute_cache_flags": OrchestratorContextMixin,
    "_finalize_stream_metrics": OrchestratorMetricsMixin,
    "set_workspace": OrchestratorVerticalMixin,
    "build_system_prompt": OrchestratorPromptingMixin,
}


class TestMroMembership:
    def test_all_mixins_in_mro(self):
        mro_names = {b.__name__ for b in AgentOrchestrator.__mro__}
        for mixin in NEW_MIXINS:
            assert mixin.__name__ in mro_names, mixin

    def test_mixins_define_no_init(self):
        # AgentOrchestrator never calls super().__init__(); mixins must stay
        # pure method/property carriers so MRO order cannot change construction.
        for mixin in NEW_MIXINS:
            assert "__init__" not in vars(mixin), mixin


class TestModuleResidency:
    def test_representative_methods_resolve_to_their_mixins(self):
        for name, mixin in RESIDENT_METHODS.items():
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith(f"{mixin.__module__.rsplit('.', 1)[-1]}.py"), (name, path)

    def test_floor_methods_still_in_class_body(self):
        # The SVC-2 delegation floors parse the class body only; the must-stay
        # methods must remain defined there (guard test enforces call counts;
        # this pins definition presence from the other direction).
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "chat",
            "chat_with_planning",
            "stream_chat",
            "save_checkpoint",
            "restore_checkpoint",
            "maybe_auto_checkpoint",
            "get_context_metrics",
            "get_available_tools",
            "set_enabled_tools",
            "is_tool_enabled",
            "_initialize_services",
            "add_message",
            "from_settings",
        ):
            assert f"def {name}(" in class_src, name


class TestNoCrossBaseShadowing:
    def test_no_method_defined_in_two_bases(self):
        # install_properties setattr shadows silently; two bases defining the
        # same name would be equally invisible. Pin: pairwise-empty overlap.
        bases = [b for b in AgentOrchestrator.__mro__ if b is not object]
        seen = {}
        for base in bases:
            for name in vars(base):
                if name.startswith("__"):
                    continue
                if name in seen:
                    raise AssertionError(
                        f"{name!r} defined in both {seen[name].__name__} " f"and {base.__name__}"
                    )
                seen[name] = base

    def test_install_properties_override_status_quo(self):
        # Today the installed descriptors shadow the class-body defs of the
        # three tool methods. If this changes, the facade semantics changed.
        static = inspect.getattr_static(AgentOrchestrator, "get_enabled_tools")
        assert isinstance(static, property)


class TestToolStrategyConstruction:
    def test_strategy_methods_construct_real_runtime_per_call(self):
        # MagicMock(spec=AgentOrchestrator) hosts auto-create attribute caches,
        # so the tool-strategy delegations must construct ToolStrategyRuntime
        # per call (a getattr-cached locator would bind to a mock runtime and
        # silently skip the real strategy logic). Pin the construction shape.
        from unittest.mock import patch

        orch = MagicMock(spec=AgentOrchestrator)
        with patch("victor.agent.services.tool_strategy_runtime.ToolStrategyRuntime") as ctor:
            ctor.return_value.sort_tools_for_kv_stability.return_value = ["t"]
            result = AgentOrchestrator._sort_tools_for_kv_stability(orch, ["x"])
        assert result == ["t"]
        ctor.assert_called_once_with(orch)


class TestImportSurface:
    def test_historical_module_attributes(self):
        import asyncio

        import victor.agent.orchestrator as mod

        for name in mod.__all__:
            assert hasattr(mod, name), name
        # Patch targets specifically.
        assert mod.asyncio is asyncio
        assert hasattr(mod.ProviderRegistry, "create")

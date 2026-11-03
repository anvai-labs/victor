"""Mirror tests for victor.agent.orchestrator_vertical.

Pins the vertical protocol/storage methods moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): context sync,
workspace switching (incl. the idle-turn guard negative), LSP chain
registration, storage protocol, and coordination suggestions.
"""

import inspect
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.vertical_context import VerticalContext


def make_orch():
    return object.__new__(AgentOrchestrator)


def vertical_context(name="vtest"):
    ctx = MagicMock(spec=VerticalContext)
    ctx.vertical_name = name
    ctx.has_tool_selection_strategy = False
    return ctx


class TestVerticalContextAndLsp:
    def test_get_vertical_context(self):
        orch = make_orch()
        ctx = vertical_context()
        orch._vertical_context = ctx
        assert AgentOrchestrator.get_vertical_context(orch) is ctx

    def test_lsp_property_and_set_without_chain(self):
        orch = make_orch()
        assert AgentOrchestrator.lsp.fget(orch) is None
        cap = MagicMock()
        AgentOrchestrator.set_lsp(orch, cap)
        assert AgentOrchestrator.lsp.fget(orch) is cap

    def test_set_lsp_registers_on_existing_chain(self):
        orch = make_orch()
        orch._middleware_chain = MagicMock()
        chain_ref = orch._middleware_chain
        cap = MagicMock()
        with patch("victor.framework.lsp_middleware.register_lsp_on_chain") as register:
            AgentOrchestrator.set_lsp(orch, cap)
        register.assert_called_once_with(orch, chain_ref)

    def test_set_middleware_chain_registers_lsp_only_when_set(self):
        orch = make_orch()
        chain = MagicMock()
        with patch("victor.framework.lsp_middleware.register_lsp_on_chain") as register:
            AgentOrchestrator.set_middleware_chain(orch, chain)
            register.assert_not_called()  # negative: no LSP yet
            orch._lsp = MagicMock()
            AgentOrchestrator.set_middleware_chain(orch, chain)
            register.assert_called_once_with(orch, chain)
        assert AgentOrchestrator.get_middleware_chain(orch) is chain


class TestCoordinationSuggestions:
    def test_deprecated_alias_warns_and_routes(self):
        orch = make_orch()
        orch._coordination_advisor_runtime = MagicMock()
        orch.mode_controller = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentOrchestrator.get_team_suggestions(orch, "feature", "low")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_prefers_existing_runtime_and_resolves_mode(self):
        orch = make_orch()
        runtime = MagicMock()
        orch._coordination_advisor_runtime = runtime
        orch.mode_controller = SimpleNamespace(current_mode=SimpleNamespace(value="explore"))
        AgentOrchestrator.get_coordination_suggestion(orch, "bugfix", "high")
        _, kwargs = runtime.suggest_for_task.call_args
        assert kwargs["task_type"] == "bugfix"
        assert kwargs["mode"] == "explore"
        assert kwargs["runtime_subject"] is orch

    def test_lazy_factory_branch(self):
        orch = make_orch()
        orch._coordination_advisor_runtime = None
        orch.mode_controller = None
        runtime = MagicMock()
        orch._factory = SimpleNamespace(create_coordination_advisor_runtime=lambda: runtime)
        result = AgentOrchestrator.get_coordination_suggestion(
            orch, "refactor", "medium", mode="build"
        )
        assert result is runtime.suggest_for_task.return_value
        assert orch._coordination_advisor_runtime is runtime


class TestVerticalContextSync:
    def test_syncs_advisor_selector_and_chain(self):
        orch = make_orch()
        ctx = vertical_context()
        advisor = MagicMock()
        selector = MagicMock()
        chain = MagicMock()
        orch._coordination_advisor = advisor
        orch.tool_selector = selector
        orch._middleware_chain = chain
        AgentOrchestrator.set_vertical_context(orch, ctx)
        advisor.set_vertical_context.assert_called_once_with(ctx)
        selector.set_vertical_context.assert_called_once_with(ctx)
        chain.set_vertical_context.assert_called_once_with(ctx)
        assert orch._vertical_context is ctx

    def test_no_advisor_is_safe(self):
        # Negative: missing optional components must not crash the sync.
        orch = make_orch()
        ctx = vertical_context()
        orch._vertical_context = None
        AgentOrchestrator.set_vertical_context(orch, ctx)
        assert orch._vertical_context is ctx

    def test_tiered_config_applied_to_context_and_controller(self):
        orch = make_orch()
        ctx = vertical_context()
        orch._vertical_context = ctx
        controller = MagicMock()
        orch._tool_access_controller = controller
        AgentOrchestrator.set_tiered_tool_config(orch, {"tiers": []})
        ctx.apply_tiered_config.assert_called_once_with({"tiers": []})
        controller.set_tiered_config.assert_called_once_with({"tiers": []})


class TestSetWorkspace:
    def _idle_orch(self):
        orch = make_orch()
        orch._session_accessor = MagicMock()
        orch._session_accessor.stream_turn_lock = MagicMock()
        orch._session_accessor.stream_turn_lock.locked.return_value = False
        orch.provider = SimpleNamespace(name="prov", base_url="http://x")
        orch.provider_name = "prov"
        orch.model = "m1"
        orch._prompt_pipeline = MagicMock()
        orch.build_system_prompt = MagicMock(return_value="SP")
        orch.reset_conversation = MagicMock()
        prompt_runtime = MagicMock()
        prompt_runtime.is_kv_optimization_enabled.return_value = True
        prompt_runtime.is_cache_optimization_enabled.return_value = True
        orch._get_prompt_builder_runtime = lambda: prompt_runtime
        return orch

    def _patches(self, orch, rl_ok=True):
        return [
            patch("victor.config.settings.set_project_root"),
            patch("victor.context.project_context.ProjectContext"),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                side_effect=None if rl_ok else RuntimeError("no rl"),
            ),
        ]

    def test_idle_switch_full_flow(self):
        from pathlib import Path
        from contextlib import ExitStack

        orch = self._idle_orch()
        with ExitStack() as stack:
            set_root = stack.enter_context(self._patches(orch)[0])
            stack.enter_context(self._patches(orch)[1])
            rl = stack.enter_context(self._patches(orch)[2])
            AgentOrchestrator.set_workspace(orch, Path("/tmp/ws"))
        set_root.assert_called_once_with(Path("/tmp/ws"))
        rl().set_repo_context.assert_called_once_with("ws")
        rl().set_session_provider.assert_called_once_with("prov", "m1", base_url="http://x")
        orch._prompt_pipeline.unfreeze.assert_called_once()
        assert orch._system_prompt == "SP"
        # KV + cache enabled -> re-freeze prompt and re-lock session tools
        assert orch._system_prompt_frozen is True
        assert orch._session_tools is None
        orch.reset_conversation.assert_called_once()

    def test_busy_turn_raises_and_changes_nothing(self):
        # Negative: a held turn lock must abort BEFORE any state changes.
        from pathlib import Path
        from contextlib import ExitStack

        orch = self._idle_orch()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "victor.agent.orchestrator_vertical.require_idle_turn",
                    side_effect=RuntimeError("turn in progress"),
                )
            )
            set_root = stack.enter_context(self._patches(orch)[0])
            try:
                AgentOrchestrator.set_workspace(orch, Path("/tmp/ws"))
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError")
        set_root.assert_not_called()
        orch.build_system_prompt.assert_not_called()
        orch.reset_conversation.assert_not_called()

    def test_rl_failure_is_non_fatal(self):
        from pathlib import Path
        from contextlib import ExitStack

        orch = self._idle_orch()
        with ExitStack() as stack:
            stack.enter_context(self._patches(orch)[0])
            stack.enter_context(self._patches(orch)[1])
            stack.enter_context(self._patches(orch, rl_ok=False)[2])
            AgentOrchestrator.set_workspace(orch, Path("/tmp/ws2"))
        orch.reset_conversation.assert_called_once()


class TestStorageProtocol:
    def test_adapter_delegation(self):
        orch = make_orch()
        adapter = MagicMock()
        orch._vertical_integration_adapter = adapter
        AgentOrchestrator.apply_vertical_middleware(orch, ["mw"])
        adapter.apply_middleware.assert_called_once_with(["mw"])
        AgentOrchestrator.apply_vertical_safety_patterns(orch, ["sp"])
        adapter.apply_safety_patterns.assert_called_once_with(["sp"])

    def test_apply_vertical_tools(self):
        orch = make_orch()
        ctx = vertical_context()
        orch._vertical_context = ctx
        AgentOrchestrator._apply_vertical_tools(orch, {"read", "grep"})
        ctx.apply_enabled_tools.assert_called_once_with({"read", "grep"})

    def test_set_get_pairs_and_internal_setters(self):
        orch = make_orch()
        AgentOrchestrator.set_middleware(orch, ["m"])
        assert AgentOrchestrator.get_middleware(orch) == ["m"]
        AgentOrchestrator._set_vertical_middleware_storage(orch, ["m2"])
        assert AgentOrchestrator.get_middleware(orch) == ["m2"]
        AgentOrchestrator.set_safety_patterns(orch, ["s"])
        assert AgentOrchestrator.get_safety_patterns(orch) == ["s"]
        AgentOrchestrator._set_safety_patterns_storage(orch, ["s2"])
        assert AgentOrchestrator.get_safety_patterns(orch) == ["s2"]
        AgentOrchestrator.set_team_specs(orch, {"t": 1})
        assert AgentOrchestrator.get_team_specs(orch) == {"t": 1}
        # Defaults on a fresh instance
        fresh = make_orch()
        assert AgentOrchestrator.get_middleware(fresh) == []
        assert AgentOrchestrator.get_safety_patterns(fresh) == []
        assert AgentOrchestrator.get_team_specs(fresh) == {}


class TestSurfacePins:
    def test_methods_live_on_mixin(self):
        # Negative pin: moved methods must not reappear in the class body.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in ("set_workspace", "set_vertical_context", "set_lsp"):
            assert f"def {name}(" not in class_src, name
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith("orchestrator_vertical.py"), (name, path)

    def test_checkpoints_stay_in_class_body(self):
        # Session-service delegation floor (3/3) must remain in the class.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "save_checkpoint",
            "restore_checkpoint",
            "maybe_auto_checkpoint",
        ):
            assert f"def {name}(" in class_src, name

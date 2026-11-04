"""Mirror tests for victor.agent.orchestrator_prompting.

Pins the skill/prompt runtime accessors moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): lazy runtime
caching, delegation pins for every method, and the fallback branches.
"""

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator


def make_orch():
    return object.__new__(AgentOrchestrator)


def patched_state(orch, **values):
    stack = ExitStack()
    for name, value in values.items():
        owner = next((b for b in type(orch).__mro__ if name in vars(b)), None)
        if owner is not None:
            stack.enter_context(patch.object(owner, name, property(lambda self, v=value: v)))
        else:
            setattr(orch, name, value)
    return stack


class TestLazyRuntimes:
    def test_skill_runtime_lazy_cached(self):
        orch = make_orch()
        adapter = MagicMock()
        runtime = MagicMock()
        with patched_state(orch, protocol_adapter=adapter):
            with patch(
                "victor.agent.services.skill_runtime.SkillRuntime",
                return_value=runtime,
            ) as ctor:
                first = AgentOrchestrator._get_skill_runtime(orch)
                second = AgentOrchestrator._get_skill_runtime(orch)
            assert first is runtime and second is runtime
            ctor.assert_called_once_with(adapter)

    def test_prompt_builder_runtime_lazy_cached(self):
        orch = make_orch()
        adapter = MagicMock()
        runtime = MagicMock()
        with patched_state(orch, protocol_adapter=adapter):
            with patch(
                "victor.agent.services.prompt_builder_runtime.PromptBuilderRuntime",
                return_value=runtime,
            ) as ctor:
                first = AgentOrchestrator._get_prompt_builder_runtime(orch)
                second = AgentOrchestrator._get_prompt_builder_runtime(orch)
            assert first is runtime and second is runtime
            ctor.assert_called_once_with(adapter)


class TestSkillDelegation:
    def _orch(self):
        orch = make_orch()
        runtime = MagicMock()
        orch._skill_runtime = runtime
        return orch, runtime

    def test_all_skill_methods_delegate(self):
        orch, runtime = self._orch()
        AgentOrchestrator.get_last_skill_match_info(orch)
        runtime.get_last_skill_match_info.assert_called_once_with()
        AgentOrchestrator.clear_active_skills(orch)
        runtime.clear_active_skills.assert_called_once_with()
        AgentOrchestrator.get_skill_user_prefix(orch)
        runtime.get_skill_user_prefix.assert_called_once_with()
        skill, skills = object(), [object()]
        AgentOrchestrator.inject_skill(orch, skill)
        runtime.inject_skill.assert_called_once_with(skill)
        AgentOrchestrator.inject_skills(orch, skills)
        runtime.inject_skills.assert_called_once_with(skills)


class TestPromptDelegation:
    def _orch(self):
        orch = make_orch()
        runtime = MagicMock()
        orch._prompt_builder_runtime = runtime
        return orch, runtime

    def test_update_and_refresh_kwargs(self):
        orch, runtime = self._orch()
        AgentOrchestrator.update_system_prompt_for_query(orch, {"task": "edit"})
        runtime.update_system_prompt_for_query.assert_called_once_with(
            query_classification={"task": "edit"}
        )
        AgentOrchestrator.refresh_system_prompt(orch, None, preserve_existing_classification=False)
        runtime.refresh_system_prompt.assert_called_once_with(
            query_classification=None, preserve_existing_classification=False
        )

    def test_sync_build_and_event(self):
        orch, runtime = self._orch()
        AgentOrchestrator._sync_prompt_builder_runtime_state(orch)
        runtime.sync_prompt_builder_runtime_state.assert_called_once_with()
        runtime.build_system_prompt.return_value = "SP"
        assert AgentOrchestrator.build_system_prompt(orch) == "SP"
        # Event emission requires the pipeline; absent -> no-op.
        orch._prompt_pipeline = None
        AgentOrchestrator._emit_prompt_used_event(orch, "SP")
        pipeline = MagicMock()
        orch._prompt_pipeline = pipeline
        AgentOrchestrator._emit_prompt_used_event(orch, "SP2")
        pipeline._emit_prompt_used_event.assert_called_once_with("SP2")


class TestShellVariantAndThinking:
    def test_shell_variant_prefers_pipeline(self):
        orch = make_orch()
        orch._prompt_pipeline = MagicMock()
        orch._prompt_pipeline.resolve_shell_variant = MagicMock(return_value="pwsh")
        assert AgentOrchestrator._resolve_shell_variant(orch, "shell") == "pwsh"

    def test_shell_variant_falls_back_to_resolver(self):
        # Branch: no pipeline -> direct resolver; resolver failure -> identity.
        orch = make_orch()
        orch._prompt_pipeline = None
        orch.tools = MagicMock()
        orch.mode_controller = MagicMock()
        with patch(
            "victor.agent.shell_resolver.resolve_shell_variant",
            return_value="bash",
        ) as resolve:
            assert AgentOrchestrator._resolve_shell_variant(orch, "shell") == "bash"
        assert resolve.call_args[0][0] == "shell"
        with patch(
            "victor.agent.shell_resolver.resolve_shell_variant",
            side_effect=RuntimeError("no resolver"),
        ):
            assert AgentOrchestrator._resolve_shell_variant(orch, "shell") == "shell"

    def test_thinking_disabled_prompt(self):
        orch = make_orch()
        orch.tool_calling_caps = SimpleNamespace(thinking_disable_prefix="/no_think")
        assert AgentOrchestrator._get_thinking_disabled_prompt(orch, "P") == "/no_think\nP"
        orch.tool_calling_caps = SimpleNamespace(thinking_disable_prefix=None)
        # Negative: no prefix -> prompt unchanged.
        assert AgentOrchestrator._get_thinking_disabled_prompt(orch, "P") == "P"


class TestSurfacePins:
    def test_methods_live_on_mixin(self):
        # Negative pin: moved methods must not reappear in the class body.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "build_system_prompt",
            "_get_skill_runtime",
            "_resolve_shell_variant",
        ):
            assert f"def {name}(" not in class_src, name
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith("orchestrator_prompting.py"), (name, path)

"""Mirror tests for victor.agent.orchestrator_protocol_conformance.

Pins the protocol-conformance accessors moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): stage/counters,
provider/model properties, hot-swap delegation, system-prompt trio, and
the MessagesProtocol accessors — plus the OrchestratorProtocol surface.
"""

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.conversation.state_machine import ConversationStage


def make_orch():
    return object.__new__(AgentOrchestrator)


def patched_state(orch, **values):
    """Class descriptors get patched; plain instance attrs get set."""
    stack = ExitStack()
    for name, value in values.items():
        if name in type(orch).__dict__:
            stack.enter_context(patch.object(type(orch), name, property(lambda self, v=value: v)))
        else:
            setattr(orch, name, value)
    return stack


class TestStageAndCounters:
    def test_get_stage_with_state(self):
        orch = make_orch()
        orch.conversation_state = MagicMock()
        orch.conversation_state.get_stage = MagicMock(return_value=ConversationStage.EXECUTION)
        assert AgentOrchestrator.get_stage(orch) == ConversationStage.EXECUTION

    def test_get_stage_defaults_to_initial(self):
        # Negative: no conversation state -> INITIAL, not a crash.
        orch = make_orch()
        orch.conversation_state = None
        assert AgentOrchestrator.get_stage(orch) == ConversationStage.INITIAL

    def test_get_tool_calls_count_reads_descriptor(self):
        orch = make_orch()
        with patched_state(orch, tool_calls_used=7):
            assert AgentOrchestrator.get_tool_calls_count(orch) == 7

    def test_get_tool_calls_count_defaults(self):
        orch = make_orch()
        with patched_state(orch, tool_calls_used=0):
            assert AgentOrchestrator.get_tool_calls_count(orch) == 0

    def test_get_tool_budget_default_50(self):
        orch = make_orch()
        assert AgentOrchestrator.get_tool_budget(orch) == 50

    def test_observed_and_modified_files(self):
        orch = make_orch()
        orch.conversation_state = None
        with patched_state(orch, observed_files=["/a", "/a"]):
            assert AgentOrchestrator.get_observed_files(orch) == {"/a"}
        assert AgentOrchestrator.get_modified_files(orch) == set()

    def test_modified_files_from_state(self):
        orch = make_orch()
        orch.conversation_state = SimpleNamespace(state=SimpleNamespace(modified_files=["/b"]))
        assert AgentOrchestrator.get_modified_files(orch) == {"/b"}

    def test_iteration_counters_with_and_without_tracker(self):
        orch = make_orch()
        orch.unified_tracker = SimpleNamespace(iteration_count=4, max_iterations=40)
        assert AgentOrchestrator.get_iteration_count(orch) == 4
        assert AgentOrchestrator.get_max_iterations(orch) == 40
        orch.unified_tracker = None
        assert AgentOrchestrator.get_iteration_count(orch) == 0
        assert AgentOrchestrator.get_max_iterations(orch) == 25


class TestProviderProtocol:
    def test_current_provider_and_model(self):
        orch = make_orch()
        orch.provider_name = "anthropic"
        orch.model = "claude-opus"
        assert AgentOrchestrator.current_provider.fget(orch) == "anthropic"
        assert AgentOrchestrator.current_model.fget(orch) == "claude-opus"

    async def test_switch_provider_delegates(self):
        orch = make_orch()
        runtime = MagicMock()
        runtime.switch_provider = AsyncMock(return_value=True)
        orch._get_provider_management_runtime = lambda: runtime
        cb = MagicMock()
        ok = await AgentOrchestrator.switch_provider(orch, "openai", "gpt-x", on_switch=cb)
        assert ok is True
        runtime.switch_provider.assert_called_once_with("openai", "gpt-x", cb)

    async def test_switch_model_delegates(self):
        orch = make_orch()
        runtime = MagicMock()
        runtime.switch_model = AsyncMock(return_value=True)
        orch._get_provider_management_runtime = lambda: runtime
        assert await AgentOrchestrator.switch_model(orch, "m2") is True
        runtime.switch_model.assert_called_once_with("m2")


class TestSystemPromptProtocol:
    def test_trio_delegates_to_prompt_builder_runtime(self):
        orch = make_orch()
        runtime = MagicMock()
        runtime.get_system_prompt = MagicMock(return_value="SP")
        orch._get_prompt_builder_runtime = lambda: runtime
        assert AgentOrchestrator.get_system_prompt(orch) == "SP"
        AgentOrchestrator.set_system_prompt(orch, "NEW")
        runtime.set_system_prompt.assert_called_once_with("NEW")
        AgentOrchestrator.append_to_system_prompt(orch, "+X")
        runtime.append_to_system_prompt.assert_called_once_with("+X")


class TestMessagesProtocol:
    def test_get_messages_and_count(self):
        orch = make_orch()
        msg = MagicMock(role="user", content="hi")
        orch.conversation = SimpleNamespace(messages=[msg])
        assert AgentOrchestrator.get_messages(orch) == [{"role": "user", "content": "hi"}]
        assert AgentOrchestrator.get_message_count(orch) == 1


class TestSurfacePins:
    def test_orchestrator_protocol_surface_still_present(self):
        # The five OrchestratorProtocol members must remain reachable on the
        # class (mixins resolve through the MRO).
        for member in (
            "chat",
            "messages",
            "reset_conversation",
            "stream_chat",
            "tool_calls_used",
        ):
            assert hasattr(AgentOrchestrator, member), member

    def test_methods_live_on_mixin(self):
        # Negative pin: none of the moved methods may be re-defined in the
        # orchestrator class body.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "get_stage",
            "switch_provider",
            "get_messages",
            "append_to_system_prompt",
        ):
            assert f"def {name}(" not in class_src, name
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith("orchestrator_protocol_conformance.py"), (name, path)

    def test_must_stay_methods_still_in_class_body(self):
        # Floor protection from the other direction: the ToolsProtocol block
        # and dunders/from_settings were deliberately NOT moved.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "get_available_tools",
            "set_enabled_tools",
            "is_tool_enabled",
            "from_settings",
        ):
            assert f"def {name}(" in class_src, name

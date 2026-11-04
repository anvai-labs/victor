"""Mirror tests for victor.agent.orchestrator_context.

Pins the context/cache accessors moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): message-count and
context-window branches, cache-flag delegation, KV helpers, context-assembly
delegation, and the embedding-store init.
"""

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator


def make_orch():
    return object.__new__(AgentOrchestrator)


def patched_state(orch, **values):
    stack = ExitStack()
    for name, value in values.items():
        if name in type(orch).__dict__:
            stack.enter_context(patch.object(type(orch), name, property(lambda self, v=value: v)))
        else:
            setattr(orch, name, value)
    return stack


class TestMessagesAndCounts:
    def test_messages_property(self):
        orch = make_orch()
        orch.conversation = SimpleNamespace(messages=["m1", "m2"])
        assert AgentOrchestrator.messages.fget(orch) == ["m1", "m2"]

    def test_message_count_delegates_to_assembly(self):
        orch = make_orch()
        with patch(
            "victor.agent.conversation.context_assembly.conversation_message_count",
            return_value=3,
        ) as fn:
            assert AgentOrchestrator._get_conversation_message_count(orch) == 3
        assert fn.call_args[0][0] is orch


class TestContextWindow:
    def test_delegates_to_context_manager(self):
        orch = make_orch()
        orch._context_manager = MagicMock()
        orch._context_manager.get_model_context_window = MagicMock(return_value=200000)
        assert AgentOrchestrator._get_model_context_window(orch) == 200000

    def test_fallback_to_provider_limits_during_init(self):
        # Branch: no context manager yet (during __init__).
        orch = make_orch()
        orch.provider_name = "anthropic"
        orch.model = "m"
        with patch(
            "victor.config.config_loaders.get_provider_limits",
            return_value=SimpleNamespace(context_window=99000),
        ):
            assert AgentOrchestrator._get_model_context_window(orch) == 99000

    def test_failure_falls_back_to_128000(self):
        # Negative: config-load failure must degrade to the safe default.
        orch = make_orch()
        orch.provider_name = "x"
        orch.model = "y"
        with patch(
            "victor.config.config_loaders.get_provider_limits",
            side_effect=RuntimeError("boom"),
        ):
            assert AgentOrchestrator._get_model_context_window(orch) == 128000

    def test_max_context_chars_and_overflow(self):
        orch = make_orch()
        orch._context_manager = MagicMock()
        orch._context_manager.get_max_context_chars = MagicMock(return_value=500)
        orch._context_manager.check_context_overflow = MagicMock(return_value=True)
        assert AgentOrchestrator._get_max_context_chars(orch) == 500
        assert AgentOrchestrator._check_context_overflow(orch, 200000) is True
        orch._context_manager.check_context_overflow.assert_called_once_with(200000)


class TestCacheFlags:
    def _orch(self):
        orch = make_orch()
        orch._get_prompt_builder_runtime = lambda: orch.runtime
        orch.runtime = MagicMock()
        return orch

    def test_cache_setting_and_flags(self):
        orch = self._orch()
        orch.runtime.check_cache_setting_enabled = MagicMock(return_value=True)
        assert AgentOrchestrator._check_cache_setting_enabled(orch) is True
        AgentOrchestrator._compute_cache_flags(orch)
        orch.runtime.compute_cache_flags.assert_called_once()

    def test_optimization_properties(self):
        orch = self._orch()
        orch.runtime.is_cache_optimization_enabled = MagicMock(return_value=True)
        orch.runtime.is_kv_optimization_enabled = MagicMock(return_value=False)
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is True
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False

    async def test_warm_up_kv_cache(self):
        orch = self._orch()
        orch.runtime.warm_up_kv_cache = AsyncMock()
        await AgentOrchestrator.warm_up_kv_cache(orch)
        orch.runtime.warm_up_kv_cache.assert_called_once()

    def test_kv_prefix_fingerprint(self):
        orch = self._orch()
        orch.runtime.kv_prefix_fingerprint = MagicMock(return_value="ab12")
        assert AgentOrchestrator._kv_prefix_fingerprint(orch) == "ab12"


class TestAssemblyDelegation:
    def test_session_tools(self):
        orch = make_orch()
        with patch(
            "victor.agent.conversation.context_assembly.session_tools",
            return_value=["read"],
        ) as fn:
            assert AgentOrchestrator.get_session_tools(orch) == ["read"]
        assert fn.call_args[0][0] is orch

    def test_assembled_messages(self):
        orch = make_orch()
        with patch(
            "victor.agent.conversation.context_assembly.assembled_messages_for_call",
            return_value=["sys"],
        ) as fn:
            result = AgentOrchestrator.get_assembled_messages(
                orch, "q", selected_tools=["t"], planned_tools=["p"], goals=["g"]
            )
        assert result == ["sys"]
        _, kwargs = fn.call_args
        assert kwargs == {
            "selected_tools": ["t"],
            "planned_tools": ["p"],
            "goals": ["g"],
        }


class TestEmbeddingStoreInit:
    def test_initializes_store_and_cache(self):
        orch = make_orch()
        orch.memory_manager = MagicMock()
        store, cache = MagicMock(), MagicMock()
        with patch(
            "victor.agent.runtime.memory_runtime.initialize_conversation_embedding_store",
            return_value=(store, cache),
        ) as fn:
            AgentOrchestrator._init_conversation_embedding_store(orch)
        assert orch._conversation_embedding_store is store
        assert orch._pending_semantic_cache is cache
        assert fn.call_args[1]["memory_manager"] is orch.memory_manager


class TestSurfacePins:
    def test_methods_live_on_mixin(self):
        # Negative pin: moved methods must not reappear in the class body.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "_get_model_context_window",
            "_compute_cache_flags",
            "get_assembled_messages",
        ):
            assert f"def {name}(" not in class_src, name
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith("orchestrator_context.py"), (name, path)

    def test_get_context_metrics_stays_in_class_body(self):
        # Context-service delegation floor (1/1) must remain in the class.
        class_src = inspect.getsource(AgentOrchestrator)
        assert "def get_context_metrics(" in class_src

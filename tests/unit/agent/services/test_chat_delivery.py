"""Delivery preserves configured component behavior without facade reach-through."""

import gc
import weakref
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.factory.chat_runtime_bindings import bind_chat_runtime_services
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_stream_executor import StreamingChatExecutor
from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.providers.base import StreamChunk


class Chunks:
    def __init__(self):
        self.emitted = []

    def generate_content_chunk(self, content, is_final=False):
        chunk = StreamChunk(
            content=content, is_final=is_final, metadata={"sequence": len(self.emitted)}
        )
        self.emitted.append(chunk)
        return chunk


class Sanitizer:
    def sanitize(self, text):
        return "" if text.startswith("<") else text.strip()

    def strip_markup(self, text):
        return text.replace("<b>", "").replace("</b>", "")

    def is_garbage_content(self, text):
        return text == "garbage"


def owner():
    result = object.__new__(AgentOrchestrator)
    result._session_accessor = SessionStateAccessor(SessionStateManager())
    result._chunk_generator = Chunks()
    result.sanitizer = Sanitizer()
    return result


@pytest.mark.parametrize("through_adapter", [False, True])
def test_binding_retains_components_without_retaining_facade(through_adapter):
    facade = owner()
    ref = weakref.ref(facade)
    supplied = OrchestratorProtocolAdapter(facade) if through_adapter else facade
    view = bind_chat_runtime_services(supplied)
    assert view.delivery.chunks is facade._chunk_generator
    assert view.delivery.sanitizer is facade.sanitizer
    assert not hasattr(view.delivery, "__dict__")
    with pytest.raises(FrozenInstanceError):
        view.delivery.chunks = Chunks()
    del supplied, facade
    gc.collect()
    assert ref() is None


def test_chunks_preserve_identity_metadata_and_session_isolation():
    first, second = owner(), owner()
    view, other = bind_chat_runtime_services(first), bind_chat_runtime_services(second)
    chunk = view.delivery.content_chunk("answer", is_final=True)
    assert chunk is first._chunk_generator.emitted[0]
    assert chunk.metadata == {"sequence": 0}
    assert chunk.is_final
    assert second._chunk_generator.emitted == []
    assert other.delivery.chunks is not view.delivery.chunks


def test_final_marker_uses_configured_factory_or_legacy_fallback():
    for delivery in (ChatDelivery(), ChatDelivery(chunks=Chunks())):
        marker = delivery.final_marker_chunk()
        assert marker.content == "" and marker.is_final
    marker = StreamChunk(content="", is_final=True, metadata={"custom": True})
    chunks = SimpleNamespace(generate_final_marker_chunk=lambda: marker)
    assert ChatDelivery(chunks=chunks).final_marker_chunk() is marker


def test_only_optional_terminal_cleanup_allows_missing_components():
    delivery = ChatDelivery()
    assert delivery.sanitize(" raw ", optional=True) == " raw "
    assert delivery.strip_markup("<b>raw</b>", optional=True) == "<b>raw</b>"
    for operation in (delivery.sanitize, delivery.strip_markup, delivery.is_garbage_content):
        with pytest.raises(TypeError, match="sanitizer"):
            operation("raw")
    with pytest.raises(TypeError, match="chunk generator"):
        delivery.content_chunk("raw")


def test_configured_component_failures_propagate_even_for_optional_cleanup():
    def fail(*args, **kwargs):
        raise ValueError("component failed")

    component = SimpleNamespace(
        sanitize=fail,
        strip_markup=fail,
        is_garbage_content=fail,
        generate_content_chunk=fail,
        generate_final_marker_chunk=fail,
    )
    delivery = ChatDelivery(chunks=component, sanitizer=component)
    for operation in (delivery.sanitize, delivery.strip_markup):
        for optional in (True, False):
            with pytest.raises(ValueError, match="component failed"):
                operation("raw", optional=optional)
    with pytest.raises(ValueError, match="component failed"):
        delivery.content_chunk("raw")
    with pytest.raises(ValueError, match="component failed"):
        delivery.final_marker_chunk()


def test_terminal_plaintext_recovery_uses_bound_sanitizer():
    facade = owner()
    runtime = SimpleNamespace(services=bind_chat_runtime_services(facade))
    executor = StreamingChatExecutor(runtime)
    del facade.sanitizer
    content, source = executor._resolve_terminal_visible_output(
        facade,
        SimpleNamespace(compaction_summary=""),
        full_content="<b>answer</b>",
        user_message="q",
    )
    assert (content, source) == ("answer", "provider_response")


def test_garbage_threshold_and_reset_use_bound_sanitizer():
    facade = owner()
    helper = ChatStreamHelperMixin()
    helper.services = bind_chat_runtime_services(facade)
    del facade.sanitizer
    bad = StreamChunk(content="garbage")
    assert helper._handle_stream_chunk(bad, 0, 2, False) == (bad, 1, False)
    assert helper._handle_stream_chunk(bad, 1, 2, False) == (None, 2, True)
    good = StreamChunk(content="answer")
    assert helper._handle_stream_chunk(good, 1, 2, False) == (good, 0, False)


async def test_request_block_uses_bound_chunk_generator_before_stream_setup():
    from victor.framework.policies import BlockPatternPolicy, MessagePolicyGate, Phase, PolicyEngine

    facade = owner()
    facade._message_policy_gate = MessagePolicyGate(
        PolicyEngine([BlockPatternPolicy([r"rm -rf"], phases={Phase.REQUEST}, reason="dangerous")])
    )
    runtime = SimpleNamespace(
        _orchestrator=facade,
        services=bind_chat_runtime_services(facade),
        _create_stream_context=MagicMock(side_effect=AssertionError("must not start stream")),
    )
    generator = facade._chunk_generator
    del facade._chunk_generator
    chunks = [chunk async for chunk in StreamingChatExecutor(runtime).run_unified("rm -rf /")]
    assert len(chunks) == 1 and chunks[0] is generator.emitted[0]
    assert chunks[0].is_final and "dangerous" in chunks[0].content
    runtime._create_stream_context.assert_not_called()

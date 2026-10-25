"""OrchestratorProtocol reconciliation (TD-033) + stringly-typed gate fixes.

The protocol was aspirational — the concrete AgentOrchestrator failed it at
runtime (model/provider_name/tool_budget absent as class attributes), so
isinstance checks rejected the real class. It now carries exactly the
verified surface. The Path/HITLNode name-string checks became isinstance.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_concrete_orchestrator_satisfies_the_protocol():
    """Every protocol member is present on the real class — the property
    that FAILED before the reconciliation (shrinking the protocol to the
    verified surface), pinned so it cannot drift back to aspiration."""
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.core.protocols import OrchestratorProtocol

    members = sorted(getattr(OrchestratorProtocol, "__protocol_attrs__", ()))
    assert members == [
        "chat",
        "messages",
        "reset_conversation",
        "stream_chat",
        "tool_calls_used",
    ]
    for member in members:
        assert hasattr(AgentOrchestrator, member), member


def test_protocol_presence_check_accepts_standins_rejects_misuse():
    from victor.core.protocols import OrchestratorProtocol

    stand_in = SimpleNamespace(
        tool_calls_used=0,
        messages=[],
        chat=lambda msg: None,
        stream_chat=lambda msg: None,
        reset_conversation=lambda: None,
    )
    assert isinstance(stand_in, OrchestratorProtocol)
    assert not isinstance("a string", OrchestratorProtocol)
    assert not isinstance({"chat": True}, OrchestratorProtocol)


def test_path_schema_conversion_is_isinstance_not_name():
    """A class NAMED 'Path' that is not pathlib.Path must not be converted
    to a file-path schema (the old name-string check matched it)."""
    import inspect

    import victor.tools.decorators as decorators

    source = inspect.getsource(decorators)
    assert 'prop_type.__name__ == "Path"' not in source
    assert "issubclass(prop_type, _Path)" in source


def test_hitl_detection_is_isinstance_not_name():
    import inspect

    import victor.workflows.runtime as runtime

    source = inspect.getsource(runtime)
    assert '__class__.__name__ == "HITLNode"' not in source
    assert "isinstance(node, HITLNode)" in source

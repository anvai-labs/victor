"""OrchestratorProtocol reconciliation (TD-033) + stringly-typed gate fixes.

The protocol was aspirational — the concrete AgentOrchestrator failed it at
runtime (model/provider_name/tool_budget absent as class attributes), so
isinstance checks rejected the real class. It now carries exactly the
verified surface. The Path/HITLNode name-string checks became isinstance.
"""

from __future__ import annotations

from pathlib import Path as RealPath
from pathlib import PurePosixPath
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


def test_path_schema_type_check_is_identity_not_name():
    """The Path branch is identity/issubclass: real Path (and subclasses)
    convert; a class merely NAMED 'Path' does not — the old name-string
    check matched it."""
    from victor.tools.decorators import _schema_type_is_path

    assert _schema_type_is_path(RealPath)

    class MyPath(RealPath):  # a real subclass still converts
        pass

    assert _schema_type_is_path(MyPath)
    assert _schema_type_is_path(PurePosixPath) is False  # pathlib sibling, not Path

    class Path:  # deliberately named Path, NOT pathlib.Path
        pass

    assert _schema_type_is_path(Path) is False
    assert _schema_type_is_path("Path") is False
    assert _schema_type_is_path("string") is False


def test_has_hitl_nodes_detects_real_hitl_nodes():
    """Cover the isinstance branch: a real HITLNode instance is detected,
    and a plain node without HITL markers is not."""
    from victor.workflows.hitl import HITLNode
    from victor.workflows.runtime import WorkflowRuntime

    node = HITLNode(id="h", name="approve")
    workflow = SimpleNamespace(nodes={"h": node})

    class Bare:
        node_type = None  # not the HITL enum; not a HITLNode

    wf_bare = SimpleNamespace(nodes=[Bare()])
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)  # no __init__ side effects
    assert runtime._has_hitl_nodes(workflow) is True
    assert runtime._has_hitl_nodes(wf_bare) is False


def test_hitl_detection_is_isinstance_not_name():
    import inspect

    import victor.workflows.runtime as runtime

    source = inspect.getsource(runtime)
    assert '__class__.__name__ == "HITLNode"' not in source
    assert "isinstance(node, HITLNode)" in source

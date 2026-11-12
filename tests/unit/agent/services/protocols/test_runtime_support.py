"""Contract checks for the runtime-support protocols (mirror suite for
``victor/agent/services/protocols/runtime_support.py``).

Pins the public protocol surface the orchestrator delegates through, and
the TerminationAuthority seam (consolidation PR 5): the recovery protocol's
``check_blocked_threshold`` must accept the optional ``guided_check``
guided-suppression predicate while remaining callable without it
(default-off parity).
"""

import inspect

from victor.agent.services.protocols.runtime_support import (
    ChunkRuntimeProtocol,
    CoordinationAdvisorRuntimeProtocol,
    PromptRuntimeProtocol,
    RLLearningRuntimeProtocol,
    StateRuntimeProtocol,
    StreamingRecoveryRuntimeProtocol,
    TaskRuntimeProtocol,
    ToolPlanningRuntimeProtocol,
)

ALL_PROTOCOLS = (
    CoordinationAdvisorRuntimeProtocol,
    ToolPlanningRuntimeProtocol,
    TaskRuntimeProtocol,
    StateRuntimeProtocol,
    PromptRuntimeProtocol,
    StreamingRecoveryRuntimeProtocol,
    ChunkRuntimeProtocol,
    RLLearningRuntimeProtocol,
)


def test_all_runtime_support_protocols_are_runtime_checkable() -> None:
    import typing

    for protocol in ALL_PROTOCOLS:
        assert issubclass(protocol, typing.Protocol), f"{protocol.__name__} must be a Protocol"
        # @runtime_checkable marks the class for structural isinstance checks.
        assert getattr(protocol, "_is_runtime_protocol", False), protocol.__name__


def test_streaming_recovery_protocol_keeps_blocked_threshold_surface() -> None:
    members = StreamingRecoveryRuntimeProtocol.__dict__

    assert {
        "filter_blocked_tool_calls",
        "check_blocked_threshold",
        "check_force_action",
    } <= members.keys()


def test_check_blocked_threshold_accepts_guided_check_and_stays_default_callable() -> None:
    """Consolidation PR 5: the guided-suppression predicate is an optional
    keyword — the protocol stays callable with the legacy two-arg shape."""
    signature = inspect.signature(StreamingRecoveryRuntimeProtocol.check_blocked_threshold)

    assert "guided_check" in signature.parameters
    assert signature.parameters["guided_check"].default is None

    # Legacy positional call shape still binds (self, ctx, all_blocked[, guided_check]).
    signature.bind(object(), object(), True)
    signature.bind(object(), object(), True, lambda: False)
    signature.bind(object(), object(), True, guided_check=lambda: True)


def test_concrete_implementations_honor_the_guided_check_contract() -> None:
    """The protocol's implementations accept the optional guided-suppression
    predicate and keep it default-None (ungated legacy behavior)."""
    from victor.agent.services.recovery_service import RecoveryService
    from victor.agent.streaming.coordinator import IterationCoordinator
    from victor.agent.streaming.handler import StreamingChatHandler

    for owner in (RecoveryService, IterationCoordinator, StreamingChatHandler):
        parameters = inspect.signature(owner.check_blocked_threshold).parameters
        assert "guided_check" in parameters, owner.__name__
        assert parameters["guided_check"].default is None, owner.__name__

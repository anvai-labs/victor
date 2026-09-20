"""Contract checks for the unified streaming runtime protocol."""

from victor.agent.services.protocols.streaming_runtime import (
    StreamingExecutionRuntimeProtocol,
)


def test_streaming_execution_protocol_excludes_retired_precheck() -> None:
    members = StreamingExecutionRuntimeProtocol.__dict__

    assert "_run_iteration_pre_checks" not in members
    assert {
        "_create_stream_context",
        "_stream_provider_response",
        "_handle_empty_response_recovery",
    } <= members.keys()

# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Governance (REQUEST/RESPONSE message phases) on the streaming path.

Tool governance already shares the non-streaming ToolPipeline; these tests
cover the message gate wired into StreamingChatExecutor: the REQUEST gate at
the top of run() and the RESPONSE gate (_govern_final_response) applied to the
final assistant output.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.chat_stream_executor import StreamingChatExecutor
from victor.framework.policies import (
    BlockPatternPolicy,
    MessagePolicyGate,
    Phase,
    PolicyEngine,
    RedactContentPolicy,
)


def _executor() -> StreamingChatExecutor:
    return StreamingChatExecutor(runtime_owner=MagicMock())


def _orch_with_gate(gate):
    """Minimal orch stub exposing the gate (None disables)."""
    return SimpleNamespace(_message_policy_gate=gate)


# -- _govern_final_response (RESPONSE phase) ---------------------------------


async def test_govern_final_response_passthrough_when_no_gate():
    exe = _executor()
    text, blocked = await exe._govern_final_response(_orch_with_gate(None), "hello")
    assert text == "hello"
    assert blocked is False


async def test_govern_final_response_passthrough_empty_text():
    gate = MessagePolicyGate(PolicyEngine([RedactContentPolicy([r"secret"])]))
    exe = _executor()
    text, blocked = await exe._govern_final_response(_orch_with_gate(gate), "")
    assert text == ""
    assert blocked is False


async def test_govern_final_response_redacts():
    gate = MessagePolicyGate(PolicyEngine([RedactContentPolicy([r"sk-\w+"], placeholder="[KEY]")]))
    exe = _executor()
    text, blocked = await exe._govern_final_response(_orch_with_gate(gate), "the key is sk-abc123")
    assert text == "the key is [KEY]"
    assert blocked is False


async def test_govern_final_response_blocks():
    gate = MessagePolicyGate(
        PolicyEngine(
            [BlockPatternPolicy([r"CONFIDENTIAL"], phases={Phase.RESPONSE}, reason="nope")]
        )
    )
    exe = _executor()
    text, blocked = await exe._govern_final_response(_orch_with_gate(gate), "this is CONFIDENTIAL")
    assert blocked is True
    assert text == "nope"


# -- REQUEST gate at top of run() -------------------------------------------


async def _drain(agen):
    chunks = []
    async for c in agen:
        chunks.append(c)
    return chunks


def _runtime_owner_with_orch(orch):
    runtime_owner = MagicMock()
    runtime_owner._orchestrator = orch
    return runtime_owner


async def test_run_request_block_short_circuits():
    # A blocked REQUEST yields a single refusal chunk and never sets up the stream.
    gate = MessagePolicyGate(
        PolicyEngine([BlockPatternPolicy([r"rm -rf"], phases={Phase.REQUEST}, reason="dangerous")])
    )
    sentinel_chunk = object()
    chunk_gen = MagicMock()
    chunk_gen.generate_content_chunk = MagicMock(return_value=sentinel_chunk)
    orch = MagicMock()
    orch._message_policy_gate = gate
    orch._chunk_generator = chunk_gen
    # _create_stream_context must never be reached on a blocked request.
    runtime_owner = _runtime_owner_with_orch(orch)
    runtime_owner._create_stream_context = MagicMock(
        side_effect=AssertionError("stream setup must not run on blocked request")
    )

    exe = StreamingChatExecutor(runtime_owner=runtime_owner)
    chunks = await _drain(exe.run_unified("please run rm -rf /"))

    assert chunks == [sentinel_chunk]
    # The refusal chunk carries the policy reason and is final.
    args, kwargs = chunk_gen.generate_content_chunk.call_args
    assert "dangerous" in args[0]
    assert kwargs.get("is_final") is True
    runtime_owner._create_stream_context.assert_not_called()

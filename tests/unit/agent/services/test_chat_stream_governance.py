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
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_runtime_services import (
    ChatCompletion,
    ChatConversation,
    ChatGovernance,
)
from victor.agent.services.chat_stream_executor import StreamingActResult, StreamingChatExecutor
from victor.core.errors import AgentError
from victor.providers.base import StreamChunk
from victor.framework.policies import (
    BlockPatternPolicy,
    MessagePolicyGate,
    Phase,
    PolicyEngine,
    RedactContentPolicy,
)


def _executor(gate=None) -> StreamingChatExecutor:
    runtime_owner = MagicMock()
    runtime_owner.services = SimpleNamespace(governance=ChatGovernance(gate=gate))
    return StreamingChatExecutor(runtime_owner=runtime_owner)


# -- _govern_final_response (RESPONSE phase) ---------------------------------


async def test_govern_final_response_passthrough_when_no_gate():
    exe = _executor()
    text = await exe._govern_final_response("hello")
    assert text == "hello"


async def test_govern_final_response_passthrough_empty_text():
    gate = MessagePolicyGate(PolicyEngine([RedactContentPolicy([r"secret"])]))
    exe = _executor(gate)
    text = await exe._govern_final_response("")
    assert text == ""


# -- REQUEST gate at top of run() -------------------------------------------


async def _drain(agen):
    chunks = []
    async for c in agen:
        chunks.append(c)
    return chunks


def _runtime_owner_with_orch(orch):
    runtime_owner = MagicMock()
    runtime_owner._orchestrator = orch
    runtime_owner.services = SimpleNamespace(
        delivery=ChatDelivery(chunks=orch._chunk_generator),
        governance=ChatGovernance(gate=orch._message_policy_gate),
    )
    return runtime_owner


async def test_run_request_block_short_circuits():
    # A blocked REQUEST yields a single refusal chunk and never sets up the stream.
    gate = MessagePolicyGate(
        PolicyEngine([BlockPatternPolicy([r"rm -rf"], phases={Phase.REQUEST}, reason="dangerous")])
    )
    sentinel_chunk = StreamChunk(content="dangerous", is_final=True)
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

    assert len(chunks) == 1
    assert chunks[0].content == "dangerous"
    assert chunks[0].metadata["agentic_loop_success"] is False
    assert chunks[0].metadata["agentic_loop_iterations"] == 0
    # The refusal chunk carries the policy reason and is final.
    args, kwargs = chunk_gen.generate_content_chunk.call_args
    assert "dangerous" in args[0]
    assert kwargs.get("is_final") is True
    runtime_owner._create_stream_context.assert_not_called()


def _live_act_executor(gate):
    """Use real ACT assembly and publication; only external provider I/O is stubbed."""
    runtime = SimpleNamespace()
    runtime.services = SimpleNamespace(
        governance=ChatGovernance(gate=gate),
        delivery=ChatDelivery(
            chunks=SimpleNamespace(
                generate_content_chunk=lambda content, is_final=False: StreamChunk(
                    content=content, is_final=is_final
                )
            ),
            sanitizer=SimpleNamespace(sanitize=lambda text: text, strip_markup=lambda text: text),
        ),
        stream_lifecycle=SimpleNamespace(is_cancelled=lambda: False),
        tool_calls=SimpleNamespace(parse_and_validate=lambda calls, text: (calls, text)),
        completion=ChatCompletion(),
        conversation=ChatConversation(),
    )
    executor = StreamingChatExecutor(runtime)
    executor._stream_provider_turn = AsyncMock(
        return_value=(None, "the key is sk-abc123", None, False)
    )
    return executor, runtime


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("policy", ["absent", "redact", "block", "empty"])
async def test_act_governs_candidate_before_history_and_delivery(policy, forced):
    gate = None
    if policy == "redact":
        gate = MessagePolicyGate(
            PolicyEngine([RedactContentPolicy([r"sk-\w+"], placeholder="[KEY]")])
        )
    elif policy == "block":
        gate = MessagePolicyGate(
            PolicyEngine(
                [BlockPatternPolicy([r"sk-\w+"], phases={Phase.RESPONSE}, reason="withheld")]
            )
        )
    elif policy == "empty":
        gate = MessagePolicyGate(PolicyEngine([RedactContentPolicy([r".+"], placeholder="")]))
    if gate:
        gate.gate_response = AsyncMock(wraps=gate.gate_response)
    executor, runtime = _live_act_executor(gate)
    orch = SimpleNamespace(add_message=MagicMock())
    result = StreamingActResult()
    stream = executor.execute_turn_streaming(
        orch,
        runtime,
        SimpleNamespace(user_message="", is_qa_task=False),
        user_message="q",
        goals=None,
        recovery=None,
        create_recovery_context=None,
        forced_task_completion=forced,
        result=result,
    )
    if policy in {"block", "empty"}:
        reason = "withheld" if policy == "block" else "no deliverable content"
        with pytest.raises(AgentError, match=reason) as exc:
            await _drain(stream)
        assert exc.value.recoverable is False
        orch.add_message.assert_not_called()
    else:
        chunks = await _drain(stream)
        expected = {"absent": "the key is sk-abc123", "redact": "the key is [KEY]", "empty": ""}[
            policy
        ]
        assert "".join(c.content for c in chunks) == expected
        assert result.full_content == expected
        assert result.turn_result.content == expected
        if expected:
            assert orch.add_message.call_args.args[:2] == ("assistant", expected)
        else:
            orch.add_message.assert_not_called()
    if gate:
        gate.gate_response.assert_awaited_once_with("the key is sk-abc123")
    executor._stream_provider_turn.assert_awaited_once()

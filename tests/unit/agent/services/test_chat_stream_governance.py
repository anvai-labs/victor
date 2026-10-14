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
the top of run() and the RESPONSE gate (ChatGovernance.govern_response) applied to the
assistant narrative before publication and tool dispatch.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_runtime_services import (
    ChatCompletion,
    ChatConversation,
    ChatGovernance,
    ChatRuntimeIntelligence,
)
from victor.agent.services.chat_stream_executor import (
    StreamingActResult,
    StreamingChatExecutor,
    _StreamingCancelled,
)
from victor.core.errors import AgentError
from victor.providers.base import StreamChunk
from victor.framework.policies import (
    BlockPatternPolicy,
    GateResult,
    MessagePolicyGate,
    Phase,
    PolicyEngine,
    RedactContentPolicy,
)


async def test_govern_response_passthrough_empty_text():
    gate = MessagePolicyGate(PolicyEngine([RedactContentPolicy([r"secret"])]))
    assert await ChatGovernance(gate=gate).govern_response("") == ""


# -- REQUEST gate at top of run() -------------------------------------------


async def _drain(agen):
    chunks = []
    async for c in agen:
        chunks.append(c)
    return chunks


def _runtime_owner_with_orch(orch):
    from victor.config.agentic_loop_settings import AgenticLoopSettings

    orch.settings.agentic_loop = AgenticLoopSettings()
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
    from victor.agent.services.metrics_service import AgentMetricsService
    from victor.agent.session_cost_tracker import SessionCostTracker

    runtime = SimpleNamespace()
    runtime.services = SimpleNamespace(
        metrics=AgentMetricsService(MagicMock(), SessionCostTracker(), {}),
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
        intelligence=ChatRuntimeIntelligence(),
    )
    executor = StreamingChatExecutor(runtime)
    executor._stream_provider_turn = AsyncMock(
        return_value=(None, "the key is sk-abc123", None, False)
    )
    return executor, runtime


@pytest.mark.parametrize(
    "mode",
    [
        "ordinary",
        "forced",
        "recovered",
        "tool-bearing",
        "natural",
        "exhausted",
        "completion-summary",
        "compaction-summary",
    ],
)
@pytest.mark.parametrize("policy", ["absent", "redact", "block", "empty", "cancel"])
async def test_act_governs_candidate_before_history_and_delivery(policy, mode):
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
        gate = MessagePolicyGate(
            PolicyEngine([RedactContentPolicy([r"the key is sk-\w+"], placeholder="")])
        )
    elif policy == "cancel":
        gate = MessagePolicyGate(PolicyEngine([]))
    if gate:
        gate.gate_response = AsyncMock(wraps=gate.gate_response)
    executor, runtime = _live_act_executor(gate)
    if policy == "cancel":

        async def cancel_response(content):
            if content == "the key is sk-abc123":
                runtime.services.stream_lifecycle.is_cancelled = lambda: True
            return GateResult(allowed=True, content=content)

        gate.gate_response.side_effect = cancel_response
    orch = SimpleNamespace(add_message=MagicMock())
    recovery = None
    provider_calls = []
    tool_calls = [{"id": "read1", "name": "read", "arguments": {}}]
    executed = []

    async def execute_tools(*args, result_holder, **kwargs):
        from victor.agent.streaming.tool_execution import ToolExecutionResult

        executed.append(kwargs)
        result_holder.result = ToolExecutionResult()
        if False:
            yield

    executor._execute_tools_turn = execute_tools
    if mode == "tool-bearing":
        executor._stream_provider_turn.return_value = (
            None,
            "the key is sk-abc123",
            tool_calls,
            False,
        )
    if mode in {"natural", "exhausted"}:
        executor._stream_provider_turn.return_value = (None, "", None, False)
        runtime._handle_empty_response_recovery = AsyncMock(return_value=(False, None, None))
        recovery = SimpleNamespace(
            check_natural_completion=lambda *args, **kwargs: (
                StreamChunk(content="the key is sk-abc123", is_final=True)
                if mode == "natural"
                else None
            ),
            handle_empty_response=lambda *args: (None, False),
            get_recovery_fallback_message=lambda ctx: "the key is sk-abc123",
        )
    if mode == "recovered":
        from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin

        async def recover_stream(**kwargs):
            provider_calls.append(1)
            yield StreamChunk(content="the key is sk-abc123")

        orch.provider = SimpleNamespace(name="unknown", stream=recover_stream)
        orch.model = "test"
        orch.max_tokens = 1024
        orch.thinking = False
        orch.get_assembled_messages = lambda **kwargs: []
        runtime._orchestrator = orch
        runtime._handle_empty_response_recovery = MethodType(
            ChatStreamHelperMixin._handle_empty_response_recovery, runtime
        )
        executor._stream_provider_turn.return_value = (None, "", None, False)
        recovery = SimpleNamespace(
            check_natural_completion=lambda *args, **kwargs: None,
            handle_empty_response=lambda *args: (None, False),
        )
    stream_ctx = SimpleNamespace(
        user_message="", is_qa_task=False, goals=None, planned_tools=None, cumulative_usage={}
    )
    summary_mode = mode in {"completion-summary", "compaction-summary"}
    if summary_mode:
        executor._stream_provider_turn.return_value = (None, "VICTOR_TASK_DONE::", None, False)
        if mode == "completion-summary":
            runtime.services.completion = ChatCompletion(
                detector=SimpleNamespace(
                    get_state=lambda: SimpleNamespace(last_summary="the key is sk-abc123")
                )
            )
        else:
            stream_ctx.compaction_summary = "the key is sk-abc123"
    result = StreamingActResult()
    stream = executor.execute_turn_streaming(
        orch,
        runtime,
        stream_ctx,
        user_message="q",
        goals=None,
        recovery=recovery,
        create_recovery_context=lambda ctx: ctx,
        forced_task_completion=mode == "forced" or summary_mode,
        result=result,
    )
    published = []

    async def consume():
        async for chunk in stream:
            published.append(chunk)
        return published

    if policy == "cancel":
        with pytest.raises(_StreamingCancelled):
            await consume()
        assert published == []
        assert not any(call.args[0] == "assistant" for call in orch.add_message.call_args_list)
        assert result.turn_result is None
    elif policy in {"block", "empty"}:
        reason = "withheld" if policy == "block" else "no deliverable content"
        with pytest.raises(AgentError, match=reason) as exc:
            await consume()
        assert published == []
        assert exc.value.recoverable is False
        assert not any(call.args[0] == "assistant" for call in orch.add_message.call_args_list)
        assert result.turn_result is None
    else:
        chunks = await consume()
        expected = {"absent": "the key is sk-abc123", "redact": "the key is [KEY]", "empty": ""}[
            policy
        ]
        assert "".join(c.content for c in chunks) == expected
        assert result.full_content == expected
        assert result.turn_result.content == expected
        if expected and mode not in {"natural", "exhausted"}:
            assert orch.add_message.call_args.args[:2] == ("assistant", expected)
        else:
            orch.add_message.assert_not_called()
    if mode == "tool-bearing" and policy not in {"block", "empty", "cancel"}:
        assert len(executed) == 1
        assert executed[0]["full_content"] == expected
        assert executed[0]["tool_calls"] == tool_calls
        assert orch.add_message.call_args.kwargs["tool_calls"] == tool_calls
    else:
        assert executed == []
    if mode == "exhausted":
        runtime._handle_empty_response_recovery.assert_awaited_once()
        if policy not in {"block", "empty", "cancel"}:
            assert result.turn_result.disposition.value == "recovery_exhausted"
    if mode == "natural":
        runtime._handle_empty_response_recovery.assert_not_awaited()
    if gate:
        assert [call.args[0] for call in gate.gate_response.await_args_list] == (
            ["VICTOR_TASK_DONE::", "the key is sk-abc123"]
            if summary_mode
            else ["the key is sk-abc123"]
        )
    executor._stream_provider_turn.assert_awaited_once()
    if mode == "recovered":
        assert provider_calls == [1]

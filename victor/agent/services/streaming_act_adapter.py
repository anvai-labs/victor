# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Streaming ACT adapter — bridges the unified loop to the service-layer streaming primitive.

FEP-0007 Addendum A (framing B): ``AgenticLoop.run_streaming`` drives a narrow framework seam
(``StreamingActProvider.stream_turn_act``) for its ACT, decoupled from streaming plumbing. This
adapter implements that seam over the service-layer
``StreamingChatExecutor.execute_turn_streaming`` (the per-turn provider + emit + tools primitive).

One adapter instance per ``run_streaming`` call: :meth:`StreamingActAdapter.prepare` runs the
per-run setup once (stream context, task intent/goals, recovery handles), and
:meth:`StreamingActAdapter.stream_turn_act` drives one turn's ACT — re-yielding its
``StreamChunk``s and mapping the produced ``TurnResult`` onto the framework outcome holder.

This module is the connective tissue for the step-3 cutover; it is not yet wired into
``ChatStreamRuntime`` (the repoint that makes ``run_streaming`` the live UI loop). Two per-run
concerns are deliberately deferred to that cutover, where they are exercised live and parity-gated:

* the governance REQUEST gate (a block must short-circuit the whole run, not one turn), and
* reconciling the buffered loop's eager completion so the unified loop runs full multi-step tasks
  (see the run/stream parity battery's cutover tripwire).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

from victor.providers.base import StreamChunk


@dataclass
class StreamActSession:
    """Per-run streaming state the ACT primitive needs, prepared once per ``run_streaming`` call.

    Mirrors the locals ``StreamingChatExecutor.run()`` sets up before its turn loop, so the
    adapter can drive ``execute_turn_streaming`` per turn without re-running the preamble.
    """

    orch: Any
    runtime_owner: Any
    stream_ctx: Any
    goals: Any
    recovery: Any
    create_recovery_context: Any


class StreamingActAdapter:
    """Implements the framework ``StreamingActProvider`` seam over the streaming executor.

    Construct one per run via :meth:`prepare`, pass it as ``AgenticLoop(streaming_act_port=...)``,
    and ``run_streaming`` calls :meth:`stream_turn_act` once per turn. Stateless across runs.
    """

    def __init__(self, executor: Any, session: StreamActSession) -> None:
        self._executor = executor
        self._session = session

    @property
    def session(self) -> StreamActSession:
        return self._session

    @classmethod
    async def prepare(
        cls, executor: Any, user_message: str, **kwargs: Any
    ) -> "StreamingActAdapter":
        """Run ``StreamingChatExecutor.run()``'s per-run preamble and capture it as a session.

        Reuses the executor's existing setup helpers (stream-context creation, task-requirement
        extraction, run guidance, task-intent/goal initialization) so the adapter produces the
        same per-turn inputs the live loop does. The governance REQUEST gate is intentionally NOT
        run here — it short-circuits the whole run and belongs in the cutover's run wrapper.
        """
        runtime_owner = executor._runtime_owner
        orch = runtime_owner._orchestrator
        recovery = getattr(orch, "_recovery_service", None) or orch._recovery_coordinator
        create_recovery_context = orch.create_recovery_context

        executor._reset_streaming_turn_state(orch)
        stream_ctx = await runtime_owner._create_stream_context(user_message, **kwargs)
        orch._current_stream_context = stream_ctx

        await executor._extract_task_requirements(orch, user_message)
        executor._apply_run_guidance(
            orch, stream_ctx, user_message, stream_ctx.max_exploration_iterations
        )
        goals = executor._initialize_task_intent(orch, stream_ctx, user_message)

        return cls(
            executor,
            StreamActSession(
                orch=orch,
                runtime_owner=runtime_owner,
                stream_ctx=stream_ctx,
                goals=goals,
                recovery=recovery,
                create_recovery_context=create_recovery_context,
            ),
        )

    async def stream_turn_act(
        self,
        *,
        query: str,
        state: Any,
        perception: Any,
        plan: Any,
        turn_index: int,
        outcome: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Drive one turn's ACT via the streaming executor, yielding chunks + producing a TurnResult.

        Implements the framework ``StreamingActProvider`` seam: increments the stream context's
        iteration counter to match the loop turn, runs ``execute_turn_streaming``, re-yields every
        chunk, and writes the produced ``TurnResult`` onto ``outcome`` for the shared EVALUATE
        phase. ``perception`` / ``plan`` / ``state`` are accepted for seam symmetry; the streaming
        ACT consumes perception via the stream context (set up in :meth:`prepare`).
        """
        from victor.agent.services.chat_stream_executor import StreamingActResult

        session = self._session
        # Keep the streaming context's iteration counter in step with the loop turn.
        session.stream_ctx.total_iterations = turn_index

        act_result = StreamingActResult()
        async for chunk in self._executor.execute_turn_streaming(
            session.orch,
            session.runtime_owner,
            session.stream_ctx,
            user_message=query,
            goals=session.goals,
            recovery=session.recovery,
            create_recovery_context=session.create_recovery_context,
            result=act_result,
        ):
            yield chunk

        outcome.turn_result = act_result.turn_result

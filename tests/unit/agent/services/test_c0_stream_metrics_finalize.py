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

"""C0: the service streaming runtime must finalize stream metrics (close the cost wire).

Provider-independent — proves the wire (finalize is called with the turn's cumulative usage),
regardless of whether a given provider returns token usage.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime


def _usage(p: int, c: int) -> dict:
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


async def test_stream_chat_finalizes_metrics_with_cumulative_usage(monkeypatch):
    orch = SimpleNamespace(_finalize_stream_metrics=MagicMock())
    rt = ServiceStreamingRuntime(orch)

    ctx = SimpleNamespace(cumulative_usage=_usage(120, 40))
    state_dict = {
        "_current_stream_context": ctx,
        "_cumulative_token_usage": _usage(0, 0),
    }
    bindings = SimpleNamespace(
        state_host=orch,
        state_dict=state_dict,
        get_capability_value=lambda name, default=None: (
            ctx if name == "current_stream_context" else default
        ),
    )
    monkeypatch.setattr(rt, "_get_runtime_bindings", lambda *a, **k: bindings)

    class _Executor:
        async def run_unified(self, user_message, **kwargs):
            if False:  # async generator that yields nothing
                yield None

    monkeypatch.setattr(rt, "get_executor", lambda: _Executor())

    async for _ in rt.stream_chat("hi"):
        pass

    # The wire is closed: metrics are finalized exactly once with the turn's usage.
    orch._finalize_stream_metrics.assert_called_once()
    called_usage = orch._finalize_stream_metrics.call_args[0][0]
    assert called_usage["prompt_tokens"] == 120
    assert called_usage["completion_tokens"] == 40


async def test_finalize_failure_does_not_break_the_stream(monkeypatch):
    orch = SimpleNamespace(_finalize_stream_metrics=MagicMock(side_effect=RuntimeError("boom")))
    rt = ServiceStreamingRuntime(orch)
    ctx = SimpleNamespace(cumulative_usage=_usage(10, 5))
    state_dict = {
        "_current_stream_context": ctx,
        "_cumulative_token_usage": _usage(0, 0),
    }
    bindings = SimpleNamespace(
        state_host=orch,
        state_dict=state_dict,
        get_capability_value=lambda name, default=None: (
            ctx if name == "current_stream_context" else default
        ),
    )
    monkeypatch.setattr(rt, "_get_runtime_bindings", lambda *a, **k: bindings)

    class _Executor:
        async def run_unified(self, user_message, **kwargs):
            if False:
                yield None

    monkeypatch.setattr(rt, "get_executor", lambda: _Executor())

    # Finalize raising must not propagate out of the stream (it's best-effort).
    async for _ in rt.stream_chat("hi"):
        pass
    orch._finalize_stream_metrics.assert_called_once()

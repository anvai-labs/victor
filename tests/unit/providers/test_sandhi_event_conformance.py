"""Contract-consumption conformance for the sandhi typed event stream (TD-0008 P1).

Every ``ChatStreamEventV1`` variant sandhi can emit must have a deliberate
victor consumption decision: consumed, surfaced, or EXPLICITLY ignored. This
suite drives ``_sandhi_stream`` with a scripted stream containing every
variant and asserts each one lands somewhere observable — so a future variant
addition fails here instead of silently vanishing (``reasoning_delta`` did
exactly that in session modality-doc-review-fixes-b4e87728; ``refusal_delta``
until this suite).
"""

from __future__ import annotations

import json

import pytest

import victor.providers.sandhi_transport as st
from victor.core.errors import ProviderError as VictorProviderError

# Every variant of ChatStreamEventV1 (sandhi crates/sandhi-core/src/chat.rs).
ALL_EVENT_KINDS = [
    "response_start",
    "text_delta",
    "reasoning_delta",
    "refusal_delta",
    "tool_call_start",
    "tool_call_arguments_delta",
    "tool_call_end",
    "usage",
    "finish",
    "error",
]


class _FakeTyped:
    def __init__(self, events):
        self._events = events

    def stream_json(self, request_json, wire_headers_json=None):
        events = self._events

        async def _gen():
            for event in events:
                yield json.dumps(event)

        return _gen()


class _Host(st.SandhiTypedProviderMixin):
    def __init__(self, events):
        self._fake = _FakeTyped(events)

    def _typed_provider(self, model):
        return self._fake

    def _sandhi_slug(self):
        return "moonshot"

    def _sandhi_timeout(self):
        return 30.0


async def _collect(events):
    host = _Host(events)
    return [chunk async for chunk in host._sandhi_stream({"model": "kimi-k3"})]


FULL_SCRIPT = [
    {"event": "response_start", "id": "resp_1", "model": "kimi-k3"},
    {"event": "reasoning_delta", "delta": "thinking about it"},
    {"event": "refusal_delta", "delta": "I cannot help with that."},
    {"event": "text_delta", "delta": "Hello"},
    {"event": "tool_call_start", "index": 0, "id": "call_1", "name": "read"},
    {"event": "tool_call_arguments_delta", "index": 0, "delta": '{"path":'},
    {"event": "tool_call_arguments_delta", "index": 0, "delta": ' "a.py"}'},
    {"event": "tool_call_end", "index": 0},
    {
        "event": "usage",
        "usage": {
            "tokens_in": 10,
            "tokens_out": 5,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        },
    },
    {"event": "finish", "reason": "tool_calls"},
]


class TestEveryVariantHasAConsumptionDecision:
    def test_script_covers_every_variant(self):
        scripted = {event["event"] for event in FULL_SCRIPT} | {"error"}
        assert scripted == set(ALL_EVENT_KINDS), (
            "FULL_SCRIPT must exercise every ChatStreamEventV1 variant; "
            f"missing={set(ALL_EVENT_KINDS) - scripted}"
        )

    @pytest.mark.asyncio
    async def test_full_script_lands_observably(self):
        chunks = await _collect(FULL_SCRIPT)

        # text_delta -> content
        assert any(chunk.content == "Hello" for chunk in chunks)
        # reasoning_delta -> metadata.reasoning_content
        assert any(
            (chunk.metadata or {}).get("reasoning_content") == "thinking about it"
            for chunk in chunks
        )
        # refusal_delta -> metadata.refusal
        assert any(
            (chunk.metadata or {}).get("refusal") == "I cannot help with that." for chunk in chunks
        )
        final = chunks[-1]
        assert final.is_final
        # tool_call_start + arguments deltas -> assembled tool call on the final chunk
        [tool_call] = final.tool_calls
        assert tool_call["id"] == "call_1"
        assert tool_call["name"] == "read"
        assert tool_call["arguments"] == {"path": "a.py"}
        # finish -> stop_reason
        assert final.stop_reason == "tool_calls"
        # usage -> final usage dict
        assert final.usage and final.usage.get("prompt_tokens") == 10

    @pytest.mark.asyncio
    async def test_response_start_and_tool_call_end_are_explicitly_ignored(self):
        # Ignored-by-decision variants must not crash, emit chunks, or warn.
        chunks = await _collect(
            [
                {"event": "response_start", "id": "r", "model": "m"},
                {"event": "tool_call_end", "index": 0},
                {"event": "finish", "reason": "stop"},
            ]
        )
        assert len(chunks) == 1 and chunks[0].is_final

    @pytest.mark.asyncio
    async def test_error_event_surfaces_as_mapped_provider_error(self):
        with pytest.raises(VictorProviderError) as excinfo:
            await _collect(
                [
                    {"event": "text_delta", "delta": "partial"},
                    {
                        "event": "error",
                        "error": {
                            "code": "upstream_error",
                            "message": "upstream status 500",
                            "retryable": True,
                            "http_status": 500,
                        },
                    },
                ]
            )
        assert "upstream status 500" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_unknown_event_kind_raises_drift_alarm(self, caplog):
        with caplog.at_level("WARNING"):
            chunks = await _collect(
                [
                    {"event": "shiny_new_variant", "delta": "?"},
                    {"event": "finish", "reason": "stop"},
                ]
            )
        assert chunks[-1].is_final  # stream survives
        assert any(
            "no victor consumption decision" in record.getMessage() for record in caplog.records
        ), "an unconsumed variant must trip the contract-drift alarm"

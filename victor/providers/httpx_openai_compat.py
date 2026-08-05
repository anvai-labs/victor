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

"""Policy/payload base for OpenAI-compatible cloud providers (gap #2).

The 11 OpenAI-compatible cloud providers (groq, together, fireworks, openrouter,
xai, mistral, deepseek, moonshot, zai, cerebras, qwen) share this base for their
model-facing policy and request shaping. Provider I/O transport (raw httpx
``/v1/chat/completions`` POST + SSE streaming) was removed in TD-0002 gap #2:
execution is owned by the Sandhi typed variant. ``SandhiOpenAICompatPolicy``
mixes in ``SandhiHttpxTransportMixin`` which overrides ``chat``/``stream`` to
build the OpenAI payload here (``_build_request_payload``) and hand it to Sandhi.

This base still owns the shared, transport-independent surface:
- Request payload assembly (``_build_request_payload``) — consumed by the Sandhi mixin
- Response/stream parsing template methods (``_parse_response``, ``_parse_stream_chunk``)
  kept as the shared API the cloud-provider subclasses decorate and tests exercise
- Template Method hooks for provider-specific behaviour

``chat``/``stream`` are left as ``NotImplementedError`` guard stubs so the policy
shell stays concrete for discovery/capability use and for direct unit testing of
the payload/parse surface; the live transport is always the Sandhi typed variant.
"""

from __future__ import annotations

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.openai_compat import (
    accumulate_tool_call_delta,
    build_openai_messages,
    convert_tools_to_openai_format,
    parse_openai_tool_calls,
)
from victor.providers.logging import ProviderLogger
from victor.providers.usage_parsing import parse_usage_dict, usage_dict_from_neutral

logger = logging.getLogger(__name__)

# Private envelope key used only across Victor's wire seam. Sandhi has already
# parsed these neutral counts at the source; the key is stripped before the raw
# provider response is exposed to callers.
TRANSPORT_NEUTRAL_USAGE_KEY = "__victor_transport_neutral_usage__"


class HttpxOpenAICompatProvider(BaseProvider):
    """Policy/payload base for OpenAI-API-compatible providers (transport owned by Sandhi).

    Subclasses must implement:
        - ``name`` (property)
        - ``supports_tools()``
        - ``supports_streaming()``

    ``chat``/``stream`` are guard stubs; the live transport is the Sandhi typed
    variant (``SandhiHttpxTransportMixin``), which consumes ``_build_request_payload``.

    Subclasses may override the Template Method hooks:
        - ``_clean_model_name(model)``       — strip provider suffixes before the API call
        - ``_get_provider_params(...)``       — temperature/max_tokens variants, extra flags
        - ``_extract_response_metadata(msg)`` — reasoning_content or other non-standard fields
        - ``_extract_stream_metadata(delta)`` — same for streaming deltas
    """

    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        provider_name: str = "",
        default_headers: Optional[Dict[str, str]] = None,
        initialize_http_client: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self._api_key = api_key
        self._provider_logger = ProviderLogger(provider_name or self.name, __name__)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if default_headers:
            headers.update(default_headers)
        if initialize_http_client:
            self.client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(timeout),
            )

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supports_tools(self) -> bool: ...

    @abstractmethod
    def supports_streaming(self) -> bool: ...

    # ── Template Method hooks (override in subclasses) ────────────────────────

    def _clean_model_name(self, model: str) -> str:
        """Strip provider-specific suffixes before the model name is sent to the API.

        Example (ZAI): "glm-5.1:coding" → "glm-5.1"
        Default: return model unchanged.
        """
        return model

    def _get_provider_params(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return the provider-specific parameter block for the request payload.

        Override to handle model-specific variants:
        - DeepSeek: skip temperature for reasoner models
        - OpenAI O-series: use max_completion_tokens instead of max_tokens
        - ZAI: add thinking: {type: "enabled"} when thinking=True

        Default returns standard OpenAI parameters.
        """
        return {"temperature": temperature, "max_tokens": max_tokens}

    def _extract_response_metadata(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract provider-specific metadata from a non-streaming response message.

        Override to capture fields like ``reasoning_content`` (ZAI, DeepSeek).
        Default returns None.
        """
        return None

    def _extract_stream_metadata(self, delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract provider-specific metadata from a streaming delta.

        Override to capture ``reasoning_content`` in thinking-mode streams.
        Default returns None.
        """
        return None

    def _filter_tools_for_model(
        self,
        model: str,
        tools: Optional[List[ToolDefinition]],
    ) -> Optional[List[ToolDefinition]]:
        """Filter or suppress tools based on model capabilities.

        Override for models that don't support function calling in certain
        configurations (e.g., DeepSeek-reasoner, O-series without tools).
        Default passes tools through unchanged.
        """
        return tools

    # ── Shared implementation ─────────────────────────────────────────────────

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        return convert_tools_to_openai_format(tools)

    def _build_request_payload(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the full request payload for /v1/chat/completions.

        Uses ``build_openai_messages()`` for complete tool-call serialization and
        orphaned-message cleanup, then calls ``_get_provider_params()`` for the
        model-specific parameter block.
        """
        formatted = build_openai_messages(messages)

        if not formatted:
            raise ValueError(
                "Cannot send request with empty messages list. "
                "This usually indicates a conversation initialization issue."
            )

        provider_params = self._get_provider_params(model, temperature, max_tokens, **kwargs)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted,
            "stream": stream,
            **provider_params,
        }

        effective_tools = self._filter_tools_for_model(model, tools)
        if effective_tools:
            payload["tools"] = self._convert_tools(effective_tools)
            payload["tool_choice"] = "auto"

        # Pass through provider-specific kwargs (exclude internal keys)
        for key, value in kwargs.items():
            if key not in {"api_key"} and value is not None and key not in payload:
                payload[key] = value

        # reasoning_effort is only valid for reasoning models. The kwarg
        # passthrough above forwards it for any OpenAI-compatible endpoint that
        # accepts it (subclasses opt in via supports_reasoning_effort); strip it
        # otherwise so it never reaches a model/endpoint that would reject it.
        if "reasoning_effort" in payload and not self.supports_reasoning_effort(model):
            payload.pop("reasoning_effort", None)

        # Log detailed message structure for debugging tool pairing issues
        tool_messages = [
            (i, m) for i, m in enumerate(formatted) if m.get("role") in ("tool", "assistant")
        ]
        if tool_messages:
            for i, msg in tool_messages:
                role = msg.get("role")
                if role == "assistant" and "tool_calls" in msg:
                    tc_ids = [tc.get("id", "") for tc in msg["tool_calls"]]
                    self._provider_logger.logger.debug(
                        "%s payload: message[%d] role=assistant tool_calls=%s",
                        self.name,
                        i,
                        tc_ids,
                    )
                elif role == "tool":
                    self._provider_logger.logger.debug(
                        "%s payload: message[%d] role=tool tool_call_id=%s name=%s",
                        self.name,
                        i,
                        msg.get("tool_call_id"),
                        msg.get("name", ""),
                    )

        # System prompt visibility: log the full payload shape (tool names + system
        # message) at INFO so the actual offer the model receives is auditable in the
        # session log. This is the single source of truth for "what the LLM sees"
        # before the API call — critical for diagnosing tool-description/registry
        # drift (e.g. a stale tool like `code_search` being advertised).
        offered_tool_names = []
        if effective_tools:
            for _tool in effective_tools:
                # ToolDefinition may expose name directly or via .function.name
                _tname = getattr(_tool, "name", None)
                if _tname is None:
                    _tname = getattr(getattr(_tool, "function", None), "name", None)
                if _tname:
                    offered_tool_names.append(_tname)

        _sys_preview = ""
        for _msg in formatted:
            if _msg.get("role") == "system":
                _content = _msg.get("content", "") or ""
                _sys_preview = _content[:160].replace("\n", " ")
                break

        self._provider_logger.logger.info(
            "%s payload: model=%s messages=%d tools=%d stream=%s "
            "tool_names=%s system_prompt_preview=%r",
            self.name,
            model,
            len(formatted),
            len(effective_tools) if effective_tools else 0,
            stream,
            offered_tool_names,
            _sys_preview,
        )
        self._provider_logger.logger.debug(
            "%s payload: model=%s messages=%d tools=%s stream=%s",
            self.name,
            model,
            len(formatted),
            tools is not None,
            stream,
        )

        return payload

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse a non-streaming /v1/chat/completions response dict."""
        neutral_usage = result.get(TRANSPORT_NEUTRAL_USAGE_KEY)
        if neutral_usage is not None:
            result = dict(result)
            result.pop(TRANSPORT_NEUTRAL_USAGE_KEY, None)
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = parse_openai_tool_calls(message.get("tool_calls"))

        # Routed through sandhi's single-sourced parser (recovers the prompt-cache
        # split; prompt_tokens stays the FULL count); native dict is the fallback.
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = (
                usage_dict_from_neutral(neutral_usage, usage_data, slug="openai")
                or parse_usage_dict("openai", usage_data)
                or {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }
            )

        metadata = self._extract_response_metadata(message)

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
            metadata=metadata,
        )

    def _parse_stream_chunk(
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
    ) -> Optional[StreamChunk]:
        """Parse one SSE chunk, accumulating tool-call deltas."""
        choices = chunk_data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        metadata = self._extract_stream_metadata(delta)

        # Accumulate tool-call fragments
        accumulate_tool_call_delta(delta, accumulated_tool_calls)

        # Finalize tool calls when the model signals completion
        final_tool_calls: Optional[List[Dict[str, Any]]] = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args_str = tc.get("arguments", "{}")
                    try:
                        parsed_args = (
                            json_loads(args_str) if isinstance(args_str, str) else args_str
                        )
                    except JSONDecodeError:
                        parsed_args = {}
                    final_tool_calls.append(
                        {
                            "id": tc.get("id"),
                            "name": tc["name"],
                            "arguments": parsed_args,
                        }
                    )

        # Parse usage from final chunk (when finish_reason is set) — routed through
        # sandhi's single-sourced parser; native dict is the fallback.
        usage = None
        if finish_reason and "usage" in chunk_data:
            usage_data = chunk_data.get("usage") or {}
            usage = parse_usage_dict("openai", usage_data) or {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
            metadata=metadata,
            usage=usage,
        )

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Chat transport is owned by the Sandhi typed variant.

        This policy/payload shell stays concrete so its discovery, capability, and
        request-shaping surface (``_build_request_payload`` and the parse template
        methods) remain directly unit-testable. Completion transport is delegated to
        the Sandhi runtime. Obtain the typed provider via ``resolve_transport_class()``
        (the dynamic ``SandhiHttpxTransportMixin`` variant mixed into every
        ``SandhiOpenAICompatPolicy`` subclass).
        """
        raise NotImplementedError(
            f"{self.name} chat() is owned by the Sandhi typed variant; "
            "use resolve_transport_class() to obtain the typed provider."
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream transport is owned by the Sandhi typed variant (see ``chat``)."""
        if False:  # pragma: no cover - async-generator marker for typing
            yield StreamChunk()
        raise NotImplementedError(
            f"{self.name} stream() is owned by the Sandhi typed variant; "
            "use resolve_transport_class() to obtain the typed provider."
        )

    async def close(self) -> None:
        """Close the underlying httpx client (idempotent)."""
        client = getattr(self, "client", None)
        if client is None or getattr(client, "is_closed", False):
            return
        try:
            await client.aclose()
        except Exception:
            logger.debug("Provider httpx client close failed (already closed?)", exc_info=True)

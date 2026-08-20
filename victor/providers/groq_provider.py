# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Groq model and prompt-budget policy over Sandhi's typed runtime."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from victor.providers.base import CompletionResponse, Message, StreamChunk, ToolDefinition
from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.payload_limiter import ProviderPayloadLimiter, TruncationStrategy
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("groq")
DEFAULT_BASE_URL = _SPEC.base_url
GROQ_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class GroqProvider(SandhiOpenAICompatPolicy):
    """Victor prompt policy for Groq; Sandhi owns transport and wire parsing."""

    CONFIG_KEY = "groq"

    @staticmethod
    def _add_queue_time(response: CompletionResponse) -> CompletionResponse:
        raw = response.raw_response or {}
        queue_time = (raw.get("usage") or {}).get("queue_time")
        if response.usage is not None and isinstance(queue_time, (int, float)):
            response.usage["queue_time_ms"] = int(queue_time * 1000)
        if "x_groq" in raw:
            response.metadata = {**(response.metadata or {}), "x_groq": raw["x_groq"]}
        return response

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        return self._add_queue_time(super()._parse_response(result, model))

    def _completion_from_typed(self, response: Dict[str, Any], model: str) -> CompletionResponse:
        return self._add_queue_time(super()._completion_from_typed(response, model))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Prompt truncation is an agent policy decision, not a provider-wire concern.
        self._payload_limiter = ProviderPayloadLimiter(
            provider_name="groq",
            max_payload_bytes=4 * 1024 * 1024,
            default_strategy=TruncationStrategy.TRUNCATE_TOOL_RESULTS,
        )

    def _limit_payload(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
    ) -> Tuple[List[Message], Optional[List[ToolDefinition]]]:
        ok, warning = self._payload_limiter.check_limit(messages, tools)
        if warning:
            self._provider_logger.logger.warning(warning)
        if ok:
            return messages, tools

        result = self._payload_limiter.truncate_if_needed(messages, tools)
        if result.warning:
            self._provider_logger.logger.warning(result.warning)
        if result.truncated:
            self._provider_logger.logger.info(
                "Truncated Groq payload: removed %s messages, saved %s bytes",
                result.messages_removed,
                result.bytes_saved,
            )
        return result.messages, result.tools

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
        messages, tools = self._limit_payload(messages, tools)
        return await super().chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
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
        messages, tools = self._limit_payload(messages, tools)
        async for chunk in super().stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            yield chunk


__all__ = ["DEFAULT_BASE_URL", "GROQ_MODELS", "GroqProvider"]

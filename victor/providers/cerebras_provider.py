# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Cerebras model and presentation policy over Sandhi's typed runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from victor.providers.base import CompletionResponse, Message, StreamChunk, ToolDefinition
from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("cerebras")
DEFAULT_BASE_URL = _SPEC.base_url
CEREBRAS_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}
THINKING_MODELS = {"qwen-3-235b-a22b-instruct-2507"}
QWEN3_THINKING_PATTERNS = (
    r"^Okay,?\s+(?:the\s+)?(?:user\s+)?(?:is\s+)?(?:asking|wants?|needs?|said)",
    r"^(?:Let\s+me\s+)?(?:think|see|check|analyze|consider)",
    r"^(?:Hmm|Well|So),?\s+",
    r"^(?:First|Now),?\s+(?:I\s+)?(?:need|should|will|'ll)",
    r"^(?:The\s+)?(?:user|question|task|request)\s+(?:is|asks?|wants?)",
    r"^(?:I\s+)?(?:need|should|will|'ll)\s+(?:first|start|begin)",
)


def _is_thinking_model(model: str) -> bool:
    normalized = model.lower()
    return any(name in normalized for name in ("qwen-3", "qwen3"))


def _looks_like_thinking(text: str) -> bool:
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in QWEN3_THINKING_PATTERNS)


def _extract_qwen3_thinking(content: str) -> Tuple[str, str]:
    """Separate a leading inline reasoning paragraph from visible response text."""
    if not content:
        return "", ""
    paragraphs = re.split(r"\n\s*\n", content, maxsplit=1)
    if len(paragraphs) == 2 and _looks_like_thinking(paragraphs[0].strip()):
        return paragraphs[0].strip(), paragraphs[1].strip()
    return "", content


@dataclass
class StreamingThinkingFilter:
    """Bounded presentation filter for models that emit reasoning inline."""

    model: str
    buffer_size: int = 150
    max_thinking_buffer: int = 2000
    _buffer: str = field(default="", init=False)
    _decided: bool = field(default=False, init=False)
    _filtering: bool = field(default=False, init=False)

    def process_chunk(self, content: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not _is_thinking_model(self.model):
            return content, None
        if self._decided and not self._filtering:
            return content, None

        self._buffer += content
        if not self._decided:
            if len(self._buffer) < self.buffer_size and "\n" not in self._buffer:
                return "", None
            self._filtering = _looks_like_thinking(self._buffer.strip())
            self._decided = True
            if not self._filtering:
                visible, self._buffer = self._buffer, ""
                return visible, None

        thinking, visible = _extract_qwen3_thinking(self._buffer)
        if thinking:
            self._buffer = ""
            self._filtering = False
            return visible, {"reasoning_content": thinking}
        if len(self._buffer) > self.max_thinking_buffer:
            visible, self._buffer = self._buffer, ""
            self._filtering = False
            return visible, None
        return "", None

    def finalize(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        content, self._buffer = self._buffer, ""
        if not content:
            return "", None
        thinking, visible = _extract_qwen3_thinking(content)
        if thinking:
            return visible, {"reasoning_content": thinking}
        return content, None


def _with_cerebras_metadata(response: CompletionResponse, model: str) -> CompletionResponse:
    raw = response.raw_response or {}
    total_time = (raw.get("time_info") or {}).get("total_time")
    if response.usage is not None and isinstance(total_time, (int, float)):
        response.usage["total_time_ms"] = int(total_time * 1000)
    if response.content and _is_thinking_model(model):
        thinking, visible = _extract_qwen3_thinking(response.content)
        if thinking:
            response.content = visible
            response.metadata = {**(response.metadata or {}), "reasoning_content": thinking}
    return response


class CerebrasProvider(SandhiOpenAICompatPolicy):
    """Victor model/UX policy for Cerebras; Sandhi owns all provider I/O."""

    CONFIG_KEY = "cerebras"

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        return _with_cerebras_metadata(super()._parse_response(result, model), model)

    def _completion_from_typed(self, response: Dict[str, Any], model: str) -> CompletionResponse:
        return _with_cerebras_metadata(super()._completion_from_typed(response, model), model)

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
        thinking_filter = StreamingThinkingFilter(model)
        async for chunk in super().stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            content, reasoning = thinking_filter.process_chunk(chunk.content)
            if chunk.is_final:
                trailing, final_reasoning = thinking_filter.finalize()
                content += trailing
                reasoning = final_reasoning or reasoning
            metadata = {**(chunk.metadata or {}), **(reasoning or {})} or None
            if content or chunk.is_final or chunk.tool_calls or metadata:
                yield chunk.model_copy(update={"content": content, "metadata": metadata})


__all__ = [
    "CEREBRAS_MODELS",
    "DEFAULT_BASE_URL",
    "QWEN3_THINKING_PATTERNS",
    "StreamingThinkingFilter",
    "THINKING_MODELS",
    "CerebrasProvider",
    "_extract_qwen3_thinking",
    "_is_thinking_model",
]

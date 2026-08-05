# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek model/repair policy over Sandhi's typed runtime."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.core.json_utils import json_loads
from victor.providers.base import CompletionResponse, Message, StreamChunk, ToolDefinition
from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

logger = logging.getLogger(__name__)
_SPEC = get_openai_compat_provider_spec("deepseek")
DEFAULT_BASE_URL = _SPEC.base_url
DEEPSEEK_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class DeepSeekProvider(SandhiOpenAICompatPolicy):
    """Victor model constraints and malformed-DSML repair for DeepSeek."""

    CONFIG_KEY = "deepseek"
    DEFAULT_TIMEOUT = 120
    RETRY_ATTEMPTS = 5
    MAX_API_TIMEOUT = 120

    def __init__(self, *args: Any, timeout: int = DEFAULT_TIMEOUT, **kwargs: Any) -> None:
        super().__init__(*args, timeout=min(timeout, self.MAX_API_TIMEOUT), **kwargs)

    def _get_provider_params(
        self, model: str, temperature: float, max_tokens: int, **kwargs: Any
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"max_tokens": max_tokens}
        if not self._is_reasoner(model):
            params["temperature"] = temperature
        return params

    @staticmethod
    def _is_reasoner(model: str) -> bool:
        normalized = model.lower()
        return "reasoner" in normalized or "r1" in normalized

    def _filter_tools_for_model(
        self, model: str, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[ToolDefinition]]:
        return None if tools and self._is_reasoner(model) else tools

    def get_context_window(self, model: str) -> int:
        return self.context_window(model)

    _DSML_OPEN = re.compile(r"<｜{1,2}DSML｜{1,2}tool_calls\s*>", re.IGNORECASE)
    _DSML_INVOKE = re.compile(
        r'<｜{1,2}DSML｜{1,2}invoke\s+name="([^"]+)"\s*>(.*?)</｜{0,2}DSML｜{0,2}invoke\s*>',
        re.DOTALL | re.IGNORECASE,
    )
    _DSML_PARAM = re.compile(
        r'<｜{1,2}DSML｜{1,2}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜{0,2}DSML｜{0,2}parameter\s*>',
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def _parse_dsml_tool_calls(cls, content: str) -> Optional[List[Dict[str, Any]]]:
        if not cls._DSML_OPEN.search(content):
            return None
        calls: List[Dict[str, Any]] = []
        for invoke in cls._DSML_INVOKE.finditer(content):
            arguments: Dict[str, Any] = {}
            for parameter in cls._DSML_PARAM.finditer(invoke.group(2)):
                raw = parameter.group(3).strip()
                if parameter.group(2).lower() == "true":
                    value: Any = raw
                else:
                    try:
                        value = json_loads(raw)
                    except Exception:
                        value = raw
                arguments[parameter.group(1).strip()] = value
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": invoke.group(1).strip(),
                    "arguments": arguments,
                }
            )
        return calls or None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        return self._repair_dsml(super()._parse_response(result, model))

    def _completion_from_typed(self, response: Dict[str, Any], model: str) -> CompletionResponse:
        return self._repair_dsml(super()._completion_from_typed(response, model))

    def _repair_dsml(self, response: CompletionResponse) -> CompletionResponse:
        if response.tool_calls:
            return response
        calls = self._parse_dsml_tool_calls(response.content or "")
        if not calls:
            return response
        logger.debug("deepseek: recovered %d tool call(s) from DSML content", len(calls))
        return response.model_copy(
            update={"content": "", "tool_calls": calls, "stop_reason": "tool_calls"}
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
        buffered: List[StreamChunk] = []
        async for chunk in super().stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            buffered.append(chunk)
        if any(chunk.tool_calls for chunk in buffered):
            for chunk in buffered:
                yield chunk
            return
        calls = self._parse_dsml_tool_calls("".join(chunk.content for chunk in buffered))
        if calls:
            yield StreamChunk(content="", tool_calls=calls, stop_reason="tool_calls", is_final=True)
            return
        for chunk in buffered:
            yield chunk


__all__ = ["DEEPSEEK_MODELS", "DEFAULT_BASE_URL", "DeepSeekProvider"]

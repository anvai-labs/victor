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

"""Hugging Face Inference API provider.

Hugging Face provides access to thousands of models through their
Inference API and Inference Endpoints.

Free Tier (Serverless):
- Rate limited but free access to popular models
- No GPU allocation guarantee
- Good for experimentation

Pro Features:
- Inference Endpoints (dedicated GPU)
- Higher rate limits
- Private model hosting

Features:
- Access to 1000s of open models
- Text generation, chat, embeddings
- OpenAI-compatible chat API (for supported models)
- Tool calling support (model dependent)

References:
- https://huggingface.co/docs/api-inference/
- https://huggingface.co/docs/inference-endpoints/
"""

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)
from victor.providers.logging import ProviderLogger

DEFAULT_BASE_URL = "https://api-inference.huggingface.co"

# Popular models available on HF Inference API
HUGGINGFACE_MODELS = {
    # Chat/Instruct models with OpenAI-compatible API
    "meta-llama/Llama-3.3-70B-Instruct": {
        "description": "Llama 3.3 70B - High quality open model",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
        "api_type": "chat",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "description": "Llama 3.1 70B - Balanced performance",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
        "api_type": "chat",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "description": "Llama 3.1 8B - Fast and efficient",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
        "api_type": "chat",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "description": "Mistral 7B - Efficient open model",
        "context_window": 32768,
        "max_output": 4096,
        "supports_tools": False,
        "api_type": "chat",
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "description": "Mixtral 8x7B MoE - Strong performance",
        "context_window": 32768,
        "max_output": 4096,
        "supports_tools": False,
        "api_type": "chat",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "description": "Qwen 2.5 72B - Top tier open model",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "api_type": "chat",
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "description": "Qwen 2.5 Coder 32B - Code specialized",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "api_type": "chat",
    },
    "microsoft/Phi-3-medium-128k-instruct": {
        "description": "Phi-3 Medium - Microsoft SLM",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": False,
        "api_type": "chat",
    },
    "google/gemma-2-27b-it": {
        "description": "Gemma 2 27B - Google's open model",
        "context_window": 8192,
        "max_output": 4096,
        "supports_tools": False,
        "api_type": "chat",
    },
    "deepseek-ai/DeepSeek-V2.5": {
        "description": "DeepSeek V2.5 - Strong MoE model",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "api_type": "chat",
    },
}


class HuggingFaceProvider(BaseProvider):
    """Provider for Hugging Face Inference API.

    Features:
    - Access to 1000s of open models
    - Free tier with rate limits
    - OpenAI-compatible chat API
    - Streaming support
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Hugging Face provider.

        Args:
            api_key: HF API token (or set HF_TOKEN / HUGGINGFACE_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("huggingface", __name__)

        # For backward compatibility, support both HF_TOKEN and HUGGINGFACE_API_KEY
        # If no explicit key and HF_TOKEN is not set, check HUGGINGFACE_API_KEY
        effective_api_key = api_key
        if effective_api_key is None:
            env_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
            # Convert empty string to None so resolver knows to check other sources
            effective_api_key = env_key if env_key else None

        # Resolve API key using unified resolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("huggingface", explicit_key=effective_api_key)

        # Log API key resolution
        self._provider_logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            # Raise detailed error with actionable suggestions
            raise APIKeyNotFoundError(
                provider="huggingface",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        self._api_key = key_result.key

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="hf",  # Will be set on chat()
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return "huggingface"

    def supports_tools(self) -> bool:
        return True  # Model dependent

    def supports_streaming(self) -> bool:
        return True

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
        """Send chat completion request to Hugging Face.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional HuggingFace parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderAuthError: If authentication fails
            ProviderRateLimitError: If rate limit is exceeded
            ProviderError: For other errors
        """
        # Use structured logging context manager
        with self._provider_logger.log_api_call(
            endpoint=f"/models/{model}/v1/chat/completions",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ) as log_success:
            try:
                # Use OpenAI-compatible Messages API
                payload = self._build_request_payload(
                    messages, model, temperature, max_tokens, tools, False, **kwargs
                )

                # HF uses model name in URL path
                url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"

                response = await self._execute_with_circuit_breaker(
                    self.client.post, url, json=payload
                )
                response.raise_for_status()

                parsed = self._parse_response(response.json(), model)

                # Log success with usage info
                tokens = parsed.usage.get("total_tokens") if parsed.usage else None
                log_success(tokens=tokens)

                return parsed

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"Hugging Face request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                error_body = e.response.text[:500] if e.response.text else ""

                # Convert to specific provider error types based on status code
                if e.response.status_code == 401:
                    raise ProviderAuthError(
                        message=f"Authentication failed: {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
                elif e.response.status_code == 429:
                    raise ProviderRateLimitError(
                        message=f"Rate limit exceeded: {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
                # Check for model loading status
                elif e.response.status_code == 503:
                    raise ProviderError(
                        message=f"Hugging Face model is loading. Please retry in a few seconds. {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
                else:
                    raise ProviderError(
                        message=f"Hugging Face HTTP error {e.response.status_code}: {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
            except Exception as e:
                raise self.classify_error(e) from e

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
        """Stream chat completion from Hugging Face."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, True, **kwargs
            )

            url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"

            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                accumulated_tool_calls: List[Dict[str, Any]] = []

                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(
                            content="",
                            tool_calls=(accumulated_tool_calls if accumulated_tool_calls else None),
                            stop_reason="stop",
                            is_final=True,
                        )
                        break

                    try:
                        chunk_data = json_loads(data_str)
                        yield self._parse_stream_chunk(chunk_data, accumulated_tool_calls)
                    except JSONDecodeError:
                        pass

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Hugging Face stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Hugging Face streaming error {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
            ) from e

    def _build_request_payload(
        self, messages, model, temperature, max_tokens, tools, stream, **kwargs
    ) -> Dict[str, Any]:
        """Build OpenAI-compatible request payload."""
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}
            if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                formatted_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": (
                                json_dumps(tc.get("arguments", {}))
                                if isinstance(tc.get("arguments"), dict)
                                else tc.get("arguments", "{}")
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(formatted_msg)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"

        return payload

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse HF response (OpenAI format)."""
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        choice = choices[0]
        message = choice.get("message", {})
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        usage = None
        if usage_data := result.get("usage"):
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=message.get("content", "") or "",
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _normalize_tool_calls(self, tool_calls) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls to Victor format."""
        if not tool_calls:
            return None
        normalized = []
        for call in tool_calls:
            if "function" in call:
                func = call["function"]
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except JSONDecodeError:
                        args = {}
                normalized.append(
                    {
                        "id": call.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                )
        return normalized if normalized else None

    def _parse_stream_chunk(self, chunk_data, accumulated_tool_calls) -> StreamChunk:
        """Parse HF stream chunk (OpenAI format)."""
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func = tc_delta["function"]
                if "name" in func:
                    accumulated_tool_calls[idx]["name"] = func["name"]
                if "arguments" in func:
                    accumulated_tool_calls[idx]["arguments"] += func["arguments"]

        final_tool_calls = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        args = json_loads(args) if isinstance(args, str) else args
                    except JSONDecodeError:
                        args = {}
                    final_tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "arguments": args,
                        }
                    )

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )

    async def close(self) -> None:
        await self.client.aclose()

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

"""Replicate provider for running open-source models.

Replicate is a neocloud platform that makes it easy to run open-source
machine learning models with a simple API.

Pricing:
- Pay-per-second billing
- No minimum spend
- Free tier with limited credits for new users

Features:
- Run any public model on Replicate
- OpenAI-compatible API for chat models
- Streaming support
- Model versioning
- Custom deployments

References:
- https://replicate.com/docs
- https://replicate.com/docs/reference/http
"""

import asyncio
import os
import time
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
from victor.providers.resolution import UnifiedApiKeyResolver, APIKeyNotFoundError
from victor.providers.logging import ProviderLogger

DEFAULT_BASE_URL = "https://api.replicate.com/v1"

# Popular models on Replicate
REPLICATE_MODELS = {
    # Meta Llama models
    "meta/llama-3.3-70b-instruct": {
        "description": "Llama 3.3 70B - Latest Meta model",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": False,
    },
    "meta/llama-3.1-405b-instruct": {
        "description": "Llama 3.1 405B - Largest open model",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": False,
    },
    "meta/llama-3-70b-instruct": {
        "description": "Llama 3 70B - Reliable performance",
        "context_window": 8192,
        "max_output": 4096,
        "supports_tools": False,
    },
    # Mistral models
    "mistralai/mixtral-8x7b-instruct-v0.1": {
        "description": "Mixtral 8x7B MoE",
        "context_window": 32768,
        "max_output": 4096,
        "supports_tools": False,
    },
    "mistralai/mistral-7b-instruct-v0.2": {
        "description": "Mistral 7B v0.2",
        "context_window": 32768,
        "max_output": 4096,
        "supports_tools": False,
    },
    # Code models
    "meta/codellama-70b-instruct": {
        "description": "Code Llama 70B - Code generation",
        "context_window": 16384,
        "max_output": 4096,
        "supports_tools": False,
    },
    # DeepSeek
    "deepseek-ai/deepseek-v3": {
        "description": "DeepSeek V3 - 671B MoE",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": False,
    },
}


class ReplicateProvider(BaseProvider):
    """Provider for Replicate API.

    Features:
    - Access to thousands of open models
    - Pay-per-second billing
    - Streaming support
    - Simple API
    """

    DEFAULT_TIMEOUT = 300  # Models may take time to cold start

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Replicate provider.

        Args:
            api_key: Replicate API token (or set REPLICATE_API_TOKEN env var)
            base_url: API endpoint
            timeout: Request timeout
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("replicate", __name__)

        # Resolve API key using unified resolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("replicate", explicit_key=api_key)

        # Log API key resolution
        self._provider_logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            # Raise detailed error with actionable suggestions
            raise APIKeyNotFoundError(
                provider="replicate",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        self._api_key = key_result.key

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="replicate",  # Will be set on chat()
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return "replicate"

    def supports_tools(self) -> bool:
        return False  # Replicate uses raw model API, no tool calling

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
        """Send chat completion request to Replicate."""
        # Use structured logging context manager
        with self._provider_logger.log_api_call(
            endpoint="/predictions",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=False,  # Replicate doesn't support tools
        ) as log_success:
            try:
                # Convert messages to prompt format for Replicate
                prompt = self._messages_to_prompt(messages)

                # Create prediction
                prediction = await self._create_prediction(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # Wait for completion
                prediction = await self._wait_for_prediction(prediction["id"])

                if prediction["status"] == "failed":
                    raise ProviderError(
                        message=f"Replicate prediction failed: {prediction.get('error', 'Unknown error')}",
                        provider=self.name,
                    )

                # Parse output
                output = prediction.get("output", "")
                if isinstance(output, list):
                    output = "".join(output)

                response = CompletionResponse(
                    content=output,
                    role="assistant",
                    model=model,
                    raw_response=prediction,
                )

                # Log success
                log_success(tokens=None)

                return response

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"Replicate request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                error_body = e.response.text[:500] if e.response.text else ""

                # Convert to specific error types
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
                else:
                    raise ProviderError(
                        message=f"Replicate HTTP error {e.response.status_code}: {error_body}",
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
        """Stream chat completion from Replicate."""
        try:
            prompt = self._messages_to_prompt(messages)

            # Create prediction with streaming
            prediction = await self._create_prediction(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            # Get stream URL
            stream_url = prediction.get("urls", {}).get("stream")
            if not stream_url:
                # Fallback to polling
                prediction = await self._wait_for_prediction(prediction["id"])
                output = prediction.get("output", "")
                if isinstance(output, list):
                    output = "".join(output)
                yield StreamChunk(
                    content=output,
                    stop_reason="stop",
                    is_final=True,
                )
                return

            # Stream from SSE endpoint
            async with self.client.stream("GET", stream_url) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("event: "):
                        _event_type = line[7:].strip()
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "done":
                            yield StreamChunk(
                                content="",
                                stop_reason="stop",
                                is_final=True,
                            )
                            break

                        yield StreamChunk(
                            content=data_str,
                            is_final=False,
                        )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Replicate stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Replicate streaming error {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
            ) from e

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a prompt string for Replicate models."""
        prompt_parts = []
        system_prompt = ""

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        # Format as Llama-style prompt
        if system_prompt:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        else:
            full_prompt = "<|begin_of_text|>"

        for msg in messages:
            if msg.role == "user":
                full_prompt += (
                    f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
                )
            elif msg.role == "assistant":
                full_prompt += (
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"
                )

        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return full_prompt

    async def _create_prediction(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a prediction on Replicate."""
        # Parse model name (owner/model or owner/model:version)
        if ":" in model:
            model_path, version = model.rsplit(":", 1)
        else:
            # Get latest version
            model_path = model
            version = None

        payload = {
            "input": {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_new_tokens": max_tokens,
            },
            "stream": stream,
        }

        if version:
            payload["version"] = version
            url = f"{self.base_url}/predictions"
        else:
            # Use model endpoint for latest version
            url = f"{self.base_url}/models/{model_path}/predictions"

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def _wait_for_prediction(
        self, prediction_id: str, poll_interval: float = 0.5
    ) -> Dict[str, Any]:
        """Wait for a prediction to complete."""
        url = f"{self.base_url}/predictions/{prediction_id}"
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                raise ProviderTimeoutError(
                    message=f"Replicate prediction timed out after {self.timeout}s",
                    provider=self.name,
                )

            response = await self.client.get(url)
            response.raise_for_status()
            prediction = response.json()

            status = prediction.get("status")
            if status in ("succeeded", "failed", "canceled"):
                return prediction

            await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        await self.client.aclose()

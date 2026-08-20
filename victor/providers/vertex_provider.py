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

"""Google Cloud Vertex AI provider for enterprise model access.

Vertex AI provides enterprise-grade access to Google's AI models with
features like VPC-SC, CMEK encryption, and SLA guarantees.

Authentication:
- Service account JSON key (GOOGLE_APPLICATION_CREDENTIALS env var)
- Application Default Credentials (ADC) via gcloud auth
- Workload Identity for GKE deployments

Features:
- Access to Gemini models (1.5 Pro, 1.5 Flash, 2.0)
- Enterprise security controls (VPC-SC, CMEK)
- Batch prediction support
- Model tuning capabilities
- Native tool/function calling

References:
- https://cloud.google.com/vertex-ai/generative-ai/docs
- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
"""

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    CacheCostModel,
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
from victor.providers.logging import ProviderLogger

# Vertex AI models available
VERTEX_MODELS = {
    "gemini-1.5-pro": {
        "description": "Gemini 1.5 Pro - Best for complex tasks",
        "context_window": 2097152,
        "max_output": 8192,
        "supports_tools": True,
    },
    "gemini-1.5-flash": {
        "description": "Gemini 1.5 Flash - Fast and efficient",
        "context_window": 1048576,
        "max_output": 8192,
        "supports_tools": True,
    },
    "gemini-2.0-flash-exp": {
        "description": "Gemini 2.0 Flash - Latest experimental",
        "context_window": 1048576,
        "max_output": 8192,
        "supports_tools": True,
    },
    "gemini-1.5-pro-002": {
        "description": "Gemini 1.5 Pro v002 - Stable version",
        "context_window": 2097152,
        "max_output": 8192,
        "supports_tools": True,
    },
    "gemini-1.5-flash-002": {
        "description": "Gemini 1.5 Flash v002 - Stable version",
        "context_window": 1048576,
        "max_output": 8192,
        "supports_tools": True,
    },
}


class VertexAIProvider(BaseProvider):
    """Provider for Google Cloud Vertex AI.

    Features:
    - Enterprise-grade Gemini model access
    - VPC-SC and CMEK encryption support
    - Multiple authentication methods
    - Native tool calling support
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Vertex AI provider.

        Args:
            project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
            location: GCP region (default: us-central1)
            api_key: Optional API key (alternative to service account auth)
            timeout: Request timeout
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("vertex", __name__)

        # Resolution order: parameter → env var → keyring → warning
        resolved_key = api_key or os.environ.get("VERTEX_API_KEY", "")
        key_source = None

        if resolved_key:
            key_source = "explicit parameter or VERTEX_API_KEY environment variable"
        else:
            try:
                from victor.config.api_keys import get_api_key

                resolved_key = get_api_key("vertex") or get_api_key("gcp") or ""
                if resolved_key:
                    key_source = "keyring (victor.config.api_keys)"
            except ImportError:
                pass

        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location
        self._api_key = resolved_key
        self._access_token: Optional[str] = None
        self._non_interactive = non_interactive

        # Determine auth method for logging
        if not self._project_id:
            self._provider_logger.logger.warning(
                "Vertex AI project ID not provided. Set GOOGLE_CLOUD_PROJECT environment variable."
            )

        auth_method = (
            "API key" if self._api_key else "Google Cloud ADC (Application Default Credentials)"
        )

        # Build base URL for Vertex AI
        base_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{self._project_id}/locations/{location}/publishers/google/models"

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="gemini-1.5-pro",  # Default model
            key_source=key_source or auth_method,
            non_interactive=non_interactive or False,
            config={
                "project_id": self._project_id,
                "location": location,
                "auth_method": auth_method,
                "timeout": timeout,
                **kwargs,
            },
        )

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

        # Initialize client - auth will be added per-request
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"Content-Type": "application/json"},
        )

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Vertex AI requests.

        Uses Application Default Credentials (ADC) or API key.
        """
        headers = {"Content-Type": "application/json"}

        if self._api_key:
            # Use API key if provided
            headers["x-goog-api-key"] = self._api_key
        else:
            # Try to get access token from ADC
            try:
                # Try using google-auth library if available
                import google.auth
                import google.auth.transport.requests

                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                request = google.auth.transport.requests.Request()
                credentials.refresh(request)
                headers["Authorization"] = f"Bearer {credentials.token}"
            except ImportError:
                # Fallback: try gcloud CLI
                import subprocess

                try:
                    result = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        token = result.stdout.strip()
                        headers["Authorization"] = f"Bearer {token}"
                    else:
                        self._provider_logger.logger.warning(
                            "Failed to get access token from gcloud CLI"
                        )
                except Exception as e:
                    self._provider_logger.logger.warning(f"Failed to get access token: {e}")

        return headers

    @property
    def name(self) -> str:
        return "vertex"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        """Vertex AI (Gemini) supports context caching (75-90% discount)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        """Vertex AI reuses KV cache for matching prompt prefixes."""
        return True

    def cache_cost_model(self) -> CacheCostModel:
        """Characterized API caching (FEP-0011): 75-90% discount (Gemini)."""
        return CacheCostModel(
            supported=True,
            read_discount=0.825,
            write_overhead=1.0,
            ttl_seconds=0.0,
            min_prefix_tokens=32768,
            prefix_granularity="token",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import VERTEX, VERTEX_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(VERTEX, target, VERTEX_DEFAULT)

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
        """Send chat completion request to Vertex AI."""
        with self._provider_logger.log_api_call(
            endpoint=f"/{model}:generateContent",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ) as log_success:
            try:
                headers = await self._get_auth_headers()
                payload = self._build_request_payload(
                    messages, model, temperature, max_tokens, tools, **kwargs
                )

                url = f"{self.base_url}/{model}:generateContent"
                response = await self._execute_with_circuit_breaker(
                    self.client.post, url, json=payload, headers=headers
                )
                response.raise_for_status()

                result = self._parse_response(response.json(), model)

                # Log success with usage info
                tokens = result.usage.get("total_tokens") if result.usage else None
                log_success(tokens=tokens)

                return result

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"Vertex AI request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                error_body = e.response.text[:500] if e.response.text else ""

                # Convert specific HTTP status codes to appropriate error types
                status_code = e.response.status_code
                if status_code in (401, 403):
                    raise ProviderAuthError(
                        message=f"Vertex AI authentication failed: {error_body}",
                        provider=self.name,
                        status_code=status_code,
                    ) from e
                elif status_code == 429:
                    raise ProviderRateLimitError(
                        message=f"Vertex AI rate limit exceeded: {error_body}",
                        provider=self.name,
                        status_code=status_code,
                    ) from e
                else:
                    raise ProviderError(
                        message=f"Vertex AI HTTP error {status_code}: {error_body}",
                        provider=self.name,
                        status_code=status_code,
                    ) from e
            except ProviderError:
                # Re-raise already-converted provider errors
                raise
            except Exception as e:
                raise ProviderError(
                    message=f"Vertex AI request failed: {str(e)}",
                    provider=self.name,
                ) from e

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
        """Stream chat completion from Vertex AI."""
        try:
            headers = await self._get_auth_headers()
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, **kwargs
            )

            url = f"{self.base_url}/{model}:streamGenerateContent?alt=sse"

            self._provider_logger.logger.debug(
                f"Vertex AI streaming request: model={model}, msgs={len(messages)}, tools={tools is not None}"
            )

            async with self.client.stream("POST", url, json=payload, headers=headers) as response:
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
                        self._provider_logger.logger.warning(
                            f"Vertex AI JSON decode error on line: {line[:100]}"
                        )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Vertex AI stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_body = e.response.text[:500] if e.response.text else ""

            # Convert specific HTTP status codes to appropriate error types
            if status_code in (401, 403):
                raise ProviderAuthError(
                    message=f"Vertex AI authentication failed: {error_body}",
                    provider=self.name,
                    status_code=status_code,
                ) from e
            elif status_code == 429:
                raise ProviderRateLimitError(
                    message=f"Vertex AI rate limit exceeded: {error_body}",
                    provider=self.name,
                    status_code=status_code,
                ) from e
            else:
                raise ProviderError(
                    message=f"Vertex AI streaming error {status_code}: {error_body}",
                    provider=self.name,
                    status_code=status_code,
                ) from e
        except ProviderError:
            # Re-raise already-converted provider errors
            raise
        except Exception as e:
            raise ProviderError(
                message=f"Vertex AI stream error: {str(e)}",
                provider=self.name,
            ) from e

    def _build_request_payload(
        self, messages, model, temperature, max_tokens, tools, **kwargs
    ) -> Dict[str, Any]:
        """Build Vertex AI request payload (Gemini format)."""
        # Convert messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc.get("name", ""),
                                    "args": tc.get("arguments", {}),
                                }
                            }
                        )
                contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool":
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": getattr(msg, "tool_name", "unknown"),
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in tools
                    ]
                }
            ]

        return payload

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse Vertex AI response."""
        candidates = result.get("candidates", [])
        if not candidates:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_content = ""
        tool_calls = []

        for part in content_parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    {
                        "id": fc.get("name", ""),
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args", {}),
                    }
                )

        usage = None
        if usage_data := result.get("usageMetadata"):
            usage = {
                "prompt_tokens": usage_data.get("promptTokenCount", 0),
                "completion_tokens": usage_data.get("candidatesTokenCount", 0),
                "total_tokens": usage_data.get("totalTokenCount", 0),
            }

        return CompletionResponse(
            content=text_content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=candidate.get("finishReason", "").lower(),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _parse_stream_chunk(self, chunk_data, accumulated_tool_calls) -> StreamChunk:
        """Parse Vertex AI stream chunk."""
        candidates = chunk_data.get("candidates", [])
        if not candidates:
            return StreamChunk(content="", is_final=False)

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        finish_reason = candidate.get("finishReason")

        text_content = ""
        for part in content_parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                accumulated_tool_calls.append(
                    {
                        "id": fc.get("name", ""),
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args", {}),
                    }
                )

        final_tool_calls = None
        if finish_reason and accumulated_tool_calls:
            final_tool_calls = accumulated_tool_calls.copy()

        return StreamChunk(
            content=text_content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason.lower() if finish_reason else None,
            is_final=finish_reason is not None,
        )

    async def close(self) -> None:
        await self.client.aclose()

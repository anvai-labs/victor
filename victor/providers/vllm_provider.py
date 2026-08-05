from __future__ import annotations

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

"""vLLM high-throughput inference server provider.

vLLM provides high-throughput LLM serving with PagedAttention for
efficient memory management. Exposes OpenAI-compatible API endpoints.

Usage:
    Start vLLM server:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-Coder-7B-Instruct \
            --port 8000 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes

    Connect with Victor:
        victor chat --provider vllm --model Qwen/Qwen2.5-Coder-7B-Instruct
        victor chat --provider vllm --endpoint http://remote-server:8000

Top tool-enabled coding models for vLLM (fp16/q8):
    1. Qwen/Qwen2.5-Coder-7B-Instruct (14GB fp16, 7GB q8)
    2. Qwen/Qwen2.5-Coder-14B-Instruct (28GB fp16, 14GB q8)
    3. deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (32GB fp16)
    4. codellama/CodeLlama-34b-Instruct-hf (68GB fp16)
    5. mistralai/Codestral-22B-v0.1 (44GB fp16)
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    CacheCostModel,
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderConnectionError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.logging import ProviderLogger

logger = logging.getLogger(__name__)

# Default vLLM endpoints to try
DEFAULT_VLLM_URLS = [
    "http://localhost:8000/v1",
    "http://127.0.0.1:8000/v1",
]

# Models that support tool calling natively
TOOL_CAPABLE_MODELS = [
    "qwen2.5-coder",
    "qwen2.5-instruct",
    "qwen3-coder",
    "llama-3.1",
    "llama-3.3",
    "deepseek-coder",
    "codestral",
    "hermes",
    "-tools",
    "-instruct",
]

# Models that may output thinking tags
THINKING_TAG_MODELS = [
    "qwen3",
    "deepseek-r1",
    "deepseek-reasoner",
]


def _model_supports_tools(model: str) -> bool:
    """Check if model likely supports tool calling.

    Args:
        model: Model name/path

    Returns:
        True if model likely supports tools
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_MODELS)


def _model_uses_thinking_tags(model: str) -> bool:
    """Check if model outputs thinking tags.

    Args:
        model: Model name/path

    Returns:
        True if model uses thinking tags
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in THINKING_TAG_MODELS)


class VLLMProvider(BaseProvider):
    """Provider for vLLM high-throughput inference server.

    Features:
        - OpenAI-compatible API
        - PagedAttention for efficient memory
        - High-throughput batch inference
        - Tool calling with --enable-auto-tool-choice flag
        - Multiple quantization support (fp16, awq, gptq)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",  # vLLM doesn't require auth
        timeout: int = 300,  # Longer timeout for large models
        max_retries: int = 2,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize vLLM provider.

        Args:
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            api_key: API key (vLLM typically doesn't require one)
            timeout: Request timeout in seconds (default: 300 for large models)
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("vllm", __name__)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.base_url = base_url or DEFAULT_VLLM_URLS[0]
        self.timeout = timeout

        # Remove trailing /v1 if present (we'll add it in requests)
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={"Content-Type": "application/json"},
        )
        self._available_model: Optional[str] = None

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model=self._available_model or "unknown",
            key_source="None (vLLM is a local server)",
            non_interactive=non_interactive or False,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

    @classmethod
    async def create(cls, **kwargs: Any) -> "VLLMProvider":
        """Factory method to create and initialize provider.

        Args:
            **kwargs: Provider configuration

        Returns:
            Initialized VLLMProvider instance

        Raises:
            ProviderConnectionError: If server is not reachable
        """
        provider = cls(**kwargs)

        # Try to connect and verify server is running
        base_url = kwargs.get("base_url")
        urls_to_try = [base_url] if base_url else DEFAULT_VLLM_URLS

        for url in urls_to_try:
            if url.endswith("/v1"):
                url = url[:-3]
            try:
                response = await provider.client.get(f"{url}/v1/models", timeout=10.0)
                if response.status_code == 200:
                    provider.base_url = url
                    data = response.json()
                    if data.get("data"):
                        provider._available_model = data["data"][0].get("id")
                    provider._provider_logger.logger.info(f"Connected to vLLM server at {url}")
                    return provider
            except Exception as e:
                provider._provider_logger.logger.debug(f"vLLM not available at {url}: {e}")
                continue

        raise ProviderConnectionError(
            message="Cannot connect to vLLM server",
            provider="vllm",
            details={
                "tried_urls": urls_to_try,
                "suggestion": (
                    "Start vLLM server with:\n"
                    "  python -m vllm.entrypoints.openai.api_server \\\n"
                    "    --model Qwen/Qwen2.5-Coder-7B-Instruct \\\n"
                    "    --port 8000"
                ),
            },
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "vllm"

    def supports_tools(self) -> bool:
        """vLLM supports tools with --enable-auto-tool-choice flag."""
        return True

    def supports_streaming(self) -> bool:
        """vLLM supports streaming."""
        return True

    def supports_prompt_caching(self) -> bool:
        """vLLM has no API-level prompt caching (no billing discount)."""
        return False

    def supports_kv_prefix_caching(self) -> bool:
        """vLLM supports Automatic Prefix Caching (APC) for KV cache reuse."""
        return True

    def kv_cache_cost_model(self) -> CacheCostModel:
        """KV-prefix caching (FEP-0011): latency-only, no billing discount."""
        return CacheCostModel(
            supported=True,
            read_discount=0.0,
            ttl_seconds=0.0,
            prefix_granularity="system_block",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import OLLAMA, VLLM_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(OLLAMA, target, VLLM_DEFAULT)

    def get_tool_output_format(self) -> Any:
        """vLLM models expect XML format (trained on Victor outputs).

        vLLM models parse <TOOL_OUTPUT> tags in responses and have been
        trained on Victor's historical XML format. Keeping this ensures
        optimal tool result cognition.
        """
        from victor.agent.format_strategies import XML_FORMAT

        return XML_FORMAT

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models loaded in vLLM server.

        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get(f"{self.base_url}/v1/models", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except Exception as e:
            self._provider_logger.logger.warning(f"Failed to list vLLM models: {e}")
            return []

    async def check_health(self) -> bool:
        """Check if vLLM server is healthy.

        Returns:
            True if server is healthy
        """
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

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

        This policy shell stays concrete for discovery/capability use; completion
        transport is delegated to the Sandhi runtime. Obtain the typed provider
        via ``resolve_transport_class()`` (e.g. ``SandhiVLLMProvider``).
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
        """Close the HTTP client."""
        await self.client.aclose()

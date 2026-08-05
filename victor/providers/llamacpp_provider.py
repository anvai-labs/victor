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

"""llama.cpp server provider for CPU-friendly local inference.

llama.cpp provides efficient CPU inference with GGUF quantized models.
Exposes an OpenAI-compatible API endpoint.

Usage:
    Start llama.cpp server:
        # Using llama-server (recommended)
        llama-server -m model.gguf --port 8080

        # Or using llama-cpp-python
        python -m llama_cpp.server --model model.gguf --port 8080

    Connect with Victor:
        victor chat --provider llamacpp --model default
        victor chat --provider llamacpp --endpoint http://localhost:8080

Recommended GGUF models for coding (Q4_K_M quantization):
    1. qwen2.5-coder-7b-instruct.Q4_K_M.gguf (4.4GB)
    2. qwen2.5-coder-3b-instruct.Q4_K_M.gguf (2.0GB)
    3. codellama-7b-instruct.Q4_K_M.gguf (4.2GB)
    4. deepseek-coder-6.7b-instruct.Q4_K_M.gguf (4.0GB)
    5. starcoder2-7b.Q4_K_M.gguf (4.3GB)

Download models from: https://huggingface.co/models?sort=trending&search=gguf
"""

from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CacheCostModel,
    CompletionResponse,
    Message,
    ProviderConnectionError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.logging import ProviderLogger

# Default llama.cpp server endpoints
DEFAULT_LLAMACPP_URLS = [
    "http://localhost:8080/v1",
    "http://127.0.0.1:8080/v1",
    "http://localhost:8000/v1",  # Alternative port
]

# Models that support tool calling (instruction-tuned)
TOOL_CAPABLE_PATTERNS = [
    "instruct",
    "chat",
    "coder",
    "-it",
    "qwen",
    "llama-3",
    "mistral",
    "deepseek",
]


def _model_supports_tools(model: str) -> bool:
    """Check if model likely supports tool calling.

    Args:
        model: Model name/path

    Returns:
        True if model likely supports tools
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_PATTERNS)


class LlamaCppProvider(BaseProvider):
    """Provider for llama.cpp server (CPU-optimized inference).

    Features:
        - OpenAI-compatible API
        - GGUF quantized model support
        - Efficient CPU inference
        - Low memory footprint with quantization
        - Cross-platform (macOS, Linux, Windows)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "not-needed",  # llama.cpp doesn't require auth
        timeout: int = 300,  # Longer timeout for CPU inference
        max_retries: int = 2,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize llama.cpp provider.

        Args:
            base_url: llama.cpp server URL (default: http://localhost:8080/v1)
            api_key: API key (not required for llama.cpp)
            timeout: Request timeout in seconds (default: 300 for CPU)
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("llamacpp", __name__)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.base_url = base_url or DEFAULT_LLAMACPP_URLS[0]
        self.timeout = timeout

        # Remove trailing /v1 if present (we'll add it in requests)
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={"Content-Type": "application/json"},
        )
        self._loaded_model: Optional[str] = None

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="llama.cpp",  # Will be updated on connection
            key_source=None,  # No API key for local server
            non_interactive=non_interactive or False,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

    @classmethod
    async def create(cls, **kwargs: Any) -> "LlamaCppProvider":
        """Factory method to create and initialize provider.

        Args:
            **kwargs: Provider configuration

        Returns:
            Initialized LlamaCppProvider instance

        Raises:
            ProviderConnectionError: If server is not reachable
        """
        provider = cls(**kwargs)

        # Try to connect and verify server is running
        base_url = kwargs.get("base_url")
        urls_to_try = [base_url] if base_url else DEFAULT_LLAMACPP_URLS

        for url in urls_to_try:
            if url.endswith("/v1"):
                url = url[:-3]
            try:
                # llama.cpp uses /health endpoint
                response = await provider.client.get(f"{url}/health", timeout=10.0)
                if response.status_code == 200:
                    provider.base_url = url
                    # Try to get loaded model info
                    try:
                        models_resp = await provider.client.get(f"{url}/v1/models", timeout=5.0)
                        if models_resp.status_code == 200:
                            data = models_resp.json()
                            if data.get("data"):
                                provider._loaded_model = data["data"][0].get("id", "default")
                    except Exception:
                        provider._loaded_model = "default"
                    provider._provider_logger.logger.info(f"Connected to llama.cpp server at {url}")
                    return provider
            except Exception as e:
                provider._provider_logger.logger.debug(f"llama.cpp not available at {url}: {e}")
                continue

        raise ProviderConnectionError(
            message="Cannot connect to llama.cpp server",
            provider="llamacpp",
            details={
                "tried_urls": urls_to_try,
                "suggestion": (
                    "Start llama.cpp server with:\n"
                    "  llama-server -m model.gguf --port 8080\n\n"
                    "Or using llama-cpp-python:\n"
                    "  pip install llama-cpp-python[server]\n"
                    "  python -m llama_cpp.server --model model.gguf"
                ),
            },
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "llamacpp"

    def supports_tools(self) -> bool:
        """llama.cpp supports tools with compatible models."""
        return True

    def supports_streaming(self) -> bool:
        """llama.cpp supports streaming."""
        return True

    def supports_prompt_caching(self) -> bool:
        """llama.cpp has no API-level prompt caching (no billing discount)."""
        return False

    def supports_kv_prefix_caching(self) -> bool:
        """llama.cpp natively reuses KV cache for matching prefixes."""
        return True

    def kv_cache_cost_model(self) -> CacheCostModel:
        """llama.cpp KV caching (FEP-0011).

        Latency-only (no billing discount), local-engine TTL is effectively the
        session lifetime, stable system prefix is the reuse unit.
        NOTE: supports_prompt_caching() stays False, so cache_cost_model()
        (API-level) correctly remains unsupported via the default.
        """
        return CacheCostModel(
            supported=True,
            read_discount=0.0,  # KV cache saves latency, not cost
            ttl_seconds=0.0,  # 0 = indefinite while the session/engine lives
            prefix_granularity="system_block",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import OLLAMA, LLAMACPP_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(OLLAMA, target, LLAMACPP_DEFAULT)

    def get_tool_output_format(self) -> Any:
        """llama.cpp models parse XML tags in responses.

        llama.cpp models have been trained on Victor's XML format
        with <TOOL_OUTPUT> tags. This format ensures optimal tool
        result parsing and cognition.
        """
        from victor.agent.format_strategies import XML_FORMAT

        return XML_FORMAT

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models loaded in llama.cpp server.

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
            self._provider_logger.logger.warning(f"Failed to list llama.cpp models: {e}")
            return []

    async def check_health(self) -> Dict[str, Any]:
        """Check llama.cpp server health.

        Returns:
            Health status dictionary
        """
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            if response.status_code == 200:
                try:
                    return response.json()
                except Exception:
                    return {"status": "ok"}
            return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_server_props(self) -> Dict[str, Any]:
        """Get llama.cpp server properties.

        Returns:
            Server properties including model info
        """
        try:
            response = await self.client.get(f"{self.base_url}/props", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Chat transport is owned by the Sandhi typed variant.

        This policy shell stays concrete for discovery/capability use; completion
        transport is delegated to the Sandhi runtime. Obtain the typed provider
        via ``resolve_transport_class()`` (e.g. ``SandhiLlamaCppProvider``).
        """
        raise NotImplementedError(
            f"{self.name} chat() is owned by the Sandhi typed variant; "
            "use resolve_transport_class() to obtain the typed provider."
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
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

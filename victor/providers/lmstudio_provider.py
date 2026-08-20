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

"""LMStudio provider implementation for local model inference.

LMStudio provides an OpenAI-compatible API but requires specialized handling:
- Endpoint probing via /v1/models (not /api/tags like Ollama)
- Tiered URL fallback for multiple LMStudio hosts
- Longer timeouts (300s) for local model inference
- Direct httpx client (not AsyncOpenAI SDK)
- Thinking tag extraction for Qwen3/DeepSeek-R1 models
- Model-aware tool calling capability detection

References:
- https://lmstudio.ai/docs/api/openai-api
- https://lmstudio.ai/docs/advanced/tool-use
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

from victor.providers.base import (
    CacheCostModel,
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.logging import ProviderLogger
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

logger = logging.getLogger(__name__)


# Models that output <think>...</think> tags
THINKING_TAG_MODELS = [
    "qwen3",
    "deepseek-r1",
    "deepseek-reasoner",
]

# Models with native tool calling support
TOOL_CAPABLE_MODELS = [
    "-tools",  # Suffix pattern (e.g., qwen3-coder-tools-30b)
    "qwen2.5-coder",
    "qwen3-coder",
    "llama3.1-tools",
    "llama3.3-tools",
    "deepseek-coder-tools",
    "deepseek-r1-tools",
    "deepseek-coder-v2-tools",
]


def _model_uses_thinking_tags(model: str) -> bool:
    """Check if a model outputs thinking tags."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in THINKING_TAG_MODELS)


def _model_supports_tools(model: str) -> bool:
    """Check if a model supports native tool calling."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_MODELS)


class LMStudioProvider(BaseProvider):
    """Provider for LMStudio local model server (OpenAI-compatible API).

    Features:
    - Tiered URL selection with health probing
    - Async factory for non-blocking initialization
    - Native tool calling support for hammer-badge models
    - JSON fallback parsing for other models
    - Thinking tag extraction for Qwen3/DeepSeek-R1 models
    - Model-aware capability detection
    """

    # Default timeout matches Ollama (local models need more time)
    DEFAULT_TIMEOUT = 300

    # Default LMStudio port
    DEFAULT_PORT = 1234

    # Initial request timeout (for model loading)
    INITIAL_REQUEST_TIMEOUT = 180  # 3 minutes for first request (model loading)

    def __init__(
        self,
        base_url: Union[str, List[str]] = "http://127.0.0.1:1234",
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str = "lm-studio",
        _skip_discovery: bool = False,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize LMStudio provider.

        Args:
            base_url: LMStudio server URL or list of URLs (first reachable is used)
            timeout: Request timeout (default: 300s for local models)
            api_key: API key (LMStudio default is "lm-studio")
            _skip_discovery: Skip endpoint discovery (for async factory)
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("lmstudio", __name__)
        if _skip_discovery:
            # Used by async factory, base_url is already resolved
            chosen_base = (
                base_url
                if isinstance(base_url, str)
                else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
            )
        else:
            chosen_base = self._select_base_url(base_url, timeout)

        super().__init__(base_url=chosen_base, timeout=timeout, **kwargs)
        self._raw_base_urls = base_url
        self._api_key = api_key
        self._models_available: Optional[bool] = None  # Set during URL discovery
        self._context_window_cache: Dict[str, int] = {}
        self._current_model: Optional[str] = None  # Track current model for capability detection
        self._model_tool_support_cache: Dict[str, bool] = {}  # Cache tool support per model

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="lmstudio-local",  # Will be set on chat()
            key_source="Local server (no API key required)",
            non_interactive=non_interactive if non_interactive is not None else True,
            config={"base_url": chosen_base, "timeout": timeout, **kwargs},
        )

        # Use httpx directly (not AsyncOpenAI SDK) for consistent behavior with Ollama
        self.client = httpx.AsyncClient(
            base_url=f"{chosen_base}/v1",  # OpenAI-compatible /v1 path
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @classmethod
    async def create(
        cls,
        base_url: Union[str, List[str]] = "http://127.0.0.1:1234",
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str = "lm-studio",
        **kwargs: Any,
    ) -> "LMStudioProvider":
        """Async factory to create LMStudioProvider with async endpoint discovery.

        Preferred over __init__ for non-blocking initialization.

        Args:
            base_url: LMStudio server URL or list of URLs
            timeout: Request timeout (default: 300s)
            api_key: API key (default: "lm-studio")
            **kwargs: Additional configuration

        Returns:
            Initialized LMStudioProvider with best available endpoint
        """
        chosen_base, models_available = await cls._select_base_url_async(base_url, timeout)
        instance = cls(
            base_url=chosen_base,
            timeout=timeout,
            api_key=api_key,
            _skip_discovery=True,
            **kwargs,
        )
        instance._models_available = models_available
        return instance

    @property
    def name(self) -> str:
        """Provider name."""
        return "lmstudio"

    def supports_tools(self) -> bool:
        """Check if current model supports tool calling.

        Returns:
            True if current model supports native tool calling
        """
        return self._supports_tools_for_model(self._current_model)

    def _supports_tools_for_model(self, model: Optional[str]) -> bool:
        """Check if specific model supports tool calling (internal).

        Args:
            model: Model name to check. If None, returns True (optimistic).

        Returns:
            True if model supports native tool calling
        """
        if not model:
            return True  # Optimistic default when model unknown

        # Check cache first
        if model in self._model_tool_support_cache:
            return self._model_tool_support_cache[model]

        # Check model name patterns
        supports = _model_supports_tools(model)
        self._model_tool_support_cache[model] = supports

        if not supports:
            self._provider_logger.logger.debug(
                f"LMStudio: Model {model} does not support native tool calling"
            )

        return supports

    def supports_streaming(self) -> bool:
        """LMStudio supports streaming."""
        return True

    def supports_prompt_caching(self) -> bool:
        """LMStudio has no API-level prompt caching (no billing discount)."""
        return False

    def supports_kv_prefix_caching(self) -> bool:
        """LMStudio reuses KV cache for GGUF models with matching prefixes."""
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
        # LMStudio loads various GGUF models; share Ollama's table since
        # the same model files are typically served.
        from victor.providers.context_windows import OLLAMA, LMSTUDIO_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(OLLAMA, target, LMSTUDIO_DEFAULT)

    def model_uses_thinking_tags(self, model: Optional[str] = None) -> bool:
        """Check if model outputs thinking tags.

        Args:
            model: Model name to check. If None, uses current model.

        Returns:
            True if model outputs <think>...</think> tags
        """
        check_model = model or self._current_model
        if not check_model:
            return False
        return _model_uses_thinking_tags(check_model)

    async def check_model_status(self, model: str) -> Dict[str, Any]:
        """Check if a model is loaded and ready.

        Args:
            model: Model name to check

        Returns:
            Status dict with 'loaded', 'loading', 'error' keys
        """
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json()

            models = data.get("data", [])
            model_info = next((m for m in models if m.get("id") == model), None)

            if model_info:
                return {
                    "loaded": True,
                    "loading": False,
                    "model": model,
                    "info": model_info,
                }

            # Model not in list - might need to be loaded
            return {
                "loaded": False,
                "loading": False,
                "model": model,
                "available_models": [m.get("id") for m in models],
            }

        except Exception as e:
            return {
                "loaded": False,
                "loading": False,
                "error": str(e),
                "model": model,
            }

    def get_context_window(self, model: str) -> int:
        """Get context window size using cached discovery or config fallback."""
        cache_key = f"{self.base_url}:{model}"
        if cache_key in self._context_window_cache:
            return self._context_window_cache[cache_key]

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("lmstudio", model)
        self._context_window_cache[cache_key] = limits.context_window
        return limits.context_window

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        """Async capability discovery via /v1/models."""
        cache_key = f"{self.base_url}:{model}"

        context_window: Optional[int] = None
        supports_tools = True
        supports_streaming = True
        raw_response: Optional[Dict[str, Any]] = None

        try:
            resp = await self.client.get("/models")
            resp.raise_for_status()
            raw_response = resp.json()

            models = raw_response.get("data", [])
            match = next(
                (m for m in models if m.get("id") == model),
                models[0] if models else None,
            )

            if match:
                capabilities = match.get("capabilities", {}) or {}
                supports_tools = bool(
                    capabilities.get("functions", True) or capabilities.get("tools", True)
                )
                supports_streaming = bool(
                    capabilities.get("streaming", True) or capabilities.get("stream", True)
                )
                context_window = self._extract_context_window(match)

        except Exception as exc:
            self._provider_logger.logger.warning(
                f"Failed to discover capabilities for {model} on {self.base_url}: {exc}"
            )

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("lmstudio", model)
        resolved_context = context_window or limits.context_window

        self._context_window_cache[cache_key] = resolved_context

        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=resolved_context,
            supports_tools=supports_tools,
            supports_streaming=supports_streaming,
            source="discovered" if context_window else "config",
            raw=raw_response,
        )

    def _extract_context_window(self, model_info: Dict[str, Any]) -> Optional[int]:
        """Extract context window from LMStudio model info."""
        candidates = [
            model_info.get("context_length"),
            model_info.get("max_context_length"),
            model_info.get("max_context_tokens"),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        return None

    def _select_base_url(self, base_url: Union[str, List[str], None], timeout: int) -> str:
        """Pick the first reachable LMStudio endpoint from a tiered list.

        Priority:
        1) LMSTUDIO_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost:1234

        Set LMSTUDIO_ENDPOINTS="http://<your-server>:1234,http://127.0.0.1:1234"
        to prioritize LAN servers.
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("LMSTUDIO_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://127.0.0.1:1234"

        # Only process base_url if no env var candidates
        if not candidates:
            if isinstance(base_url, (list, tuple)):
                candidates = [str(u).strip() for u in base_url if str(u).strip()]
            elif isinstance(base_url, str):
                if "," in base_url:
                    candidates = [u.strip() for u in base_url.split(",") if u.strip()]
                else:
                    candidates = [base_url]
            else:
                candidates = [str(base_url)]

        for url in candidates:
            try:
                with httpx.Client(timeout=httpx.Timeout(2)) as client:
                    # LMStudio uses /v1/models for model listing (OpenAI-compatible)
                    resp = client.get(f"{url}/v1/models")
                    resp.raise_for_status()

                    # Verify we got a valid response with models
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        model_names = [m.get("id", "unknown") for m in models[:3]]
                        self._provider_logger.logger.info(
                            f"LMStudio base URL selected: {url} "
                            f"(models: {', '.join(model_names)}{'...' if len(models) > 3 else ''})"
                        )
                        self._models_available = True
                    else:
                        self._provider_logger.logger.warning(
                            f"LMStudio server at {url} has NO MODELS LOADED. "
                            "Please load a model in LMStudio before using this provider."
                        )
                        self._models_available = False
                    return url
            except Exception as exc:
                self._provider_logger.logger.warning(
                    f"LMStudio endpoint {url} not reachable ({exc}); trying next."
                )

        fallback = (
            base_url
            if isinstance(base_url, str)
            else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
        )
        logger.error(
            f"No LMStudio endpoints reachable from: {candidates}. Falling back to {fallback}"
        )
        return fallback

    @classmethod
    async def _select_base_url_async(
        cls, base_url: Union[str, List[str], None], timeout: int
    ) -> Tuple[str, Optional[bool]]:
        """Async version of _select_base_url for non-blocking endpoint discovery.

        Priority:
        1) LMSTUDIO_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost:1234

        Returns:
            Tuple of (url, models_available) where models_available is:
            - True if models are loaded
            - False if no models loaded
            - None if couldn't determine
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("LMSTUDIO_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://127.0.0.1:1234"

        # Only process base_url if no env var candidates
        if not candidates:
            if isinstance(base_url, (list, tuple)):
                candidates = [str(u).strip() for u in base_url if str(u).strip()]
            elif isinstance(base_url, str):
                if "," in base_url:
                    candidates = [u.strip() for u in base_url.split(",") if u.strip()]
                else:
                    candidates = [base_url]
            else:
                candidates = [str(base_url)]

        for url in candidates:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(2)) as client:
                    # LMStudio uses /v1/models for model listing (OpenAI-compatible)
                    resp = await client.get(f"{url}/v1/models")
                    resp.raise_for_status()

                    # Verify we got a valid response with models
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        model_names = [m.get("id", "unknown") for m in models[:3]]
                        logger.info(
                            f"LMStudio base URL selected (async): {url} "
                            f"(models: {', '.join(model_names)}{'...' if len(models) > 3 else ''})"
                        )
                        return url, True
                    else:
                        logger.warning(
                            f"LMStudio server at {url} has NO MODELS LOADED. "
                            "Please load a model in LMStudio before using this provider."
                        )
                        return url, False
            except Exception as exc:
                logger.warning(f"LMStudio endpoint {url} not reachable ({exc}); trying next.")

        fallback = (
            base_url
            if isinstance(base_url, str)
            else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
        )
        logger.error(
            f"No LMStudio endpoints reachable from: {candidates}. Falling back to {fallback}"
        )
        return fallback, None

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on LMStudio server.

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If request fails
        """
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            result = response.json()
            return result.get("data", [])
        except Exception as e:
            raise ProviderError(
                message=f"LMStudio failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

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
        via ``resolve_transport_class()`` (e.g. ``SandhiLMStudioProvider``).
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
        """Close HTTP client."""
        await self.client.aclose()

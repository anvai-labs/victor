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

"""Ollama provider implementation for local model inference."""

import asyncio
from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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
from victor.providers.ollama_capability_detector import TOOL_SUPPORT_PATTERNS


class OllamaProvider(BaseProvider):
    """Provider for Ollama local model server."""

    def __init__(
        self,
        base_url: Union[str, List[str]] = "http://localhost:11434",
        timeout: int = 300,
        _skip_discovery: bool = False,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL or list/comma-separated URLs.
                     For synchronous init, the first URL is used without verification.
                     Use OllamaProvider.create() for async discovery of reachable endpoint.
            timeout: Request timeout (longer for local models)
            _skip_discovery: Deprecated flag (kept for compatibility).
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        import os

        # Initialize structured logger
        self._provider_logger = ProviderLogger("ollama", __name__)

        # Resolve candidates (logic from _select_base_url but without network I/O)
        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("OLLAMA_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if not candidates:
            if base_url is None:
                candidates = ["http://localhost:11434"]
            elif isinstance(base_url, (list, tuple)):
                candidates = [str(u).strip() for u in base_url if str(u).strip()]
            elif isinstance(base_url, str):
                if "," in base_url:
                    candidates = [u.strip() for u in base_url.split(",") if u.strip()]
                else:
                    candidates = [base_url]
            else:
                candidates = [str(base_url)]

        if not candidates:
            candidates = ["http://localhost:11434"]

        # Pick first candidate (blindly)
        chosen_base = candidates[0]

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="local",  # Ollama runs local models
            key_source=None,  # No API key for local server
            non_interactive=non_interactive or False,
            config={
                "base_url": base_url,
                "timeout": timeout,
                **kwargs,
            },
        )

        super().__init__(base_url=chosen_base, timeout=timeout, **kwargs)
        self._raw_base_urls = base_url
        self._models_without_tools: set = set()  # Cache models that don't support tools
        self._models_without_thinking: set = set()  # Cache models that reject think mode
        self._context_window_cache: Dict[str, int] = {}  # Cache model context windows
        self._base_url_for_client = chosen_base
        self._timeout_for_client = timeout
        self._client_loop_id: int | None = None
        self.client = self._make_client()

    def _make_client(self) -> httpx.AsyncClient:
        """Create a fresh httpx AsyncClient."""
        return httpx.AsyncClient(
            base_url=self._base_url_for_client,
            timeout=httpx.Timeout(self._timeout_for_client),
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Get httpx client, recreating if the event loop has changed.

        Handles run_sync_in_thread creating new event loops — the old client's
        connection pool references the dead loop and must be replaced.
        """
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            return self.client

        if self._client_loop_id != loop_id:
            self.client = self._make_client()
            self._client_loop_id = loop_id

        return self.client

    @classmethod
    async def create(
        cls,
        base_url: Union[str, List[str]] = "http://localhost:11434",
        timeout: int = 300,
        **kwargs: Any,
    ) -> "OllamaProvider":
        """Async factory to create OllamaProvider with async endpoint discovery.

        Preferred over __init__ for non-blocking initialization.

        Args:
            base_url: Ollama server URL or list/comma-separated URLs
            timeout: Request timeout (longer for local models)
            **kwargs: Additional configuration

        Returns:
            Initialized OllamaProvider with best available endpoint
        """
        chosen_base = await cls._select_base_url_async(base_url, timeout)
        return cls(base_url=chosen_base, timeout=timeout, _skip_discovery=True, **kwargs)

    @property
    def name(self) -> str:
        """Provider name."""
        return "ollama"

    def supports_tools(self) -> bool:
        """Ollama supports tool calling for compatible models."""
        return True

    def supports_streaming(self) -> bool:
        """Ollama supports streaming."""
        return True

    def supports_prompt_caching(self) -> bool:
        """Ollama has no API-level prompt caching (no billing discount)."""
        return False

    def supports_kv_prefix_caching(self) -> bool:
        """Ollama reuses KV cache via llama.cpp for matching prefixes."""
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
        from victor.providers.context_windows import OLLAMA, OLLAMA_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(OLLAMA, target, OLLAMA_DEFAULT)

    def get_tool_output_format(self) -> Any:
        """Ollama models trained on XML format - use XML for optimal cognition.

        Ollama models (and other local models) have been trained on Victor's
        historical XML format with <TOOL_OUTPUT> tags and ═══ delimiters.
        Removing this format may hurt model cognition and tool result parsing.
        """
        from victor.agent.format_strategies import XML_FORMAT

        return XML_FORMAT

    def get_context_window(self, model: str) -> int:
        """Get context window size using cached discovery or config fallback."""
        cache_key = f"{self.base_url}:{model}"

        if cache_key in self._context_window_cache:
            return self._context_window_cache[cache_key]

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("ollama", model)
        self._context_window_cache[cache_key] = limits.context_window
        return limits.context_window

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        """Async capability discovery using /api/show."""
        cache_key = f"{self.base_url}:{model}"

        context_window: Optional[int] = None
        supports_tools = True  # Default: Ollama supports tools for capable models
        raw_response: Optional[Dict[str, Any]] = None

        try:
            resp = await self._get_client().post("/api/show", json={"name": model})
            resp.raise_for_status()
            raw_response = resp.json()

            context_window = self._parse_context_window(raw_response)
            template = raw_response.get("template", "") or ""
            supports_tools = self._detect_tool_support(template)

        except Exception as exc:
            self._provider_logger.logger.warning(
                f"Failed to discover capabilities for {model} on {self.base_url}: {exc}"
            )

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("ollama", model)
        resolved_context = context_window or limits.context_window

        self._context_window_cache[cache_key] = resolved_context

        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=resolved_context,
            supports_tools=supports_tools,
            supports_streaming=True,
            source="discovered" if context_window else "config",
            raw=raw_response,
        )

    def _parse_context_window(self, response_data: Dict[str, Any]) -> Optional[int]:
        """Extract context window from Ollama /api/show response."""
        parameters = response_data.get("parameters", "")
        if "num_ctx" in parameters:
            for line in parameters.split("\n"):
                if "num_ctx" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[-1])
                        except (ValueError, IndexError):
                            continue

        model_info = response_data.get("model_info", {})
        for key, value in model_info.items():
            if "context_length" in key.lower():
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue

        return None

    def _detect_tool_support(self, template: str) -> bool:
        """Detect native tool support from model template."""
        if not template:
            return True  # default optimistic

        for pattern in TOOL_SUPPORT_PATTERNS:
            try:
                if pattern and re.search(pattern, template):
                    return True
            except Exception:
                continue

        return False

    def _select_base_url(self, base_url: Union[str, List[str], None], timeout: int) -> str:
        """Pick the first reachable Ollama endpoint from a tiered list.

        Priority:
        1) OLLAMA_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost

        Set OLLAMA_ENDPOINTS="http://<your-server>:11434,http://localhost:11434"
        to prioritize LAN servers.
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("OLLAMA_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://localhost:11434"

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
                with httpx.Client(base_url=url, timeout=httpx.Timeout(2)) as client:
                    resp = client.get("/api/tags")
                    resp.raise_for_status()
                    self._provider_logger.logger.debug(f"Ollama base URL selected: {url}")
                    return url
            except Exception as exc:
                self._provider_logger.logger.warning(
                    f"Ollama endpoint {url} not reachable ({exc}); trying next."
                )

        self._provider_logger.logger.error(
            f"No Ollama endpoints reachable from: {candidates}. Falling back to {base_url}"
        )
        return base_url

    @classmethod
    async def _select_base_url_async(
        cls, base_url: Union[str, List[str], None], timeout: int
    ) -> str:
        """Async version of _select_base_url for non-blocking endpoint discovery.

        Priority:
        1) OLLAMA_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost

        Set OLLAMA_ENDPOINTS="http://<your-server>:11434,http://localhost:11434"
        to prioritize LAN servers.
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("OLLAMA_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://localhost:11434"

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

        # Use module logger since this is a classmethod
        from victor.providers.logging import logger as provider_logger

        for url in candidates:
            try:
                async with httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(2)) as client:
                    resp = await client.get("/api/tags")
                    resp.raise_for_status()
                    provider_logger.debug(f"Ollama base URL selected (async): {url}")
                    return url
            except Exception as exc:
                provider_logger.warning(
                    f"Ollama endpoint {url} not reachable ({exc}); trying next."
                )

        provider_logger.error(
            f"No Ollama endpoints reachable from: {candidates}. Falling back to {base_url}"
        )
        return base_url if isinstance(base_url, str) else str(base_url)

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on Ollama server.

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If request fails
        """
        try:
            response = await self._get_client().get("/api/tags")
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            raise ProviderError(
                message=f"Failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def pull_model(self, model: str) -> AsyncIterator[Dict[str, Any]]:
        """Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Yields:
            Progress updates

        Raises:
            ProviderError: If pull fails
        """
        try:
            payload = {"name": model, "stream": True}

            async with self._get_client().stream("POST", "/api/pull", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            yield json_loads(line)
                        except JSONDecodeError:
                            continue

        except Exception as e:
            raise ProviderError(
                message=f"Failed to pull model: {str(e)}",
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
        via ``resolve_transport_class()`` (e.g. ``SandhiOllamaProvider``).
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

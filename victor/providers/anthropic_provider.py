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

"""Anthropic Claude provider implementation."""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.providers.base import (
    BaseProvider,
    CacheCostModel,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.openai_compat import convert_tools_to_anthropic_format
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)
from victor.providers.logging import ProviderLogger
from victor.providers.oauth_manager import OAuthTokenManager

logger = logging.getLogger(__name__)

# Curated fallback used when live ``/v1/models`` discovery is unavailable
# (offline, auth failure, or the anthropic SDK absent). Context-window facts
# are otherwise resolved via ``context_windows.lookup``; this only enumerates
# the known model ids and human-readable names. Sourced from Anthropic's
# Models overview (platform.claude.com), current as of 2026-07.
_ANTHROPIC_STATIC_MODELS: List[Dict[str, Any]] = [
    {
        "id": "claude-fable-5",
        "name": "Claude Fable 5",
        "description": "Frontier flagship (Mythos-class) for the hardest long-horizon agentic work",
        "context_window": 1_000_000,
        "max_output_tokens": 131_072,
    },
    {
        "id": "claude-opus-4-8",
        "name": "Claude Opus 4.8",
        "description": "Most capable model for complex reasoning and agentic coding",
        "context_window": 1_000_000,
        "max_output_tokens": 131_072,
    },
    {
        "id": "claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "description": "Default model; best combination of speed and intelligence",
        "context_window": 1_000_000,
        "max_output_tokens": 131_072,
    },
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "description": "Previous-generation balanced model",
        "context_window": 1_000_000,
        "max_output_tokens": 65_536,
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "name": "Claude Haiku 4.5",
        "description": "Fastest model with near-frontier intelligence",
        "context_window": 200_000,
        "max_output_tokens": 65_536,
    },
]


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    # Cloud provider timeout
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        auth_mode: str = "api_key",
        oauth_source: str = "victor",
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            auth_mode: Authentication mode — "api_key" (default) or "oauth"
            oauth_source: OAuth token source — "victor" or "claude-code"
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("anthropic", __name__)
        self._oauth_manager: Optional[OAuthTokenManager] = None
        self._auth_mode = auth_mode
        self._sandhi_auth_scheme = "bearer" if auth_mode == "oauth" else "api_key"

        if auth_mode == "oauth":
            self._oauth_manager = OAuthTokenManager("anthropic", token_source=oauth_source)
            cached = self._oauth_manager._load_cached()
            if cached is not None and not cached.is_expired:
                self._api_key = cached.access_token
            else:
                self._api_key = "oauth-pending"

            self._provider_logger.log_provider_init(
                model="claude",
                key_source=f"oauth/{oauth_source}",
                non_interactive=False,
                config={
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "auth_mode": "oauth",
                    **kwargs,
                },
            )
        else:
            # Resolve API key using unified resolver
            resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
            key_result = resolver.get_api_key("anthropic", explicit_key=api_key)

            # Log API key resolution
            self._provider_logger.log_api_key_resolution(key_result)

            if key_result.key is None:
                # Raise detailed error with actionable suggestions
                raise APIKeyNotFoundError(
                    provider="anthropic",
                    sources_attempted=key_result.sources_attempted,
                    non_interactive=key_result.non_interactive,
                )

            self._api_key = key_result.key

            # Log provider initialization
            self._provider_logger.log_provider_init(
                model="claude",  # Will be set on chat()
                key_source=key_result.source_detail,
                non_interactive=key_result.non_interactive,
                config={
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    **kwargs,
                },
            )

        super().__init__(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    async def _ensure_valid_token(self) -> None:
        """Refresh OAuth token if needed. No-op for api_key mode.

        The refreshed token lands on ``self._api_key``, which the Sandhi typed
        handle reads when it is (re)constructed. The policy shell owns no
        provider generation client (TD-0002 deletion gate), so there is no SDK
        client to mutate here.
        """
        if self._oauth_manager is None:
            return
        token = await self._oauth_manager.get_valid_token()
        if token != self._api_key:
            self._api_key = token

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    def supports_tools(self) -> bool:
        """Anthropic supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
        return True

    def supports_vision(self) -> bool:
        """Claude 3+ models support image input."""
        return True

    @staticmethod
    def _serialize_message(msg: "Message") -> Dict[str, Any]:
        """Serialize a Message to Anthropic API format, handling images."""
        if msg.role == "user" and msg.images:
            content: List[Dict[str, Any]] = []
            for data_uri in msg.images:
                # Strip the data URI prefix to get raw base64
                if "," in data_uri:
                    header, b64_data = data_uri.split(",", 1)
                    media_type = (
                        header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                    )
                else:
                    b64_data, media_type = data_uri, "image/png"
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    }
                )
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            return {"role": "user", "content": content}
        return {"role": msg.role, "content": msg.content}

    def supports_prompt_caching(self) -> bool:
        """Anthropic explicit cache_control (90% read, 1.25x write premium, 5m-1h TTL)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        """Anthropic reuses KV cache for matching prompt prefixes."""
        return True

    def cache_cost_model(self) -> CacheCostModel:
        """Anthropic explicit ``cache_control`` (FEP-0011).

        90% read discount, 1.25x write premium, 5m–1h TTL, 1024-token minimum
        prefix, cached at system-block granularity.
        """
        return CacheCostModel(
            supported=True,
            read_discount=0.9,
            write_overhead=1.25,
            ttl_seconds=300.0,  # 5-minute floor (1h with extended caching)
            min_prefix_tokens=1024,
            prefix_granularity="system_block",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import (
            ANTHROPIC,
            ANTHROPIC_DEFAULT,
            lookup,
        )

        target = model or getattr(self, "_current_model", None)
        return lookup(ANTHROPIC, target, ANTHROPIC_DEFAULT)

    def _build_request_params(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the Messages API request params shared by ``chat()`` and ``stream()``.

        Single source of truth for prompt assembly (deduped from the former inline
        blocks in both call sites): system message extraction with ``cache_control``,
        message serialization (including image content blocks), and tool conversion
        with the cache boundary placed at the stable/dynamic tier edge — tools are
        sorted FULL -> COMPACT -> STUB; caching the FULL+COMPACT prefix means STUB
        tools can change per-turn without invalidating the cached prefix.
        """
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append(self._serialize_message(msg))

        request_params: Dict[str, Any] = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        if system_message:
            # Content block format with cache_control for prefix caching. Anthropic
            # caches the prefix (tools -> system -> messages) at 90% discount; the
            # ephemeral TTL (5 min) refreshes on each use.
            request_params["system"] = [
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if tools:
            converted = self._convert_tools(tools)
            if converted:
                cache_idx = self._find_cache_boundary(tools, converted)
                if 0 <= cache_idx < len(converted):
                    converted[cache_idx]["cache_control"] = {"type": "ephemeral"}
            request_params["tools"] = converted

        return request_params

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to Anthropic format."""
        return convert_tools_to_anthropic_format(tools)

    @staticmethod
    def _find_cache_boundary(
        tools: List[ToolDefinition],
        converted: List[Dict[str, Any]],
    ) -> int:
        """Find the index for cache_control placement at the stable/dynamic boundary.

        Tools are sorted FULL -> COMPACT -> STUB. The cache boundary is placed
        on the last FULL or COMPACT tool so Anthropic caches the stable prefix.
        STUB tools after the boundary can change per-turn without cache invalidation.
        """
        last_stable = len(converted) - 1
        for i, tool_def in enumerate(tools):
            level = getattr(tool_def, "schema_level", None)
            if level == "stub" and i > 0:
                last_stable = i - 1
                break
        return last_stable

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Anthropic Claude models.

        Resolution order -- Sandhi owns the catalog **data** (TD-0004 Phase A); Victor
        owns the catalog **policy** (shaping the facts for its consumers):

        1. **Sandhi catalog** -- curated, versioned model data from
           ``sandhi_gateway.provider_models_json``, when the installed binding exposes it.
        2. **Live SDK discovery** -- the Anthropic SDK's ``/v1/models`` endpoint
           (enrichment/fallback when the Sandhi catalog is absent).
        3. **Curated static list** (``_ANTHROPIC_STATIC_MODELS``) -- offline fallback.

        Transport (``chat``/``stream``) stays Sandhi-owned; any SDK client here is
        discovery-only (transient), so the provider owns no generation client (TD-0002
        deletion gate holds).

        Returns:
            List of available models with metadata
        """
        catalog = self._models_from_sandhi()
        if catalog is not None:
            return catalog
        try:
            await self._ensure_valid_token()
            # Lazy import: keeps `import victor` (and CLI cold-start) off the
            # anthropic SDK; it is only needed for the fallback discovery path.
            from anthropic import AsyncAnthropic

            async with AsyncAnthropic(
                api_key=None if self._auth_mode == "oauth" else self._api_key,
                auth_token=self._api_key if self._auth_mode == "oauth" else None,
                base_url=getattr(self, "base_url", None),
                timeout=getattr(self, "timeout", self.DEFAULT_TIMEOUT),
            ) as client:
                page = await client.models.list()
                return [self._model_from_sdk(m) for m in (getattr(page, "data", None) or [])]
        except Exception as exc:  # offline, auth failure, missing SDK, or bad response
            logger.debug("anthropic live model discovery unavailable; using static list: %s", exc)
            return [dict(model) for model in _ANTHROPIC_STATIC_MODELS]

    def _models_from_sandhi(self) -> Optional[List[Dict[str, Any]]]:
        """Victor-shaped models from the Sandhi catalog, or ``None`` to fall back.

        The Sandhi catalog (TD-0004) carries curated model *facts* (id, context window,
        max output, capabilities). Victor applies catalog *policy* (shared shaping in
        ``victor.providers.sandhi_catalog``). Returns ``None`` when the installed Sandhi
        binding predates the catalog surface, so ``list_models`` falls through to live
        SDK discovery / the static list.
        """
        from victor.providers.sandhi_catalog import models_from_sandhi

        return models_from_sandhi(self.name)

    @staticmethod
    def _model_from_sdk(model: Any) -> Dict[str, Any]:
        """Map an Anthropic SDK ``ModelInfo`` to Victor's model-dict shape.

        ``getattr`` fallbacks keep this robust to SDK version drift; ``id`` is
        the only required field on the live ``/v1/models`` response.
        """
        return {
            "id": model.id,
            "name": getattr(model, "display_name", None) or model.id,
            "type": getattr(model, "type", "model"),
            "context_window": getattr(model, "max_input_tokens", None),
            "max_output_tokens": getattr(model, "max_tokens", None),
        }

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

        This policy shell stays concrete for auth/capability use; completion
        transport is delegated to the Sandhi runtime. Obtain the typed provider
        via ``resolve_transport_class()`` (e.g. ``SandhiAnthropicProvider``).
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
        """No provider client to close; transport is owned by the Sandhi handle."""
        pass

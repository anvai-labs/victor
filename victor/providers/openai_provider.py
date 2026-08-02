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

"""OpenAI GPT provider implementation."""

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from victor.providers.base import (
    BaseProvider,
    CacheCostModel,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)
from victor.providers.logging import ProviderLogger
from victor.providers.oauth_manager import OAuthTokenManager


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT models."""

    # Cloud provider timeout
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        auth_mode: str = "api_key",
        oauth_source: str = "victor",
        oauth_tokens: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            auth_mode: Authentication mode — "api_key" (default) or "oauth"
            oauth_source: OAuth token source — "victor" or "codex"
            oauth_tokens: Pre-obtained OAuth tokens (optional, for auth_mode="oauth")
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("openai", __name__)

        # OAuth token manager (None when using api_key mode)
        self._oauth_manager: Optional[OAuthTokenManager] = None
        self._auth_mode = auth_mode

        if auth_mode == "oauth":
            # OAuth mode uses ChatGPT subscription via Codex API.
            if base_url is None:
                base_url = "https://chatgpt.com/backend-api/codex"

            self._oauth_manager = OAuthTokenManager("openai", token_source=oauth_source)
            # Use pre-obtained tokens or load cached (sync-safe)
            if oauth_tokens is not None:
                resolved_key = oauth_tokens.access_token
            else:
                cached = self._oauth_manager._load_cached()
                if cached is not None and not cached.is_expired:
                    resolved_key = cached.access_token
                else:
                    # Placeholder — actual login deferred to _ensure_valid_token()
                    # which runs inside the async context (chat/stream)
                    resolved_key = "oauth-pending"

            self._api_key = resolved_key

            self._provider_logger.log_provider_init(
                model="gpt",
                key_source="oauth",
                non_interactive=False,
                config={
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "organization": organization,
                    "auth_mode": "oauth",
                    **kwargs,
                },
            )
        else:
            # Resolve API key using unified resolver (existing path)
            resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
            key_result = resolver.get_api_key("openai", explicit_key=api_key)

            # Log API key resolution
            self._provider_logger.log_api_key_resolution(key_result)

            if key_result.key is None:
                raise APIKeyNotFoundError(
                    provider="openai",
                    sources_attempted=key_result.sources_attempted,
                    non_interactive=key_result.non_interactive,
                )

            self._api_key = key_result.key

            self._provider_logger.log_provider_init(
                model="gpt",
                key_source=key_result.source_detail,
                non_interactive=key_result.non_interactive,
                config={
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "organization": organization,
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

        # Build extra headers for OAuth / Codex backend
        default_headers = None
        if auth_mode == "oauth":
            import platform

            from victor import __version__

            default_headers = {
                "originator": "victor",
                "User-Agent": f"victor/{__version__} ({platform.system()})",
            }
            # Auth and endpoint protocol are independent explicit choices. ChatGPT subscription
            # OAuth uses the item/event-shaped Responses protocol; API keys retain Chat
            # Completions unless a caller explicitly chooses otherwise.
            self._sandhi_protocol = "chatgpt_responses"
            self._wire_headers = dict(default_headers)
            account_id = self._oauth_manager.get_chatgpt_account_id(oauth_tokens)
            if isinstance(account_id, str) and account_id:
                self._wire_headers["ChatGPT-Account-ID"] = account_id

        self.client = AsyncOpenAI(
            api_key=self._api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )

    async def _ensure_valid_token(self) -> None:
        """Refresh OAuth token if needed. No-op for api_key mode."""
        if self._oauth_manager is None:
            return
        token = await self._oauth_manager.get_valid_token()
        if token != self.client.api_key:
            self.client.api_key = token
            self._api_key = token
        account_id = self._oauth_manager.get_chatgpt_account_id()
        if isinstance(account_id, str) and account_id:
            self._wire_headers["ChatGPT-Account-ID"] = account_id
        else:
            self._wire_headers.pop("ChatGPT-Account-ID", None)

    def _is_o_series_model(self, model: str) -> bool:
        """Check if model is an O-series reasoning model.

        O-series models have different parameter requirements:
        - Use max_completion_tokens instead of max_tokens
        - Don't support temperature parameter
        - Don't support tools/function calling

        Args:
            model: Model name

        Returns:
            True if model is O-series
        """
        model_lower = model.lower()
        return any(model_lower.startswith(prefix) for prefix in ["o1", "o3"])

    def _uses_max_completion_tokens(self, model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens.

        GPT-5.x and O-series models use max_completion_tokens.
        GPT-5.x still supports temperature and tools unlike O-series.
        """
        model_lower = model.lower()
        return any(model_lower.startswith(prefix) for prefix in ["o1", "o3", "gpt-5", "gpt5"])

    def supports_reasoning_effort(self, model: Optional[str] = None) -> bool:
        """OpenAI reasoning models (o-series, GPT-5.x) accept ``reasoning_effort``."""
        if not model:
            return False
        return self._uses_max_completion_tokens(model)

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    def supports_tools(self) -> bool:
        """OpenAI supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True

    def supports_prompt_caching(self) -> bool:
        """OpenAI automatic prefix caching (90% discount, 1024 min tokens, 5m-24h TTL)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        """OpenAI reuses KV cache server-side for matching prefixes."""
        return True

    def cache_cost_model(self) -> CacheCostModel:
        """OpenAI automatic prefix caching (FEP-0011).

        90% read discount, ~1024-token minimum prefix, 5m–24h TTL, automatic
        (no explicit cache_control), cached at token granularity.
        """
        return CacheCostModel(
            supported=True,
            read_discount=0.9,
            write_overhead=1.0,  # no explicit write premium; automatic
            ttl_seconds=300.0,  # 5-minute floor (up to 24h while active)
            min_prefix_tokens=1024,
            prefix_granularity="token",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import OPENAI, OPENAI_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(OPENAI, target, OPENAI_DEFAULT)

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models.

        Resolution order -- Sandhi owns the catalog **data** (TD-0004 Phase A); Victor
        owns the catalog **policy**:

        1. **Sandhi catalog** -- curated model data via
           ``sandhi_gateway.provider_models_json``, when the installed binding exposes it.
        2. **Live SDK discovery** -- the OpenAI API's model list, filtered to
           chat-capable models (fallback when the Sandhi catalog is absent).

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If the catalog is absent and the API request fails
        """
        catalog = self._models_from_sandhi()
        if catalog is not None:
            return catalog
        try:
            response = await self.client.models.list()
            # Filter to GPT models and format consistently
            models = []
            for model in response.data:
                model_id = model.id
                # Filter to chat-capable GPT models
                if any(
                    prefix in model_id for prefix in ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"]
                ):
                    models.append(
                        {
                            "id": model_id,
                            "name": model_id,
                            "owned_by": model.owned_by,
                            "created": model.created,
                        }
                    )
            # Sort by name for consistent output
            models.sort(key=lambda x: x["id"])
            return models
        except Exception as e:
            raise ProviderError(
                message=f"Failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    def _models_from_sandhi(self) -> Optional[List[Dict[str, Any]]]:
        """Victor-shaped models from the Sandhi catalog, or ``None`` to fall back.

        Shared catalog policy lives in ``victor.providers.sandhi_catalog``; ``None``
        means the installed Sandhi binding predates the catalog surface (or has no
        OpenAI data), so ``list_models`` falls through to live SDK discovery.
        """
        from victor.providers.sandhi_catalog import models_from_sandhi

        return models_from_sandhi(self.name)

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
        via ``resolve_transport_class()`` (e.g. ``SandhiOpenAIProvider``).
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
        await self.client.close()

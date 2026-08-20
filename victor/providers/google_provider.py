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

"""Google Gemini provider implementation.

Requires optional dependency: pip install victor[google]

Note: This module uses the new google-genai SDK (google.genai) instead of
the deprecated google-generativeai package.
"""

import logging
import warnings
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Suppress Google SDK warning about non-text parts (we handle it correctly)
warnings.filterwarnings(
    "ignore",
    message=".*non-text parts in the response.*",
    category=UserWarning,
    module="google_genai.types",
)

from victor.providers.base import (
    CacheCostModel,
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)
from victor.providers.logging import ProviderLogger

# Curated offline fallback (mirrors the Sandhi catalog lineup): used when the
# installed sandhi-gateway predates the TD-0004 catalog surface.
_GOOGLE_STATIC_MODELS: List[Dict[str, Any]] = [
    {
        "id": "gemini-3-pro",
        "name": "Gemini 3 Pro",
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
    },
    {
        "id": "gemini-3-flash",
        "name": "Gemini 3 Flash",
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
    },
]


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models.

    Safety Settings:
        The `safety_level` parameter controls content filtering:
        - "block_none": No blocking (least restrictive, for development)
        - "block_few": Block only high probability harmful content
        - "block_some": Block medium and above (default)
        - "block_most": Block low and above (most restrictive)

    Example:
        # For code generation without safety blocks
        provider = GoogleProvider(api_key=key, safety_level="block_none")
    """

    # Cloud provider timeout
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        safety_level: str = "block_none",
        non_interactive: Optional[bool] = None,
        auth_mode: str = "api_key",
        oauth_tokens: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize Google provider.

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
            safety_level: Safety filter level - "block_none", "block_few",
                         "block_some", or "block_most" (default: "block_none")
            non_interactive: Force non-interactive mode (None = auto-detect)
            auth_mode: Authentication mode - "api_key" or "oauth"
            oauth_tokens: Pre-obtained OAuth tokens (optional)
            **kwargs: Additional configuration

        Raises:
            ImportError: If google-genai package is not installed
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("google", __name__)
        self._oauth_manager = None
        self._current_token = None
        # Sandhi owns transport: pick the auth scheme the typed variant passes to the Gemini
        # handle. OAuth/ADC → Authorization: Bearer; API key → x-goog-api-key.
        self._sandhi_auth_scheme = "bearer" if auth_mode == "oauth" else "api_key"

        if auth_mode == "oauth":
            # OAuth mode: use Google subscription (AI Pro/Ultra) instead of API key
            from victor.providers.oauth_manager import OAuthTokenManager

            self._oauth_manager = OAuthTokenManager("google")

            # Load cached token
            access_token = None
            refresh_token = None
            if oauth_tokens and hasattr(oauth_tokens, "access_token"):
                access_token = oauth_tokens.access_token
                refresh_token = getattr(oauth_tokens, "refresh_token", None)
            else:
                cached = self._oauth_manager._load_cached()
                if cached and not cached.is_expired:
                    access_token = cached.access_token
                    refresh_token = cached.refresh_token

            # Build google.oauth2.credentials for native SDK integration
            from google.oauth2.credentials import Credentials as OAuthCredentials

            creds = OAuthCredentials(
                token=access_token or "oauth-pending",
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=(
                    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j" ".apps.googleusercontent.com"
                ),
                client_secret="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",
            )
            self._google_credentials = creds
            self._current_token = access_token
            # The access token (or a placeholder until _ensure_valid_token refreshes it) is what
            # Sandhi sends as Authorization: Bearer via the typed Gemini handle.
            self._api_key = access_token or "oauth-pending"

            self._provider_logger.log_provider_init(
                model="gemini",
                key_source="oauth",
                non_interactive=non_interactive,
                config={"timeout": timeout, "safety_level": safety_level, **kwargs},
            )

            super().__init__(api_key=self._api_key, timeout=timeout, **kwargs)
        else:
            # Standard API key mode
            self._google_credentials = None

            resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
            key_result = resolver.get_api_key("google", explicit_key=api_key)

            self._provider_logger.log_api_key_resolution(key_result)

            if key_result.key is None:
                raise APIKeyNotFoundError(
                    provider="google",
                    sources_attempted=key_result.sources_attempted,
                    non_interactive=key_result.non_interactive,
                )

            self._api_key = key_result.key

            self._provider_logger.log_provider_init(
                model="gemini",
                key_source=key_result.source_detail,
                non_interactive=key_result.non_interactive,
                config={"timeout": timeout, "safety_level": safety_level, **kwargs},
            )

            super().__init__(api_key=self._api_key, timeout=timeout, **kwargs)

    @property
    def name(self) -> str:
        """Provider name."""
        return "google"

    def supports_tools(self) -> bool:
        """Google Gemini supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """Google supports streaming."""
        return True

    def supports_prompt_caching(self) -> bool:
        """Gemini hybrid caching (75-90% read, $4.50/1M/hr storage, 32K min, custom TTL)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        """Gemini reuses KV cache for matching prompt prefixes."""
        return True

    def cache_cost_model(self) -> CacheCostModel:
        """Characterized API caching (FEP-0011): 75-90% read, 32K min, custom TTL."""
        return CacheCostModel(
            supported=True,
            read_discount=0.825,
            write_overhead=1.0,
            ttl_seconds=0.0,
            min_prefix_tokens=32768,
            prefix_granularity="token",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import GOOGLE, GOOGLE_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(GOOGLE, target, GOOGLE_DEFAULT)

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Google Gemini models.

        Resolution order -- Sandhi owns the catalog **data** (TD-0004 Phase A); Victor
        owns the catalog **policy**:

        1. **Sandhi catalog** -- curated model data via
           ``sandhi_gateway.provider_models_json``, when the installed binding exposes it.
        2. **Curated static list** (``_GOOGLE_STATIC_MODELS``) -- offline fallback
           mirroring the catalog lineup.

        Returns:
            List of available models with metadata
        """
        catalog = self._models_from_sandhi()
        if catalog is not None:
            return catalog
        return [dict(model) for model in _GOOGLE_STATIC_MODELS]

    def _models_from_sandhi(self) -> Optional[List[Dict[str, Any]]]:
        """Victor-shaped models from the Sandhi catalog, or ``None`` to fall back.

        Shared catalog policy lives in ``victor.providers.sandhi_catalog``; ``None``
        means the installed Sandhi binding predates the catalog surface (or has no
        Gemini data), so ``list_models`` falls through to the static list.
        """
        from victor.providers.sandhi_catalog import models_from_sandhi

        return models_from_sandhi(self.name)

    async def _ensure_valid_token(self) -> None:
        """Ensure OAuth token is valid, refreshing if needed."""
        if self._oauth_manager is None:
            return
        token = await self._oauth_manager.get_valid_token()
        if token != self._current_token:
            from google.oauth2.credentials import Credentials as OAuthCredentials

            self._google_credentials = OAuthCredentials(
                token=token,
                refresh_token=(
                    self._google_credentials.refresh_token if self._google_credentials else None
                ),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=(
                    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j" ".apps.googleusercontent.com"
                ),
                client_secret="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",
            )
            self._current_token = token
            # Keep _api_key in sync so the Sandhi Gemini handle receives the refreshed token.
            self._api_key = token

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
        via ``resolve_transport_class()`` (e.g. ``SandhiGoogleProvider``).
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
        """Close connections (Gemini client doesn't need explicit closing)."""
        pass

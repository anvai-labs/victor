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

"""Qwen host policy over Sandhi's typed OpenAI-compatible runtime.

Sandhi owns provider identity, aliases, endpoints, HTTP/SSE, typed codecs,
errors, retries, and usage metering. Victor retains model/context policy and
credential acquisition, including Qwen Coding Plan OAuth.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Optional

import sandhi_gateway

from victor.providers.oauth_manager import OAuthTokenManager
from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy


def _qwen_base_urls() -> MappingProxyType:
    """Read stable Qwen wire endpoints from Sandhi's provider descriptor."""
    descriptor = json.loads(sandhi_gateway.provider_descriptor_json("qwen"))
    options = (descriptor.get("extensions") or {}).get("endpoint_options") or {}
    return MappingProxyType({str(name): str(url) for name, url in options.items()})


QWEN_BASE_URLS = _qwen_base_urls()
QWEN_OAUTH_CONFIG = MappingProxyType({"oauth_base_url": "https://chat.qwen.ai"})
QWEN_MODELS = get_openai_compat_provider_spec("qwen").models


class QwenProvider(SandhiOpenAICompatPolicy):
    """Qwen model/auth policy; all provider execution is typed Sandhi FFI."""

    CONFIG_KEY = "qwen"
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        auth_mode: str = "api_key",
        oauth_tokens: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self._oauth_manager: Optional[OAuthTokenManager] = None
        if auth_mode == "oauth":
            self._oauth_manager = OAuthTokenManager("qwen")
            if oauth_tokens is not None:
                api_key = oauth_tokens.access_token
            else:
                cached = self._oauth_manager._load_cached()
                api_key = (
                    cached.access_token
                    if cached is not None and not cached.is_expired
                    else "oauth-pending"
                )
            base_url = base_url or QWEN_BASE_URLS["portal"]

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            non_interactive=non_interactive,
            **kwargs,
        )

    async def _ensure_valid_token(self) -> None:
        """Refresh host-owned OAuth credentials before Sandhi selects its handle."""
        if self._oauth_manager is None:
            return
        token = await self._oauth_manager.get_valid_token()
        if token != self._api_key:
            self._api_key = token
            self.api_key = token


__all__ = ["QWEN_BASE_URLS", "QWEN_MODELS", "QWEN_OAUTH_CONFIG", "QwenProvider"]

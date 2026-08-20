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

"""Victor policy shell for Sandhi-catalogued OpenAI-compatible providers.

Victor owns model-facing policy and normalized framework objects. The shared
request-shaping helpers are inherited from :class:`HttpxOpenAICompatProvider`,
but execution is always delegated to Sandhi's typed runtime. Concrete classes
only select a validated policy key so imports and registry identities remain
stable.
"""

from __future__ import annotations

from abc import ABC
from json import JSONDecodeError
from typing import Any, ClassVar, Dict, List, Optional

from victor.core.json_utils import json_loads
from victor.providers.base import CacheCostModel, StreamChunk
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.logging import ProviderLogger
from victor.providers.openai_compat import parse_openai_tool_calls
from victor.providers.openai_compat_model_policy import (
    OpenAICompatProviderSpec,
    get_openai_compat_provider_spec,
)
from victor.providers.resolution import APIKeyNotFoundError, UnifiedApiKeyResolver
from victor.providers.sandhi_transport import SandhiHttpxTransportMixin


class SandhiOpenAICompatPolicy(SandhiHttpxTransportMixin, HttpxOpenAICompatProvider, ABC):
    """OpenAI-compatible adapter whose static policy comes from validated config."""

    CONFIG_KEY: ClassVar[str]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        spec = get_openai_compat_provider_spec(self.CONFIG_KEY)
        self._spec = spec
        self._api_key = ""

        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key(spec.credential_provider, explicit_key=api_key)
        if key_result.key is None:
            raise APIKeyNotFoundError(
                provider=spec.credential_provider,
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )
        self._api_key = key_result.key

        custom_headers: Dict[str, str] = {}
        for option_name, header_name in spec.header_options.items():
            option_value = kwargs.pop(option_name, None)
            if option_value is not None:
                custom_headers[header_name] = str(option_value)
        self._wire_headers = dict(custom_headers)

        resolved_base_url = base_url or spec.base_url
        resolved_timeout = timeout if timeout is not None else spec.timeout
        resolved_retries = max_retries if max_retries is not None else spec.max_retries
        super().__init__(
            api_key=self._api_key,
            base_url=resolved_base_url,
            timeout=resolved_timeout,
            max_retries=resolved_retries,
            provider_name=spec.slug,
            default_headers=custom_headers,
            initialize_http_client=False,
            **kwargs,
        )

        self._provider_logger = ProviderLogger(spec.slug, self.__class__.__module__)
        self._provider_logger.log_api_key_resolution(key_result)
        self._provider_logger.log_provider_init(
            model=spec.default_model,
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={
                "base_url": resolved_base_url,
                "timeout": resolved_timeout,
                "max_retries": resolved_retries,
                **kwargs,
            },
        )

    @classmethod
    def provider_spec(cls) -> OpenAICompatProviderSpec:
        """Return this compatibility class's immutable provider policy."""
        return get_openai_compat_provider_spec(cls.CONFIG_KEY)

    @property
    def _policy_spec(self) -> OpenAICompatProviderSpec:
        """The effective provider policy spec, safe to query before ``__init__``.

        Capability and context facts live in validated config (``provider_spec()``);
        ``__init__`` copies that spec onto ``self._spec``. Discovery queries these
        facts on partially-constructed instances (before credentials or HTTP exist),
        so this falls back to the class-level spec until ``_spec`` is set. Centralizing
        the fallback keeps the pre-init-safe contract explicit rather than duplicated.
        """
        return getattr(self, "_spec", None) or self.provider_spec()

    @property
    def name(self) -> str:
        return self._spec.slug

    def supports_tools(self) -> bool:
        return self._policy_spec.capabilities.tools

    def supports_streaming(self) -> bool:
        return self._policy_spec.capabilities.streaming

    def supports_prompt_caching(self) -> bool:
        return self._policy_spec.capabilities.prompt_caching

    def supports_kv_prefix_caching(self) -> bool:
        return self._policy_spec.capabilities.kv_prefix_caching

    def cache_cost_model(self) -> CacheCostModel:
        policy = self._policy_spec.cache
        return CacheCostModel(
            supported=self.supports_prompt_caching(),
            read_discount=policy.read_discount,
            write_overhead=policy.write_overhead,
            ttl_seconds=policy.ttl_seconds,
            min_prefix_tokens=policy.min_prefix_tokens,
            max_cache_tokens=policy.max_cache_tokens,
            prefix_granularity=policy.prefix_granularity,
        )

    def context_window(self, model: Optional[str] = None) -> int:
        # Context budgeting is also queried on an uninitialized instance by
        # discovery code, so it must not depend on credentials or HTTP setup.
        spec = self._policy_spec
        target = model or getattr(self, "_current_model", None)
        if target:
            for model_prefix, tokens in spec.context_window_routes:
                if target.startswith(model_prefix):
                    return tokens
        return spec.default_context_window

    def get_default_model(self) -> str:
        return self._spec.default_model

    def get_supported_models(self) -> List[str]:
        return list(self._spec.models)

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Compatibility shim for callers of the former provider-private helper."""
        normalized = parse_openai_tool_calls(tool_calls)
        if not normalized:
            return normalized
        # The former adapters treated malformed model-generated JSON as an empty
        # argument object. Preserve that boundary behavior while the shared parser
        # retains its more diagnostic {"raw": ...} representation elsewhere.
        for call in normalized:
            arguments = call.get("arguments")
            if isinstance(arguments, dict) and set(arguments) == {"raw"}:
                try:
                    json_loads(str(arguments["raw"]))
                except JSONDecodeError:
                    call["arguments"] = {}
        return normalized

    def _parse_stream_chunk(
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
    ) -> Optional[StreamChunk]:
        chunk = super()._parse_stream_chunk(chunk_data, accumulated_tool_calls)
        if chunk is None:
            return StreamChunk(content="", is_final=False)
        return chunk

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models -- Sandhi catalog data first, Victor's admitted policy as fallback.

        Resolution order (TD-0004 Phase A): the Sandhi catalog carries curated model
        facts for first-party vendors (deepseek/xai/mistral/zai/qwen/...); aggregators
        (openrouter/together/fireworks) deliberately have no catalog there, so they --
        and any sandhi version predating the surface -- fall back to Victor's admitted
        model policy. Neither tier performs I/O.
        """
        catalog = self._models_from_sandhi()
        if catalog is not None:
            return catalog
        return [
            {"id": model_id, **dict(model_info)}
            for model_id, model_info in self._spec.models.items()
        ]

    def _models_from_sandhi(self) -> Optional[List[Dict[str, Any]]]:
        """Victor-shaped models from the Sandhi catalog, or ``None`` to fall back.

        Shared catalog policy lives in ``victor.providers.sandhi_catalog``; ``None``
        means the installed Sandhi binding predates the catalog surface or carries no
        curated lineup for this slug (e.g. aggregators), so ``list_models`` falls
        through to the spec-admitted policy list.
        """
        from victor.providers.sandhi_catalog import models_from_sandhi

        return models_from_sandhi(self.name)


__all__ = ["SandhiOpenAICompatPolicy"]

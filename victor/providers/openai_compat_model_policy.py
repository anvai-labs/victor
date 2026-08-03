# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Victor-owned model policy for Sandhi OpenAI-compatible providers.

Provider identity, aliases, endpoint family, base URL, and wire capabilities are
loaded from Sandhi's typed descriptor. This module retains only agent-facing model,
context-budget, cache-economics, timeout, and credential-selection policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json

import yaml

try:
    import sandhi_gateway as _sandhi_gateway  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover - installation is validated by provider creation
    raise RuntimeError("sandhi-gateway 0.1.5 is required for provider policy") from exc


class OpenAICompatConfigError(ValueError):
    """Raised when the packaged OpenAI-compatible provider policy is invalid."""


@dataclass(frozen=True)
class OpenAICompatCapabilities:
    tools: bool
    streaming: bool
    prompt_caching: bool = False
    kv_prefix_caching: bool = False


@dataclass(frozen=True)
class OpenAICompatCachePolicy:
    read_discount: float = 0.0
    write_overhead: float = 1.0
    ttl_seconds: float = 0.0
    min_prefix_tokens: int = 0
    max_cache_tokens: int = 0
    prefix_granularity: str = "token"


@dataclass(frozen=True)
class OpenAICompatProviderSpec:
    key: str
    slug: str
    aliases: tuple[str, ...]
    credential_provider: str
    base_url: str
    default_model: str
    timeout: int
    max_retries: int
    capabilities: OpenAICompatCapabilities
    default_context_window: int
    context_window_routes: tuple[tuple[str, int], ...]
    models: Mapping[str, Mapping[str, Any]]
    cache: OpenAICompatCachePolicy = field(default_factory=OpenAICompatCachePolicy)
    header_options: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    endpoint_options: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "openai_compat_model_policy.yaml"


def _wire_descriptor(name: str) -> Mapping[str, Any]:
    """Read transport facts from Sandhi, the sole owner of this catalog."""
    try:
        descriptor = json.loads(_sandhi_gateway.provider_descriptor_json(name))
    except Exception as exc:
        raise OpenAICompatConfigError(
            f"Sandhi has no typed provider descriptor for {name!r}: {exc}"
        ) from exc
    if descriptor.get("endpoint_family") != "openai_chat_completions":
        raise OpenAICompatConfigError(f"{name!r} is not an OpenAI-compatible Sandhi provider")
    return descriptor


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OpenAICompatConfigError(f"{path} must be a mapping")
    return value


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenAICompatConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenAICompatConfigError(f"{path} must be a positive integer")
    return value


def _optional_bool(config: Mapping[str, Any], key: str, path: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise OpenAICompatConfigError(f"{path}.{key} must be boolean")
    return value


def _load_specs(path: Path = _CONFIG_PATH) -> Mapping[str, OpenAICompatProviderSpec]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise OpenAICompatConfigError(f"failed to load {path}: {exc}") from exc

    root = _require_mapping(raw, "root")
    if root.get("version") != 1:
        raise OpenAICompatConfigError("version must be 1")
    providers = _require_mapping(root.get("providers"), "providers")
    if not providers:
        raise OpenAICompatConfigError("providers must not be empty")

    parsed: dict[str, OpenAICompatProviderSpec] = {}
    claimed_names: dict[str, str] = {}
    for key, untyped_config in providers.items():
        key = _require_text(key, "providers key").lower()
        config = _require_mapping(untyped_config, f"providers.{key}")
        descriptor = _wire_descriptor(key)
        slug = _require_text(descriptor.get("slug"), f"sandhi.{key}.slug").lower()
        base_url = _require_text(descriptor.get("base_url"), f"sandhi.{key}.base_url")
        aliases = tuple(str(alias) for alias in descriptor.get("aliases", []))
        wire_capabilities = _require_mapping(
            descriptor.get("capabilities", {}), f"sandhi.{key}.capabilities"
        )

        models_raw = _require_mapping(config.get("models"), f"providers.{key}.models")
        models: dict[str, Mapping[str, Any]] = {}
        for model_name, model_metadata in models_raw.items():
            model_name = _require_text(model_name, f"providers.{key}.models key")
            metadata = dict(
                _require_mapping(model_metadata, f"providers.{key}.models.{model_name}")
            )
            context_window = metadata.get("context_window")
            if context_window is not None:
                _require_positive_int(
                    context_window, f"providers.{key}.models.{model_name}.context_window"
                )
            models[model_name] = MappingProxyType(metadata)

        routes_value = config.get("context_window_routes", [])
        if not isinstance(routes_value, list):
            raise OpenAICompatConfigError(f"providers.{key}.context_window_routes must be a list")
        context_window_routes: list[tuple[str, int]] = []
        for index, untyped_route in enumerate(routes_value):
            route_path = f"providers.{key}.context_window_routes.{index}"
            route = _require_mapping(untyped_route, route_path)
            context_window_routes.append(
                (
                    _require_text(route.get("model_prefix"), f"{route_path}.model_prefix"),
                    _require_positive_int(route.get("tokens"), f"{route_path}.tokens"),
                )
            )

        default_model = _require_text(config.get("default_model"), f"providers.{key}.default_model")
        if default_model not in models:
            raise OpenAICompatConfigError(f"providers.{key}.default_model must appear in models")

        cache_raw = _require_mapping(config.get("cache", {}), f"providers.{key}.cache")
        capabilities = OpenAICompatCapabilities(
            tools=bool(wire_capabilities.get("tools", False)),
            streaming=bool(wire_capabilities.get("streaming", False)),
            prompt_caching=_optional_bool(cache_raw, "supported", f"providers.{key}.cache"),
            kv_prefix_caching=_optional_bool(
                cache_raw, "kv_prefix_caching", f"providers.{key}.cache"
            ),
        )
        cache = OpenAICompatCachePolicy(
            read_discount=float(cache_raw.get("read_discount", 0.0)),
            write_overhead=float(cache_raw.get("write_overhead", 1.0)),
            ttl_seconds=float(cache_raw.get("ttl_seconds", 0.0)),
            min_prefix_tokens=int(cache_raw.get("min_prefix_tokens", 0)),
            max_cache_tokens=int(cache_raw.get("max_cache_tokens", 0)),
            prefix_granularity=str(cache_raw.get("prefix_granularity", "token")),
        )
        if not 0.0 <= cache.read_discount <= 1.0:
            raise OpenAICompatConfigError(
                f"providers.{key}.cache.read_discount must be between 0 and 1"
            )
        if cache.write_overhead < 1.0:
            raise OpenAICompatConfigError(
                f"providers.{key}.cache.write_overhead must be at least 1"
            )

        headers_raw = _require_mapping(
            _require_mapping(descriptor.get("extensions", {}), f"sandhi.{key}.extensions").get(
                "header_options", {}
            ),
            f"sandhi.{key}.extensions.header_options",
        )
        header_options = {
            _require_text(option, f"providers.{key}.header_options key"): _require_text(
                header, f"providers.{key}.header_options.{option}"
            )
            for option, header in headers_raw.items()
        }
        protected_headers = {"authorization", "content-type", "host"}
        for header in header_options.values():
            if header.lower() in protected_headers:
                raise OpenAICompatConfigError(
                    f"sandhi.{key}.header_options cannot override transport header {header!r}"
                )

        endpoints_raw = _require_mapping(
            _require_mapping(descriptor.get("extensions", {}), f"sandhi.{key}.extensions").get(
                "endpoint_options", {}
            ),
            f"sandhi.{key}.extensions.endpoint_options",
        )
        endpoint_options = {
            _require_text(option, f"sandhi.{key}.endpoint_options key"): _require_text(
                endpoint, f"sandhi.{key}.endpoint_options.{option}"
            ).rstrip("/")
            for option, endpoint in endpoints_raw.items()
        }

        for claimed_name in dict.fromkeys((key, slug, *aliases)):
            if claimed_name in claimed_names:
                raise OpenAICompatConfigError(
                    f"provider name {claimed_name!r} is claimed by both "
                    f"{claimed_names[claimed_name]!r} and {key!r}"
                )
            claimed_names[claimed_name] = key

        parsed[key] = OpenAICompatProviderSpec(
            key=key,
            slug=slug,
            aliases=aliases,
            credential_provider=key,
            base_url=base_url.rstrip("/"),
            default_model=default_model,
            timeout=_require_positive_int(config.get("timeout"), f"providers.{key}.timeout"),
            max_retries=_require_positive_int(
                config.get("max_retries"), f"providers.{key}.max_retries"
            ),
            capabilities=capabilities,
            default_context_window=_require_positive_int(
                config.get("default_context_window"),
                f"providers.{key}.default_context_window",
            ),
            context_window_routes=tuple(context_window_routes),
            models=MappingProxyType(models),
            cache=cache,
            header_options=MappingProxyType(header_options),
            endpoint_options=MappingProxyType(endpoint_options),
        )

    return MappingProxyType(parsed)


_SPECS = _load_specs()


def get_openai_compat_provider_spec(name: str) -> OpenAICompatProviderSpec:
    """Return a validated spec by config key, slug, or alias."""
    normalized = name.strip().lower()
    direct = _SPECS.get(normalized)
    if direct is not None:
        return direct
    for spec in _SPECS.values():
        if normalized == spec.slug or normalized in spec.aliases:
            return spec
    available = ", ".join(sorted(_SPECS))
    raise KeyError(
        f"Unknown configured OpenAI-compatible provider {name!r}; available: {available}"
    )


def list_openai_compat_provider_specs() -> Mapping[str, OpenAICompatProviderSpec]:
    """Return the immutable configured-provider mapping."""
    return _SPECS


__all__ = [
    "OpenAICompatCachePolicy",
    "OpenAICompatCapabilities",
    "OpenAICompatConfigError",
    "OpenAICompatProviderSpec",
    "get_openai_compat_provider_spec",
    "list_openai_compat_provider_specs",
]

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unit tests for the shared local-provider context-window cache helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from victor.providers.runtime_capabilities import cached_config_context_window


def test_cache_miss_resolves_config_limits_and_memoizes() -> None:
    cache: dict[str, int] = {}
    with patch(
        "victor.config.config_loaders.get_provider_limits",
        return_value=SimpleNamespace(context_window=32768),
    ) as limits:
        first = cached_config_context_window(
            cache, provider="ollama", base_url="http://h:1", model="qwen"
        )
        second = cached_config_context_window(
            cache, provider="ollama", base_url="http://h:1", model="qwen"
        )
    assert first == second == 32768
    limits.assert_called_once_with("ollama", "qwen")
    assert cache == {"http://h:1:qwen": 32768}


def test_cache_is_keyed_per_endpoint() -> None:
    cache = {"http://a:1:m": 4096}
    with patch(
        "victor.config.config_loaders.get_provider_limits",
        return_value=SimpleNamespace(context_window=8192),
    ) as limits:
        assert (
            cached_config_context_window(cache, provider="x", base_url="http://a:1", model="m")
            == 4096
        )
        assert (
            cached_config_context_window(cache, provider="x", base_url="http://b:1", model="m")
            == 8192
        )
    limits.assert_called_once_with("x", "m")


def test_real_config_lookup_returns_positive_window() -> None:
    cache: dict[str, int] = {}
    window = cached_config_context_window(
        cache, provider="lmstudio", base_url="http://localhost:1234", model="any-model"
    )
    assert window > 0 and cache["http://localhost:1234:any-model"] == window

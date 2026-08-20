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

"""FEP-0020 Phase 4 — provider egress routing through a sandhi gateway."""

from __future__ import annotations

import pytest

from victor.observability.gateway_client import (
    GatewayRoute,
    build_gateway_routed_provider,
)


def test_openai_kwargs_targets_gateway_v1() -> None:
    route = GatewayRoute(base_url="http://localhost:8600", virtual_key="vk-alice", enabled=True)
    kwargs = route.openai_kwargs()
    # Base must carry /v1 so the OpenAI client posts to …/v1/chat/completions.
    assert kwargs["base_url"] == "http://localhost:8600/v1"
    assert kwargs["api_key"] == "vk-alice"


def test_openai_kwargs_strips_trailing_slash() -> None:
    route = GatewayRoute(base_url="http://gw:8600/", virtual_key="vk", enabled=True)
    assert route.openai_kwargs()["base_url"] == "http://gw:8600/v1"


def test_disabled_route_raises() -> None:
    route = GatewayRoute(base_url="http://gw:8600", virtual_key="vk")  # enabled defaults False
    with pytest.raises(RuntimeError, match="disabled"):
        build_gateway_routed_provider(route)


def test_routed_provider_points_at_gateway() -> None:
    route = GatewayRoute(
        base_url="http://gw.test:8600", virtual_key="vk-alice-secret", enabled=True
    )
    provider = build_gateway_routed_provider(route, provider="openai")
    # The provider's egress client is aimed at the gateway, not api.openai.com.
    assert "gw.test:8600/v1" in str(provider.client.base_url)


def test_overrides_win_over_route() -> None:
    route = GatewayRoute(base_url="http://gw:8600", virtual_key="vk", enabled=True)
    provider = build_gateway_routed_provider(
        route, provider="openai", base_url="http://other:9000/v1"
    )
    assert "other:9000" in str(provider.client.base_url)


def test_secret_not_reprd() -> None:
    route = GatewayRoute(base_url="http://gw:8600", virtual_key="super-secret", enabled=True)
    assert "super-secret" not in repr(route)

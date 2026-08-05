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

"""G6 (sandhi typed-integration gap ledger): descriptor-backed capabilities.

``BaseProvider.discover_capabilities()`` defaulted to config-only for every
provider; capability facts for Sandhi-routed providers now come from Sandhi's
typed descriptor (the wire truth the transport honors), with the config-only
base as fallback when the binding or descriptor is absent.
"""

from __future__ import annotations

import json

from victor.providers import sandhi_transport
from victor.providers.sandhi_transport import SandhiAnthropicProvider


class _FakeGateway:
    """Minimal sandhi_gateway stand-in serving one descriptor."""

    def __init__(self, descriptor=None, error=False):
        self._descriptor = descriptor
        self._error = error

    def provider_descriptor_json(self, slug: str) -> str:
        if self._error or self._descriptor is None:
            raise RuntimeError(f"unknown provider: {slug}")
        return json.dumps(self._descriptor)


def _provider() -> SandhiAnthropicProvider:
    # Uninitialized real instance — the repo's established pattern for
    # capability queries (no credentials or transport construction needed).
    return SandhiAnthropicProvider.__new__(SandhiAnthropicProvider)


DESCRIPTOR = {
    "schema_version": "1",
    "slug": "anthropic",
    "endpoint_family": "anthropic_messages",
    "capabilities": {
        "streaming": True,
        "tools": True,
        "vision": True,
        "reasoning": True,
        "prompt_cache_usage": True,
    },
}


class TestDescriptorBackedDiscovery:
    async def test_capabilities_come_from_descriptor(self, monkeypatch):
        monkeypatch.setattr(sandhi_transport, "_sg", _FakeGateway(DESCRIPTOR))
        caps = await _provider().discover_capabilities("claude-sonnet-5")

        assert caps.source == "sandhi_descriptor"
        assert caps.supports_tools is True
        assert caps.supports_streaming is True
        assert caps.raw["endpoint_family"] == "anthropic_messages"

    async def test_descriptor_flags_override_hardcoded_supports(self, monkeypatch):
        gated = dict(DESCRIPTOR, capabilities={"streaming": True, "tools": False})
        monkeypatch.setattr(sandhi_transport, "_sg", _FakeGateway(gated))
        caps = await _provider().discover_capabilities("claude-sonnet-5")

        # AnthropicProvider hardcodes supports_tools()=True; the descriptor
        # (wire truth) wins on the discovery surface.
        assert caps.supports_tools is False
        assert caps.source == "sandhi_descriptor"

    async def test_missing_binding_falls_back_to_config(self, monkeypatch):
        monkeypatch.setattr(sandhi_transport, "_sg", None)
        caps = await _provider().discover_capabilities("claude-sonnet-5")

        assert caps.source == "config"

    async def test_descriptor_error_falls_back_to_config(self, monkeypatch):
        monkeypatch.setattr(sandhi_transport, "_sg", _FakeGateway(error=True))
        caps = await _provider().discover_capabilities("claude-sonnet-5")

        assert caps.source == "config"

    async def test_compat_policy_shell_shares_the_override(self):
        from victor.providers.sandhi_transport import (
            SandhiHttpxTransportMixin,
            SandhiTypedProviderMixin,
        )

        # One override covers both transport paths: the compat shells' mixin
        # inherits the typed mixin where discover_capabilities lives.
        assert issubclass(SandhiHttpxTransportMixin, SandhiTypedProviderMixin)
        assert "discover_capabilities" in SandhiTypedProviderMixin.__dict__

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

"""FEP-0020 Phase 4 — route provider egress through a ``sandhi`` gateway.

The "provider transport migration" (ADR-0047 D10), OpenAI-compat first. Instead
of hitting the provider's real endpoint directly, Victor points its **existing**
OpenAI-compatible provider at a running ``sandhi`` usage gateway (Phase 3,
``victor gateway serve``) and authenticates with a per-user **virtual key**. The
gateway resolves the key to a subject/team, enforces the token budget, forwards
to the real upstream holding the shared key, and meters usage per subject.

Only the *egress endpoint* changes — Victor keeps its Python adapter, so prompt
assembly, tool translation, FEP-0011 hints, agent-aware selection, and
**streaming** all work unchanged (the gateway passes SSE through byte-for-byte).
Routing is opt-in (``GatewayRoute.enabled``); the default path is direct, so this
is non-breaking.

Scope note (ADR-0047 D10). This ships the OpenAI-compat *endpoint reroute*, which
covers OpenAI proper plus the ~20 providers that speak the Chat Completions wire
format, with full streaming. Migrating Victor's in-process adapters onto the
Rust ``sandhi-providers`` transport directly (via a PyO3 async binding) is the
**deferred tail** of D10 — its incremental value over metering (Phase 2) + the
reverse proxy (Phase 3) is marginal, and bridging streaming Rust futures across
PyO3 is costly; it stays a named-trigger deferral.
"""

import logging
from typing import Any, Dict

from pydantic import BaseModel, SecretStr

logger = logging.getLogger(__name__)


class GatewayRoute(BaseModel):
    """How to reach a running ``sandhi`` gateway with a per-user virtual key."""

    base_url: str  # the gateway root, e.g. http://localhost:8600 (no /v1 suffix)
    virtual_key: SecretStr  # the bearer token registered on the gateway for this user
    enabled: bool = False  # opt-in; default direct (non-breaking)

    def openai_kwargs(self) -> Dict[str, str]:
        """Provider kwargs that point an OpenAI-compat adapter at the gateway.

        The gateway serves ``POST /v1/chat/completions``; the OpenAI client appends
        ``/chat/completions`` to its ``base_url``, so the base must carry the ``/v1``.
        """
        return {
            "base_url": self.base_url.rstrip("/") + "/v1",
            "api_key": self.virtual_key.get_secret_value(),
        }


def build_gateway_routed_provider(
    route: GatewayRoute,
    *,
    provider: str = "openai",
    **overrides: Any,
) -> Any:
    """Construct a Victor provider whose egress is routed through the gateway.

    ``provider`` selects the Victor adapter slug (OpenAI-compat first). Extra
    ``overrides`` (e.g. ``timeout``) pass through to the provider; an explicit
    ``base_url``/``api_key`` override wins over the route's.

    Raises ``RuntimeError`` if the route is disabled — callers gate on
    ``route.enabled`` and fall back to their direct provider when off.
    """
    if not route.enabled:
        raise RuntimeError(
            "gateway routing is disabled (GatewayRoute.enabled is False); "
            "use the direct provider."
        )
    from victor.providers.registry import get_provider_registry

    kwargs: Dict[str, Any] = {**route.openai_kwargs(), **overrides}
    logger.debug(
        "routing provider '%s' egress through sandhi gateway at %s",
        provider,
        kwargs.get("base_url"),
    )
    return get_provider_registry().create(provider, **kwargs)

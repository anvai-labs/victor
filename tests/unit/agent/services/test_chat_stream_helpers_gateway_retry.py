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

"""Gateway rate-limit retry semantics for _stream_with_rate_limit_retry.

Gateway-routed providers enforce their own rate limits upstream, so the
client must NOT retry (max_retries collapses to 0); direct-API providers
keep the bounded retry ladder. Pins the behavior of the
``uses_gateway(self._orchestrator.provider)`` branch (the line the boundary
ratchet and the changed-file coverage gate both watch).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_runtime_services import ChatStreamMetrics
from victor.core.errors import ProviderRateLimitError


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator, attempts):
        self._orchestrator = orchestrator
        self._attempts = attempts
        self.services = SimpleNamespace(
            delivery=ChatDelivery(
                chunks=None,
                sanitizer=SimpleNamespace(
                    is_garbage_content=lambda _c: False, sanitize=lambda c: c
                ),
            ),
            metrics=ChatStreamMetrics(orchestrator._metrics_collector),
            stream_lifecycle=SimpleNamespace(
                rate_limit_wait_time=MagicMock(return_value=0.0)
            ),
        )

    async def _stream_provider_response_inner(self, tools, provider_kwargs, stream_ctx):
        self._attempts.append(1)
        raise ProviderRateLimitError("429 slow down")


def _orchestrator(gateway: bool):
    provider = SimpleNamespace()
    if gateway:
        # uses_gateway traverses __dict__ for an extra_config.gateway entry.
        provider = SimpleNamespace(extra_config={"gateway": {"url": "https://gw"}})
    return SimpleNamespace(
        provider=provider,
        _metrics_collector=SimpleNamespace(record_first_token=MagicMock()),
    )


@pytest.mark.parametrize("gateway,expected_attempts", [(True, 1), (False, 4)])
async def test_gateway_providers_do_not_retry_rate_limits(gateway, expected_attempts):
    attempts: list[int] = []
    helper = _Helper(_orchestrator(gateway), attempts)

    with pytest.raises(ProviderRateLimitError):
        await helper._stream_with_rate_limit_retry(tools=None, provider_kwargs={}, stream_ctx=None)

    assert len(attempts) == expected_attempts

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

"""Gateway rate-limit policy is resolved through the bound lifecycle capability."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.factory.chat_runtime_bindings import bind_chat_runtime_services
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.core.errors import ProviderRateLimitError


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator, attempts):
        self.provider = orchestrator.provider
        self._attempts = attempts
        self._session_accessor = SessionStateAccessor(SessionStateManager())
        self._provider_service = SimpleNamespace(get_rate_limit_wait_time=lambda _: 0.0)
        self.services = bind_chat_runtime_services(self)

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

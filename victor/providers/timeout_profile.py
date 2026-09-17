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

"""Named timeout slots for the provider request lifecycle (co-design review
item 18).

Today these eight numbers live scattered across five files as bare literals
(``ProviderFactoryConfig.timeout``, ``BaseProvider.__init__``'s ``timeout``
default, a hardcoded ``stream_idle_timeout_secs`` in sandhi_transport.py,
``BaseProvider``'s ``circuit_breaker_recovery_timeout`` default,
``ProviderRetryConfig``'s delay bounds, and the dead ``Timeouts.HTTP_LLM_API``
/ ``Timeouts.HTTP_EMBEDDING`` constants in victor/config/timeouts.py). This
dataclass gives them one documented home with the CURRENT effective value at
each site preserved exactly, so nothing about runtime behavior changes by
introducing it.

This phase is additive/documentation-only: ``TimeoutProfile`` is not yet
threaded through ``BaseProvider``/``ResilientProvider``/the factory to
replace those literals — that is a follow-up once the inverted invariant
below has its own fix proposal.

Known invariant, DOCUMENTED not fixed here (fixing it changes outage
behavior under load and needs its own proposal): ``ResilientProvider``'s own
docstring says ``request_timeout`` should be set LOWER than the SDK-level
timeout so ``asyncio.wait_for`` fires first with a clean ``TimeoutError``
instead of racing the SDK's own internal timeout. In practice
``ProviderFactory`` passes the same ``ProviderFactoryConfig.timeout`` value
into both ``provider_kwargs["timeout"]`` (the SDK-level slot,
``BaseProvider.timeout``) and ``request_timeout`` (see factory.py's
``ProviderFactory.create`` and ``ResilientProvider.__init__``), so the two
slots are usually equal rather than correctly ordered — the fast-clean-
timeout property the docstring promises does not actually hold by default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeoutProfile:
    """A provider's full set of request-lifecycle timeout slots, in seconds.

    Attributes:
        request_timeout_seconds: Outer ``asyncio.wait_for`` ceiling applied
            by ``ResilientProvider`` around a provider call. Current default
            mirrors ``ProviderFactoryConfig.timeout`` (factory.py).
        sdk_timeout_seconds: Timeout passed to the underlying SDK/HTTP
            client. Current default mirrors ``BaseProvider.__init__``'s
            ``timeout`` parameter default (base.py).
        stream_idle_timeout_seconds: Maximum gap between streamed chunks
            before the sandhi gateway transport gives up. Current default
            mirrors the literal in sandhi_transport.py.
        circuit_breaker_recovery_seconds: Grace period a tripped circuit
            breaker waits before allowing a trial request through. Current
            default mirrors ``BaseProvider``'s
            ``circuit_breaker_recovery_timeout`` parameter default, which
            also matches ``victor.config.timeouts.Timeouts.CIRCUIT_BREAKER_RECOVERY``.
        retry_base_delay_seconds: Initial backoff delay between retries.
            Current default mirrors ``ProviderRetryConfig.base_delay_seconds``
            (resilience.py).
        retry_max_delay_seconds: Backoff delay cap. Current default mirrors
            ``ProviderRetryConfig.max_delay_seconds`` (resilience.py).
        llm_api_http_timeout_seconds: Outer HTTP-client ceiling documented
            for LLM chat/completion calls. Current default mirrors
            ``victor.config.timeouts.Timeouts.HTTP_LLM_API`` — previously a
            dead constant with no consumer; this field is its first one.
        embedding_http_timeout_seconds: Outer HTTP-client ceiling documented
            for embedding calls. Current default mirrors
            ``victor.config.timeouts.Timeouts.HTTP_EMBEDDING`` — previously
            a dead constant with no consumer; this field is its first one.
    """

    request_timeout_seconds: float = 120.0
    sdk_timeout_seconds: float = 60.0
    stream_idle_timeout_seconds: float = 90.0
    circuit_breaker_recovery_seconds: float = 30.0
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 60.0
    llm_api_http_timeout_seconds: float = 300.0
    embedding_http_timeout_seconds: float = 120.0

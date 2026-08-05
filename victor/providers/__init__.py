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

"""LLM Provider abstraction layer."""

from victor.providers.base import (
    BaseProvider,
    Message,
    CompletionResponse,
    StreamChunk,
    ProviderError,
    ProviderNotFoundError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.core.circuit_breaker import CircuitBreakerError, CircuitState
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)
from victor.providers.registry import ProviderRegistry

__all__ = [
    # Base classes
    "BaseProvider",
    # HttpxOpenAICompatProvider is intentionally NOT eagerly imported here to
    # avoid pulling ProviderLogger → resolution → settings into the providers
    # package import chain. Import it directly:
    #   from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
    "Message",
    "CompletionResponse",
    "StreamChunk",
    # Error classes
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    # Registry
    "ProviderRegistry",
]

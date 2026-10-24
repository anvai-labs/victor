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

"""Provider message vocabulary — the kernel types every layer speaks.

The canonical location for the LLM message vocabulary (self-analysis
round 2, W4), following the ``victor.core.shared_types`` contract: pure
data — pydantic models, one frozen dataclass, error re-exports — with NO
``victor.providers.*`` dependencies, side-effect-free imports, and
backward-compat re-exports left at the original module
(``victor.providers.base``), so its 200+ import sites stay unchanged.

Why a kernel: the compiler-verified call graph ranked ``Message`` (474
in-edges), ``StreamChunk`` (380) and ``CompletionResponse`` (268) as the
most load-bearing types in the system — and every import of them dragged
``providers/base.py``'s machinery (circuit breakers, YAML context-limit
tables, error taxonomy) along. The runtime can now speak messages
without importing provider infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Provider error classes - re-exported from victor/core/errors for backward
# compatibility; all error classes are DEFINED there (single source of truth).
from victor.core.errors import (
    ModelNotFoundError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidResponseError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

__all__ = [
    "Message",
    "ToolDefinition",
    "CompletionResponse",
    "StreamChunk",
    "CacheCostModel",
    "ProviderError",
    "ProviderNotFoundError",
    "ModelNotFoundError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderInvalidResponseError",
]


class Message(BaseModel):
    """Standard message format across all providers."""

    role: str = Field(
        ...,
        description=(
            "Message role. OpenAI Chat Completions supports developer, system, user, "
            "assistant, tool, and legacy function; other provider families may differ."
        ),
    )
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional name for the message sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool calls requested by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call being responded to"
    )
    images: Optional[List[str]] = Field(
        default=None,
        description="Image data URIs (data:image/png;base64,...) for multimodal user messages",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Internal metadata (e.g. MessageSource). Never serialised to providers.",
        exclude=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return self.model_dump()


class ToolDefinition(BaseModel):
    """Standard tool definition format.

    The optional schema_level tracks which verbosity tier was used to
    generate the description and parameters. It is excluded from
    serialization (never sent to providers) — used only for internal
    cache-aware ordering.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="What the tool does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for tool parameters")
    schema_level: Optional[str] = Field(default=None, exclude=True)


class CompletionResponse(BaseModel):
    """Standard completion response format."""

    content: str = Field(..., description="Generated content")
    role: str = Field(default="assistant", description="Response role")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls requested")
    stop_reason: Optional[str] = Field(None, description="Why generation stopped")
    usage: Optional[Dict[str, int | bool]] = Field(
        None, description="Token usage stats and reasoning inclusion convention"
    )
    model: Optional[str] = Field(None, description="Model used")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata (e.g., reasoning_content)"
    )


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    content: str = Field(default="", description="Incremental content")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: Optional[str] = None
    is_final: bool = Field(default=False, description="Is this the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata (e.g., reasoning_content)"
    )
    usage: Optional[Dict[str, int | bool]] = Field(
        default=None,
        description="Token usage stats (typically on final chunk). Keys: prompt_tokens, completion_tokens, total_tokens, cache_creation_input_tokens, cache_read_input_tokens",
    )


@dataclass(frozen=True)
class CacheCostModel:
    """Characterized caching economics for a provider capability (FEP-0011).

    Replaces branching on a single boolean with a value object the prompt
    assembler can optimize against (pruning aggressiveness, stable/dynamic
    prefix split, prefix freeze size). See FEP-0011 for the full design.

    Phase 1: additive only. ``BaseProvider.cache_cost_model()`` defaults
    to ``CacheCostModel(supported=self.supports_prompt_caching())`` so existing
    providers behave identically; providers may override to advertise real
    numbers. The assembler still reads the booleans today; consumption is a
    later phase.

    Attributes:
        supported: Whether this caching capability is available at all.
        read_discount: Fraction saved on a cached read (0.0–1.0).
        write_overhead: Cost multiplier for a cache write (>=1.0; 1.0 = none).
        ttl_seconds: Lifetime of a cache entry (0.0 = no TTL / indefinite).
        min_prefix_tokens: Minimum stable prefix to be eligible for caching.
        max_cache_tokens: Maximum cacheable tokens (0 = no published limit).
        prefix_granularity: Smallest cacheable unit ("token"|"message"|"system_block").
    """

    supported: bool = False
    read_discount: float = 0.0
    write_overhead: float = 1.0
    ttl_seconds: float = 0.0
    min_prefix_tokens: int = 0
    max_cache_tokens: int = 0
    prefix_granularity: str = "token"

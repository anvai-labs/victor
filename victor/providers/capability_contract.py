"""Canonical, immutable capability surface for a provider+model pair.

Wave 4: Consolidates ToolCallingCapabilities (agent/tool_calling/base.py) and
ProviderRuntimeCapabilities (providers/runtime_capabilities.py) into a single
frozen Pydantic model.  ProviderState carries this as an additive field;
existing capabilities / runtime_capabilities fields are preserved for backward
compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from victor.agent.tool_calling.base import ToolCallingCapabilities
    from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities


class ProviderCapabilityContract(BaseModel, frozen=True):
    """Canonical, immutable capability surface for a provider+model pair.

    Consolidates fields from both ToolCallingCapabilities and
    ProviderRuntimeCapabilities so callers have a single source of truth.

    Usage::

        contract = ProviderCapabilityContract.from_tool_calling(caps, runtime)
        if contract.native_tool_calls:
            ...
    """

    provider: str
    model: str
    context_window: int

    # From ToolCallingCapabilities
    native_tool_calls: bool = False
    streaming_tool_calls: bool = False
    parallel_tool_calls: bool = False
    json_fallback_parsing: bool = False
    xml_fallback_parsing: bool = False
    thinking_mode: bool = False
    requires_strict_prompting: bool = False

    # From ProviderRuntimeCapabilities
    supports_streaming: bool = False
    source: str = "config"

    @classmethod
    def from_tool_calling(
        cls,
        caps: "Optional[ToolCallingCapabilities]",
        runtime: "Optional[ProviderRuntimeCapabilities]",
        provider: str = "",
        model: str = "",
        context_window: int = 0,
    ) -> "ProviderCapabilityContract":
        """Build a contract by merging both source structs.

        ``runtime`` is used as the primary source for ``provider``, ``model``,
        and ``context_window``.  ``caps`` is used for tool-calling flags.
        Explicit kwargs override values from both structs.

        Args:
            caps:           ToolCallingCapabilities instance (may be None).
            runtime:        ProviderRuntimeCapabilities instance (may be None).
            provider:       Override for provider name.
            model:          Override for model name.
            context_window: Override for context window size.
        """
        resolved_provider = provider or (runtime.provider if runtime else "unknown")
        resolved_model = model or (runtime.model if runtime else "unknown")
        resolved_cw = context_window or (runtime.context_window if runtime else 0)

        return cls(
            provider=resolved_provider,
            model=resolved_model,
            context_window=resolved_cw,
            native_tool_calls=getattr(caps, "native_tool_calls", False),
            streaming_tool_calls=getattr(caps, "streaming_tool_calls", False),
            parallel_tool_calls=getattr(caps, "parallel_tool_calls", False),
            json_fallback_parsing=getattr(caps, "json_fallback_parsing", False),
            xml_fallback_parsing=getattr(caps, "xml_fallback_parsing", False),
            thinking_mode=getattr(caps, "thinking_mode", False),
            requires_strict_prompting=getattr(caps, "requires_strict_prompting", False),
            supports_streaming=getattr(runtime, "supports_streaming", False),
            source=getattr(runtime, "source", "config"),
        )

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

"""Adaptive serializer with automatic format selection.

Main entry point for the token-optimized serialization system.
Automatically selects the most token-efficient format based on:
- Data structure and characteristics
- Model/provider capabilities and preferences
- Tool-specific configurations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    SerializationResult,
    DataCharacteristics,
    FORMAT_DESCRIPTIONS,
)
from victor.processing.serialization.analyzer import DataAnalyzer, get_data_analyzer
from victor.processing.serialization.capabilities import (
    CapabilityRegistry,
    get_capability_registry,
    is_serialization_enabled,
)
from victor.processing.serialization.formats import get_format_registry
from victor.processing.serialization.formats.base import EncodingResult
from victor.processing.serialization.tool_config import (
    ToolSerializationRegistry,
    get_tool_serialization_registry,
)
from victor.processing.serialization.intelligent_selector import (
    IntelligentFormatSelector,
    SelectionContext as IntelligentContext,
    get_intelligent_selector,
)
from victor.processing.serialization.metrics import (
    record_serialization_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SerializationContext:
    """Context for serialization decisions.

    Provides information about the current request context
    to enable intelligent format selection.
    """

    # Provider and model
    provider: Optional[str] = None
    model: Optional[str] = None

    # Tool information
    tool_name: Optional[str] = None
    tool_category: Optional[str] = None
    tool_operation: Optional[str] = None  # Operation within tool (e.g., "log" for git)

    # Override configuration (takes precedence)
    config_override: Optional[SerializationConfig] = None

    # Hints for format selection
    prefer_readable: bool = False  # Prefer human-readable formats
    prefer_compact: bool = False  # Prefer most compact format
    prefer_speed: bool = False  # Prefer faster encoding
    max_tokens: Optional[int] = None  # Token budget for output

    # Context pressure (for intelligent selection)
    remaining_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    response_token_reserve: int = 4096

    # Metrics recording
    record_metrics: bool = True  # Record serialization metrics


@dataclass
class SerializationMetrics:
    """Metrics collected during serialization."""

    # Size metrics
    original_json_chars: int = 0
    original_json_tokens: int = 0
    serialized_chars: int = 0
    serialized_tokens: int = 0

    # Savings
    char_savings_percent: float = 0.0
    token_savings_percent: float = 0.0

    # Timing (if measured)
    analysis_time_ms: float = 0.0
    encoding_time_ms: float = 0.0

    # Format selection
    format_selected: str = ""
    selection_reason: str = ""
    candidates_considered: List[str] = field(default_factory=list)


class AdaptiveSerializer:
    """Adaptive serializer with automatic format selection.

    Analyzes data and selects the most token-efficient format
    based on data characteristics and model/provider capabilities.

    Usage:
        serializer = AdaptiveSerializer()
        context = SerializationContext(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tool_name="database_tool",
        )
        result = serializer.serialize(data, context)
        print(f"Saved {result.estimated_savings_percent}% tokens")
    """

    def __init__(
        self,
        analyzer: Optional[DataAnalyzer] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        tool_registry: Optional[ToolSerializationRegistry] = None,
        intelligent_selector: Optional[IntelligentFormatSelector] = None,
    ):
        """Initialize serializer.

        Args:
            analyzer: DataAnalyzer instance (uses global if not provided)
            capability_registry: CapabilityRegistry instance
            tool_registry: ToolSerializationRegistry instance
            intelligent_selector: IntelligentFormatSelector instance
        """
        self._analyzer = analyzer or get_data_analyzer()
        self._capability_registry = capability_registry or get_capability_registry()
        self._tool_registry = tool_registry or get_tool_serialization_registry()
        self._intelligent_selector = intelligent_selector or get_intelligent_selector()
        self._format_registry = get_format_registry()
        self._metrics: Optional[SerializationMetrics] = None
        self._last_context: Optional[SerializationContext] = None
        self._last_characteristics: Optional[DataCharacteristics] = None
        # Stats tracking
        self._serialization_count: int = 0
        self._total_tokens_saved: int = 0
        self._total_original_tokens: int = 0
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
        self._format_stats: Dict[str, Dict[str, Any]] = {}

    def serialize(
        self,
        data: Any,
        context: Optional[SerializationContext] = None,
        format_override: Optional[SerializationFormat] = None,
    ) -> SerializationResult:
        """Serialize data with automatic format selection.

        Args:
            data: Data to serialize
            context: Serialization context for format selection
            format_override: Force specific format (bypasses auto-selection)

        Returns:
            SerializationResult with serialized content and metrics
        """
        context = context or SerializationContext()
        self._metrics = SerializationMetrics()

        # Check if serialization is globally disabled via Settings
        if not is_serialization_enabled():
            try:
                content = json.dumps(data, indent=2, default=str)
            except Exception:
                content = str(data)
            self._metrics.format_selected = SerializationFormat.JSON.value
            self._metrics.selection_reason = "globally_disabled"
            return SerializationResult(
                content=content,
                format=SerializationFormat.JSON,
                format_hint=None,
                original_json_estimate=0,
                serialized_tokens=len(content) // 4,
                estimated_savings_percent=0.0,
            )

        # Check if tool wants serialization disabled
        if context.tool_name:
            tool_config = self._tool_registry.get_tool_config(context.tool_name)
            if not tool_config.should_serialize(context.tool_operation):
                # Return plain JSON for tools that opt out
                try:
                    content = json.dumps(data, indent=2, default=str)
                except Exception:
                    content = str(data)
                self._metrics.format_selected = SerializationFormat.JSON.value
                self._metrics.selection_reason = "tool_disabled"
                return SerializationResult(
                    content=content,
                    format=SerializationFormat.JSON,
                    format_hint=None,
                    original_json_estimate=0,
                    serialized_tokens=len(content) // 4,
                    estimated_savings_percent=0.0,
                )

        # Get configuration for this context (now includes tool config)
        config = self._get_config(context)

        # Analyze data
        characteristics = self._analyzer.analyze(data)
        self._metrics.original_json_chars = characteristics.estimated_json_chars
        self._metrics.original_json_tokens = characteristics.estimated_json_tokens

        # Select format
        if format_override:
            selected_format = format_override
            self._metrics.selection_reason = "format_override"
        elif config.preferred_format:
            selected_format = config.preferred_format
            self._metrics.selection_reason = "config_preferred"
        else:
            selected_format = self._select_format(data, characteristics, config, context)

        self._metrics.format_selected = selected_format.value

        # Store for later retrieval
        self._last_context = context
        self._last_characteristics = characteristics

        # Encode data
        result = self._encode(data, characteristics, config, selected_format)

        # Calculate savings
        if result.success:
            self._calculate_savings(result, characteristics)

        # Record metrics for learning (if enabled)
        if context.record_metrics and result.success:
            try:
                record_serialization_metrics(
                    self._metrics,
                    context,
                    characteristics,
                )
            except Exception as e:
                logger.debug(f"Failed to record metrics: {e}")

        serialization_result = self._build_result(result, selected_format, config, characteristics)

        # Update internal statistics
        if result.success:
            self._update_stats(serialization_result, context)

        return serialization_result

    def serialize_for_tool(
        self,
        data: Any,
        tool_name: str,
        operation: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> SerializationResult:
        """Convenience method for tool output serialization.

        Args:
            data: Tool output data
            tool_name: Name of the tool
            operation: Operation within tool (e.g., "log" for git)
            provider: Provider name
            model: Model name

        Returns:
            SerializationResult
        """
        context = SerializationContext(
            provider=provider,
            model=model,
            tool_name=tool_name,
            tool_operation=operation,
        )
        return self.serialize(data, context)

    def get_last_metrics(self) -> Optional[SerializationMetrics]:
        """Get metrics from last serialization.

        Returns:
            SerializationMetrics or None if no serialization performed
        """
        return self._metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get serialization statistics.

        Returns:
            Dict with serialization stats including totals, by-tool, and by-format breakdowns.
        """
        avg_savings = 0.0
        if self._total_original_tokens > 0:
            avg_savings = (self._total_tokens_saved / self._total_original_tokens) * 100

        # Build by_format list sorted by usage
        format_list = [
            {
                "format": fmt,
                "usage_count": stats.get("count", 0),
                "avg_savings_percent": stats.get("avg_savings", 0.0),
            }
            for fmt, stats in self._format_stats.items()
        ]
        format_list.sort(key=lambda x: x["usage_count"], reverse=True)

        return {
            "total_serializations": self._serialization_count,
            "total_tokens_saved": self._total_tokens_saved,
            "avg_savings_percent": avg_savings,
            "by_tool": self._tool_stats,
            "by_format": format_list,
        }

    def reset_stats(self) -> None:
        """Reset serialization statistics."""
        self._serialization_count = 0
        self._total_tokens_saved = 0
        self._total_original_tokens = 0
        self._tool_stats = {}
        self._format_stats = {}

    def _update_stats(
        self,
        result: "SerializationResult",
        context: SerializationContext,
    ) -> None:
        """Update internal statistics after a serialization.

        Args:
            result: The serialization result
            context: The serialization context used
        """
        self._serialization_count += 1

        # Calculate tokens saved
        original = result.original_json_estimate
        serialized = result.serialized_tokens
        saved = max(0, original - serialized)
        self._total_tokens_saved += saved
        self._total_original_tokens += original

        savings_pct = result.estimated_savings_percent * 100

        # Update tool stats
        if context.tool_name:
            if context.tool_name not in self._tool_stats:
                self._tool_stats[context.tool_name] = {
                    "usage_count": 0,
                    "total_savings": 0,
                    "avg_savings_percent": 0.0,
                    "preferred_format": result.format.value,
                }
            ts = self._tool_stats[context.tool_name]
            ts["usage_count"] += 1
            ts["total_savings"] += saved
            # Running average
            ts["avg_savings_percent"] = (
                ts["avg_savings_percent"] * (ts["usage_count"] - 1) + savings_pct
            ) / ts["usage_count"]
            ts["preferred_format"] = result.format.value

        # Update format stats
        fmt_name = result.format.value
        if fmt_name not in self._format_stats:
            self._format_stats[fmt_name] = {
                "count": 0,
                "total_savings": 0,
                "avg_savings": 0.0,
            }
        fs = self._format_stats[fmt_name]
        fs["count"] += 1
        fs["total_savings"] += saved
        fs["avg_savings"] = (fs["avg_savings"] * (fs["count"] - 1) + savings_pct) / fs["count"]

    def _get_config(self, context: SerializationContext) -> SerializationConfig:
        """Get serialization config for context.

        Hierarchical resolution:
        1. Start with provider/model capabilities
        2. Apply tool-specific config (from tool_serialization section)
        3. Apply context hints (prefer_readable, prefer_compact)
        4. Apply config_override (highest priority)

        Args:
            context: Serialization context

        Returns:
            SerializationConfig with resolved settings
        """
        # Start with capabilities from registry
        if context.provider:
            capabilities = self._capability_registry.get_capabilities(
                context.provider, context.model
            )
            config = capabilities.to_config()
        else:
            config = SerializationConfig()

        # Apply tool-specific config
        if context.tool_name:
            tool_config = self._tool_registry.get_tool_config(context.tool_name)

            # Merge tool config - tool config can override format preferences
            if tool_config.preferred_format:
                config.preferred_format = tool_config.preferred_format
            if tool_config.preferred_formats:
                config.allowed_formats = tool_config.preferred_formats
            config.min_array_size_for_tabular = tool_config.min_rows_for_tabular
            config.min_savings_threshold = tool_config.min_savings_threshold
            config.include_format_hint = tool_config.include_format_hint

        # Apply context hints
        if context.prefer_readable:
            # Prefer markdown table for readability
            if SerializationFormat.MARKDOWN_TABLE not in config.allowed_formats:
                config.allowed_formats.insert(0, SerializationFormat.MARKDOWN_TABLE)

        if context.prefer_compact:
            # Prioritize CSV and TOON
            config.allowed_formats = [
                SerializationFormat.CSV,
                SerializationFormat.TOON,
                SerializationFormat.JSON_MINIFIED,
                SerializationFormat.JSON,
            ]
            config.min_savings_threshold = 0.10  # More aggressive

        # Apply override config
        if context.config_override:
            config = config.merge_with(context.config_override)

        return config

    def _select_format(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
        context: SerializationContext,
    ) -> SerializationFormat:
        """Select best format for data using intelligent selection.

        Uses multi-factor scoring:
        1. Base encoder suitability scores
        2. Historical performance from metrics
        3. Current context (token pressure, urgency)

        Args:
            data: Data to serialize
            characteristics: Data analysis results
            config: Serialization configuration
            context: Serialization context

        Returns:
            Selected SerializationFormat
        """
        # Collect encoder suitability scores
        encoder_scores: Dict[SerializationFormat, float] = {}
        for fmt in config.allowed_formats:
            encoder = self._format_registry.get_encoder(fmt)
            if encoder and encoder.can_encode(data, characteristics):
                encoder_scores[fmt] = encoder.suitability_score(characteristics, config)

        if not encoder_scores:
            self._metrics.selection_reason = "no_suitable_encoders"
            return config.fallback_format

        # Build intelligent selection context
        intelligent_context = IntelligentContext(
            tool_name=context.tool_name,
            tool_operation=context.tool_operation,
            provider=context.provider,
            model=context.model,
            remaining_context_tokens=context.remaining_context_tokens,
            max_context_tokens=context.max_context_tokens,
            response_token_reserve=context.response_token_reserve,
            data_size_bytes=len(str(data)),
            estimated_json_tokens=characteristics.estimated_json_tokens,
            prefer_speed=context.prefer_speed,
            prefer_savings=context.prefer_compact,
            prefer_readability=context.prefer_readable,
        )

        # Use intelligent selector
        selected_format, reason = self._intelligent_selector.select_best_format(
            data=data,
            characteristics=characteristics,
            config=config,
            context=intelligent_context,
            encoder_scores=encoder_scores,
        )

        self._metrics.selection_reason = reason
        return selected_format

    def _encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
        format_id: SerializationFormat,
    ) -> EncodingResult:
        """Encode data using selected format.

        Args:
            data: Data to encode
            characteristics: Data characteristics
            config: Configuration
            format_id: Format to use

        Returns:
            EncodingResult from encoder
        """
        encoder = self._format_registry.get_encoder(format_id)

        if not encoder:
            # Fall back to JSON if encoder not found
            logger.warning(f"Encoder not found for {format_id}, falling back to JSON")
            encoder = self._format_registry.get_encoder(SerializationFormat.JSON)

        if not encoder:
            # Ultimate fallback - use json.dumps directly
            try:
                content = json.dumps(data, default=str)
                return EncodingResult(content=content, success=True)
            except Exception as e:
                return EncodingResult(content="", success=False, error=str(e))

        # Invoke pre-encode hooks
        self._format_registry.invoke_hooks("pre_encode", data, characteristics, config)

        # Encode
        result = encoder.encode(data, characteristics, config)

        # Invoke post-encode hooks
        self._format_registry.invoke_hooks("post_encode", result, encoder)

        return result

    def _calculate_savings(
        self,
        result: EncodingResult,
        characteristics: DataCharacteristics,
    ) -> None:
        """Calculate token savings.

        Args:
            result: Encoding result
            characteristics: Data characteristics
        """
        self._metrics.serialized_chars = len(result.content)
        self._metrics.serialized_tokens = result.metadata.get(
            "estimated_tokens", len(result.content) // 4
        )

        # Calculate savings
        if characteristics.estimated_json_chars > 0:
            self._metrics.char_savings_percent = (
                (characteristics.estimated_json_chars - self._metrics.serialized_chars)
                / characteristics.estimated_json_chars
            ) * 100

        if characteristics.estimated_json_tokens > 0:
            self._metrics.token_savings_percent = (
                (characteristics.estimated_json_tokens - self._metrics.serialized_tokens)
                / characteristics.estimated_json_tokens
            ) * 100

    def _build_result(
        self,
        encoding_result: EncodingResult,
        format_id: SerializationFormat,
        config: SerializationConfig,
        characteristics: DataCharacteristics,
    ) -> SerializationResult:
        """Build final serialization result.

        Args:
            encoding_result: Raw encoding result
            format_id: Format used
            config: Configuration used
            characteristics: Data characteristics

        Returns:
            Complete SerializationResult
        """
        # Generate format hint if enabled
        format_hint = None
        if config.include_format_hint and format_id != SerializationFormat.JSON:
            encoder = self._format_registry.get_encoder(format_id)
            if encoder:
                format_hint = encoder.get_format_hint(config)
            else:
                format_hint = f"Format: {FORMAT_DESCRIPTIONS.get(format_id, format_id.value)}"

        # Build warnings list
        warnings = []
        if not encoding_result.success:
            warnings.append(f"Encoding failed: {encoding_result.error}")

        return SerializationResult(
            content=encoding_result.content,
            format=format_id,
            format_hint=format_hint,
            original_json_estimate=characteristics.estimated_json_tokens,
            serialized_tokens=self._metrics.serialized_tokens if self._metrics else 0,
            estimated_savings_percent=(
                self._metrics.token_savings_percent / 100 if self._metrics else 0
            ),
            reference_table=encoding_result.reference_table,
            characteristics=characteristics if config.debug_mode else None,
            warnings=warnings,
        )


# Global serializer instance
_serializer: Optional[AdaptiveSerializer] = None


def get_adaptive_serializer() -> AdaptiveSerializer:
    """Get the global adaptive serializer.

    Returns:
        AdaptiveSerializer instance
    """
    global _serializer
    if _serializer is None:
        _serializer = AdaptiveSerializer()
    return _serializer


def reset_adaptive_serializer() -> None:
    """Reset the global serializer (mainly for testing)."""
    global _serializer
    _serializer = None


# Convenience function
def serialize(
    data: Any,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tool_name: Optional[str] = None,
    format_override: Optional[SerializationFormat] = None,
) -> SerializationResult:
    """Convenience function for quick serialization.

    Args:
        data: Data to serialize
        provider: Provider name
        model: Model name
        tool_name: Tool name
        format_override: Force specific format

    Returns:
        SerializationResult
    """
    serializer = get_adaptive_serializer()
    context = SerializationContext(
        provider=provider,
        model=model,
        tool_name=tool_name,
    )
    return serializer.serialize(data, context, format_override)

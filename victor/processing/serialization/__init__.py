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

"""Token-optimized serialization system for LLM communication.

This module provides an adaptive serialization layer that automatically
selects the most token-efficient format based on:
- Data structure characteristics
- Model/provider capabilities
- Tool-specific configurations
- User preferences

Supported formats:
- JSON (standard, universal compatibility)
- TOON (Token-Oriented Object Notation, 30-60% savings for tabular data)
- CSV (maximum compression for flat data)
- Markdown Table (readable tabular format)
- Minified JSON (10-20% savings, drop-in replacement)
- Reference Encoded (40-70% savings for repetitive data)

Usage:
    from victor.processing.serialization import AdaptiveSerializer, SerializationContext

    # Create serializer with context
    serializer = AdaptiveSerializer()
    context = SerializationContext(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        tool_name="database_tool",
    )

    # Serialize data (format auto-selected)
    result = serializer.serialize(data, context)
    print(f"Format: {result.format}, Tokens saved: {result.estimated_savings_percent*100}%")

    # Or specify format explicitly
    result = serializer.serialize(data, context, format_override=SerializationFormat.TOON)

    # Quick serialization with convenience function
    from victor.processing.serialization import serialize
    result = serialize(data, provider="anthropic", model="claude-sonnet-4-20250514")
"""

from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    SerializationResult,
    DataCharacteristics,
    DataStructureType,
    FORMAT_DESCRIPTIONS,
)
from victor.processing.serialization.analyzer import DataAnalyzer, get_data_analyzer
from victor.processing.serialization.capabilities import (
    ModelSerializationCapabilities,
    CapabilityRegistry,
    get_capability_registry,
    reset_capability_registry,
    config_from_settings,
    is_serialization_enabled,
)
from victor.processing.serialization.adaptive import (
    AdaptiveSerializer,
    SerializationContext,
    SerializationMetrics,
    get_adaptive_serializer,
    reset_adaptive_serializer,
    serialize,
)
from victor.processing.serialization.tool_config import (
    ToolOutputType,
    ToolSerializationConfig,
    ToolSerializationRegistry,
    get_tool_serialization_registry,
    reset_tool_serialization_registry,
)
from victor.processing.serialization.metrics import (
    SerializationMetricRecord,
    SerializationMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
    record_serialization_metrics,
)
from victor.processing.serialization.intelligent_selector import (
    IntelligentFormatSelector,
    SelectionContext,
    FormatScore,
    get_intelligent_selector,
    reset_intelligent_selector,
)

__all__ = [
    # Core types
    "SerializationFormat",
    "SerializationConfig",
    "SerializationResult",
    "DataCharacteristics",
    "DataStructureType",
    "FORMAT_DESCRIPTIONS",
    # Analyzer
    "DataAnalyzer",
    "get_data_analyzer",
    # Capabilities
    "ModelSerializationCapabilities",
    "CapabilityRegistry",
    "get_capability_registry",
    "reset_capability_registry",
    "config_from_settings",
    "is_serialization_enabled",
    # Main interface
    "AdaptiveSerializer",
    "SerializationContext",
    "SerializationMetrics",
    "get_adaptive_serializer",
    "reset_adaptive_serializer",
    "serialize",
    # Tool-specific config
    "ToolOutputType",
    "ToolSerializationConfig",
    "ToolSerializationRegistry",
    "get_tool_serialization_registry",
    "reset_tool_serialization_registry",
    # Metrics
    "SerializationMetricRecord",
    "SerializationMetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
    "record_serialization_metrics",
    # Intelligent selector
    "IntelligentFormatSelector",
    "SelectionContext",
    "FormatScore",
    "get_intelligent_selector",
    "reset_intelligent_selector",
]

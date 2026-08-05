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

"""Core types and enums for the serialization system.

Defines the fundamental data structures used throughout the
token-optimized serialization system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SerializationFormat(Enum):
    """Supported serialization formats.

    Each format has different trade-offs between:
    - Token efficiency
    - LLM compatibility
    - Human readability
    - Structure preservation
    """

    # Standard JSON - universal compatibility, baseline
    JSON = "json"

    # Minified JSON - removes whitespace, 10-20% savings
    JSON_MINIFIED = "json_minified"

    # TOON - Token-Oriented Object Notation, 30-60% savings for tabular data
    TOON = "toon"

    # CSV - maximum compression for flat data, 40-70% savings
    CSV = "csv"

    # Markdown table - readable tabular format
    MARKDOWN_TABLE = "markdown_table"

    # Raw text - pass-through for non-structured data
    RAW = "raw"

    # Reference encoding - for highly repetitive data
    REFERENCE_ENCODED = "reference_encoded"


class DataStructureType(Enum):
    """Classification of data structure types.

    Used to determine optimal serialization format.
    """

    # List of objects with identical keys
    UNIFORM_ARRAY = "uniform_array"

    # List of primitive values
    PRIMITIVE_ARRAY = "primitive_array"

    # Nested/heterogeneous structure
    NESTED_OBJECT = "nested_object"

    # Simple key-value pairs
    FLAT_OBJECT = "flat_object"

    # Plain text content
    TEXT = "text"

    # Mixed or unknown structure
    MIXED = "mixed"

    # Empty or null data
    EMPTY = "empty"


@dataclass
class DataCharacteristics:
    """Analysis results describing data structure characteristics.

    Used by the adaptive serializer to select optimal format.
    """

    # Structure classification
    structure_type: DataStructureType = DataStructureType.MIXED

    # Size metrics
    total_elements: int = 0
    nesting_depth: int = 0
    unique_keys: int = 0
    total_keys: int = 0

    # For arrays
    array_length: int = 0
    array_uniformity: float = 0.0  # 0.0-1.0, 1.0 = perfectly uniform

    # Repetition metrics (for reference encoding)
    key_repetition_ratio: float = 0.0  # How often keys repeat
    value_repetition_ratio: float = 0.0  # How often values repeat
    common_values: Dict[str, int] = field(default_factory=dict)  # value -> count

    # Content characteristics
    has_nested_objects: bool = False
    has_nested_arrays: bool = False
    has_null_values: bool = False
    has_special_chars: bool = False  # Chars that need escaping

    # Size estimates
    estimated_json_tokens: int = 0
    estimated_json_chars: int = 0

    # Field information (for uniform arrays)
    field_names: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)

    def is_tabular(self) -> bool:
        """Check if data is suitable for tabular format (TOON/CSV)."""
        return (
            self.structure_type == DataStructureType.UNIFORM_ARRAY
            and self.array_uniformity >= 0.9
            and not self.has_nested_objects
            and not self.has_nested_arrays
        )

    def is_flat(self) -> bool:
        """Check if data is flat (no nesting)."""
        return self.nesting_depth <= 1 and not self.has_nested_objects

    def has_high_repetition(self) -> bool:
        """Check if data has high value repetition (good for reference encoding)."""
        return self.value_repetition_ratio >= 0.5

    def estimated_savings(self, target_format: SerializationFormat) -> float:
        """Estimate token savings for a target format.

        Returns:
            Estimated percentage savings (0.0-1.0)
        """
        if target_format == SerializationFormat.JSON:
            return 0.0
        elif target_format == SerializationFormat.JSON_MINIFIED:
            return 0.15  # ~15% from whitespace removal
        elif target_format == SerializationFormat.TOON:
            if self.is_tabular():
                # TOON excels with tabular data
                base_savings = 0.45
                # More fields = more savings (field names declared once)
                field_bonus = min(0.15, self.unique_keys * 0.02)
                # More rows = more savings
                row_bonus = min(0.10, self.array_length * 0.005)
                return min(0.65, base_savings + field_bonus + row_bonus)
            else:
                return 0.20  # Modest savings for non-tabular
        elif target_format == SerializationFormat.CSV:
            if self.is_tabular() and not self.has_special_chars:
                return 0.55  # CSV is most compact for clean tabular
            return 0.0  # CSV not suitable
        elif target_format == SerializationFormat.MARKDOWN_TABLE:
            if self.is_tabular():
                return 0.30  # Less efficient than CSV but readable
            return 0.0
        elif target_format == SerializationFormat.REFERENCE_ENCODED:
            if self.has_high_repetition():
                return 0.40 + (self.value_repetition_ratio * 0.30)
            return 0.0
        return 0.0


@dataclass
class SerializationConfig:
    """Configuration for serialization behavior.

    Can be set at multiple levels:
    - Global defaults
    - Provider-specific
    - Model-specific
    - Tool-specific
    """

    # Preferred format (None = auto-select)
    preferred_format: Optional[SerializationFormat] = None

    # Formats to consider for auto-selection (order = preference)
    allowed_formats: List[SerializationFormat] = field(
        default_factory=lambda: [
            SerializationFormat.TOON,
            SerializationFormat.CSV,
            SerializationFormat.JSON_MINIFIED,
            SerializationFormat.JSON,
        ]
    )

    # Formats to never use
    disabled_formats: Set[SerializationFormat] = field(default_factory=set)

    # Minimum array size to consider tabular formats
    min_array_size_for_tabular: int = 3

    # Minimum estimated savings to switch from JSON
    min_savings_threshold: float = 0.20

    # Include format hint in output (helps LLM parse)
    include_format_hint: bool = True

    # Format hint template
    format_hint_template: str = "Data format: {format_name}. {format_description}"

    # Maximum nesting depth for non-JSON formats
    max_nesting_for_compact: int = 1

    # Enable reference encoding for repetitive values
    enable_reference_encoding: bool = True

    # Minimum repetition ratio to use reference encoding
    min_repetition_for_references: float = 0.5

    # Fallback format if preferred fails
    fallback_format: SerializationFormat = SerializationFormat.JSON

    # Debug mode - include analysis metadata
    debug_mode: bool = False

    def merge_with(self, other: "SerializationConfig") -> "SerializationConfig":
        """Merge with another config (other takes precedence for set values)."""
        return SerializationConfig(
            preferred_format=other.preferred_format or self.preferred_format,
            allowed_formats=other.allowed_formats or self.allowed_formats,
            disabled_formats=self.disabled_formats | other.disabled_formats,
            min_array_size_for_tabular=(
                other.min_array_size_for_tabular
                if other.min_array_size_for_tabular != 3
                else self.min_array_size_for_tabular
            ),
            min_savings_threshold=(
                other.min_savings_threshold
                if other.min_savings_threshold != 0.20
                else self.min_savings_threshold
            ),
            include_format_hint=other.include_format_hint,
            format_hint_template=other.format_hint_template or self.format_hint_template,
            max_nesting_for_compact=(
                other.max_nesting_for_compact
                if other.max_nesting_for_compact != 1
                else self.max_nesting_for_compact
            ),
            enable_reference_encoding=other.enable_reference_encoding,
            min_repetition_for_references=(
                other.min_repetition_for_references
                if other.min_repetition_for_references != 0.5
                else self.min_repetition_for_references
            ),
            fallback_format=other.fallback_format,
            debug_mode=other.debug_mode or self.debug_mode,
        )


@dataclass
class SerializationResult:
    """Result of serialization operation.

    Contains the serialized data plus metadata about the serialization.
    """

    # Serialized content
    content: str

    # Format used
    format: SerializationFormat

    # Format hint for LLM (if enabled)
    format_hint: Optional[str] = None

    # Metrics
    original_json_estimate: int = 0  # Estimated JSON tokens
    serialized_tokens: int = 0  # Estimated tokens in result
    estimated_savings_percent: float = 0.0

    # For reference encoding
    reference_table: Optional[Dict[str, str]] = None

    # Analysis metadata (if debug mode)
    characteristics: Optional[DataCharacteristics] = None

    # Any warnings during serialization
    warnings: List[str] = field(default_factory=list)

    @property
    def full_content(self) -> str:
        """Get content with format hint if present."""
        if self.format_hint:
            return f"{self.format_hint}\n\n{self.content}"
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "format": self.format.value,
            "content_length": len(self.content),
            "original_estimate": self.original_json_estimate,
            "serialized_tokens": self.serialized_tokens,
            "savings_percent": round(self.estimated_savings_percent * 100, 1),
            "has_format_hint": self.format_hint is not None,
            "has_reference_table": self.reference_table is not None,
            "warnings": self.warnings,
        }


# Format descriptions for hints
FORMAT_DESCRIPTIONS: Dict[SerializationFormat, str] = {
    SerializationFormat.JSON: "Standard JSON object/array notation.",
    SerializationFormat.JSON_MINIFIED: "Compact JSON without whitespace.",
    SerializationFormat.TOON: (
        "TOON (Token-Oriented Object Notation): "
        "Arrays declared as name[count]{fields}: followed by comma-separated rows."
    ),
    SerializationFormat.CSV: ("CSV format: Header row followed by comma-separated data rows."),
    SerializationFormat.MARKDOWN_TABLE: ("Markdown table with | separators and header divider."),
    SerializationFormat.RAW: "Raw text content.",
    SerializationFormat.REFERENCE_ENCODED: (
        "Reference-encoded: Common values defined in refs={} block, " "referenced by key in data."
    ),
}

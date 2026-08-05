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

"""TOON (Token-Oriented Object Notation) format encoder.

TOON achieves 30-60% token savings for tabular data by:
- Declaring field names once in a header
- Using comma-separated values for each row
- Minimizing syntax overhead (no repeated keys)

Format:
    arrayName[count]{field1,field2,field3}:
    value1,value2,value3
    value4,value5,value6
    ...

Example:
    users[3]{id,name,email}:
    1,John,john@example.com
    2,Jane,jane@example.com
    3,Bob,bob@example.com
"""

from __future__ import annotations

import logging
from typing import Any

from victor.processing.serialization.formats.base import FormatEncoder, EncodingResult
from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
    DataStructureType,
)

logger = logging.getLogger(__name__)


class TOONEncoder(FormatEncoder):
    """TOON (Token-Oriented Object Notation) encoder.

    Optimized for uniform arrays of objects. Achieves 30-60% token
    savings by declaring field names once and using CSV-like rows.
    """

    format_id = SerializationFormat.TOON
    format_name = "TOON"
    format_description = (
        "Token-Oriented Object Notation: "
        "Header declares fields once, rows use comma-separated values. "
        "Format: name[count]{fields}: followed by value rows."
    )

    # Capabilities
    supports_nested = False  # TOON is for flat tabular data
    supports_arrays = True
    supports_special_chars = True  # With proper escaping
    max_nesting_depth = 1

    # Token efficiency (30-60% better than JSON for tabular)
    base_efficiency = 0.55

    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """Check if TOON can encode this data.

        TOON requires:
        - Array of objects (uniform structure)
        - High uniformity (>90% same keys)
        - No nested objects or arrays

        Args:
            data: Data to check
            characteristics: Data analysis results

        Returns:
            True if TOON can efficiently encode this data
        """
        if not isinstance(data, list):
            return False

        if not data:
            return False

        # Must be uniform array of dicts
        if characteristics.structure_type != DataStructureType.UNIFORM_ARRAY:
            return False

        # Must have high uniformity
        if characteristics.array_uniformity < 0.9:
            return False

        # No nested structures
        if characteristics.has_nested_objects or characteristics.has_nested_arrays:
            return False

        return True

    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data to TOON format.

        Args:
            data: List of dicts to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            EncodingResult with TOON content
        """
        if not isinstance(data, list) or not data:
            return EncodingResult(
                content="",
                success=False,
                error="TOON requires non-empty list of objects",
            )

        try:
            # Get field names from first object or characteristics
            if characteristics.field_names:
                fields = characteristics.field_names
            else:
                first_item = data[0]
                if not isinstance(first_item, dict):
                    return EncodingResult(
                        content="",
                        success=False,
                        error="TOON requires list of objects",
                    )
                fields = list(first_item.keys())

            # Build header: name[count]{fields}:
            header = f"data[{len(data)}]{{{','.join(fields)}}}:"

            # Build rows
            rows = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                row_values = []
                for field in fields:
                    value = item.get(field, "")
                    row_values.append(self._escape_value(value))
                rows.append(",".join(row_values))

            # Combine
            content = header + "\n" + "\n".join(rows)

            return EncodingResult(
                content=content,
                success=True,
                metadata={
                    "encoder": "toon",
                    "field_count": len(fields),
                    "row_count": len(rows),
                    "char_count": len(content),
                    "estimated_tokens": self.estimate_tokens(content),
                },
            )

        except Exception as e:
            logger.warning(f"TOON encoding failed: {e}")
            return EncodingResult(
                content="",
                success=False,
                error=str(e),
            )

    def _escape_value(self, value: Any) -> str:
        """Escape a value for TOON format.

        Handles:
        - None → empty string
        - Commas → quoted string
        - Newlines → escaped
        - Quotes → escaped

        Args:
            value: Value to escape

        Returns:
            Escaped string representation
        """
        if value is None:
            return ""

        str_value = str(value)

        # Check if quoting is needed
        needs_quoting = False
        if "," in str_value or "\n" in str_value or '"' in str_value:
            needs_quoting = True

        if needs_quoting:
            # Escape quotes and wrap in quotes
            escaped = str_value.replace('"', '""')
            return f'"{escaped}"'

        return str_value

    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """Calculate TOON suitability score.

        TOON excels with:
        - Uniform arrays (same keys in each object)
        - Many rows (amortizes header cost)
        - Many fields (more savings per row)
        - No nesting

        Args:
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            Suitability score (0.0-1.0)
        """
        if not characteristics.is_tabular():
            return 0.0

        base_score = 0.6

        # Bonus for more rows (header cost is amortized)
        if characteristics.array_length >= 10:
            base_score += 0.15
        elif characteristics.array_length >= 5:
            base_score += 0.10
        elif characteristics.array_length >= 3:
            base_score += 0.05

        # Bonus for more fields (more savings per row)
        if characteristics.unique_keys >= 8:
            base_score += 0.10
        elif characteristics.unique_keys >= 5:
            base_score += 0.05

        # Perfect uniformity bonus
        if characteristics.array_uniformity == 1.0:
            base_score += 0.05

        # Penalty for special characters (need escaping)
        if characteristics.has_special_chars:
            base_score -= 0.05

        return min(0.95, max(0.0, base_score))

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens for TOON content.

        TOON is more token-efficient than JSON:
        - ~4 chars per token on average

        Args:
            content: TOON string

        Returns:
            Estimated token count
        """
        return int(len(content) / 4)

    def validate_output(self, content: str) -> bool:
        """Validate TOON output format.

        Checks:
        - Has header line with name[count]{fields}:
        - Has data rows

        Args:
            content: TOON string to validate

        Returns:
            True if valid TOON format
        """
        if not content:
            return False

        lines = content.strip().split("\n")
        if not lines:
            return False

        # Check header format: name[count]{fields}:
        header = lines[0]
        if "[" not in header or "{" not in header or "}" not in header:
            return False
        if not header.endswith(":"):
            return False

        return True

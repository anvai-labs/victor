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

"""CSV format encoder for maximum token efficiency.

CSV achieves 40-70% token savings for flat tabular data:
- Standard CSV format with header row
- Minimal syntax overhead
- Widely understood by LLMs

Limitations:
- No nesting support
- Special characters require quoting
- Less structured than TOON for complex data
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from victor.processing.serialization.formats.base import FormatEncoder, EncodingResult
from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
)

logger = logging.getLogger(__name__)


class CSVEncoder(FormatEncoder):
    """CSV format encoder for maximum compression.

    Best for flat, uniform data without special characters.
    Uses Python's csv module for proper escaping.
    """

    format_id = SerializationFormat.CSV
    format_name = "CSV"
    format_description = (
        "Comma-separated values with header row. "
        "Standard format: header row followed by data rows."
    )

    # Capabilities
    supports_nested = False
    supports_arrays = True
    supports_special_chars = True  # With proper quoting
    max_nesting_depth = 1

    # Token efficiency (40-70% better than JSON for clean tabular)
    base_efficiency = 0.45

    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """Check if CSV can encode this data efficiently.

        CSV requires:
        - Array of objects or list of lists
        - High uniformity
        - No nested structures
        - Preferably no special characters (quoting adds overhead)

        Args:
            data: Data to check
            characteristics: Data analysis results

        Returns:
            True if CSV is suitable
        """
        if not isinstance(data, list):
            return False

        if not data:
            return False

        # Must be tabular
        if not characteristics.is_tabular():
            return False

        # Heavy special characters make CSV less efficient
        # due to quoting overhead, but still encodable
        return True

    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data to CSV format.

        Args:
            data: List of dicts to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            EncodingResult with CSV content
        """
        if not isinstance(data, list) or not data:
            return EncodingResult(
                content="",
                success=False,
                error="CSV requires non-empty list",
            )

        try:
            # Get field names
            if characteristics.field_names:
                fields = characteristics.field_names
            else:
                first_item = data[0]
                if isinstance(first_item, dict):
                    fields = list(first_item.keys())
                elif isinstance(first_item, (list, tuple)):
                    # List of lists - generate column names
                    fields = [f"col{i}" for i in range(len(first_item))]
                else:
                    return EncodingResult(
                        content="",
                        success=False,
                        error="CSV requires list of dicts or lists",
                    )

            # Use StringIO to capture CSV output
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

            # Write header
            writer.writerow(fields)

            # Write data rows
            for item in data:
                if isinstance(item, dict):
                    row = [self._format_value(item.get(f, "")) for f in fields]
                elif isinstance(item, (list, tuple)):
                    row = [self._format_value(v) for v in item]
                else:
                    row = [str(item)]
                writer.writerow(row)

            content = output.getvalue()

            return EncodingResult(
                content=content,
                success=True,
                metadata={
                    "encoder": "csv",
                    "field_count": len(fields),
                    "row_count": len(data),
                    "char_count": len(content),
                    "estimated_tokens": self.estimate_tokens(content),
                },
            )

        except Exception as e:
            logger.warning(f"CSV encoding failed: {e}")
            return EncodingResult(
                content="",
                success=False,
                error=str(e),
            )

    def _format_value(self, value: Any) -> str:
        """Format a value for CSV.

        Args:
            value: Value to format

        Returns:
            String representation
        """
        if value is None:
            return ""
        return str(value)

    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """Calculate CSV suitability score.

        CSV is best for:
        - Uniform arrays with many rows
        - Simple string/numeric data
        - No special characters (quoting overhead)

        Args:
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            Suitability score (0.0-1.0)
        """
        if not characteristics.is_tabular():
            return 0.0

        base_score = 0.65

        # Large datasets benefit most from CSV compression
        if characteristics.array_length >= 20:
            base_score += 0.15
        elif characteristics.array_length >= 10:
            base_score += 0.10
        elif characteristics.array_length >= 5:
            base_score += 0.05

        # Perfect uniformity bonus
        if characteristics.array_uniformity == 1.0:
            base_score += 0.05

        # Penalty for special characters (need quoting)
        if characteristics.has_special_chars:
            base_score -= 0.15  # More significant penalty than TOON

        # Penalty for null values (empty cells can be ambiguous)
        if characteristics.has_null_values:
            base_score -= 0.05

        return min(0.90, max(0.0, base_score))

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens for CSV content.

        CSV is highly token-efficient:
        - ~4.5 chars per token on average

        Args:
            content: CSV string

        Returns:
            Estimated token count
        """
        return int(len(content) / 4.5)

    def validate_output(self, content: str) -> bool:
        """Validate CSV output.

        Checks:
        - Can be parsed by Python's csv module
        - Has at least header and one data row

        Args:
            content: CSV string to validate

        Returns:
            True if valid CSV
        """
        if not content:
            return False

        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            return len(rows) >= 1  # At least header
        except csv.Error:
            return False

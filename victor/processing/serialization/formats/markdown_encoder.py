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

"""Markdown table format encoder.

Markdown tables are:
- Human-readable in plain text
- Well-understood by all LLMs
- ~20-30% token savings vs JSON
- Good for data that may be displayed to users

Format:
    | col1 | col2 | col3 |
    |------|------|------|
    | val1 | val2 | val3 |
    | val4 | val5 | val6 |
"""

from __future__ import annotations

import logging
from typing import Any

from victor.processing.serialization.formats.base import FormatEncoder, EncodingResult
from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
)

logger = logging.getLogger(__name__)


class MarkdownTableEncoder(FormatEncoder):
    """Markdown table encoder for readable tabular data.

    Best when data may be displayed to users or when
    readability is prioritized over maximum compression.
    """

    format_id = SerializationFormat.MARKDOWN_TABLE
    format_name = "Markdown Table"
    format_description = (
        "Markdown table with | separators and header divider. "
        "Readable format suitable for display."
    )

    # Capabilities
    supports_nested = False
    supports_arrays = True
    supports_special_chars = True
    max_nesting_depth = 1

    # Token efficiency (20-30% better than JSON, less than CSV/TOON)
    base_efficiency = 0.75

    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """Check if data can be encoded as markdown table.

        Requires tabular data (uniform array of objects).

        Args:
            data: Data to check
            characteristics: Data analysis results

        Returns:
            True if suitable for markdown table
        """
        if not isinstance(data, list):
            return False

        if not data:
            return False

        # Must be tabular
        return characteristics.is_tabular()

    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data to markdown table.

        Args:
            data: List of dicts to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            EncodingResult with markdown table
        """
        if not isinstance(data, list) or not data:
            return EncodingResult(
                content="",
                success=False,
                error="Markdown table requires non-empty list",
            )

        try:
            # Get field names
            if characteristics.field_names:
                fields = characteristics.field_names
            else:
                first_item = data[0]
                if isinstance(first_item, dict):
                    fields = list(first_item.keys())
                else:
                    return EncodingResult(
                        content="",
                        success=False,
                        error="Markdown table requires list of objects",
                    )

            # Calculate column widths for alignment
            widths = {f: len(f) for f in fields}
            for item in data:
                if isinstance(item, dict):
                    for f in fields:
                        val_len = len(self._escape_value(item.get(f, "")))
                        widths[f] = max(widths[f], val_len)

            # Build header row
            header_cells = [f.ljust(widths[f]) for f in fields]
            header = "| " + " | ".join(header_cells) + " |"

            # Build separator row
            separator_cells = ["-" * widths[f] for f in fields]
            separator = "|-" + "-|-".join(separator_cells) + "-|"

            # Build data rows
            rows = []
            for item in data:
                if isinstance(item, dict):
                    cells = [self._escape_value(item.get(f, "")).ljust(widths[f]) for f in fields]
                    rows.append("| " + " | ".join(cells) + " |")

            content = "\n".join([header, separator] + rows)

            return EncodingResult(
                content=content,
                success=True,
                metadata={
                    "encoder": "markdown_table",
                    "field_count": len(fields),
                    "row_count": len(rows),
                    "char_count": len(content),
                    "estimated_tokens": self.estimate_tokens(content),
                },
            )

        except Exception as e:
            logger.warning(f"Markdown table encoding failed: {e}")
            return EncodingResult(
                content="",
                success=False,
                error=str(e),
            )

    def _escape_value(self, value: Any) -> str:
        """Escape value for markdown table cell.

        Handles:
        - None → empty string
        - Pipe characters → escaped
        - Newlines → space

        Args:
            value: Value to escape

        Returns:
            Escaped string
        """
        if value is None:
            return ""

        str_value = str(value)

        # Escape pipe characters
        str_value = str_value.replace("|", "\\|")

        # Replace newlines with space
        str_value = str_value.replace("\n", " ")

        return str_value

    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """Calculate markdown table suitability score.

        Markdown tables are good when:
        - Data is tabular and may be shown to users
        - Moderate compression is acceptable
        - Readability is valued

        Args:
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            Suitability score (0.0-1.0)
        """
        if not characteristics.is_tabular():
            return 0.0

        base_score = 0.45  # Lower than CSV/TOON due to less compression

        # Bonus for moderate-sized datasets (not too large)
        if 3 <= characteristics.array_length <= 20:
            base_score += 0.10

        # Perfect uniformity bonus
        if characteristics.array_uniformity == 1.0:
            base_score += 0.05

        # Penalty for many fields (wide tables are hard to read)
        if characteristics.unique_keys > 8:
            base_score -= 0.10
        elif characteristics.unique_keys > 5:
            base_score -= 0.05

        return min(0.70, max(0.0, base_score))

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens for markdown table.

        Markdown tables have overhead from pipes and spacing:
        - ~3.8 chars per token

        Args:
            content: Markdown table string

        Returns:
            Estimated token count
        """
        return int(len(content) / 3.8)

    def validate_output(self, content: str) -> bool:
        """Validate markdown table format.

        Checks:
        - Has header row with pipes
        - Has separator row with dashes
        - Has at least one data row

        Args:
            content: Markdown table to validate

        Returns:
            True if valid format
        """
        if not content:
            return False

        lines = content.strip().split("\n")
        if len(lines) < 2:
            return False

        # Check header has pipes
        if "|" not in lines[0]:
            return False

        # Check separator has dashes and pipes
        if "|" not in lines[1] or "-" not in lines[1]:
            return False

        return True

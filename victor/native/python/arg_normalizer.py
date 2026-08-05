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

"""Pure Python argument normalizer implementation.

Provides JSON repair and type coercion for tool call arguments.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional, Tuple

from victor.native.observability import InstrumentedAccelerator
from victor.native.protocols import CoercedType, CoercedValue

# Patterns for quote repair
SINGLE_QUOTE_PATTERN = re.compile(r"(?<!\\)'")
UNESCAPED_QUOTE_PATTERN = re.compile(r'(?<!\\)"(?=[^"]*"[^"]*$)')


class PythonArgumentNormalizer(InstrumentedAccelerator):
    """Pure Python implementation of ArgumentNormalizerProtocol.

    Provides JSON repair and type coercion using Python's json and ast modules.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"

    def get_version(self) -> Optional[str]:
        return self._version

    def normalize_json(self, value: str) -> Tuple[str, bool]:
        """Normalize a potentially malformed JSON string.

        Attempts multiple repair strategies in order:
        1. Valid JSON (fast path)
        2. AST literal eval
        3. Quote replacement
        4. Manual repair

        Args:
            value: Potentially malformed JSON string

        Returns:
            Tuple of (normalized JSON string, success boolean)
        """
        with self._timed_call("json_repair"):
            # Fast path: already valid JSON
            try:
                json.loads(value)
                return value, True
            except (json.JSONDecodeError, ValueError):
                pass

            # Try AST literal eval (handles Python literals like dicts/lists)
            try:
                python_obj = ast.literal_eval(value)
                json_str = json.dumps(python_obj)
                # Validate the result
                json.loads(json_str)
                return json_str, True
            except (ValueError, SyntaxError, TypeError):
                pass

            # Try quote replacement
            repaired = self.repair_quotes(value)
            try:
                json.loads(repaired)
                return repaired, True
            except (json.JSONDecodeError, ValueError):
                pass

            # Manual repair attempt
            try:
                repaired = self._manual_repair(value)
                json.loads(repaired)
                return repaired, True
            except (json.JSONDecodeError, ValueError):
                pass

            return value, False

    def coerce_type(self, value: str) -> CoercedValue:
        """Coerce a string value to its likely type.

        Detects: int, float, bool, null, or keeps as string.

        Args:
            value: String value to coerce

        Returns:
            CoercedValue with detected type
        """
        with self._timed_call("type_coercion"):
            stripped = value.strip()

            # Check for null
            if stripped.lower() in ("null", "none", "nil"):
                return CoercedValue(
                    value=None,
                    original=value,
                    coerced_type=CoercedType.NULL,
                    confidence=1.0,
                )

            # Check for boolean
            if stripped.lower() == "true":
                return CoercedValue(
                    value=True,
                    original=value,
                    coerced_type=CoercedType.BOOL,
                    confidence=1.0,
                )
            if stripped.lower() == "false":
                return CoercedValue(
                    value=False,
                    original=value,
                    coerced_type=CoercedType.BOOL,
                    confidence=1.0,
                )

            # Check for integer
            try:
                int_val = int(stripped)
                return CoercedValue(
                    value=int_val,
                    original=value,
                    coerced_type=CoercedType.INT,
                    confidence=1.0,
                )
            except ValueError:
                pass

            # Check for float
            try:
                float_val = float(stripped)
                return CoercedValue(
                    value=float_val,
                    original=value,
                    coerced_type=CoercedType.FLOAT,
                    confidence=1.0,
                )
            except ValueError:
                pass

            # Check for JSON object/array
            try:
                json_val = json.loads(stripped)
                if isinstance(json_val, dict):
                    return CoercedValue(
                        value=json_val,
                        original=value,
                        coerced_type=CoercedType.DICT,
                        confidence=1.0,
                    )
                if isinstance(json_val, list):
                    return CoercedValue(
                        value=json_val,
                        original=value,
                        coerced_type=CoercedType.LIST,
                        confidence=1.0,
                    )
            except (json.JSONDecodeError, ValueError):
                pass

            # Default to string
            return CoercedValue(
                value=stripped,
                original=value,
                coerced_type=CoercedType.STRING,
                confidence=1.0,
            )

    def repair_quotes(self, value: str) -> str:
        """Repair mismatched or incorrect quotes in JSON.

        Handles:
        - Single quotes → double quotes
        - Unescaped quotes within strings

        Args:
            value: String with potential quote issues

        Returns:
            String with repaired quotes
        """
        with self._timed_call("quote_repair"):
            # Replace single quotes with double quotes
            # Be careful not to replace escaped single quotes
            result = value

            # First, handle the case where single quotes are used for JSON
            if "'" in result and '"' not in result:
                # Likely using single quotes instead of double quotes
                result = result.replace("'", '"')
            elif "'" in result:
                # Mixed quotes - try to be smart about it
                result = self._smart_quote_replace(result)

            return result

    def _smart_quote_replace(self, value: str) -> str:
        """Intelligently replace quotes, preserving string content.

        This handles cases like:
        {"key": 'value'} -> {"key": "value"}
        {'key': "value"} -> {"key": "value"}
        """
        result = []
        in_string = False
        string_char = None
        i = 0

        while i < len(value):
            char = value[i]

            if char in ('"', "'") and (i == 0 or value[i - 1] != "\\"):
                if not in_string:
                    # Starting a string
                    in_string = True
                    string_char = char
                    result.append('"')  # Always use double quotes
                elif char == string_char:
                    # Ending the string
                    in_string = False
                    string_char = None
                    result.append('"')
                else:
                    # Different quote inside string - escape it
                    if char == '"':
                        result.append('\\"')
                    else:
                        result.append(char)
            else:
                result.append(char)

            i += 1

        return "".join(result)

    def _manual_repair(self, value: str) -> str:
        """Attempt manual JSON repair for common issues.

        Handles:
        - Trailing commas
        - Missing quotes around keys
        - Unquoted string values
        """
        result = value.strip()

        # Remove trailing commas before } or ]
        result = re.sub(r",(\s*[}\]])", r"\1", result)

        # Try to add quotes around unquoted keys
        result = re.sub(
            r"(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
            r'\1"\2":',
            result,
        )

        return result

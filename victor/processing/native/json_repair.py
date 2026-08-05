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

"""JSON repair and type coercion functions with native acceleration."""

from functools import lru_cache
import json
from typing import Any, List, Optional, Tuple

from victor.core.json_utils import json_loads
from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# =============================================================================
# JSON REPAIR FUNCTIONS
# =============================================================================


def repair_json(input_str: str) -> str:
    """Repair malformed JSON by converting Python-style syntax to valid JSON.

    Handles:
    - Single quotes -> double quotes
    - Python True/False/None -> JSON true/false/null

    Args:
        input_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    if _NATIVE_AVAILABLE:
        return _native.repair_json(input_str)

    # Pure Python fallback
    result = input_str

    # Fast path: check if already valid
    try:
        json_loads(result)
        return result
    except json.JSONDecodeError:
        pass

    # Replace Python literals
    result = result.replace("True", "true")
    result = result.replace("False", "false")
    result = result.replace("None", "null")

    # Replace single quotes with double quotes (simple approach)
    # This is a simplified version; the Rust implementation handles edge cases better
    in_string = False
    output = []
    i = 0
    while i < len(result):
        c = result[i]
        if c == "\\" and i + 1 < len(result):
            output.append(c)
            output.append(result[i + 1])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            output.append(c)
        elif c == "'" and not in_string:
            output.append('"')
        else:
            output.append(c)
        i += 1

    return "".join(output)


def extract_json_objects(text: str) -> List[Tuple[int, int, str]]:
    """Extract JSON objects from mixed text content.

    Args:
        text: Text that may contain JSON objects

    Returns:
        List of (start_pos, end_pos, json_string) tuples for each found object
    """
    if _NATIVE_AVAILABLE:
        return _native.extract_json_objects(text)

    # Pure Python fallback
    results: List[Tuple[int, int, str]] = []
    i = 0

    while i < len(text):
        if text[i] in "{[":
            # Find matching bracket
            match = _find_json_end(text, i)
            if match:
                end, json_str = match
                # Validate
                try:
                    json_loads(json_str)
                    results.append((i, end, json_str))
                    i = end
                    continue
                except json.JSONDecodeError:
                    # Try repairing
                    repaired = repair_json(json_str)
                    try:
                        json_loads(repaired)
                        results.append((i, end, repaired))
                        i = end
                        continue
                    except json.JSONDecodeError:
                        pass
        i += 1

    return results


def _find_json_end(text: str, start: int) -> Optional[Tuple[int, str]]:
    """Find the end of a JSON structure starting at given position."""
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if c == "\\":
            escape_next = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return (i + 1, text[start : i + 1])

    return None


# =============================================================================
# TYPE COERCION (v0.4.0 - Argument Normalizer Hot Path)
# =============================================================================


def coerce_string_type(value: str) -> Tuple[str, str, Optional[str]]:
    """Coerce a string to its appropriate type.

    Fast detection and coercion of string values to bool/int/float/null.
    Uses Rust implementation when available for ~3-5x speedup.

    Args:
        value: String value to coerce

    Returns:
        Tuple of (type_name, coerced_str, error_or_none)
        - type_name: "null", "bool", "int", "float", or "string"
        - coerced_str: The value as a string in canonical form
        - error_or_none: Error message if coercion failed, else None
    """
    if _NATIVE_AVAILABLE:
        return _coerce_string_type_native_hot_path(value)

    return _coerce_string_type_python_fallback(value)


@lru_cache(maxsize=4096)
def _coerce_string_type_native_hot_path(value: str) -> Tuple[str, str, Optional[str]]:
    """Use a cached scalar fast path before paying native FFI overhead.

    Tiny scalar strings are extremely common in tool arguments, and Python↔Rust
    boundary costs can dominate the actual coercion work for these values.
    For scalar inputs we match the native semantics directly in Python and only
    cross the FFI boundary for structured JSON candidates.
    """
    stripped = value.strip()
    lower = stripped.lower()

    if lower in ("none", "null", "nil"):
        return ("null", "null", None)

    if lower == "true":
        return ("bool", "true", None)
    if lower == "false":
        return ("bool", "false", None)

    if stripped and stripped[0] not in "{[":
        try:
            int_val = int(stripped)
            return ("int", str(int_val), None)
        except ValueError:
            pass

        try:
            float_val = float(stripped)
            return ("float", str(float_val), None)
        except ValueError:
            pass

        return ("string", stripped, None)

    if _NATIVE_AVAILABLE and hasattr(_native, "coerce_string_type"):
        type_name, coerced_str, third = _native.coerce_string_type(value)
        return (
            type_name,
            coerced_str,
            None if isinstance(third, float) else third,
        )

    return ("string", stripped, None)


def _coerce_string_type_python_fallback(value: str) -> Tuple[str, str, Optional[str]]:
    """Pure Python fallback used when the native module is unavailable."""
    # Check for null
    if value.lower() in ("none", "null", "nil"):
        return ("null", "null", None)

    # Check for bool
    if value.lower() in ("true", "yes", "on", "1"):
        return ("bool", "true", None)
    if value.lower() in ("false", "no", "off", "0"):
        return ("bool", "false", None)

    # Check for int
    try:
        int_val = int(value)
        return ("int", str(int_val), None)
    except ValueError:
        pass

    # Check for float
    try:
        float_val = float(value)
        return ("float", str(float_val), None)
    except ValueError:
        pass

    # Default to string
    return ("string", value, None)

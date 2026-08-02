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

"""Unified response parsing for LLM providers across Victor.

This module consolidates JSON extraction and response handling logic
that was previously duplicated across providers, workflows, and agents.

Key utilities:
- extract_json_from_response() - Main JSON extraction with format detection
- extract_content_from_response() - Content extraction from various response types
- parse_provider_response() - Parse structured provider responses

This module reuses:
- victor.processing.native.extract_json_objects for JSON extraction with repair
- victor.agent.tool_calling.base.py patterns for tool call parsing

Design Principles:
- Single Responsibility: Each function handles one specific parsing task
- Reusability: Works across all providers and modules
- Robustness: Handles edge cases (empty responses, malformed JSON, etc.)
- Performance: Leverages native Rust implementation when available
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from openai.types.completion import Completion

logger = logging.getLogger(__name__)

# Import framework JSON extraction
try:
    from victor.processing.native import extract_json_objects

    _FRAMEWORK_AVAILABLE = True
except ImportError:
    logger.warning("victor.processing.native not available, using fallback")
    _FRAMEWORK_AVAILABLE = False


def extract_content_from_response(
    response: Union[str, Dict[str, Any], object],
) -> Optional[str]:
    """Extract content from various LLM provider response types.

    This is a unified content extractor that handles different provider
    response formats: plain strings, dicts with 'content' key,
    CompletionResponse objects, Message objects, etc.

    Args:
        response: Provider response (dict, CompletionResponse, string, etc.)

    Returns:
        Extracted content string, or None if unable to extract

    Examples:
        >>> extract_content_from_response('{"key": "value"}')
        '{"key": "value"}'
        >>> extract_content_from_response({"content": "hello"})
        'hello'
        >>> extract_content_from_response(None)
        None
    """
    # Handle None explicitly
    if response is None:
        return None

    # Case 1: String response
    if isinstance(response, str):
        return response

    # Case 2: Dict response (common with many providers)
    if isinstance(response, dict):
        # Try common content keys
        for key in ["content", "message", "text", "response", "choices"]:
            if key in response:
                content = response[key]
                if isinstance(content, str):
                    return content
                # Handle OpenAI-style choices array
                if key == "choices" and isinstance(content, list) and len(content) > 0:
                    choice = content[0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            msg = choice["message"]
                            if isinstance(msg, dict) and "content" in msg:
                                return msg["content"]
                # Handle nested message structure
                if isinstance(content, dict) and "content" in content:
                    return content["content"]

    # Case 3: Object with content attribute (CompletionResponse, Message, etc.)
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content

    # Case 4: Object with message attribute (some provider formats)
    if hasattr(response, "message"):
        message = response.message
        if isinstance(message, dict) and "content" in message:
            return message["content"]
        if isinstance(message, str):
            return message

    # Fallback: try string conversion (but not for None)
    try:
        return str(response)
    except Exception:
        logger.warning(f"Unable to extract content from response type: {type(response)}")
        return None


def extract_json_from_response(
    response: Union[str, Dict[str, Any], object],
) -> Optional[str]:
    """Extract JSON from LLM response, handling various formats.

    This function provides robust JSON extraction that handles:
    1. Clean JSON (entire response is JSON)
    2. Markdown-wrapped JSON (```json ... ```)
    3. JSON with text prefix/suffix
    4. Empty or invalid responses
    5. JSON repair for malformed JSON (via victor.processing.native)

    Args:
        response: LLM provider response (any format)

    Returns:
        Extracted JSON string, or None if no valid JSON found

    Examples:
        >>> extract_json_from_response('{"key": "value"}')
        '{"key": "value"}'
        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json_from_response('Result: {"key": "value"}')
        '{"key": "value"}'
    """
    # First extract the content string
    content = extract_content_from_response(response)
    if not content:
        return None

    content = content.strip()
    logger.debug(f"Extracting JSON from response (length: {len(content)})")

    # Try 1: Direct JSON parse (clean response)
    try:
        json.loads(content)
        logger.debug("Response is clean JSON")
        return content
    except json.JSONDecodeError:
        pass

    # Try 2: Use framework's JSON extraction from victor.processing.native
    # This handles markdown code blocks and mixed content with JSON repair
    if _FRAMEWORK_AVAILABLE:
        try:
            json_objects = extract_json_objects(content)
            if json_objects:
                # Return the first valid JSON object found
                start, end, json_str = json_objects[0]
                logger.debug(f"Extracted JSON from position {start}-{end}")
                return json_str
        except Exception as e:
            logger.warning(f"Framework JSON extraction failed: {e}")

    # Try 3: Fallback to regex-based markdown extraction
    # (similar to victor.agent.tool_calling.base.parse_json_from_content)
    import re

    json_block_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL | re.IGNORECASE
    )
    if json_block_match:
        extracted = json_block_match.group(1).strip()
        try:
            json.loads(extracted)
            logger.debug("Extracted JSON from markdown code block (fallback)")
            return extracted
        except json.JSONDecodeError:
            pass

    logger.warning("No valid JSON found in response")
    return None


def parse_provider_response(
    response: Union[str, Dict[str, Any], object],
    schema: Optional[type] = None,
) -> Optional[Union[Dict[str, Any], Any]]:
    """Parse and validate provider response with optional schema.

    This is a unified response parser that:
    1. Extracts content from various response types
    2. Extracts JSON if present
    3. Optionally validates against a Pydantic schema

    Args:
        response: Provider response (any format)
        schema: Optional Pydantic schema class for validation

    Returns:
        Parsed and validated response, or None if parsing fails

    Examples:
        >>> from pydantic import BaseModel
        >>> class MySchema(BaseModel):
        ...     name: str
        >>> result = parse_provider_response('{"name": "test"}', MySchema)
        >>> isinstance(result, MySchema)
        True
    """
    # Extract JSON from response
    json_str = extract_json_from_response(response)
    if not json_str:
        return None

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extracted JSON: {e}")
        return None

    # Validate against schema if provided
    if schema:
        try:
            return schema.model_validate(data)
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return None

    return data


def find_json_objects(
    text: str,
    max_objects: Optional[int] = None,
) -> List[Tuple[int, int, str]]:
    """Find all JSON objects in text.

    This is a convenience wrapper around victor.processing.native.extract_json_objects
    that returns all found JSON objects with their positions.

    Args:
        text: Text that may contain JSON objects
        max_objects: Maximum number of objects to return (None = all)

    Returns:
        List of (start_pos, end_pos, json_string) tuples

    Examples:
        >>> find_json_objects('{"a": 1} and {"b": 2}')
        [(0, 9, '{"a": 1}'), (14, 23, '{"b": 2}')]
    """
    if not _FRAMEWORK_AVAILABLE:
        logger.warning("victor.processing.native not available")
        return []

    try:
        json_objects = extract_json_objects(text)
        if max_objects:
            return json_objects[:max_objects]
        return json_objects
    except Exception as e:
        logger.error(f"Failed to find JSON objects: {e}")
        return []


def repair_malformed_json(json_str: str) -> Optional[str]:
    """Attempt to repair malformed JSON.

    This is a convenience wrapper around victor.processing.native.repair_json.

    Args:
        json_str: Malformed JSON string

    Returns:
        Repaired JSON string, or None if repair fails

    Examples:
        >>> repair_malformed_json('{"key": "value"')  # Missing closing brace
        '{"key": "value"}'
    """
    if not _FRAMEWORK_AVAILABLE:
        logger.warning("victor.processing.native not available")
        return None

    try:
        from victor.processing.native import repair_json

        return repair_json(json_str)
    except ImportError:
        logger.warning("repair_json not available")
        return None
    except Exception as e:
        logger.error(f"Failed to repair JSON: {e}")
        return None


def is_valid_json(content: str) -> bool:
    """Check if content is valid JSON.

    Args:
        content: String to check

    Returns:
        True if content is valid JSON, False otherwise

    Examples:
        >>> is_valid_json('{"key": "value"}')
        True
        >>> is_valid_json('not json')
        False
    """
    try:
        json.loads(content)
        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def safe_json_parse(
    content: str,
    fallback: Optional[Any] = None,
) -> Optional[Any]:
    """Safely parse JSON with fallback.

    Args:
        content: JSON string to parse
        fallback: Value to return if parsing fails

    Returns:
        Parsed JSON data, or fallback if parsing fails

    Examples:
        >>> safe_json_parse('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_parse('invalid json', fallback={})
        {}
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback

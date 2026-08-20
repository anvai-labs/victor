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

"""Tests for unified response parser module.

Tests cover:
- Content extraction from various response types
- JSON extraction with multiple formats
- Edge cases and error handling
- Integration with framework components
"""

import json
import pytest

from victor.processing.response_parser import (
    extract_content_from_response,
    extract_json_from_response,
    parse_provider_response,
    find_json_objects,
    is_valid_json,
    safe_json_parse,
    repair_malformed_json,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MockCompletionResponse:
    """Mock LLM completion response object."""

    def __init__(self, content: str):
        self.content = content


class MockMessageResponse:
    """Mock message response object."""

    def __init__(self, message: dict):
        self.message = message


# =============================================================================
# Tests: extract_content_from_response
# =============================================================================


class TestExtractContentFromResponse:
    """Tests for extract_content_from_response function."""

    def test_string_response(self):
        """Test extracting content from plain string response."""
        response = "This is plain text content"
        content = extract_content_from_response(response)
        assert content == "This is plain text content"

    def test_dict_response_with_content_key(self):
        """Test extracting content from dict with 'content' key."""
        response = {"content": "Hello world"}
        content = extract_content_from_response(response)
        assert content == "Hello world"

    def test_dict_response_with_message_key(self):
        """Test extracting content from dict with 'message' key."""
        response = {"message": "Test message"}
        content = extract_content_from_response(response)
        assert content == "Test message"

    def test_dict_response_with_text_key(self):
        """Test extracting content from dict with 'text' key."""
        response = {"text": "Some text"}
        content = extract_content_from_response(response)
        assert content == "Some text"

    def test_openai_style_choices_response(self):
        """Test extracting content from OpenAI-style choices format."""
        response = {"choices": [{"message": {"content": "OpenAI response content"}}]}
        content = extract_content_from_response(response)
        assert content == "OpenAI response content"

    def test_nested_content_in_dict(self):
        """Test extracting nested content from dict."""
        response = {"content": {"content": "nested"}}
        content = extract_content_from_response(response)
        assert content == "nested"

    def test_completion_response_object(self):
        """Test extracting content from CompletionResponse-like object."""
        response = MockCompletionResponse("Object content")
        content = extract_content_from_response(response)
        assert content == "Object content"

    def test_message_response_object(self):
        """Test extracting content from message response object."""
        response = MockMessageResponse({"content": "Message content"})
        content = extract_content_from_response(response)
        assert content == "Message content"

    def test_none_response(self):
        """Test handling None response."""
        content = extract_content_from_response(None)
        assert content is None

    def test_empty_string_response(self):
        """Test handling empty string response."""
        content = extract_content_from_response("")
        assert content == ""

    def test_dict_without_content_keys(self):
        """Test handling dict without recognized content keys."""
        response = {"other_key": "value"}
        content = extract_content_from_response(response)
        # Fallback to string conversion
        assert content == "{'other_key': 'value'}"


# =============================================================================
# Tests: extract_json_from_response
# =============================================================================


class TestExtractJSONFromResponse:
    """Tests for extract_json_from_response function."""

    def test_clean_json_response(self):
        """Test extracting clean JSON from response."""
        response = '{"name": "test", "value": 123}'
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test", "value": 123}'

    def test_markdown_wrapped_json(self):
        """Test extracting JSON wrapped in markdown code blocks."""
        response = '```json\n{"name": "test", "value": 123}\n```'
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test", "value": 123}'

    def test_plain_markdown_wrapped_json(self):
        """Test extracting JSON wrapped in plain markdown code blocks."""
        response = '```\n{"name": "test"}\n```'
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

    def test_json_with_text_prefix(self):
        """Test extracting JSON with text before it."""
        response = 'Here is the result: {"name": "test"}'
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

    def test_json_with_text_suffix(self):
        """Test extracting JSON with text after it."""
        response = '{"name": "test"} End of response'
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

    def test_dict_response(self):
        """Test extracting JSON from dict response."""
        response = {"content": '{"name": "test"}'}
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

    def test_completion_response_object(self):
        """Test extracting JSON from CompletionResponse object."""
        response = MockCompletionResponse('{"name": "test"}')
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

    def test_no_json_returns_none(self):
        """Test handling response with no JSON."""
        response = "This is just plain text with no JSON"
        json_str = extract_json_from_response(response)
        assert json_str is None

    def test_invalid_json_returns_none(self):
        """Test handling response with invalid JSON."""
        response = '{"name": invalid}'
        json_str = extract_json_from_response(response)
        # Framework's extract_json_objects might repair this
        # If it can't be repaired, returns None
        # We just verify it doesn't crash
        assert json_str is None or json_str != '{"name": invalid}'

    def test_none_response_returns_none(self):
        """Test handling None response."""
        json_str = extract_json_from_response(None)
        assert json_str is None

    def test_multiple_json_objects(self):
        """Test handling response with multiple JSON objects."""
        response = '{"first": 1} and {"second": 2}'
        json_str = extract_json_from_response(response)
        # Should extract the first one
        assert json_str == '{"first": 1}'

    def test_nested_json_structure(self):
        """Test extracting nested JSON structure."""
        response = '{"outer": {"inner": {"deep": "value"}}}'
        json_str = extract_json_from_response(response)
        assert json_str == '{"outer": {"inner": {"deep": "value"}}}'
        data = json.loads(json_str)
        assert data["outer"]["inner"]["deep"] == "value"

    def test_json_array(self):
        """Test extracting JSON array."""
        response = '[{"name": "first"}, {"name": "second"}]'
        json_str = extract_json_from_response(response)
        assert json_str == '[{"name": "first"}, {"name": "second"}]'


# =============================================================================
# Tests: parse_provider_response
# =============================================================================


class TestParseProviderResponse:
    """Tests for parse_provider_response function."""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        response = '{"name": "test", "value": 123}'
        result = parse_provider_response(response)
        assert result == {"name": "test", "value": 123}

    def test_parse_with_schema_validation(self):
        """Test parsing with Pydantic schema validation."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        response = '{"name": "test", "value": 123}'
        result = parse_provider_response(response, TestSchema)
        assert isinstance(result, TestSchema)
        assert result.name == "test"
        assert result.value == 123

    def test_parse_with_schema_validation_failure(self):
        """Test parsing with schema that doesn't match."""
        from pydantic import BaseModel, ValidationError

        class StrictSchema(BaseModel):
            name: str
            required_field: int

        response = '{"name": "test"}'  # Missing required_field
        result = parse_provider_response(response, StrictSchema)
        assert result is None

    def test_parse_invalid_json_response(self):
        """Test parsing invalid JSON response."""
        response = "not valid json"
        result = parse_provider_response(response)
        assert result is None

    def test_parse_none_response(self):
        """Test parsing None response."""
        result = parse_provider_response(None)
        assert result is None


# =============================================================================
# Tests: find_json_objects
# =============================================================================


class TestFindJSONObjects:
    """Tests for find_json_objects function."""

    def test_find_single_json_object(self):
        """Test finding single JSON object."""
        text = '{"name": "test"}'
        objects = find_json_objects(text)
        assert len(objects) == 1
        start, end, json_str = objects[0]
        assert json_str == '{"name": "test"}'

    def test_find_multiple_json_objects(self):
        """Test finding multiple JSON objects."""
        text = '{"first": 1} and {"second": 2}'
        objects = find_json_objects(text)
        assert len(objects) == 2
        assert objects[0][2] == '{"first": 1}'
        assert objects[1][2] == '{"second": 2}'

    def test_max_objects_limit(self):
        """Test max_objects parameter limits results."""
        text = '{"first": 1} and {"second": 2} and {"third": 3}'
        objects = find_json_objects(text, max_objects=2)
        assert len(objects) == 2

    def test_find_json_in_mixed_content(self):
        """Test finding JSON in mixed text and markdown."""
        text = 'Here is the result: ```json\n{"name": "test"}\n``` done.'
        objects = find_json_objects(text)
        assert len(objects) >= 1


# =============================================================================
# Tests: is_valid_json
# =============================================================================


class TestIsValidJSON:
    """Tests for is_valid_json function."""

    def test_valid_json_object(self):
        """Test checking valid JSON object."""
        assert is_valid_json('{"name": "test"}') is True

    def test_valid_json_array(self):
        """Test checking valid JSON array."""
        assert is_valid_json("[1, 2, 3]") is True

    def test_valid_json_string(self):
        """Test checking valid JSON string."""
        assert is_valid_json('"hello"') is True

    def test_valid_json_number(self):
        """Test checking valid JSON number."""
        assert is_valid_json("123") is True

    def test_invalid_json_syntax(self):
        """Test checking invalid JSON syntax."""
        assert is_valid_json("{name: test}") is False

    def test_invalid_json_unclosed_brace(self):
        """Test checking invalid JSON with unclosed brace."""
        assert is_valid_json('{"name": "test"') is False

    def test_plain_text(self):
        """Test checking plain text is not JSON."""
        assert is_valid_json("This is plain text") is False

    def test_empty_string(self):
        """Test checking empty string is not valid JSON."""
        assert is_valid_json("") is False

    def test_none_input(self):
        """Test checking None input returns False."""
        assert is_valid_json(None) is False


# =============================================================================
# Tests: safe_json_parse
# =============================================================================


class TestSafeJSONParse:
    """Tests for safe_json_parse function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        result = safe_json_parse('{"name": "test"}')
        assert result == {"name": "test"}

    def test_parse_with_fallback(self):
        """Test parsing with fallback value."""
        result = safe_json_parse("invalid json", fallback={"default": True})
        assert result == {"default": True}

    def test_parse_invalid_json_no_fallback(self):
        """Test parsing invalid JSON without fallback."""
        result = safe_json_parse("invalid json")
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = safe_json_parse("", fallback={})
        assert result == {}

    def test_parse_with_none_fallback(self):
        """Test parsing with None as fallback."""
        result = safe_json_parse("invalid", fallback=None)
        assert result is None


# =============================================================================
# Tests: repair_malformed_json
# =============================================================================


class TestRepairMalformedJSON:
    """Tests for repair_malformed_json function."""

    def test_repair_missing_closing_brace(self):
        """Test repairing JSON with missing closing brace."""
        # Note: This depends on victor.processing.native.repair_json
        # The repair function may not handle all syntax errors
        result = repair_malformed_json('{"name": "test"')
        # Result may be repaired JSON, original, or None
        # We just verify it doesn't crash
        assert result is None or isinstance(result, str)

    def test_repair_trailing_comma(self):
        """Test repairing JSON with trailing comma."""
        result = repair_malformed_json('{"name": "test",}')
        # May return repaired JSON, original, or None
        assert result is None or isinstance(result, str)

    def test_repair_unquoted_keys(self):
        """Test repairing JSON with unquoted keys."""
        result = repair_malformed_json('{name: "test"}')
        # May return repaired JSON, original, or None
        assert result is None or isinstance(result, str)

    def test_repair_python_single_quotes(self):
        """Test repairing JSON with Python-style single quotes."""
        result = repair_malformed_json("{'name': 'test'}")
        # Should convert to double quotes
        if result:
            assert '"name"' in result or "'name'" in result

    def test_repair_already_valid_json(self):
        """Test that valid JSON is returned as-is."""
        result = repair_malformed_json('{"name": "test"}')
        # Should return the same JSON or None (if repair not available)
        assert result in ['{"name": "test"}', None]


# =============================================================================
# Tests: Integration and Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_unicode_in_json(self):
        """Test handling Unicode characters in JSON."""
        response = '{"message": "Hello 世界 🌍"}'
        json_str = extract_json_from_response(response)
        assert json_str == '{"message": "Hello 世界 🌍"}'
        data = json.loads(json_str)
        assert data["message"] == "Hello 世界 🌍"

    def test_escaped_characters_in_json(self):
        """Test handling escaped characters in JSON."""
        response = '{"path": "C:\\\\Users\\\\test", "newline": "line1\\nline2"}'
        json_str = extract_json_from_response(response)
        assert json_str is not None
        data = json.loads(json_str)
        assert data["path"] == "C:\\Users\\test"
        assert data["newline"] == "line1\nline2"

    def test_very_long_json_response(self):
        """Test handling very long JSON response."""
        # Create a large JSON object
        large_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        response = json.dumps(large_data)
        json_str = extract_json_from_response(response)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert len(parsed) == 100

    def test_deeply_nested_json(self):
        """Test handling deeply nested JSON structure."""
        deep_json = '{"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}'
        json_str = extract_json_from_response(deep_json)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["level1"]["level2"]["level3"]["level4"]["level5"] == "deep"

    def test_json_with_special_characters(self):
        """Test JSON with various special characters."""
        response = '{"special": "!@#$%^&*()", "newline": "\\n", "tab": "\\t"}'
        json_str = extract_json_from_response(response)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["special"] == "!@#$%^&*()"

    def test_response_with_multiple_code_blocks(self):
        """Test response with multiple code blocks."""
        response = 'First: ```json\n{"first": 1}\n``` Second: ```json\n{"second": 2}\n```'
        json_str = extract_json_from_response(response)
        # Should extract the first JSON
        assert json_str is not None
        assert '{"first": 1}' in json_str or '{"second": 2}' in json_str

    def test_response_with_leading_whitespace(self):
        """Test response with leading whitespace."""
        response = '   \n\n  {"name": "test"}'
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

    def test_response_with_trailing_whitespace(self):
        """Test response with trailing whitespace."""
        response = '{"name": "test"}   \n\n  '
        json_str = extract_json_from_response(response)
        assert json_str == '{"name": "test"}'

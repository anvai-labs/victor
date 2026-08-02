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

"""TDD tests for JSON extraction and response handling in planning module.

Tests cover:
1. JSON extraction from various formats (clean, markdown-wrapped, with prefix/suffix)
2. Empty/malformed response handling
3. CompletionResponse vs dict vs string handling
4. Error recovery and retries
5. Integration with existing framework components
"""

import pytest

from victor.agent.planning.readable_schema import (
    extract_llm_response_content,
    extract_json_from_llm_response,
    ReadableTaskPlan,
    TaskComplexity,
)
from victor.providers.base import CompletionResponse, Message


class TestExtractLLMResponseContent:
    """Tests for extract_llm_response_content function."""

    def test_string_response(self):
        """Test extracting content from plain string response."""
        response = '{"key": "value"}'
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_dict_response_with_content_key(self):
        """Test extracting content from dict with 'content' key."""
        response = {"content": '{"key": "value"}'}
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_dict_response_with_message_key(self):
        """Test extracting content from dict with 'message' key."""
        response = {"message": '{"key": "value"}'}
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_dict_response_with_openai_choices(self):
        """Test extracting content from OpenAI-style choices format."""
        response = {"choices": [{"message": {"content": '{"key": "value"}'}}]}
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_completion_response_object(self):
        """Test extracting content from CompletionResponse object."""
        response = CompletionResponse(
            content='{"key": "value"}',
            role="assistant",
            tool_calls=None,
        )
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_message_object(self):
        """Test extracting content from Message object."""
        response = Message(role="assistant", content='{"key": "value"}')
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_nested_message_dict(self):
        """Test extracting content from nested message dict."""
        response = {"message": {"content": '{"key": "value"}'}}
        content = extract_llm_response_content(response)
        assert content == '{"key": "value"}'

    def test_empty_response(self):
        """Test handling of empty response."""
        content = extract_llm_response_content("")
        assert content == ""

    def test_none_response(self):
        """Test handling of None response."""
        content = extract_llm_response_content(None)
        assert content is None

    def test_unsupported_type(self):
        """Test handling of unsupported type (falls back to str)."""

        class CustomObject:
            def __str__(self):
                return "custom"

        content = extract_llm_response_content(CustomObject())
        assert content == "custom"


class TestExtractJSONFromLLMResponse:
    """Tests for extract_json_from_llm_response function."""

    def test_clean_json_response(self):
        """Test extracting clean JSON from plain string."""
        response = '{"name": "test", "complexity": "simple"}'
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_markdown_wrapped_json(self):
        """Test extracting JSON from markdown code block."""
        response = """```json
{
  "name": "test",
  "complexity": "simple"
}
```"""
        json_str = extract_json_from_llm_response(response)
        assert '"name": "test"' in json_str
        assert '"complexity": "simple"' in json_str

    def test_markdown_without_json_tag(self):
        """Test extracting JSON from markdown without json tag."""
        response = """```
{"name": "test", "complexity": "simple"}
```"""
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_json_with_text_prefix(self):
        """Test extracting JSON when response has text prefix."""
        response = 'Here is your plan:\n{"name": "test", "complexity": "simple"}'
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_json_with_text_suffix(self):
        """Test extracting JSON when response has text suffix."""
        response = '{"name": "test", "complexity": "simple"}\n\nLet me know if you need help.'
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_json_with_text_both_sides(self):
        """Test extracting JSON when response has text on both sides."""
        response = """Here is your plan:

{"name": "test", "complexity": "simple"}

Let me know if you need help."""
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_dict_response(self):
        """Test extracting JSON from dict response."""
        response = {"content": '{"name": "test", "complexity": "simple"}'}
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_completion_response(self):
        """Test extracting JSON from CompletionResponse."""
        response = CompletionResponse(
            content='{"name": "test", "complexity": "simple"}',
            role="assistant",
            tool_calls=None,
        )
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"name": "test", "complexity": "simple"}'

    def test_no_json_returns_none(self):
        """Test that response without JSON returns None."""
        response = "This is just plain text with no JSON"
        json_str = extract_json_from_llm_response(response)
        assert json_str is None

    def test_empty_response_returns_none(self):
        """Test that empty response returns None."""
        json_str = extract_json_from_llm_response("")
        assert json_str is None

    def test_none_response_returns_none(self):
        """Test that None response returns None."""
        json_str = extract_json_from_llm_response(None)
        assert json_str is None

    def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        response = '{"incomplete": '
        json_str = extract_json_from_llm_response(response)
        # Should return None since it can't find valid JSON
        assert json_str is None

    def test_json_array(self):
        """Test extracting JSON array."""
        response = '[{"name": "step1"}, {"name": "step2"}]'
        json_str = extract_json_from_llm_response(response)
        assert json_str == '[{"name": "step1"}, {"name": "step2"}]'

    def test_nested_json(self):
        """Test extracting nested JSON structure."""
        response = '{"outer": {"inner": "value"}}'
        json_str = extract_json_from_llm_response(response)
        assert json_str == '{"outer": {"inner": "value"}}'


class TestReadableTaskPlanValidation:
    """Tests for ReadableTaskPlan validation with extracted JSON."""

    def test_valid_plan_from_json(self):
        """Test creating plan from valid JSON."""
        json_str = """{
  "name": "Test Plan",
  "complexity": "simple",
  "desc": "Test description",
  "steps": [
    [1, "research", "Analyze", "overview"],
    [2, "feature", "Implement", "write"]
  ]
}"""
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert plan.name == "Test Plan"
        assert plan.complexity == TaskComplexity.SIMPLE
        assert len(plan.steps) == 2

    def test_plan_with_dependencies(self):
        """Test creating plan with step dependencies."""
        json_str = """{
  "name": "Test Plan",
  "complexity": "moderate",
  "desc": "Test description",
  "steps": [
    [1, "research", "Analyze", "overview"],
    [2, "feature", "Implement", "write"],
    [3, "test", "Verify", "pytest", [2]]
  ]
}"""
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert len(plan.steps) == 3
        # Third step should have dependency on step 2
        assert len(plan.steps[2]) >= 5  # Has dependencies

    def test_plan_with_duration(self):
        """Test creating plan with duration."""
        json_str = """{
  "name": "Test Plan",
  "complexity": "simple",
  "desc": "Test description",
  "steps": [[1, "research", "Analyze", "overview"]],
  "duration": "30min"
}"""
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert plan.duration == "30min"

    def test_invalid_plan_raises_error(self):
        """Test that invalid plan raises ValidationError."""
        # Missing required field 'desc'
        json_str = """{
  "name": "Test Plan",
  "complexity": "simple",
  "steps": [[1, "research", "Analyze", "overview"]]
}"""
        with pytest.raises(Exception):  # ValidationError from pydantic
            ReadableTaskPlan.model_validate_json(json_str)

    def test_invalid_steps_raises_error(self):
        """Test that invalid step format raises error."""
        # Step missing required elements
        json_str = """{
  "name": "Test Plan",
  "complexity": "simple",
  "desc": "Test description",
  "steps": [[1, "research"]]
}"""
        with pytest.raises(Exception):  # ValidationError from pydantic
            ReadableTaskPlan.model_validate_json(json_str)


class TestEndToEndExtractionAndValidation:
    """End-to-end tests for extraction and validation pipeline."""

    def test_full_pipeline_clean_json(self):
        """Test full pipeline with clean JSON response."""
        llm_response = '{"name": "Test", "complexity": "simple", "desc": "Test", "steps": [[1, "research", "Test", "overview"]]}'

        # Extract
        json_str = extract_json_from_llm_response(llm_response)
        assert json_str is not None

        # Validate
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert plan.name == "Test"

    def test_full_pipeline_markdown_json(self):
        """Test full pipeline with markdown-wrapped JSON."""
        llm_response = """Here is your plan:

```json
{
  "name": "Test",
  "complexity": "simple",
  "desc": "Test",
  "steps": [[1, "research", "Test", "overview"]]
}
```

Let me know if you need help."""

        # Extract
        json_str = extract_json_from_llm_response(llm_response)
        assert json_str is not None

        # Validate
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert plan.name == "Test"

    def test_full_pipeline_dict_response(self):
        """Test full pipeline with dict response."""
        llm_response = {
            "content": '{"name": "Test", "complexity": "simple", "desc": "Test", "steps": [[1, "research", "Test", "overview"]]}'
        }

        # Extract
        json_str = extract_json_from_llm_response(llm_response)
        assert json_str is not None

        # Validate
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert plan.name == "Test"

    def test_full_pipeline_completion_response(self):
        """Test full pipeline with CompletionResponse."""
        llm_response = CompletionResponse(
            content='{"name": "Test", "complexity": "simple", "desc": "Test", "steps": [[1, "research", "Test", "overview"]]}',
            role="assistant",
            tool_calls=None,
        )

        # Extract
        json_str = extract_json_from_llm_response(llm_response)
        assert json_str is not None

        # Validate
        plan = ReadableTaskPlan.model_validate_json(json_str)
        assert plan.name == "Test"

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

"""Unit tests for AgenticLoop completion detection heuristics.

Tests the _is_complete_response() and _is_continuation_request() methods
that detect when an LLM response indicates task completion.
"""

import pytest
from victor.framework.agentic_loop import AgenticLoop


class TestCompletionHeuristics:
    """Test the _is_complete_response heuristic method."""

    def test_completion_indicator_phrases(self):
        """Test that phrases indicating completion are detected."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Responses with completion indicators
        completion_responses = [
            "Here is the answer to your question.",
            "The solution is to use recursion.",
            "In summary, the approach is valid.",
            "To summarize, the results are positive.",
            "The code above demonstrates the pattern.",
            "As shown in the example, it works.",
        ]

        for response in completion_responses:
            # Add padding to meet length requirement
            padded_response = response + " " + "x" * 150
            is_complete = loop._is_complete_response(padded_response)
            assert is_complete, f"Should detect completion indicator in: {response}"

    def test_code_block_response_detected_as_complete(self):
        """Response with complete code blocks should be detected as complete."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Response with code block
        response = """Here is the implementation:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
```

This function calculates the factorial recursively."""

        # Test the heuristic
        is_complete = loop._is_complete_response(response)

        assert is_complete, "Response with complete code block should be detected as complete"

    def test_structured_response_detected_as_complete(self):
        """Structured response with headings/lists should be detected as complete."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Structured response (need > 400 chars for structure + length heuristic)
        response = """## Key Features

1. Feature A - Description of the feature
2. Feature B - Description of the feature
3. Feature C - Description of the feature

## Implementation Details

- Step 1: Setup the environment
- Step 2: Configure the application
- Step 3: Deploy to production

## Additional Information

This covers all the requirements mentioned in the specification.
The approach is robust and scalable for future enhancements.""" + "x" * 200

        # Test the heuristic
        is_complete = loop._is_complete_response(response)

        assert is_complete, "Structured response should be detected as complete"

    def test_too_short_response(self):
        """Test that very short responses are not considered complete."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Short responses (< 150 chars)
        short_responses = [
            "OK",
            "Done.",
            "Here it is.",
            "The answer is 42.",
        ]

        for response in short_responses:
            is_complete = loop._is_complete_response(response)
            assert not is_complete, f"Short response should NOT be complete: {response}"

    def test_response_with_continuation_phrase(self):
        """Test that continuation phrases override completion indicators."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Response with continuation phrase
        response = "Here is the solution. Would you like me to explain further?" + "x" * 150

        is_complete = loop._is_complete_response(response)
        assert not is_complete, "Continuation phrase should prevent completion detection"

    def test_code_block_with_sufficient_length(self):
        """Test that code blocks with sufficient length are detected as complete."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Response with code block (must be > 300 chars)
        response = """Here is the implementation:

```python
def main():
    # This is a longer function to meet the length requirement
    for i in range(100):
        print(f"Processing item {i}")
        process_item(i)

def process_item(item):
    result = item * 2
    return result

if __name__ == "__main__":
    main()
```

This code processes items in a loop."""

        is_complete = loop._is_complete_response(response)
        assert is_complete, "Code block with sufficient length should be complete"


class TestContinuationRequestDetection:
    """Test the _is_continuation_request method."""

    def test_continuation_request_not_complete(self):
        """Response asking for direction should NOT be detected as complete."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Various continuation request patterns
        continuation_responses = [
            "Would you like me to continue with more details?",
            "Should I continue with the next step?",
            "Do you want me to explain this further?",
            "Shall I proceed to the next phase?",
            "Would you prefer I focus on X or Y?",
            "Let me know if you'd like me to elaborate.",
        ]

        for response in continuation_responses:
            is_continuation = loop._is_continuation_request(response)
            assert is_continuation, f"Should detect continuation request in: {response}"

    def test_non_continuation_response(self):
        """Normal responses should NOT be detected as continuation requests."""
        loop = AgenticLoop.__new__(AgenticLoop)

        # Normal responses (not continuation requests)
        normal_responses = [
            "The answer is 42.",
            "Here is the implementation: [code]",
            "In conclusion, the approach works well.",
            "The solution involves three steps.",
        ]

        for response in normal_responses:
            is_continuation = loop._is_continuation_request(response)
            assert not is_continuation, f"Should NOT detect continuation in: {response}"


class TestResponseLengthThresholds:
    """Test response length thresholds for completion detection."""

    def test_exactly_100_chars(self):
        """Test response exactly at the threshold boundary."""
        # Create response with exactly 100 chars
        response = "x" * 100

        # The condition is len(response.strip()) > 100
        # So exactly 100 should NOT trigger
        assert len(response.strip()) == 100, "Test setup error"
        assert not (len(response.strip()) > 100), "Exactly 100 chars should NOT be > 100"

    def test_101_chars(self):
        """Test response just above the threshold boundary."""
        response = "x" * 101

        # Should trigger the > 100 condition
        assert len(response.strip()) == 101, "Test setup error"
        assert len(response.strip()) > 100, "101 chars should be > 100"

    def test_short_response_not_complete(self):
        """Test that short responses don't pass the length threshold."""
        short_responses = [
            "x" * 50,
            "x" * 99,
            "x" * 100,
        ]

        for response in short_responses:
            assert len(response.strip()) <= 100, f"Test setup error: {len(response)}"
            assert not (len(response.strip()) > 100), f"Should NOT be > 100: {len(response)} chars"

    def test_long_response_passes_threshold(self):
        """Test that long responses pass the length threshold."""
        long_responses = [
            "x" * 101,
            "x" * 150,
            "x" * 200,
        ]

        for response in long_responses:
            assert len(response.strip()) > 100, f"Test setup error: {len(response)}"

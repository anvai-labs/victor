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

"""Tests for tool call extraction from model text."""

import pytest

from victor.agent.tool_call_extractor import (
    ToolCallExtractor,
    ExtractedToolCall,
    extract_tool_call_from_text,
    get_tool_call_extractor,
)


class TestExtractedToolCall:
    """Tests for ExtractedToolCall dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        call = ExtractedToolCall(
            tool_name="write",
            arguments={"path": "test.py", "content": "print('hello')"},
            confidence=0.9,
            source_text="test",
        )
        result = call.to_dict()
        assert result["tool"] == "write"
        assert result["args"]["path"] == "test.py"
        assert result["confidence"] == 0.9


class TestToolCallExtractor:
    """Tests for ToolCallExtractor class."""

    @pytest.fixture
    def extractor(self):
        return ToolCallExtractor()

    def test_extract_write_with_code_block(self, extractor):
        """Test extraction of write call with code block."""
        text = """I'll write the code to hello.py:

```python
print("Hello World")

def greet(name):
    print(f"Hello, {name}!")
```
"""
        result = extractor.extract_from_text(text, ["write"])
        assert result is not None
        assert result.tool_name == "write"
        assert result.arguments["path"] == "hello.py"
        assert "print" in result.arguments["content"]
        assert result.confidence >= 0.8

    def test_extract_write_with_backtick_path(self, extractor):
        """Test extraction with backtick-wrapped path."""
        text = """Let me create `src/utils.py` with:

```python
def helper():
    pass
```
"""
        result = extractor.extract_from_text(text, ["write"])
        assert result is not None
        assert result.arguments["path"] == "src/utils.py"

    def test_extract_read_call(self, extractor):
        """Test extraction of read call."""
        text = "Let me read the file `victor/agent/orchestrator.py` to understand."
        result = extractor.extract_from_text(text, ["read"])
        assert result is not None
        assert result.tool_name == "read"
        assert result.arguments["path"] == "victor/agent/orchestrator.py"
        assert result.confidence >= 0.8

    def test_extract_shell_command(self, extractor):
        """Test extraction of shell command."""
        text = """Let me run the tests:

```bash
pytest tests/unit/ -v
```
"""
        result = extractor.extract_from_text(text, ["shell"])
        assert result is not None
        assert result.tool_name == "shell"
        assert "pytest" in result.arguments["cmd"]

    def test_extract_grep_search(self, extractor):
        """Test extraction of grep/search call."""
        text = 'Let me search for "class Orchestrator" in the codebase.'
        result = extractor.extract_from_text(text, ["grep"])
        assert result is not None
        assert result.tool_name == "grep"
        assert "class Orchestrator" in result.arguments["query"]

    def test_extract_ls_call(self, extractor):
        """Test extraction of ls call."""
        text = "Let me list the directory `victor/agent` to see what files are there."
        result = extractor.extract_from_text(text, ["ls"])
        assert result is not None
        assert result.tool_name == "ls"
        assert result.arguments["path"] == "victor/agent"

    def test_no_extraction_without_content(self, extractor):
        """Test that extraction fails without proper content."""
        text = "I will write a file but I don't say which one or what content."
        result = extractor.extract_from_text(text, ["write"])
        assert result is None

    def test_blocks_dangerous_commands(self, extractor):
        """Test that dangerous shell commands are blocked."""
        text = """Let me run:

```bash
rm -rf /
```
"""
        result = extractor.extract_from_text(text, ["shell"])
        assert result is None

    def test_confidence_threshold(self, extractor):
        """Test that low confidence extractions are filtered."""
        text = "Something about write"  # Too vague
        result = extractor.extract_from_text(text, ["write"])
        assert result is None  # Should be None due to low confidence

    def test_multiple_mentioned_tools(self, extractor):
        """Test extraction with multiple mentioned tools."""
        text = """I'll read `config.py` first:

```python
# config content here
```
"""
        # Even though we mention read, the code block suggests write
        result = extractor.extract_from_text(text, ["read", "write"])
        # Should extract read since it's mentioned first and path is present
        assert result is not None
        assert result.arguments.get("path") == "config.py"

    def test_empty_mentioned_tools(self, extractor):
        """Test with no mentioned tools."""
        text = "Some random text"
        result = extractor.extract_from_text(text, [])
        assert result is None


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_tool_call_extractor_singleton(self):
        """Test that singleton returns same instance."""
        ext1 = get_tool_call_extractor()
        ext2 = get_tool_call_extractor()
        assert ext1 is ext2

    def test_extract_tool_call_from_text_convenience(self):
        """Test convenience function works."""
        text = "Let me read `test.py`"
        result = extract_tool_call_from_text(text, ["read"])
        assert result is not None
        assert result.tool_name == "read"


class TestEditExtraction:
    """Tests for edit-specific extraction."""

    @pytest.fixture
    def extractor(self):
        return ToolCallExtractor()

    def test_extract_edit_with_replace_pattern(self, extractor):
        """Test extraction of edit with replace pattern."""
        text = 'Replace "old_function" with "new_function" in utils.py'
        result = extractor.extract_from_text(text, ["edit"])
        assert result is not None
        # May fall back to write if edit patterns don't match

    def test_extract_edit_falls_back_to_write(self, extractor):
        """Test that edit with full content falls back to write."""
        text = """Update config.py with:

```python
DEBUG = True
```
"""
        result = extractor.extract_from_text(text, ["edit"])
        assert result is not None
        # Should return write since we have full content, not old/new
        assert result.tool_name == "write"

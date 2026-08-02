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

"""Tests for conversation export functionality."""

import json
import os
import tempfile
from datetime import datetime

import pytest

from victor.agent.conversation_export import (
    ConversationExporter,
    ConversationExport,
    ConversationMessage,
    ExportFormat,
    get_exporter,
)


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        ConversationMessage(
            role="user",
            content="Hello, can you help me write a Python function?",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        ),
        ConversationMessage(
            role="assistant",
            content="Sure! What kind of function would you like me to write?",
            timestamp=datetime(2024, 1, 15, 10, 30, 15),
            tool_calls=[
                {
                    "name": "read_file",
                    "arguments": {"path": "test.py"},
                    "result": "content",
                }
            ],
        ),
        ConversationMessage(
            role="user",
            content="A function to calculate fibonacci numbers.",
            timestamp=datetime(2024, 1, 15, 10, 31, 0),
        ),
    ]


@pytest.fixture
def sample_conversation(sample_messages):
    """Create a sample conversation export."""
    return ConversationExport(
        messages=sample_messages,
        title="Test Conversation",
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        model="claude-3-sonnet",
        provider="anthropic",
        session_id="test-session-123",
    )


@pytest.fixture
def exporter():
    """Create an exporter instance."""
    return ConversationExporter()


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_format_values(self):
        """Test all format values exist."""
        assert ExportFormat.MARKDOWN.value == "markdown"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.TEXT.value == "text"


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = ConversationMessage(
            role="user",
            content="Hello",
            timestamp=datetime.now(),
        )

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        msg = ConversationMessage(
            role="assistant",
            content="Let me check that file.",
            tool_calls=[{"name": "read_file", "arguments": {"path": "test.py"}}],
        )

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read_file"


class TestConversationExport:
    """Tests for ConversationExport dataclass."""

    def test_create_export(self, sample_messages):
        """Test creating an export."""
        export = ConversationExport(
            messages=sample_messages,
            title="My Conversation",
            model="gpt-4",
        )

        assert export.title == "My Conversation"
        assert len(export.messages) == 3
        assert export.model == "gpt-4"


class TestMarkdownExport:
    """Tests for Markdown export."""

    def test_basic_export(self, exporter, sample_conversation):
        """Test basic Markdown export."""
        result = exporter.export(sample_conversation, ExportFormat.MARKDOWN)

        assert "# Test Conversation" in result
        assert "**User**" in result
        assert "**Assistant**" in result
        assert "Hello, can you help me" in result

    def test_includes_metadata(self, exporter, sample_conversation):
        """Test that metadata is included."""
        result = exporter.export(sample_conversation, ExportFormat.MARKDOWN, include_metadata=True)

        assert "claude-3-sonnet" in result
        assert "anthropic" in result

    def test_excludes_metadata(self, exporter, sample_conversation):
        """Test excluding metadata."""
        result = exporter.export(sample_conversation, ExportFormat.MARKDOWN, include_metadata=False)

        assert "Session Info" not in result

    def test_includes_tool_calls(self, exporter, sample_conversation):
        """Test tool calls are included."""
        result = exporter.export(
            sample_conversation, ExportFormat.MARKDOWN, include_tool_calls=True
        )

        assert "Tool Calls" in result
        assert "read_file" in result

    def test_excludes_tool_calls(self, exporter, sample_conversation):
        """Test excluding tool calls."""
        result = exporter.export(
            sample_conversation, ExportFormat.MARKDOWN, include_tool_calls=False
        )

        assert "<details>" not in result


class TestJSONExport:
    """Tests for JSON export."""

    def test_basic_export(self, exporter, sample_conversation):
        """Test basic JSON export."""
        result = exporter.export(sample_conversation, ExportFormat.JSON)

        data = json.loads(result)
        assert "version" in data
        assert "messages" in data
        assert len(data["messages"]) == 3

    def test_includes_metadata(self, exporter, sample_conversation):
        """Test metadata in JSON export."""
        result = exporter.export(sample_conversation, ExportFormat.JSON, include_metadata=True)

        data = json.loads(result)
        assert "metadata" in data
        assert data["metadata"]["model"] == "claude-3-sonnet"

    def test_message_structure(self, exporter, sample_conversation):
        """Test message structure in JSON."""
        result = exporter.export(sample_conversation, ExportFormat.JSON)

        data = json.loads(result)
        msg = data["messages"][0]
        assert msg["role"] == "user"
        assert "content" in msg

    def test_tool_calls_in_json(self, exporter, sample_conversation):
        """Test tool calls in JSON export."""
        result = exporter.export(sample_conversation, ExportFormat.JSON, include_tool_calls=True)

        data = json.loads(result)
        assistant_msg = data["messages"][1]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["name"] == "read_file"


class TestHTMLExport:
    """Tests for HTML export."""

    def test_basic_export(self, exporter, sample_conversation):
        """Test basic HTML export."""
        result = exporter.export(sample_conversation, ExportFormat.HTML)

        assert "<!DOCTYPE html>" in result
        assert "<title>Test Conversation</title>" in result
        assert "Hello, can you help me" in result

    def test_has_styling(self, exporter, sample_conversation):
        """Test HTML has CSS styling."""
        result = exporter.export(sample_conversation, ExportFormat.HTML)

        assert "<style>" in result
        assert ".message" in result
        assert ".user" in result
        assert ".assistant" in result

    def test_escapes_html(self, exporter):
        """Test HTML special characters are escaped."""
        conv = ConversationExport(
            messages=[
                ConversationMessage(
                    role="user",
                    content="<script>alert('xss')</script>",
                )
            ],
            title="Test <script>",
        )

        result = exporter.export(conv, ExportFormat.HTML)

        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestTextExport:
    """Tests for plain text export."""

    def test_basic_export(self, exporter, sample_conversation):
        """Test basic text export."""
        result = exporter.export(sample_conversation, ExportFormat.TEXT)

        assert "Test Conversation" in result
        assert "USER" in result
        assert "ASSISTANT" in result

    def test_includes_timestamps(self, exporter, sample_conversation):
        """Test timestamps in text export."""
        result = exporter.export(sample_conversation, ExportFormat.TEXT)

        assert "10:30:00" in result
        assert "10:30:15" in result


class TestExportToFile:
    """Tests for file export."""

    def test_export_to_markdown(self, exporter, sample_conversation):
        """Test exporting to Markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "export.md")
            result = exporter.export_to_file(sample_conversation, file_path)

            assert os.path.exists(result)
            with open(result, "r") as f:
                content = f.read()
            assert "# Test Conversation" in content

    def test_export_to_json(self, exporter, sample_conversation):
        """Test exporting to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "export.json")
            result = exporter.export_to_file(sample_conversation, file_path)

            assert os.path.exists(result)
            with open(result, "r") as f:
                data = json.load(f)
            assert "messages" in data

    def test_export_to_html(self, exporter, sample_conversation):
        """Test exporting to HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "export.html")
            result = exporter.export_to_file(sample_conversation, file_path)

            assert os.path.exists(result)
            with open(result, "r") as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content

    def test_auto_detect_format(self, exporter, sample_conversation):
        """Test format auto-detection from extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test .md
            md_path = os.path.join(tmpdir, "test.md")
            exporter.export_to_file(sample_conversation, md_path)
            with open(md_path, "r") as f:
                assert "# " in f.read()

            # Test .json
            json_path = os.path.join(tmpdir, "test.json")
            exporter.export_to_file(sample_conversation, json_path)
            with open(json_path, "r") as f:
                assert json.load(f)

    def test_creates_directories(self, exporter, sample_conversation):
        """Test that directories are created as needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "subdir", "deep", "export.md")
            result = exporter.export_to_file(sample_conversation, file_path)

            assert os.path.exists(result)


class TestFromMessageList:
    """Tests for from_message_list factory."""

    def test_basic_conversion(self):
        """Test basic message list conversion."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        export = ConversationExporter.from_message_list(messages)

        assert len(export.messages) == 2
        assert export.messages[0].role == "user"
        assert export.messages[1].content == "Hi there!"

    def test_with_metadata(self):
        """Test conversion with metadata."""
        messages = [{"role": "user", "content": "Test"}]

        export = ConversationExporter.from_message_list(
            messages,
            title="My Chat",
            model="gpt-4",
            provider="openai",
        )

        assert export.title == "My Chat"
        assert export.model == "gpt-4"
        assert export.provider == "openai"

    def test_with_timestamps(self):
        """Test conversion with timestamps."""
        messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-01-15T10:30:00",
            }
        ]

        export = ConversationExporter.from_message_list(messages)

        assert export.messages[0].timestamp is not None
        assert export.messages[0].timestamp.year == 2024


class TestGlobalExporter:
    """Tests for global exporter function."""

    def test_get_exporter_returns_instance(self):
        """Test getting global exporter."""
        exporter = get_exporter()
        assert isinstance(exporter, ConversationExporter)

    def test_get_exporter_singleton(self):
        """Test singleton behavior."""
        exporter1 = get_exporter()
        exporter2 = get_exporter()
        assert exporter1 is exporter2

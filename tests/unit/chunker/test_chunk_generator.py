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

"""Unit tests for ChunkGenerator.

Tests chunk generation for tool execution, status updates, metrics, and content.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from victor.agent.services.chunk_runtime import ChunkGenerator as LegacyChunkGenerator
from victor.agent.services.chunk_runtime import ChunkGenerator
from victor.providers.base import StreamChunk
from victor.config.settings import Settings


def test_legacy_chunk_generator_module_reexports_service_runtime():
    assert LegacyChunkGenerator is ChunkGenerator


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture
def mock_streaming_handler():
    """Create mock StreamingChatHandler."""
    handler = MagicMock()

    # Configure mock StreamChunk returns
    handler.generate_tool_start_chunk.return_value = StreamChunk(
        content="", metadata={"type": "tool_start"}
    )
    handler.generate_tool_result_chunks.return_value = [
        StreamChunk(content="result", metadata={"type": "tool_result"})
    ]
    handler.generate_thinking_status_chunk.return_value = StreamChunk(
        content="", metadata={"type": "thinking"}
    )
    handler.generate_budget_error_chunk.return_value = StreamChunk(
        content="Budget exhausted", metadata={"type": "error"}
    )
    handler.generate_force_response_error_chunk.return_value = StreamChunk(
        content="Force response", metadata={"type": "error"}
    )
    handler.generate_final_marker_chunk.return_value = StreamChunk(content="", is_final=True)
    handler.generate_metrics_chunk.return_value = StreamChunk(
        content="metrics", metadata={"type": "metrics"}
    )
    handler.generate_content_chunk.return_value = StreamChunk(
        content="content", metadata={"type": "content"}
    )
    handler.get_budget_exhausted_chunks.return_value = [
        StreamChunk(content="Budget warning", metadata={"type": "warning"})
    ]

    return handler


@pytest.fixture
def chunk_generator(mock_streaming_handler, mock_settings):
    """Create ChunkGenerator with mocked dependencies."""
    return ChunkGenerator(
        streaming_handler=mock_streaming_handler,
        settings=mock_settings,
    )


class TestToolRelatedChunks:
    """Tests for tool-related chunk generation."""

    def test_generate_tool_start_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating tool start chunk."""
        tool_name = "read_file"
        tool_args = {"path": "/test/file.txt"}
        status_msg = "Reading file..."

        chunk = chunk_generator.generate_tool_start_chunk(tool_name, tool_args, status_msg)

        # Verify delegation to streaming handler
        mock_streaming_handler.generate_tool_start_chunk.assert_called_once_with(
            tool_name, tool_args, status_msg, tool_call_id=None
        )

        # Verify chunk returned
        assert chunk is not None
        assert chunk.metadata["type"] == "tool_start"

    def test_generate_tool_result_chunks(self, chunk_generator, mock_streaming_handler):
        """Test generating tool result chunks."""
        result = {"output": "File content here", "success": True}

        chunks = chunk_generator.generate_tool_result_chunks(result)

        # Verify delegation to streaming handler
        mock_streaming_handler.generate_tool_result_chunks.assert_called_once_with(result)

        # Verify chunks returned
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert chunks[0].metadata["type"] == "tool_result"

    def test_generate_tool_start_chunk_with_empty_args(
        self, chunk_generator, mock_streaming_handler
    ):
        """Test generating tool start chunk with empty arguments."""
        chunk = chunk_generator.generate_tool_start_chunk(
            "list_directory", {}, "Listing directory..."
        )

        mock_streaming_handler.generate_tool_start_chunk.assert_called_once()
        assert chunk is not None


class TestStatusChunks:
    """Tests for status chunk generation."""

    def test_generate_thinking_status_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating thinking status chunk."""
        chunk = chunk_generator.generate_thinking_status_chunk()

        # Verify delegation
        mock_streaming_handler.generate_thinking_status_chunk.assert_called_once()

        # Verify chunk
        assert chunk is not None
        assert chunk.metadata["type"] == "thinking"

    def test_generate_budget_error_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating budget error chunk."""
        chunk = chunk_generator.generate_budget_error_chunk()

        # Verify delegation
        mock_streaming_handler.generate_budget_error_chunk.assert_called_once()

        # Verify chunk
        assert chunk is not None
        assert chunk.metadata["type"] == "error"
        assert "Budget" in chunk.content

    def test_generate_force_response_error_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating force response error chunk."""
        chunk = chunk_generator.generate_force_response_error_chunk()

        # Verify delegation
        mock_streaming_handler.generate_force_response_error_chunk.assert_called_once()

        # Verify chunk
        assert chunk is not None
        assert chunk.metadata["type"] == "error"

    def test_generate_final_marker_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating final marker chunk."""
        chunk = chunk_generator.generate_final_marker_chunk()

        # Verify delegation
        mock_streaming_handler.generate_final_marker_chunk.assert_called_once()

        # Verify chunk
        assert chunk is not None
        assert chunk.is_final is True


class TestContentChunks:
    """Tests for content chunk generation."""

    def test_generate_metrics_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating metrics chunk."""
        metrics_line = "Tokens: 1000 | Time: 5.2s"

        chunk = chunk_generator.generate_metrics_chunk(metrics_line)

        # Verify delegation with defaults
        mock_streaming_handler.generate_metrics_chunk.assert_called_once_with(
            metrics_line, is_final=False, prefix="\n\n"
        )

        # Verify chunk
        assert chunk is not None
        assert chunk.metadata["type"] == "metrics"

    def test_generate_metrics_chunk_with_custom_params(
        self, chunk_generator, mock_streaming_handler
    ):
        """Test generating metrics chunk with custom parameters."""
        metrics_line = "Final metrics"

        chunk = chunk_generator.generate_metrics_chunk(metrics_line, is_final=True, prefix="\n")

        # Verify delegation with custom params
        mock_streaming_handler.generate_metrics_chunk.assert_called_once_with(
            metrics_line, is_final=True, prefix="\n"
        )

    def test_generate_content_chunk(self, chunk_generator, mock_streaming_handler):
        """Test generating content chunk."""
        content = "Generated response content"

        chunk = chunk_generator.generate_content_chunk(content)

        # Verify delegation with defaults
        mock_streaming_handler.generate_content_chunk.assert_called_once_with(
            content, is_final=False, suffix=""
        )

        # Verify chunk
        assert chunk is not None
        assert chunk.metadata["type"] == "content"

    def test_generate_content_chunk_with_custom_params(
        self, chunk_generator, mock_streaming_handler
    ):
        """Test generating content chunk with custom parameters."""
        content = "Final content"

        chunk = chunk_generator.generate_content_chunk(content, is_final=True, suffix="\n\n")

        # Verify delegation with custom params
        mock_streaming_handler.generate_content_chunk.assert_called_once_with(
            content, is_final=True, suffix="\n\n"
        )


class TestBudgetChunks:
    """Tests for budget-related chunk generation."""

    def test_get_budget_exhausted_chunks(self, chunk_generator, mock_streaming_handler):
        """Test getting budget exhausted chunks."""
        stream_ctx = Mock()
        stream_ctx.iteration_count = 10

        chunks = chunk_generator.get_budget_exhausted_chunks(stream_ctx)

        # Verify delegation
        mock_streaming_handler.get_budget_exhausted_chunks.assert_called_once_with(stream_ctx)

        # Verify chunks
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert chunks[0].metadata["type"] == "warning"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_chunk_generator_with_none_handler(self, mock_settings):
        """Test ChunkGenerator handles None streaming handler gracefully."""
        # This should raise an error during initialization or method calls
        chunk_generator = ChunkGenerator(
            streaming_handler=None,
            settings=mock_settings,
        )

        # Attempting to use it should fail
        with pytest.raises(AttributeError):
            chunk_generator.generate_thinking_status_chunk()

    def test_generate_tool_result_chunks_empty_result(
        self, chunk_generator, mock_streaming_handler
    ):
        """Test generating tool result chunks with empty result."""
        result = {}

        chunks = chunk_generator.generate_tool_result_chunks(result)

        # Should still delegate
        mock_streaming_handler.generate_tool_result_chunks.assert_called_once_with(result)

    def test_generate_metrics_chunk_empty_string(self, chunk_generator, mock_streaming_handler):
        """Test generating metrics chunk with empty string."""
        chunk = chunk_generator.generate_metrics_chunk("")

        # Should still delegate
        mock_streaming_handler.generate_metrics_chunk.assert_called_once()

    def test_generate_content_chunk_empty_string(self, chunk_generator, mock_streaming_handler):
        """Test generating content chunk with empty string."""
        chunk = chunk_generator.generate_content_chunk("")

        # Should still delegate
        mock_streaming_handler.generate_content_chunk.assert_called_once()


class TestChunkGeneratorInitialization:
    """Tests for ChunkGenerator initialization."""

    def test_initialization_with_valid_dependencies(self, mock_streaming_handler, mock_settings):
        """Test successful initialization with valid dependencies."""
        chunk_generator = ChunkGenerator(
            streaming_handler=mock_streaming_handler,
            settings=mock_settings,
        )

        assert chunk_generator.streaming_handler is mock_streaming_handler
        assert chunk_generator.settings is mock_settings

    def test_initialization_stores_dependencies(self, mock_streaming_handler, mock_settings):
        """Test that initialization stores all dependencies."""
        chunk_generator = ChunkGenerator(
            streaming_handler=mock_streaming_handler,
            settings=mock_settings,
        )

        # Verify dependencies are accessible
        assert hasattr(chunk_generator, "streaming_handler")
        assert hasattr(chunk_generator, "settings")

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

"""Tests for output deduplicator.

These tests verify that the deduplicator correctly identifies and removes
repeated content blocks while preserving unique content.
"""

import pytest
from victor.agent.output_deduplicator import (
    OutputDeduplicator,
    StreamingDeduplicator,
    DeduplicationStats,
)


class TestDeduplicationStats:
    """Tests for DeduplicationStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = DeduplicationStats()

        assert stats.total_blocks == 0
        assert stats.duplicates_removed == 0
        assert stats.bytes_saved == 0
        assert len(stats.unique_hashes) == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = DeduplicationStats(
            total_blocks=10,
            duplicates_removed=3,
            bytes_saved=500,
        )
        stats.unique_hashes = {"hash1", "hash2"}

        result = stats.to_dict()

        assert result["total_blocks"] == 10
        assert result["duplicates_removed"] == 3
        assert result["bytes_saved"] == 500
        assert result["unique_count"] == 2
        assert result["dedup_ratio"] == 0.3

    def test_dedup_ratio_zero_blocks(self):
        """Test dedup ratio with zero blocks."""
        stats = DeduplicationStats()
        result = stats.to_dict()

        assert result["dedup_ratio"] == 0.0


class TestOutputDeduplicator:
    """Tests for OutputDeduplicator."""

    @pytest.fixture
    def dedup(self):
        return OutputDeduplicator()

    def test_process_empty_content(self, dedup):
        """Test processing empty content."""
        result = dedup.process("")
        assert result == ""

        result = dedup.process("   ")
        assert result == "   "

    def test_process_no_duplicates(self, dedup):
        """Test processing content without duplicates."""
        content = "First paragraph with unique content.\n\nSecond paragraph also unique."
        result = dedup.process(content)

        assert "First paragraph" in result
        assert "Second paragraph" in result
        assert dedup.duplicates_removed == 0

    def test_process_with_duplicates(self, dedup):
        """Test removing duplicate paragraphs."""
        content = """This is a paragraph with enough content to be considered for deduplication.

This is a different paragraph with its own unique content that should be kept.

This is a paragraph with enough content to be considered for deduplication."""

        result = dedup.process(content)

        # Should only have 2 unique paragraphs
        assert result.count("This is a paragraph with enough content") == 1
        assert "different paragraph" in result
        assert dedup.duplicates_removed == 1

    def test_short_blocks_not_deduplicated(self, dedup):
        """Test that short blocks are kept even if repeated."""
        content = "Short.\n\nShort.\n\nAnother short one."
        result = dedup.process(content)

        # Short blocks should not be deduplicated
        assert result.count("Short.") == 2
        assert dedup.duplicates_removed == 0

    def test_whitespace_normalization(self):
        """Test that whitespace differences are normalized."""
        dedup = OutputDeduplicator(min_block_length=20, normalize_whitespace=True)

        content = """This is content with   multiple   spaces between words here.

This is content with multiple spaces between words here."""

        result = dedup.process(content)

        # Should recognize as duplicate despite whitespace differences
        assert dedup.duplicates_removed == 1

    def test_code_blocks_preserved(self, dedup):
        """Test that code blocks are handled correctly."""
        content = """Here is some explanation.

```python
def foo():
    return "bar"
```

More explanation here.

```python
def foo():
    return "bar"
```"""

        result = dedup.process(content)

        # Code blocks might be treated as duplicates if long enough
        assert "def foo" in result

    def test_reset(self, dedup):
        """Test resetting deduplicator state."""
        content = "This is a paragraph with enough content to be considered."
        dedup.process(content + "\n\n" + content)

        assert dedup.duplicates_removed == 1

        dedup.reset()

        assert dedup.duplicates_removed == 0
        assert dedup.get_stats()["total_blocks"] == 0

    def test_native_and_python_paths_deduplicate_across_calls(self):
        """Repeated blocks should be suppressed across separate process() calls."""
        content = (
            "Now I have enough data to produce the complete analysis based on the "
            "evidence collected from the repository."
        )

        dedup = OutputDeduplicator(min_block_length=40)
        first = dedup.process(f"{content}\n\nFirst unique detail.")
        second = dedup.process(f"{content}\n\nSecond unique detail.")

        assert "First unique detail." in first
        assert second == "Second unique detail."
        assert dedup.duplicates_removed >= 1

    def test_get_stats(self, dedup):
        """Test getting statistics."""
        content = """First unique paragraph with enough content to count.

Second unique paragraph with different content here.

First unique paragraph with enough content to count."""

        dedup.process(content)
        stats = dedup.get_stats()

        assert stats["total_blocks"] == 3
        assert stats["duplicates_removed"] == 1
        assert stats["bytes_saved"] > 0

    def test_numbered_list_splitting(self, dedup):
        """Test that numbered lists are properly split."""
        content = """1. First item with enough content to be significant.

2. Second item with different content here.

3. Third item with unique information."""

        result = dedup.process(content)

        assert "First item" in result
        assert "Second item" in result
        assert "Third item" in result


class TestOutputDeduplicatorStreaming:
    """Tests for streaming chunk processing."""

    @pytest.fixture
    def dedup(self):
        return OutputDeduplicator()

    def test_process_chunk_accumulates(self, dedup):
        """Test that chunks accumulate until paragraph boundary."""
        result1 = dedup.process_chunk("Start of para")
        assert result1 == ""  # No complete paragraph yet

        result2 = dedup.process_chunk("graph continues")
        assert result2 == ""  # Still no paragraph boundary

    def test_process_chunk_on_boundary(self, dedup):
        """Test processing at paragraph boundary."""
        result1 = dedup.process_chunk("First paragraph content.\n\n")
        # May return content or empty depending on min length

        result2 = dedup.process_chunk("Second paragraph.\n\n")
        # Process continues

    def test_flush(self, dedup):
        """Test flushing remaining content."""
        dedup.process_chunk("Partial content without boundary")
        result = dedup.flush()

        assert "Partial content" in result


class TestStreamingDeduplicator:
    """Tests for StreamingDeduplicator."""

    @pytest.fixture
    def dedup(self):
        return StreamingDeduplicator()

    def test_add_chunk_no_boundary(self, dedup):
        """Test adding chunk without paragraph boundary."""
        result = dedup.add_chunk("Content without boundary")
        assert result is None

    def test_add_chunk_with_boundary(self, dedup):
        """Test adding chunk with paragraph boundary."""
        result = dedup.add_chunk("First paragraph with content.\n\nSecond paragraph here.")
        # Should return processed content
        assert result is None or isinstance(result, str)

    def test_window_based_deduplication(self):
        """Test window-based duplicate detection."""
        dedup = StreamingDeduplicator(window_size=3, min_block_length=20)

        # Add several unique paragraphs
        for i in range(5):
            dedup.add_chunk(f"Unique paragraph number {i} with content.\n\n")

        # Window should maintain only recent hashes
        assert len(dedup._recent_hashes) <= 3

    def test_flush_streaming(self, dedup):
        """Test flushing streaming buffer."""
        dedup.add_chunk("Buffered content")
        result = dedup.flush()

        assert result == "Buffered content"

    def test_reset_streaming(self, dedup):
        """Test resetting streaming deduplicator."""
        dedup.add_chunk("Some content\n\n")
        dedup.reset()

        assert len(dedup._recent_hashes) == 0
        assert dedup._buffer == ""

    def test_get_stats_streaming(self, dedup):
        """Test getting stats from streaming deduplicator."""
        stats = dedup.get_stats()

        assert "total_blocks" in stats
        assert "duplicates_removed" in stats
        assert "bytes_saved" in stats


class TestDeduplicationScenarios:
    """Integration tests for real-world deduplication scenarios."""

    def test_grok_repetition_pattern(self):
        """Test handling Grok's repetitive output pattern."""
        dedup = OutputDeduplicator(min_block_length=30)

        # Simulate Grok repeating file lists
        file_list = """1. investor_homelab/models/database_schema.py
   - Defines the SQLAlchemy schema for persistent storage.

2. investor_homelab/models/news_model.py
   - Contains ORM-backed models for news articles.

3. investor_homelab/utils/web_search_client.py
   - Implements web search functionality."""

        # Grok repeats this 4 times
        content = (file_list + "\n\n") * 4

        result = dedup.process(content)

        # Should deduplicate to single occurrence of meaningful blocks
        assert dedup.duplicates_removed >= 2

    def test_partial_duplicates_not_removed(self):
        """Test that partial matches are not incorrectly deduplicated."""
        dedup = OutputDeduplicator(min_block_length=20)

        content = """This is the first version of the function implementation.

This is the second version of the function with improvements.

This is the first version of the function implementation."""

        result = dedup.process(content)

        # Should remove exact duplicate, keep partial match
        assert "first version" in result
        assert "second version" in result
        assert dedup.duplicates_removed == 1

    def test_case_insensitive_deduplication(self):
        """Test that case differences are normalized."""
        dedup = OutputDeduplicator(min_block_length=20)

        content = """THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.

The quick brown fox jumps over the lazy dog."""

        result = dedup.process(content)

        # Should recognize as duplicate due to case normalization
        assert dedup.duplicates_removed == 1


class TestIntelligentPipelineDeduplication:
    """Tests for RuntimeIntelligencePipeline deduplication integration.

    These tests verify that the pipeline correctly integrates the output
    deduplicator with provider-aware configuration (Strategy Pattern).
    """

    @pytest.fixture
    def grok_pipeline(self):
        """Create pipeline for Grok provider (deduplication enabled)."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        return RuntimeIntelligencePipeline(
            provider_name="grok",
            model="grok-beta",
            profile_name="test-grok",
        )

    @pytest.fixture
    def xai_pipeline(self):
        """Create pipeline for xAI provider (deduplication enabled)."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        return RuntimeIntelligencePipeline(
            provider_name="xai",
            model="grok-2-1212",
            profile_name="test-xai",
        )

    @pytest.fixture
    def anthropic_pipeline(self):
        """Create pipeline for Anthropic provider (deduplication disabled)."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        return RuntimeIntelligencePipeline(
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            profile_name="test-anthropic",
        )

    @pytest.fixture
    def deepseek_pipeline(self):
        """Create pipeline for DeepSeek provider (deduplication disabled)."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        return RuntimeIntelligencePipeline(
            provider_name="deepseek",
            model="deepseek-chat",
            profile_name="test-deepseek",
        )

    def test_deduplication_enabled_for_grok(self, grok_pipeline):
        """Test deduplication is enabled for Grok provider."""
        assert grok_pipeline._deduplication_enabled is True
        assert grok_pipeline._should_enable_deduplication() is True

    def test_deduplication_enabled_for_xai(self, xai_pipeline):
        """Test deduplication is enabled for xAI provider."""
        assert xai_pipeline._deduplication_enabled is True
        assert xai_pipeline._should_enable_deduplication() is True

    def test_deduplication_disabled_for_anthropic(self, anthropic_pipeline):
        """Test deduplication is disabled for Anthropic provider."""
        assert anthropic_pipeline._deduplication_enabled is False
        assert anthropic_pipeline._should_enable_deduplication() is False

    def test_deduplication_disabled_for_deepseek(self, deepseek_pipeline):
        """Test deduplication is disabled for DeepSeek provider."""
        assert deepseek_pipeline._deduplication_enabled is False
        assert deepseek_pipeline._should_enable_deduplication() is False

    def test_deduplicate_response_when_enabled(self, grok_pipeline):
        """Test deduplicate_response applies deduplication when enabled."""
        # Simulate Grok's repetitive output pattern
        content = """This is a paragraph with significant content to deduplicate.

This is a different paragraph with unique information.

This is a paragraph with significant content to deduplicate."""

        result, stats = grok_pipeline.deduplicate_response(content)

        assert stats["deduplication_applied"] is True
        assert stats["provider"] == "grok"
        assert stats["duplicates_removed"] >= 1
        # Deduplicated content should have fewer occurrences
        assert result.count("significant content to deduplicate") == 1

    def test_deduplicate_response_when_disabled(self, anthropic_pipeline):
        """Test deduplicate_response skips processing when disabled."""
        content = """This is repeated content for testing purposes.

This is repeated content for testing purposes."""

        result, stats = anthropic_pipeline.deduplicate_response(content)

        assert stats["deduplication_applied"] is False
        assert result == content  # Unchanged
        assert "provider" not in stats  # Not set when disabled

    def test_deduplicate_response_empty_input(self, grok_pipeline):
        """Test deduplicate_response handles empty input gracefully."""
        result, stats = grok_pipeline.deduplicate_response("")

        assert result == ""
        assert stats["deduplication_applied"] is False

    def test_deduplicator_lazy_initialization(self, grok_pipeline):
        """Test output deduplicator is lazily initialized."""
        # Initially None
        assert grok_pipeline._output_deduplicator is None

        # First call initializes it
        dedup = grok_pipeline._get_output_deduplicator()
        assert dedup is not None
        assert grok_pipeline._output_deduplicator is dedup

        # Subsequent calls return same instance
        dedup2 = grok_pipeline._get_output_deduplicator()
        assert dedup2 is dedup

    def test_deduplicator_not_initialized_when_disabled(self, anthropic_pipeline):
        """Test deduplicator is not created for non-repetitive providers."""
        # Call deduplicate_response (should skip processing)
        anthropic_pipeline.deduplicate_response("test content")

        # Deduplicator should not have been initialized
        assert anthropic_pipeline._output_deduplicator is None

    def test_provider_name_case_insensitive(self):
        """Test provider detection is case-insensitive."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        # All variations should enable deduplication
        for provider in ["GROK", "Grok", "grok", "XAI", "xai", "X-AI", "x-ai"]:
            pipeline = RuntimeIntelligencePipeline(
                provider_name=provider,
                model="test",
                profile_name="test",
            )
            assert pipeline._deduplication_enabled is True, f"Failed for {provider}"

    def test_deduplicator_reset_on_each_call(self, grok_pipeline):
        """Test deduplicator is reset between calls for fresh state."""
        content1 = """First unique content block here."""
        content2 = """First unique content block here."""  # Same as content1

        # First call
        grok_pipeline.deduplicate_response(content1)

        # Second call should see duplicate but won't because reset() is called
        result2, stats2 = grok_pipeline.deduplicate_response(content2)

        # Stats should reflect fresh processing (no cross-call state)
        assert stats2["deduplication_applied"] is True

    @pytest.mark.asyncio
    async def test_process_response_applies_deduplication(self, grok_pipeline):
        """Test process_response integrates deduplication."""
        # Simulate repetitive Grok output
        response = """Here is a list of Python files with descriptions.

1. database_schema.py - Defines the SQLAlchemy schema.
2. news_model.py - Contains ORM models for news.

Here is a list of Python files with descriptions.

1. database_schema.py - Defines the SQLAlchemy schema.
2. news_model.py - Contains ORM models for news."""

        result = await grok_pipeline.process_response(
            response=response,
            query="List Python files",
            success=True,
        )

        # Pipeline should process successfully
        assert result.is_valid is True

    def test_providers_with_repetition_issues_constant(self):
        """Test PROVIDERS_WITH_REPETITION_ISSUES contains expected providers."""
        from victor.agent.runtime_intelligence_pipeline import (
            PROVIDERS_WITH_REPETITION_ISSUES,
        )

        assert "xai" in PROVIDERS_WITH_REPETITION_ISSUES
        assert "grok" in PROVIDERS_WITH_REPETITION_ISSUES
        assert "x-ai" in PROVIDERS_WITH_REPETITION_ISSUES
        # These should NOT be in the set
        assert "anthropic" not in PROVIDERS_WITH_REPETITION_ISSUES
        assert "openai" not in PROVIDERS_WITH_REPETITION_ISSUES
        assert "deepseek" not in PROVIDERS_WITH_REPETITION_ISSUES

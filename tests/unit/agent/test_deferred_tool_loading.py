# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for budget-aware compaction with deferred tool loading (P2-3).

TDD approach: Tests written first, then implementation.

Deferred tool loading replaces large tool results with lightweight placeholders
that can be loaded on-demand. This reduces context usage while preserving
the ability to access previous tool results when needed.
"""

from victor.agent.deferred_tool_loading import (
    ToolResultPlaceholder,
    DeferredToolResult,
    DeferredLoadingConfig,
    DeferredLoadingManager,
    should_defer_tool_result,
    create_placeholder_for_tool_result,
    restore_tool_result,
    get_deferred_result_store,
)


class TestToolResultPlaceholder:
    """Test suite for ToolResultPlaceholder dataclass."""

    def test_placeholder_has_required_fields(self):
        """Placeholder should have all required fields."""
        placeholder = ToolResultPlaceholder(
            tool_name="read",
            tool_args={"path": "file.py"},
            original_length=5000,
            result_id="result_123",
        )

        assert placeholder.tool_name == "read"
        assert placeholder.tool_args == {"path": "file.py"}
        assert placeholder.original_length == 5000
        assert placeholder.result_id == "result_123"

    def test_placeholder_to_string(self):
        """Placeholder should convert to informative string."""
        placeholder = ToolResultPlaceholder(
            tool_name="code_search",
            tool_args={"query": "foo"},
            original_length=8500,
            result_id="result_456",
        )

        result_str = str(placeholder)
        assert "code_search" in result_str
        assert "8500" in result_str  # original length


class TestDeferredLoadingConfig:
    """Test suite for DeferredLoadingConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = DeferredLoadingConfig()

        assert config.min_size_to_defer == 1000
        assert config.defer_tool_results is True
        assert max(config.defer_roles) == "tool"  # Only defer tool results by default

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = DeferredLoadingConfig(
            min_size_to_defer=500,
            defer_tool_results=True,
        )

        assert config.min_size_to_defer == 500
        assert config.defer_tool_results is True


class TestShouldDeferToolResult:
    """Test suite for should_defer_tool_result function."""

    def test_defer_large_tool_result(self):
        """Large tool results should be deferred."""
        result = "x" * 1500  # 1500 chars

        assert should_defer_tool_result(
            role="tool",
            content=result,
            config=DeferredLoadingConfig(min_size_to_defer=1000),
        )

    def test_not_defer_small_tool_result(self):
        """Small tool results should not be deferred."""
        result = "x" * 500  # 500 chars

        assert not should_defer_tool_result(
            role="tool",
            content=result,
            config=DeferredLoadingConfig(min_size_to_defer=1000),
        )

    def test_not_defer_non_tool_role(self):
        """Non-tool roles should not be deferred."""
        result = "x" * 2000

        for role in ["user", "assistant", "system"]:
            assert not should_defer_tool_result(
                role=role,
                content=result,
                config=DeferredLoadingConfig(),
            )

    def test_not_defer_when_disabled(self):
        """Should not defer when disabled in config."""
        config = DeferredLoadingConfig(defer_tool_results=False)
        result = "x" * 2000

        assert not should_defer_tool_result(
            role="tool",
            content=result,
            config=config,
        )


class TestCreatePlaceholderForToolResult:
    """Test suite for create_placeholder_for_tool_result function."""

    def test_placeholder_content_format(self):
        """Placeholder should have informative content."""
        placeholder = create_placeholder_for_tool_result(
            tool_name="read",
            tool_args={"path": "/path/to/file.py"},
            content="x" * 5000,
            result_id="result_123",
        )

        assert "read" in placeholder.content
        assert "5000" in placeholder.content  # original length
        assert "result_123" in placeholder.content or "deferred" in placeholder.content.lower()

    def test_placeholder_is_shorter(self):
        """Placeholder should be much shorter than original."""
        original = "x" * 5000
        placeholder = create_placeholder_for_tool_result(
            tool_name="grep",
            tool_args={"pattern": "foo"},
            content=original,
            result_id="result_456",
        )

        assert len(placeholder.content) < 500  # Much shorter


class TestDeferredLoadingManager:
    """Test suite for DeferredLoadingManager class."""

    def test_store_and_retrieve_tool_result(self):
        """Should store and retrieve tool results."""
        manager = DeferredLoadingManager()

        result_id = manager.store_result(
            tool_name="read",
            tool_args={"path": "file.py"},
            content="File content here",
        )

        retrieved = manager.get_result(result_id)
        assert retrieved is not None
        assert retrieved.content == "File content here"
        assert retrieved.tool_name == "read"

    def test_get_nonexistent_result_returns_none(self):
        """Getting nonexistent result should return None."""
        manager = DeferredLoadingManager()

        result = manager.get_result("nonexistent_id")
        assert result is None

    def test_defer_and_restore_workflow(self):
        """Full workflow: defer, then restore."""
        manager = DeferredLoadingManager()

        # Store a large tool result
        original_content = "x" * 5000
        result_id = manager.store_result(
            tool_name="code_search",
            tool_args={"query": "test"},
            content=original_content,
        )

        # Create placeholder
        placeholder = create_placeholder_for_tool_result(
            tool_name="code_search",
            tool_args={"query": "test"},
            content=original_content,
            result_id=result_id,
        )

        # Restore from placeholder
        restored = restore_tool_result(placeholder, manager)

        assert restored == original_content

    def test_manager_limits_stored_results(self):
        """Manager should limit stored results (LRU eviction)."""
        config = DeferredLoadingConfig(max_stored_results=3)
        manager = DeferredLoadingManager(config=config)

        # Store 5 results
        result_ids = []
        for i in range(5):
            result_id = manager.store_result(
                tool_name="read",
                tool_args={"path": f"file{i}.py"},
                content=f"Content {i}",
            )
            result_ids.append(result_id)

        # First 2 should be evicted (max 3)
        assert manager.get_result(result_ids[0]) is None
        assert manager.get_result(result_ids[1]) is None

        # Last 3 should still be present
        assert manager.get_result(result_ids[2]) is not None
        assert manager.get_result(result_ids[3]) is not None
        assert manager.get_result(result_ids[4]) is not None

    def test_get_stats(self):
        """Manager should provide statistics."""
        manager = DeferredLoadingManager()

        # Store some results
        for i in range(3):
            manager.store_result(
                tool_name="read",
                tool_args={"path": f"file{i}.py"},
                content="x" * 1000,
            )

        stats = manager.get_stats()
        assert stats["total_stored"] == 3
        assert stats["total_bytes_saved"] > 0


class TestRestoreToolResult:
    """Test suite for restore_tool_result function."""

    def test_restore_from_placeholder(self):
        """Should restore original content from placeholder."""
        manager = DeferredLoadingManager()
        original = "Original content here"

        result_id = manager.store_result(
            tool_name="read",
            tool_args={"path": "file.py"},
            content=original,
        )

        placeholder = ToolResultPlaceholder(
            tool_name="read",
            tool_args={"path": "file.py"},
            original_length=len(original),
            result_id=result_id,
        )

        restored = restore_tool_result(placeholder, manager)
        assert restored == original

    def test_restore_missing_result_returns_none(self):
        """Restoring missing result should return None."""
        manager = DeferredLoadingManager()

        placeholder = ToolResultPlaceholder(
            tool_name="read",
            tool_args={"path": "file.py"},
            original_length=100,
            result_id="nonexistent",
        )

        restored = restore_tool_result(placeholder, manager)
        assert restored is None


class TestGetDeferredResultStore:
    """Test suite for get_deferred_result_store function."""

    def test_returns_manager_instance(self):
        """Should return a DeferredLoadingManager instance."""
        store = get_deferred_result_store()
        assert isinstance(store, DeferredLoadingManager)

    def test_returns_same_instance(self):
        """Should return the same singleton instance."""
        store1 = get_deferred_result_store()
        store2 = get_deferred_result_store()

        assert store1 is store2


class TestDeferredToolResult:
    """Test suite for DeferredToolResult dataclass."""

    def test_deferred_result_structure(self):
        """DeferredToolResult should have correct structure."""
        result = DeferredToolResult(
            tool_name="grep",
            tool_args={"pattern": "test"},
            content="Match found",
            timestamp=1234567890,
        )

        assert result.tool_name == "grep"
        assert result.tool_args == {"pattern": "test"}
        assert result.content == "Match found"
        assert result.timestamp == 1234567890


class TestIntegrationScenarios:
    """Integration tests for deferred loading scenarios."""

    def test_compact_messages_with_deferred_loading(self):
        """Integration: compact messages and defer large tool results."""
        from victor.providers.base import Message

        manager = DeferredLoadingManager()
        config = DeferredLoadingConfig(min_size_to_defer=1000)

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Read file"),
            Message(role="tool", content="x" * 2000, tool_call_id="call_1"),
            Message(role="assistant", content="Let me analyze"),
            Message(role="tool", content="y" * 500, tool_call_id="call_2"),  # Small, not deferred
        ]

        # Process messages (would be done by compactor)
        compacted = []
        for msg in messages:
            if msg.role == "tool" and should_defer_tool_result(
                role=msg.role,
                content=msg.content,
                config=config,
            ):
                # Store and replace with placeholder
                result_id = manager.store_result(
                    tool_name="read",
                    tool_args={},
                    content=msg.content,
                )
                placeholder = create_placeholder_for_tool_result(
                    tool_name="read",
                    tool_args={},
                    content=msg.content,
                    result_id=result_id,
                )
                compacted.append(
                    Message(
                        role=msg.role,
                        content=placeholder.content,
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                compacted.append(msg)

        # First tool result should be deferred (shorter content)
        assert len(compacted[2].content) < 1000
        # Second tool result should NOT be deferred (original content)
        assert compacted[4].content == "y" * 500

    def test_restore_compacted_results(self):
        """Integration: restore deferred results when needed."""
        manager = DeferredLoadingManager()

        # Simulate storing a large result
        original = "x" * 5000
        result_id = manager.store_result(
            tool_name="code_search",
            tool_args={"query": "test"},
            content=original,
        )

        # Create placeholder
        placeholder = ToolResultPlaceholder(
            tool_name="code_search",
            tool_args={"query": "test"},
            original_length=len(original),
            result_id=result_id,
        )

        # Restore
        restored = restore_tool_result(placeholder, manager)
        assert restored == original
        assert len(restored) == 5000

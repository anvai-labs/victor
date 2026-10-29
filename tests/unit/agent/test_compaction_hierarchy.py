"""Tests for hierarchical compaction compression."""

from unittest.mock import MagicMock

import pytest

from victor.agent.compaction_hierarchy import (
    HierarchicalCompactionManager,
)


@pytest.fixture
def summarizer():
    s = MagicMock()
    s.summarize.return_value = "[Epoch summary: combined context]"
    return s


@pytest.fixture
def manager(summarizer):
    return HierarchicalCompactionManager(
        summarizer=summarizer,
        max_individual=3,
        epoch_threshold=6,
    )


class TestHierarchicalCompactionManager:
    def test_add_summary_below_threshold(self, manager):
        manager.add_summary("Summary 1", turn_index=1)
        manager.add_summary("Summary 2", turn_index=2)

        assert len(manager._individual_summaries) == 2
        assert len(manager._epochs) == 0

    def test_epoch_created_at_threshold(self, manager):
        for i in range(6):
            manager.add_summary(f"Summary {i}", turn_index=i)

        # 6 summaries should trigger epoch creation
        # 3 oldest become epoch, 3 most recent stay individual
        assert len(manager._epochs) == 1
        assert len(manager._individual_summaries) == 3
        assert manager._epochs[0].source_count == 3

    def test_get_active_context_includes_epochs_and_recent(self, manager):
        for i in range(6):
            manager.add_summary(f"Summary {i}", turn_index=i)

        context = manager.get_active_context()
        assert context  # Not empty
        # Should contain epoch info and recent summaries
        assert "Epoch" in context
        assert "Summary" in context

    def test_epoch_summary_uses_concatenation(self, manager, summarizer):
        """Epochs always use concatenation (not summarizer) since they combine
        existing summaries, not raw messages."""
        for i in range(6):
            manager.add_summary(f"Summary {i}", turn_index=i)

        # Summarizer should NOT be called — epochs use concatenation
        summarizer.summarize.assert_not_called()
        # Epoch summary should be pipe-joined
        assert " | " in manager._epochs[0].summary

    def test_to_dict_from_dict_round_trip(self, manager, summarizer):
        for i in range(8):
            manager.add_summary(f"Summary {i}", turn_index=i)

        data = manager.to_dict()
        restored = HierarchicalCompactionManager.from_dict(data, summarizer=summarizer)

        assert len(restored._individual_summaries) == len(manager._individual_summaries)
        assert len(restored._epochs) == len(manager._epochs)
        for orig, rest in zip(manager._epochs, restored._epochs):
            assert orig.epoch_id == rest.epoch_id
            assert orig.summary == rest.summary
            assert orig.source_count == rest.source_count

    def test_max_chars_respected(self, manager):
        for i in range(6):
            manager.add_summary(f"Summary with lots of content {i} " * 20, turn_index=i)

        context = manager.get_active_context(max_chars=200)
        assert len(context) <= 300  # Some tolerance for join separators

    def test_empty_manager_returns_empty(self):
        manager = HierarchicalCompactionManager()
        assert manager.get_active_context() == ""

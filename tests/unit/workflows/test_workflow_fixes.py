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

"""Tests for workflow fixes identified during MODE testing.

These tests verify the framework-level fixes implemented to address issues
found during EXPLORE, PLAN, and BUILD mode testing with multiple providers.

Issue Reference: /Users/vijaysingh/.claude/plans/workflow-test-issues.md
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_project_nested(tmp_path):
    """Create a nested project structure like investor_homelab/investor_homelab."""
    # Create nested structure
    outer = tmp_path / "my_project"
    outer.mkdir()
    inner = outer / "my_project"
    inner.mkdir()
    (inner / "__init__.py").write_text("# Package init")
    (inner / "utils").mkdir()
    (inner / "utils" / "__init__.py").write_text("# Utils")
    (inner / "utils" / "client.py").write_text("class Client: pass")
    (inner / "models").mkdir()
    (inner / "models" / "user.py").write_text("class User: pass")

    # Create files at outer level too
    (outer / "setup.py").write_text("# Setup")
    (outer / "tests").mkdir()
    (outer / "tests" / "test_client.py").write_text("# Tests")

    return outer


# =============================================================================
# Edit Tool Parameter Normalization Tests
# =============================================================================


class TestEditToolParameterNormalization:
    """Tests for edit tool parameter normalization (Issue #2)."""

    def test_normalize_path_alias_to_canonical(self):
        """Test normalizing 'file_path' to 'path'."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"type": "create", "file_path": "/test/file.py", "content": "code"}
        result = normalize_edit_operation(op)

        assert "path" in result
        assert result["path"] == "/test/file.py"

    def test_normalize_file_alias(self):
        """Test normalizing 'file' to 'path'."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"type": "create", "file": "/test/file.py", "content": "code"}
        result = normalize_edit_operation(op)

        assert "path" in result
        assert result["path"] == "/test/file.py"

    def test_normalize_old_new_str_aliases(self):
        """Test normalizing 'old' and 'new' to 'old_str' and 'new_str'."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"type": "replace", "old": "foo", "new": "bar"}
        result = normalize_edit_operation(op)

        assert "old_str" in result
        assert "new_str" in result
        assert result["old_str"] == "foo"
        assert result["new_str"] == "bar"

    def test_infer_type_from_old_str_key(self):
        """Test type inference when 'old_str' present but no 'type'."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"old_str": "foo", "new_str": "bar"}
        result = normalize_edit_operation(op)

        assert result["type"] == "replace"

    def test_infer_type_from_old_key(self):
        """Test type inference when 'old' present but no 'type'."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"old": "foo", "new": "bar"}
        result = normalize_edit_operation(op)

        assert result["type"] == "replace"

    def test_infer_type_from_new_path_key(self):
        """Test type inference when 'new_path' present but no 'type'."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"path": "/old/path", "new_path": "/new/path"}
        result = normalize_edit_operation(op)

        assert result["type"] == "rename"

    def test_infer_create_type_from_content(self):
        """Test type inference when only 'content' present."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        op = {"path": "/test/file.py", "content": "code"}
        result = normalize_edit_operation(op)

        assert result["type"] == "create"

    def test_normalize_type_aliases(self):
        """Test normalizing operation type aliases."""
        from victor.tools.file_editor_tool import normalize_edit_operation

        # Test various type aliases
        cases = [
            ({"type": "write", "content": "x"}, "create"),
            ({"type": "add", "content": "x"}, "create"),
            ({"type": "update", "old_str": "a", "new_str": "b"}, "modify"),
            ({"type": "remove", "path": "x"}, "delete"),
            ({"type": "move", "path": "x", "new_path": "y"}, "rename"),
            ({"type": "find_replace", "old_str": "a", "new_str": "b"}, "replace"),
        ]

        for op, expected_type in cases:
            result = normalize_edit_operation(op)
            assert result["type"] == expected_type, f"Failed for {op}"

    def test_normalize_operations_list(self):
        """Test normalizing a list of operations."""
        from victor.tools.file_editor_tool import normalize_edit_operations

        ops = [
            {"file_path": "/a.py", "content": "a"},
            {"old": "x", "new": "y"},
        ]

        result = normalize_edit_operations(ops)

        assert len(result) == 2
        assert result[0]["path"] == "/a.py"
        assert result[0]["type"] == "create"
        assert result[1]["old_str"] == "x"
        assert result[1]["type"] == "replace"


# =============================================================================
# Mode-Aware Loop Detection Tests
# =============================================================================


class TestModeAwareLoopDetection:
    """Tests for mode-aware loop detection (Issue #1)."""

    def test_loop_threshold_default(self):
        """Test default loop threshold without mode multiplier."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()
        threshold = tracker._get_loop_threshold()
        assert threshold == DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD

    def test_loop_patience_limits_default(self):
        """Default patience limits should match shared blocked-loop defaults."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        assert tracker.get_loop_patience_limits() == (
            DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
            DEFAULT_BLOCKED_TOTAL_THRESHOLD,
        )

    def test_loop_threshold_with_mode_multiplier(self):
        """Test loop threshold with PLAN mode multiplier (2.5x)."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()
        tracker.set_mode_exploration_multiplier(2.5)  # PLAN mode

        threshold = tracker._get_loop_threshold()
        # Base 4 * 2.5 = 10
        assert threshold == 10

    def test_loop_patience_limits_scale_with_mode_multiplier(self):
        """Patience limits should preserve the shared ratio when exploration increases."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()
        tracker.set_mode_exploration_multiplier(2.5)

        assert tracker.get_loop_patience_limits() == (10, 15)

    def test_loop_threshold_with_explore_multiplier(self):
        """Test loop threshold with EXPLORE mode multiplier (3.0x)."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()
        tracker.set_mode_exploration_multiplier(3.0)  # EXPLORE mode

        threshold = tracker._get_loop_threshold()
        # Base 4 * 3.0 = 12
        assert threshold == 12

    def test_max_overlapping_reads_mode_aware(self):
        """Test max overlapping reads increases with mode multiplier."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Get default max (depends on config, typically 3)
        default_max = tracker.max_overlapping_reads
        assert default_max >= 3

        # PLAN mode multiplier
        tracker.set_mode_exploration_multiplier(2.5)
        plan_max = tracker.max_overlapping_reads
        # Should be at least 2.5x the default
        assert plan_max >= default_max

    def test_max_searches_per_prefix_mode_aware(self):
        """Test max searches per prefix increases with mode multiplier."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Get default max (depends on config, typically 2)
        default_max = tracker.max_searches_per_prefix
        assert default_max >= 2

        # EXPLORE mode multiplier
        tracker.set_mode_exploration_multiplier(3.0)
        explore_max = tracker.max_searches_per_prefix
        # Should be at least 3x the default
        assert explore_max >= default_max


# =============================================================================
# Grounding Stopword Filter Tests
# =============================================================================


class TestGroundingStopwordFilter:
    """Tests for grounding stopword filtering (Issue #3)."""

    def test_common_words_in_stopwords(self):
        """Test that common English words are in stopwords set."""
        from victor.agent.grounding_verifier import GROUNDING_STOPWORDS

        common_words = ["that", "handles", "should", "has", "the", "is", "are", "was"]
        for word in common_words:
            assert word in GROUNDING_STOPWORDS, f"'{word}' should be in stopwords"

    def test_tech_description_words_in_stopwords(self):
        """Test that common tech description words are in stopwords."""
        from victor.agent.grounding_verifier import GROUNDING_STOPWORDS

        tech_words = [
            "handles",
            "manages",
            "provides",
            "supports",
            "allows",
            "creates",
            "returns",
            "contains",
            "includes",
            "uses",
        ]
        for word in tech_words:
            assert word in GROUNDING_STOPWORDS, f"'{word}' should be in stopwords"

    def test_prepositions_in_stopwords(self):
        """Test that prepositions are in stopwords."""
        from victor.agent.grounding_verifier import GROUNDING_STOPWORDS

        prepositions = ["in", "on", "at", "to", "for", "of", "with", "by", "from"]
        for word in prepositions:
            assert word in GROUNDING_STOPWORDS, f"'{word}' should be in stopwords"

    def test_stopwords_count(self):
        """Test that stopwords set has reasonable size."""
        from victor.agent.grounding_verifier import GROUNDING_STOPWORDS

        # Should have at least 200 common words
        assert len(GROUNDING_STOPWORDS) >= 200

    def test_code_identifiers_not_in_stopwords(self):
        """Test that actual code identifiers are NOT in stopwords."""
        from victor.agent.grounding_verifier import GROUNDING_STOPWORDS

        # These look like code symbols, not common English words
        code_identifiers = [
            "RSSClient",
            "fetch_feed",
            "parse_xml",
            "get_articles",
            "UserModel",
            "async_handler",
            "calculate_total",
        ]
        for identifier in code_identifiers:
            assert identifier.lower() not in GROUNDING_STOPWORDS


# =============================================================================
# Context-Aware Loop Signatures Tests
# =============================================================================


class TestContextAwareLoopSignatures:
    """Tests for context-aware loop signatures (Issue #4)."""

    def test_signature_includes_stage(self):
        """Test that signatures include conversation stage by default."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        sig = tracker._get_signature("read_file", {"path": "/test.py"})

        # Should include stage in signature
        assert "stage:" in sig

    def test_signature_excludes_volatile_fields(self):
        """Test that volatile fields like offset/limit are excluded."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Same file, different offsets should produce same signature
        sig1 = tracker._get_signature("read_file", {"path": "/test.py", "offset": 0})
        sig2 = tracker._get_signature("read_file", {"path": "/test.py", "offset": 100})

        assert sig1 == sig2

    def test_signature_differs_by_file(self):
        """Test that different files produce different signatures."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        sig1 = tracker._get_signature("read_file", {"path": "/test1.py"})
        sig2 = tracker._get_signature("read_file", {"path": "/test2.py"})

        # Base signature before stage should differ
        assert sig1 != sig2

    def test_signature_stage_awareness(self):
        """Test that same operation in different stages produces different sig."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker
        from victor.agent.unified_task_tracker import ConversationStage

        tracker = UnifiedTaskTracker()

        # In READING stage
        tracker._progress.stage = ConversationStage.READING
        sig_reading = tracker._get_signature("read_file", {"path": "/test.py"})

        # In EXECUTING stage
        tracker._progress.stage = ConversationStage.EXECUTING
        sig_executing = tracker._get_signature("read_file", {"path": "/test.py"})

        assert sig_reading != sig_executing

    def test_signature_without_stage(self):
        """Test signature generation without stage for backward compat."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        sig = tracker._get_signature("read_file", {"path": "/test.py"}, include_stage=False)

        assert "stage:" not in sig


# =============================================================================
# Multi-Root Path Resolution Tests
# =============================================================================


class TestMultiRootPathResolution:
    """Tests for multi-root path resolution (Issue #6)."""

    def test_auto_detect_nested_project(self, temp_project_nested):
        """Test auto-detection of nested project structure."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)

        # Should auto-detect my_project/my_project/ as additional root
        assert len(resolver.additional_roots) == 1
        assert resolver.additional_roots[0].name == "my_project"

    def test_resolve_from_additional_root(self, temp_project_nested):
        """Test resolving paths from additional root."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)

        # This file exists at my_project/my_project/utils/client.py
        # When accessed as utils/client.py from my_project/, should find it
        result = resolver.resolve("utils/client.py")

        assert result.exists
        assert result.is_file
        assert result.was_normalized  # Resolved from additional root

    def test_resolve_from_primary_root_first(self, temp_project_nested):
        """Test that primary cwd is searched first."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)

        # setup.py exists at my_project/setup.py (primary)
        result = resolver.resolve("setup.py")

        assert result.exists
        assert not result.was_normalized  # Found in primary

    def test_add_search_root(self, temp_project_nested):
        """Test manually adding search root."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)
        tests_dir = temp_project_nested / "tests"

        initial_count = len(resolver.additional_roots)
        resolver.add_search_root(tests_dir)

        assert len(resolver.additional_roots) == initial_count + 1

    def test_remove_search_root(self, temp_project_nested):
        """Test removing search root."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)
        initial_count = len(resolver.additional_roots)

        if initial_count > 0:
            root_to_remove = resolver.additional_roots[0]
            result = resolver.remove_search_root(root_to_remove)
            assert result is True
            assert len(resolver.additional_roots) == initial_count - 1

    def test_search_roots_property(self, temp_project_nested):
        """Test search_roots property includes cwd and additional roots."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)
        roots = resolver.search_roots

        assert roots[0] == resolver.cwd
        assert len(roots) == 1 + len(resolver.additional_roots)

    def test_normalization_applied_description(self, temp_project_nested):
        """Test that normalization description includes root info."""
        from victor.protocols.path_resolver import PathResolver

        resolver = PathResolver(cwd=temp_project_nested)

        # Resolve from additional root
        result = resolver.resolve("utils/client.py")

        if result.was_normalized:
            assert "resolved_from" in result.normalization_applied


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkflowFixesIntegration:
    """Integration tests combining multiple fixes."""

    def test_edit_normalization_in_tool_execution(self):
        """Test that edit normalization works in actual tool context."""
        from victor.tools.file_editor_tool import (
            normalize_edit_operations,
            EDIT_PARAMETER_ALIASES,
            EDIT_TYPE_ALIASES,
        )

        # Simulate what DeepSeek might send
        deepseek_ops = [
            {
                "file_path": "/project/main.py",  # Uses file_path instead of path
                "content": "print('hello')",
                # Missing "type" field
            },
            {
                "old": "foo",  # Uses old instead of old_str
                "new": "bar",  # Uses new instead of new_str
                "type": "find_replace",  # Uses alias
            },
        ]

        normalized = normalize_edit_operations(deepseek_ops)

        # First op should have normalized path and inferred type
        assert "path" in normalized[0]
        assert normalized[0]["type"] == "create"

        # Second op should have normalized keys and type
        assert "old_str" in normalized[1]
        assert "new_str" in normalized[1]
        assert normalized[1]["type"] == "replace"

    def test_mode_aware_exploration_limits(self):
        """Test that mode multipliers affect all exploration limits."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Get defaults
        default_threshold = tracker._get_loop_threshold()
        default_reads = tracker.max_overlapping_reads
        default_searches = tracker.max_searches_per_prefix

        # Set PLAN mode multiplier
        tracker.set_mode_exploration_multiplier(2.5)

        # All limits should increase
        assert tracker._get_loop_threshold() >= default_threshold
        assert tracker.max_overlapping_reads >= default_reads
        assert tracker.max_searches_per_prefix >= default_searches

    def test_grounding_stopwords_prevent_false_positives(self):
        """Test that stopwords prevent false symbol_not_found issues."""
        from victor.agent.grounding_verifier import GROUNDING_STOPWORDS

        # These words appeared in actual error logs
        false_positive_words = [
            "that",
            "handles",
            "should",
            "has",
            "provides",
            "supports",
            "allows",
            "enables",
            "creates",
            "returns",
        ]

        for word in false_positive_words:
            assert (
                word in GROUNDING_STOPWORDS
            ), f"'{word}' should be filtered to prevent false positives"

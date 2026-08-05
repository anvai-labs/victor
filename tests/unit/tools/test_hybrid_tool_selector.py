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

"""Tests for HybridToolSelector and HybridSelectorConfig.

Covers HIGH-002: Unified Tool Selection Architecture - Release 3, Phase 8.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.base import ToolDefinition
from victor.tools.hybrid_tool_selector import HybridSelectorConfig, HybridToolSelector


class TestHybridSelectorConfig:
    """Tests for HybridSelectorConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridSelectorConfig()
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.min_semantic_tools == 3
        assert config.min_keyword_tools == 2
        assert config.max_total_tools == 15
        assert config.enable_rl is True
        assert config.rl_boost_weight == 0.15

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridSelectorConfig(
            semantic_weight=0.8,
            keyword_weight=0.2,
            min_semantic_tools=5,
            min_keyword_tools=3,
            max_total_tools=20,
            enable_rl=False,
            rl_boost_weight=0.1,
        )
        assert config.semantic_weight == 0.8
        assert config.keyword_weight == 0.2
        assert config.min_semantic_tools == 5
        assert config.min_keyword_tools == 3
        assert config.max_total_tools == 20
        assert config.enable_rl is False
        assert config.rl_boost_weight == 0.1

    def test_invalid_semantic_weight_above_1(self):
        """Test that semantic_weight > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="semantic_weight must be in"):
            HybridSelectorConfig(semantic_weight=1.5)

    def test_invalid_semantic_weight_below_0(self):
        """Test that semantic_weight < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="semantic_weight must be in"):
            HybridSelectorConfig(semantic_weight=-0.1)

    def test_invalid_keyword_weight_above_1(self):
        """Test that keyword_weight > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="keyword_weight must be in"):
            HybridSelectorConfig(keyword_weight=1.5)

    def test_invalid_keyword_weight_below_0(self):
        """Test that keyword_weight < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="keyword_weight must be in"):
            HybridSelectorConfig(keyword_weight=-0.1)

    def test_invalid_min_semantic_tools_negative(self):
        """Test that negative min_semantic_tools raises ValueError."""
        with pytest.raises(ValueError, match="min_semantic_tools must be >= 0"):
            HybridSelectorConfig(min_semantic_tools=-1)

    def test_invalid_min_keyword_tools_negative(self):
        """Test that negative min_keyword_tools raises ValueError."""
        with pytest.raises(ValueError, match="min_keyword_tools must be >= 0"):
            HybridSelectorConfig(min_keyword_tools=-1)

    def test_invalid_max_total_tools_zero(self):
        """Test that max_total_tools < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_total_tools must be >= 1"):
            HybridSelectorConfig(max_total_tools=0)

    def test_invalid_rl_boost_weight_above_0_5(self):
        """Test that rl_boost_weight > 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="rl_boost_weight must be in"):
            HybridSelectorConfig(rl_boost_weight=0.6)

    def test_invalid_rl_boost_weight_below_0(self):
        """Test that rl_boost_weight < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="rl_boost_weight must be in"):
            HybridSelectorConfig(rl_boost_weight=-0.1)


class TestHybridToolSelector:
    """Tests for HybridToolSelector functionality."""

    @pytest.fixture
    def mock_semantic_selector(self):
        """Create mock semantic selector."""
        selector = MagicMock()
        selector.select_tools = AsyncMock(return_value=[])
        selector.record_tool_execution = MagicMock()
        selector.close = AsyncMock()
        return selector

    @pytest.fixture
    def mock_keyword_selector(self):
        """Create mock keyword selector."""
        selector = MagicMock()
        selector.select_tools = AsyncMock(return_value=[])
        selector.record_tool_execution = MagicMock()
        selector.close = AsyncMock()
        return selector

    @pytest.fixture
    def tool_definitions(self):
        """Create sample tool definitions."""
        return [
            ToolDefinition(name="read_file", description="Read a file", parameters={}),
            ToolDefinition(name="write_file", description="Write a file", parameters={}),
            ToolDefinition(name="code_search", description="Search code", parameters={}),
            ToolDefinition(name="git_status", description="Git status", parameters={}),
            ToolDefinition(name="shell", description="Run shell command", parameters={}),
        ]

    @pytest.fixture
    def hybrid_selector(self, mock_semantic_selector, mock_keyword_selector):
        """Create HybridToolSelector with mocked components."""
        config = HybridSelectorConfig(enable_rl=False)  # Disable RL for most tests
        return HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

    def test_init_with_defaults(self, mock_semantic_selector, mock_keyword_selector):
        """Test initialization with default config."""
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
        )
        assert selector.semantic is mock_semantic_selector
        assert selector.keyword is mock_keyword_selector
        assert selector.config.semantic_weight == 0.7
        assert selector.config.enable_rl is True

    def test_init_with_custom_config(self, mock_semantic_selector, mock_keyword_selector):
        """Test initialization with custom config."""
        config = HybridSelectorConfig(
            semantic_weight=0.6,
            keyword_weight=0.4,
            enable_rl=False,
        )
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )
        assert selector.config.semantic_weight == 0.6
        assert selector.config.keyword_weight == 0.4
        assert selector.config.enable_rl is False

    @pytest.mark.asyncio
    async def test_select_tools_empty_results(
        self, hybrid_selector, mock_semantic_selector, mock_keyword_selector
    ):
        """Test select_tools when both selectors return empty."""
        mock_semantic_selector.select_tools.return_value = []
        mock_keyword_selector.select_tools.return_value = []

        context = MagicMock()
        context.task_type = "analysis"

        result = await hybrid_selector.select_tools("test prompt", context)

        assert result == []
        mock_semantic_selector.select_tools.assert_called_once()
        mock_keyword_selector.select_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_tools_blends_results(
        self,
        hybrid_selector,
        mock_semantic_selector,
        mock_keyword_selector,
        tool_definitions,
    ):
        """Test that select_tools blends semantic and keyword results."""
        # Semantic returns first 3 tools
        mock_semantic_selector.select_tools.return_value = tool_definitions[:3]
        # Keyword returns last 3 tools (with overlap on code_search)
        mock_keyword_selector.select_tools.return_value = tool_definitions[2:]

        context = MagicMock()
        context.task_type = "action"

        result = await hybrid_selector.select_tools("test prompt", context)

        # Should contain deduplicated tools from both sources
        result_names = [t.name for t in result]
        assert "read_file" in result_names
        assert "write_file" in result_names
        assert "code_search" in result_names
        assert "git_status" in result_names
        assert "shell" in result_names
        # No duplicates
        assert len(result_names) == len(set(result_names))

    @pytest.mark.asyncio
    async def test_select_tools_caps_to_max_total_tools(
        self, mock_semantic_selector, mock_keyword_selector
    ):
        """Test that select_tools respects max_total_tools."""
        # Create many tools
        many_tools = [
            ToolDefinition(name=f"tool_{i}", description=f"Tool {i}", parameters={})
            for i in range(20)
        ]
        mock_semantic_selector.select_tools.return_value = many_tools[:10]
        mock_keyword_selector.select_tools.return_value = many_tools[10:]

        config = HybridSelectorConfig(max_total_tools=5, enable_rl=False)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        context = MagicMock()
        result = await selector.select_tools("test prompt", context)

        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_select_tools_ensures_minimum_semantic(
        self, mock_semantic_selector, mock_keyword_selector, tool_definitions
    ):
        """Test that minimum semantic tools are included."""
        mock_semantic_selector.select_tools.return_value = tool_definitions[:3]
        mock_keyword_selector.select_tools.return_value = tool_definitions[3:5]

        config = HybridSelectorConfig(min_semantic_tools=3, enable_rl=False)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        context = MagicMock()
        result = await selector.select_tools("test prompt", context)

        # Should have at least 3 semantic tools
        semantic_names = {t.name for t in tool_definitions[:3]}
        result_names = {t.name for t in result}
        assert len(semantic_names & result_names) >= 3

    def test_get_supported_features(self, hybrid_selector):
        """Test get_supported_features returns all features enabled."""
        features = hybrid_selector.get_supported_features()

        assert features.supports_semantic_matching is True
        assert features.supports_context_awareness is True
        assert features.supports_cost_optimization is True
        assert features.supports_usage_learning is True
        assert features.supports_workflow_patterns is True
        assert features.requires_embeddings is True

    def test_record_tool_execution_delegates_to_both(
        self, hybrid_selector, mock_semantic_selector, mock_keyword_selector
    ):
        """Test that record_tool_execution delegates to both selectors."""
        context = {"task_type": "analysis"}

        hybrid_selector.record_tool_execution("read_file", True, context)

        mock_semantic_selector.record_tool_execution.assert_called_once_with(
            "read_file", True, context
        )
        mock_keyword_selector.record_tool_execution.assert_called_once_with(
            "read_file", True, context
        )

    @pytest.mark.asyncio
    async def test_close_delegates_to_both(
        self, hybrid_selector, mock_semantic_selector, mock_keyword_selector
    ):
        """Test that close delegates to both selectors."""
        await hybrid_selector.close()

        mock_semantic_selector.close.assert_called_once()
        mock_keyword_selector.close.assert_called_once()

    def test_ensure_minimum_tools_adds_semantic(self, hybrid_selector, tool_definitions):
        """Test _ensure_minimum_tools adds semantic tools when below minimum."""
        # Only keyword tools in blended (no semantic)
        blended = tool_definitions[3:5]  # git_status, shell
        semantic_tools = tool_definitions[:3]  # read_file, write_file, code_search
        keyword_tools = tool_definitions[3:5]  # git_status, shell

        result = hybrid_selector._ensure_minimum_tools(blended, semantic_tools, keyword_tools)

        # Should have added semantic tools to meet minimum of 3
        result_names = [t.name for t in result]
        # Original keyword tools should still be there
        assert "git_status" in result_names
        assert "shell" in result_names
        # Some semantic tools should be added
        semantic_added = sum(
            1 for name in result_names if name in ["read_file", "write_file", "code_search"]
        )
        assert semantic_added >= 1  # At least 1 semantic added

    def test_ensure_minimum_tools_adds_keyword(
        self, mock_semantic_selector, mock_keyword_selector, tool_definitions
    ):
        """Test _ensure_minimum_tools adds keyword tools when below minimum."""
        config = HybridSelectorConfig(min_keyword_tools=3, enable_rl=False)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        # Only semantic tools in blended (no keyword)
        blended = tool_definitions[:2]  # read_file, write_file
        semantic_tools = tool_definitions[:2]
        keyword_tools = tool_definitions[2:5]  # code_search, git_status, shell

        result = selector._ensure_minimum_tools(blended, semantic_tools, keyword_tools)

        # Should have added keyword tools to meet minimum of 3
        result_names = [t.name for t in result]
        keyword_added = sum(
            1 for name in result_names if name in ["code_search", "git_status", "shell"]
        )
        assert keyword_added >= 1  # At least some keyword added


class TestHybridToolSelectorRLIntegration:
    """Tests for HybridToolSelector RL integration."""

    @pytest.fixture
    def mock_semantic_selector(self):
        """Create mock semantic selector."""
        selector = MagicMock()
        selector.select_tools = AsyncMock(return_value=[])
        selector.record_tool_execution = MagicMock()
        selector.close = AsyncMock()
        return selector

    @pytest.fixture
    def mock_keyword_selector(self):
        """Create mock keyword selector."""
        selector = MagicMock()
        selector.select_tools = AsyncMock(return_value=[])
        selector.record_tool_execution = MagicMock()
        selector.close = AsyncMock()
        return selector

    @pytest.fixture
    def tool_definitions(self):
        """Create sample tool definitions."""
        return [
            ToolDefinition(name="read_file", description="Read a file", parameters={}),
            ToolDefinition(name="write_file", description="Write a file", parameters={}),
            ToolDefinition(name="code_search", description="Search code", parameters={}),
            ToolDefinition(name="git_status", description="Git status", parameters={}),
            ToolDefinition(name="shell", description="Run shell command", parameters={}),
        ]

    def test_rl_disabled_skips_learner(self, mock_semantic_selector, mock_keyword_selector):
        """Test that RL disabled skips learner initialization."""
        config = HybridSelectorConfig(enable_rl=False)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        learner = selector._get_rl_learner()
        assert learner is None

    @patch("victor.framework.rl.coordinator.get_rl_coordinator")
    def test_rl_enabled_gets_learner(
        self, mock_get_coordinator, mock_semantic_selector, mock_keyword_selector
    ):
        """Test that RL enabled gets learner from coordinator."""
        mock_learner = MagicMock()
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner
        mock_get_coordinator.return_value = mock_coordinator

        config = HybridSelectorConfig(enable_rl=True)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        learner = selector._get_rl_learner()

        assert learner is mock_learner
        mock_coordinator.get_learner.assert_called_once_with("tool_selector")

    @patch("victor.framework.rl.coordinator.get_rl_coordinator")
    def test_rl_learner_cached(
        self, mock_get_coordinator, mock_semantic_selector, mock_keyword_selector
    ):
        """Test that RL learner is cached after first access."""
        mock_learner = MagicMock()
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner
        mock_get_coordinator.return_value = mock_coordinator

        config = HybridSelectorConfig(enable_rl=True)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        # First access
        learner1 = selector._get_rl_learner()
        # Second access
        learner2 = selector._get_rl_learner()

        assert learner1 is learner2
        # Only called once due to caching
        assert mock_coordinator.get_learner.call_count == 1

    def test_apply_rl_boost_empty_tools(self, mock_semantic_selector, mock_keyword_selector):
        """Test _apply_rl_boost with empty tools list."""
        config = HybridSelectorConfig(enable_rl=False)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        result = selector._apply_rl_boost([], "analysis")
        assert result == []

    def test_apply_rl_boost_no_learner(
        self, mock_semantic_selector, mock_keyword_selector, tool_definitions
    ):
        """Test _apply_rl_boost when learner is None."""
        config = HybridSelectorConfig(enable_rl=False)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        # Original order should be preserved
        result = selector._apply_rl_boost(tool_definitions, "analysis")
        assert result == tool_definitions

    @patch("victor.framework.rl.coordinator.get_rl_coordinator")
    def test_apply_rl_boost_exploration_mode(
        self,
        mock_get_coordinator,
        mock_semantic_selector,
        mock_keyword_selector,
        tool_definitions,
    ):
        """Test _apply_rl_boost in exploration mode shuffles top 3."""
        mock_learner = MagicMock()
        mock_learner.should_explore.return_value = True
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner
        mock_get_coordinator.return_value = mock_coordinator

        config = HybridSelectorConfig(enable_rl=True)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        # Force learner initialization
        selector._get_rl_learner()

        # Call multiple times to verify shuffle happens (probabilistic)
        results = []
        for _ in range(10):
            result = selector._apply_rl_boost(tool_definitions.copy(), "analysis")
            results.append(tuple(t.name for t in result[:3]))

        # All results should contain same tools (just possibly shuffled)
        expected_names = {
            tool_definitions[0].name,
            tool_definitions[1].name,
            tool_definitions[2].name,
        }
        for r in results:
            assert set(r) == expected_names

    @patch("victor.framework.rl.coordinator.get_rl_coordinator")
    def test_apply_rl_boost_exploitation_mode(
        self,
        mock_get_coordinator,
        mock_semantic_selector,
        mock_keyword_selector,
        tool_definitions,
    ):
        """Test _apply_rl_boost in exploitation mode uses Q-values."""
        mock_learner = MagicMock()
        mock_learner.should_explore.return_value = False
        # Return rankings that would boost 'shell' to top
        mock_learner.get_tool_rankings.return_value = [
            ("shell", 0.9, 0.8),  # High Q-value for shell
            ("read_file", 0.5, 0.5),
            ("write_file", 0.4, 0.5),
            ("code_search", 0.3, 0.5),
            ("git_status", 0.2, 0.5),
        ]
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner
        mock_get_coordinator.return_value = mock_coordinator

        config = HybridSelectorConfig(enable_rl=True, rl_boost_weight=0.3)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        # Force learner initialization
        selector._get_rl_learner()

        result = selector._apply_rl_boost(tool_definitions, "analysis")

        # Shell should be boosted higher due to high Q-value
        result_names = [t.name for t in result]
        # The exact position depends on the math, but shell should be higher than original (position 4)
        assert "shell" in result_names[:4]  # At least in top 4

    @patch("victor.framework.rl.coordinator.get_rl_coordinator")
    def test_record_rl_outcome(
        self, mock_get_coordinator, mock_semantic_selector, mock_keyword_selector
    ):
        """Test _record_rl_outcome records to coordinator."""
        mock_learner = MagicMock()
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner
        mock_get_coordinator.return_value = mock_coordinator

        config = HybridSelectorConfig(enable_rl=True)
        selector = HybridToolSelector(
            semantic_selector=mock_semantic_selector,
            keyword_selector=mock_keyword_selector,
            config=config,
        )

        context = {
            "task_type": "action",
            "task_completed": True,
            "grounding_score": 0.8,
        }

        selector._record_rl_outcome("read_file", True, context)

        # Should have called record_outcome on coordinator
        mock_coordinator.record_outcome.assert_called_once()
        call_args = mock_coordinator.record_outcome.call_args
        assert call_args[0][0] == "tool_selector"  # learner_name
        assert call_args[1]["vertical"] == "coding"

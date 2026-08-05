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

"""Unit tests for workflow node caching."""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from victor.workflows import (
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContext,
    WorkflowDefinition,
    WorkflowBuilder,
    WorkflowExecutor,
    TransformNode,
    ConditionNode,
    AgentNode,
    WorkflowCacheConfig,
    WorkflowCache,
    WorkflowCacheManager,
    get_workflow_cache_manager,
    configure_workflow_cache,
)


class TestWorkflowCacheConfig:
    """Test WorkflowCacheConfig class."""

    def test_default_config(self):
        """Default config has expected values."""
        config = WorkflowCacheConfig()

        assert config.enabled is False
        assert config.ttl_seconds == 3600
        assert config.max_entries == 500
        assert "transform" in config.cacheable_node_types
        assert "condition" in config.cacheable_node_types

    def test_custom_config(self):
        """Custom config values are applied."""
        config = WorkflowCacheConfig(
            enabled=True,
            ttl_seconds=7200,
            max_entries=1000,
            cacheable_node_types={"transform"},
        )

        assert config.enabled is True
        assert config.ttl_seconds == 7200
        assert config.max_entries == 1000
        assert config.cacheable_node_types == {"transform"}

    def test_excluded_context_keys(self):
        """Excluded context keys have defaults."""
        config = WorkflowCacheConfig()

        assert "_internal" in config.excluded_context_keys
        assert "_debug" in config.excluded_context_keys


class TestWorkflowCache:
    """Test WorkflowCache class."""

    def test_cache_disabled_by_default(self):
        """Cache is disabled by default."""
        cache = WorkflowCache()

        assert cache._cache is None
        assert cache.config.enabled is False

    def test_cache_enabled_with_config(self):
        """Cache is enabled with explicit config."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        assert cache._cache is not None
        assert cache.config.enabled is True

    def test_is_cacheable_transform_node(self):
        """TransformNode is cacheable."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        assert cache.is_cacheable(node) is True

    def test_is_cacheable_condition_node(self):
        """ConditionNode is cacheable."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = ConditionNode(id="condition", name="Condition")
        assert cache.is_cacheable(node) is True

    def test_is_not_cacheable_agent_node(self):
        """AgentNode is not cacheable."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = AgentNode(id="agent", name="Agent")
        assert cache.is_cacheable(node) is False

    def test_is_not_cacheable_when_disabled(self):
        """Nothing is cacheable when cache is disabled."""
        cache = WorkflowCache()

        node = TransformNode(id="transform", name="Transform")
        assert cache.is_cacheable(node) is False

    def test_get_returns_none_when_disabled(self):
        """get() returns None when cache is disabled."""
        cache = WorkflowCache()
        node = TransformNode(id="transform", name="Transform")

        result = cache.get(node, {"key": "value"})

        assert result is None

    def test_set_returns_false_when_disabled(self):
        """set() returns False when cache is disabled."""
        cache = WorkflowCache()
        node = TransformNode(id="transform", name="Transform")
        node_result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"processed": True},
        )

        success = cache.set(node, {"key": "value"}, node_result)

        assert success is False

    def test_set_and_get_transform_node(self):
        """Can set and get cached transform node result."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context = {"count": 5}
        node_result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"count": 10},
        )

        # Set the result
        success = cache.set(node, context, node_result)
        assert success is True

        # Get the result
        cached = cache.get(node, context)
        assert cached is not None
        assert cached.node_id == "transform"
        assert cached.output == {"count": 10}

    def test_get_miss_for_different_context(self):
        """Cache miss when context differs."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context1 = {"count": 5}
        context2 = {"count": 10}  # Different value

        node_result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"count": 10},
        )

        # Set with context1
        cache.set(node, context1, node_result)

        # Get with context2 should miss
        cached = cache.get(node, context2)
        assert cached is None

    def test_does_not_cache_failed_results(self):
        """Failed results are not cached."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context = {"count": 5}
        failed_result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.FAILED,
            error="Transform failed",
        )

        # Try to set failed result
        success = cache.set(node, context, failed_result)
        assert success is False

        # Should not be cached
        cached = cache.get(node, context)
        assert cached is None

    def test_does_not_cache_non_cacheable_nodes(self):
        """Non-cacheable nodes are not cached."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = AgentNode(id="agent", name="Agent")
        context = {"task": "analyze"}
        node_result = NodeResult(
            node_id="agent",
            status=ExecutorNodeStatus.COMPLETED,
            output="Analysis complete",
        )

        # Try to set
        success = cache.set(node, context, node_result)
        assert success is False

        # Should not be cached
        cached = cache.get(node, context)
        assert cached is None

    def test_invalidate_node(self):
        """Can invalidate cache entries for a specific node."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context = {"count": 5}
        node_result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"count": 10},
        )

        # Set the result
        cache.set(node, context, node_result)

        # Invalidate
        count = cache.invalidate("transform")
        assert count == 1

        # Should be gone
        cached = cache.get(node, context)
        assert cached is None

    def test_clear(self):
        """Can clear all cache entries."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node1 = TransformNode(id="transform1", name="Transform 1")
        node2 = TransformNode(id="transform2", name="Transform 2")

        cache.set(
            node1,
            {"a": 1},
            NodeResult(
                node_id="transform1",
                status=ExecutorNodeStatus.COMPLETED,
                output={"a": 2},
            ),
        )
        cache.set(
            node2,
            {"b": 2},
            NodeResult(
                node_id="transform2",
                status=ExecutorNodeStatus.COMPLETED,
                output={"b": 4},
            ),
        )

        # Clear all
        count = cache.clear()
        assert count == 2

        # Both should be gone
        assert cache.get(node1, {"a": 1}) is None
        assert cache.get(node2, {"b": 2}) is None

    def test_get_stats(self):
        """Can get cache statistics."""
        config = WorkflowCacheConfig(enabled=True, max_entries=100)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context = {"count": 5}
        node_result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"count": 10},
        )

        # Generate some stats
        cache.get(node, context)  # Miss
        cache.set(node, context, node_result)  # Set
        cache.get(node, context)  # Hit

        stats = cache.get_stats()

        assert stats["enabled"] is True
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["current_size"] == 1
        assert stats["max_size"] == 100

    def test_cache_key_uses_node_id(self):
        """Cache key includes node ID."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        # Two nodes with same context should have different cache keys
        node1 = TransformNode(id="transform1", name="Transform 1")
        node2 = TransformNode(id="transform2", name="Transform 2")
        context = {"count": 5}

        result1 = NodeResult(
            node_id="transform1",
            status=ExecutorNodeStatus.COMPLETED,
            output={"result": 1},
        )
        result2 = NodeResult(
            node_id="transform2",
            status=ExecutorNodeStatus.COMPLETED,
            output={"result": 2},
        )

        cache.set(node1, context, result1)
        cache.set(node2, context, result2)

        # Should get different results
        cached1 = cache.get(node1, context)
        cached2 = cache.get(node2, context)

        assert cached1.output == {"result": 1}
        assert cached2.output == {"result": 2}

    def test_excludes_private_context_keys(self):
        """Private context keys are excluded from cache key."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context1 = {"count": 5, "_internal": "value1"}
        context2 = {"count": 5, "_internal": "value2"}  # Different private value

        result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"count": 10},
        )

        # Set with context1
        cache.set(node, context1, result)

        # Get with context2 should hit (private keys ignored)
        cached = cache.get(node, context2)
        assert cached is not None

    def test_handles_large_context_values(self):
        """Large context values are hashed for cache key."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        large_value = "x" * 2000  # Large string
        context = {"data": large_value}

        result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"processed": True},
        )

        # Should not raise
        cache.set(node, context, result)
        cached = cache.get(node, context)
        assert cached is not None


class TestWorkflowCacheManager:
    """Test WorkflowCacheManager class."""

    def test_get_cache_creates_new(self):
        """get_cache creates new cache for unknown workflow."""
        manager = WorkflowCacheManager()

        cache = manager.get_cache("my_workflow")

        assert cache is not None
        assert isinstance(cache, WorkflowCache)

    def test_get_cache_returns_same(self):
        """get_cache returns same cache for same workflow."""
        manager = WorkflowCacheManager()

        cache1 = manager.get_cache("my_workflow")
        cache2 = manager.get_cache("my_workflow")

        assert cache1 is cache2

    def test_get_cache_with_config(self):
        """get_cache applies custom config."""
        manager = WorkflowCacheManager()
        config = WorkflowCacheConfig(enabled=True, ttl_seconds=7200)

        cache = manager.get_cache("my_workflow", config)

        assert cache.config.enabled is True
        assert cache.config.ttl_seconds == 7200

    def test_clear_workflow(self):
        """clear_workflow clears specific workflow cache."""
        config = WorkflowCacheConfig(enabled=True)
        manager = WorkflowCacheManager(config)

        cache = manager.get_cache("my_workflow")
        node = TransformNode(id="transform", name="Transform")
        cache.set(
            node,
            {"x": 1},
            NodeResult(
                node_id="transform",
                status=ExecutorNodeStatus.COMPLETED,
                output={"x": 2},
            ),
        )

        count = manager.clear_workflow("my_workflow")

        assert count == 1

    def test_clear_all(self):
        """clear_all clears all workflow caches."""
        config = WorkflowCacheConfig(enabled=True)
        manager = WorkflowCacheManager(config)

        cache1 = manager.get_cache("workflow1")
        cache2 = manager.get_cache("workflow2")

        node = TransformNode(id="transform", name="Transform")
        cache1.set(
            node,
            {"x": 1},
            NodeResult(
                node_id="transform",
                status=ExecutorNodeStatus.COMPLETED,
                output={"x": 2},
            ),
        )
        cache2.set(
            node,
            {"y": 1},
            NodeResult(
                node_id="transform",
                status=ExecutorNodeStatus.COMPLETED,
                output={"y": 2},
            ),
        )

        count = manager.clear_all()

        assert count == 2

    def test_get_all_stats(self):
        """get_all_stats returns stats for all workflows."""
        config = WorkflowCacheConfig(enabled=True)
        manager = WorkflowCacheManager(config)

        manager.get_cache("workflow1")
        manager.get_cache("workflow2")

        stats = manager.get_all_stats()

        assert "workflow1" in stats
        assert "workflow2" in stats
        assert stats["workflow1"]["enabled"] is True


class TestWorkflowExecutorWithCache:
    """Test WorkflowExecutor with caching enabled."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        return MagicMock()

    @pytest.fixture
    def mock_sub_agent_result(self):
        """Create a mock SubAgent result."""
        result = MagicMock()
        result.success = True
        result.summary = "Task completed"
        result.error = None
        result.tool_calls_used = 5
        return result

    def test_executor_with_cache_config(self, mock_orchestrator):
        """Executor can be initialized with cache config."""
        config = WorkflowCacheConfig(enabled=True)
        executor = WorkflowExecutor(mock_orchestrator, cache_config=config)

        assert executor.cache is not None
        assert executor.cache.config.enabled is True

    def test_executor_with_cache_instance(self, mock_orchestrator):
        """Executor can be initialized with cache instance."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)
        executor = WorkflowExecutor(mock_orchestrator, cache=cache)

        assert executor.cache is cache

    def test_get_cache_stats(self, mock_orchestrator):
        """Can get cache stats from executor."""
        config = WorkflowCacheConfig(enabled=True)
        executor = WorkflowExecutor(mock_orchestrator, cache_config=config)

        stats = executor.get_cache_stats()

        assert stats["enabled"] is True

    def test_get_cache_stats_when_disabled(self, mock_orchestrator):
        """get_cache_stats returns disabled when no cache."""
        executor = WorkflowExecutor(mock_orchestrator)

        stats = executor.get_cache_stats()

        assert stats["enabled"] is False

    @pytest.mark.asyncio
    async def test_execute_caches_transform_node(self, mock_orchestrator, mock_sub_agent_result):
        """Transform node results are cached."""
        config = WorkflowCacheConfig(enabled=True)
        executor = WorkflowExecutor(mock_orchestrator, cache_config=config)

        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = (
            WorkflowBuilder("test")
            .add_transform(
                "transform",
                lambda ctx: {"doubled": ctx.get("count", 0) * 2},
                next_nodes=["agent"],
            )
            .add_agent("agent", "executor", "Do something")
            .build()
        )

        # Execute first time
        await executor.execute(workflow, {"count": 5})

        # Check cache stats
        stats = executor.get_cache_stats()
        assert stats["sets"] >= 1  # Transform node was cached

    @pytest.mark.asyncio
    async def test_execute_uses_cached_transform_result(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """Transform node uses cached result on re-execution with same context."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        # Manually test cache behavior for transform node
        node = TransformNode(
            id="transform",
            name="Transform",
            transform=lambda ctx: {"doubled": ctx.get("count", 0) * 2},
        )
        context = {"count": 5}
        result = NodeResult(
            node_id="transform",
            status=ExecutorNodeStatus.COMPLETED,
            output={"doubled": 10},
        )

        # First call - should miss, then set
        cached = cache.get(node, context)
        assert cached is None  # Miss

        cache.set(node, context, result)

        # Second call - should hit
        cached = cache.get(node, context)
        assert cached is not None  # Hit
        assert cached.output == {"doubled": 10}

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_execute_does_not_cache_agent_nodes(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """Agent node results are not cached."""
        config = WorkflowCacheConfig(enabled=True)
        executor = WorkflowExecutor(mock_orchestrator, cache_config=config)

        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowBuilder("test").add_agent("agent", "executor", "Do something").build()

        # Execute twice
        await executor.execute(workflow, {"task": "analyze"})
        await executor.execute(workflow, {"task": "analyze"})

        # Agent should have been called twice (no caching)
        assert mock_sub_agents.spawn.call_count == 2

        # Cache stats should show skipped
        stats = executor.get_cache_stats()
        assert stats["skipped_non_cacheable"] >= 2

    @pytest.mark.asyncio
    async def test_execute_caches_condition_node(self, mock_orchestrator, mock_sub_agent_result):
        """Condition node results are cached (test cache directly)."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        # Manually test cache behavior for condition node
        node = ConditionNode(
            id="condition",
            name="Condition",
            condition=lambda ctx: "branch_a" if ctx.get("value", 0) > 5 else "branch_b",
            branches={"branch_a": "agent_a", "branch_b": "agent_b"},
        )

        # Check node is cacheable
        assert cache.is_cacheable(node) is True

        context = {"value": 10}
        result = NodeResult(
            node_id="condition",
            status=ExecutorNodeStatus.COMPLETED,
            output={"branch": "branch_a", "next_node": "agent_a"},
        )

        # First call - should miss
        cached = cache.get(node, context)
        assert cached is None

        # Set result
        cache.set(node, context, result)

        # Second call - should hit
        cached = cache.get(node, context)
        assert cached is not None
        assert cached.output == {"branch": "branch_a", "next_node": "agent_a"}

        stats = cache.get_stats()
        assert stats["hits"] == 1


class TestGlobalCacheManager:
    """Test global cache manager functions."""

    def test_get_workflow_cache_manager(self):
        """get_workflow_cache_manager returns global instance."""
        manager1 = get_workflow_cache_manager()
        manager2 = get_workflow_cache_manager()

        assert manager1 is manager2

    def test_configure_workflow_cache(self):
        """configure_workflow_cache sets global config."""
        config = WorkflowCacheConfig(enabled=True, ttl_seconds=9999)

        configure_workflow_cache(config)

        manager = get_workflow_cache_manager()
        cache = manager.get_cache("test_workflow")

        # Note: The global manager may have been replaced
        # Check the cache inherits the config
        assert cache.config.enabled is True


class TestCacheKeyGeneration:
    """Test cache key generation edge cases."""

    def test_different_node_ids_different_keys(self):
        """Different node IDs produce different cache keys."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node1 = TransformNode(id="transform1", name="Transform 1")
        node2 = TransformNode(id="transform2", name="Transform 2")
        context = {"x": 1}

        key1 = cache._make_cache_key(node1, context)
        key2 = cache._make_cache_key(node2, context)

        assert key1 != key2

    def test_different_contexts_different_keys(self):
        """Different contexts produce different cache keys."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context1 = {"x": 1}
        context2 = {"x": 2}

        key1 = cache._make_cache_key(node, context1)
        key2 = cache._make_cache_key(node, context2)

        assert key1 != key2

    def test_same_inputs_same_key(self):
        """Same inputs produce same cache key."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context = {"x": 1}

        key1 = cache._make_cache_key(node, context)
        key2 = cache._make_cache_key(node, context)

        assert key1 == key2

    def test_context_order_does_not_affect_key(self):
        """Context key order does not affect cache key."""
        config = WorkflowCacheConfig(enabled=True)
        cache = WorkflowCache(config)

        node = TransformNode(id="transform", name="Transform")
        context1 = {"a": 1, "b": 2}
        context2 = {"b": 2, "a": 1}  # Same data, different order

        key1 = cache._make_cache_key(node, context1)
        key2 = cache._make_cache_key(node, context2)

        assert key1 == key2

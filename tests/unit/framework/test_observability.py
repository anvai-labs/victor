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

"""Unit tests for observability manager."""

import pytest

from victor.framework.observability import (
    ObservabilityConfig,
    ObservabilityManager,
)


class TestObservabilityManager:
    """Tests for ObservabilityManager."""

    def setup_method(self):
        """Reset singleton before each test."""
        ObservabilityManager.reset()

    def test_get_instance_returns_singleton(self):
        """Test that get_instance returns singleton."""
        manager1 = ObservabilityManager.get_instance()
        manager2 = ObservabilityManager.get_instance()
        assert manager1 is manager2

    def test_reset_singleton(self):
        """Test resetting singleton."""
        manager1 = ObservabilityManager.get_instance()
        ObservabilityManager.reset()
        manager2 = ObservabilityManager.get_instance()
        assert manager1 is not manager2

    def test_manager_initialization(self):
        """Test manager initialization."""
        config = ObservabilityConfig(max_history_size=100)
        manager = ObservabilityManager.get_instance(config=config)
        assert manager._config.max_history_size == 100

    def test_list_sources(self):
        """Test listing sources."""
        manager = ObservabilityManager.get_instance()
        sources = manager.list_sources()
        assert isinstance(sources, list)

    def test_get_stats(self):
        """Test getting manager statistics."""
        manager = ObservabilityManager.get_instance()
        stats = manager.get_stats()
        assert "collection_count" in stats
        assert "registered_sources" in stats

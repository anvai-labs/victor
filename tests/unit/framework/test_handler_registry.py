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

"""Unit tests for HandlerRegistry."""

import pytest
from unittest.mock import MagicMock

from victor.framework.handler_registry import (
    HandlerSpec,
    HandlerRegistry,
    get_handler_registry,
    register_vertical_handlers,
    register_global_handler,
)


class TestHandlerRegistryInit:
    """Tests for HandlerRegistry initialization."""

    def test_init_default(self):
        """Initialize with default parameters."""
        registry = HandlerRegistry()

        assert registry._vertical_handlers == {}
        assert registry._global_handlers == {}
        assert registry._specs == {}

    def test_get_instance_singleton(self):
        """get_instance returns singleton."""
        registry1 = HandlerRegistry.get_instance()
        registry2 = HandlerRegistry.get_instance()

        assert registry1 is registry2


class TestHandlerRegistration:
    """Tests for handler registration."""

    def setup_method(self):
        """Reset registry before each test."""
        # Clear singleton for testing
        HandlerRegistry._instance = None

    def teardown_method(self):
        """Reset registry after each test."""
        HandlerRegistry._instance = None

    def test_register_vertical(self):
        """Register handlers from a vertical."""
        registry = HandlerRegistry()

        # Mock handler
        handler = MagicMock()
        handler.__class__.__name__ = "MockHandler"

        registry.register_vertical(
            "test",
            {"handler1": handler},
            category="test_category",
            description="Test handlers",
        )

        # Check handler is registered
        assert registry.get_handler("test", "handler1") is handler

        # Check spec
        spec = registry.get_spec("test", "handler1")
        assert spec.name == "handler1"
        assert spec.category == "test_category"

    def test_register_global_handler(self):
        """Register a global handler."""
        registry = HandlerRegistry()

        handler = MagicMock()

        registry.register_global("global_handler", handler, category="global")

        # Can get from any vertical
        assert registry.get_handler("any_vertical", "global_handler") is handler
        assert registry.get_handler("", "global_handler") is handler

    def test_register_vertical_overwrites(self):
        """Registering same vertical twice overwrites."""
        registry = HandlerRegistry()

        handler1 = MagicMock()
        handler2 = MagicMock()

        registry.register_vertical("test", {"h1": handler1})
        assert registry.get_handler("test", "h1") is handler1

        registry.register_vertical("test", {"h1": handler2})
        assert registry.get_handler("test", "h1") is handler2


class TestHandlerRetrieval:
    """Tests for handler retrieval."""

    def setup_method(self):
        """Reset and setup registry before each test."""
        HandlerRegistry._instance = None
        self.registry = HandlerRegistry()

        # Setup test handlers
        self.handler1 = MagicMock()
        self.handler2 = MagicMock()

        self.registry.register_vertical(
            "test",
            {
                "h1": self.handler1,
                "h2": self.handler2,
            },
        )

        self.global_handler = MagicMock()
        self.registry.register_global("gh", self.global_handler)

    def teardown_method(self):
        """Reset registry after each test."""
        HandlerRegistry._instance = None

    def test_get_handler_from_vertical(self):
        """Get handler from its vertical."""
        handler = self.registry.get_handler("test", "h1")
        assert handler is self.handler1

    def test_get_handler_falls_back_to_global(self):
        """Get handler falls back to global if not in vertical."""
        handler = self.registry.get_handler("other_vertical", "gh")
        assert handler is self.global_handler

    def test_get_handler_not_found(self):
        """Get handler returns None if not found."""
        handler = self.registry.get_handler("test", "nonexistent")
        assert handler is None

    def test_get_vertical_handlers(self):
        """Get all handlers for a vertical."""
        handlers = self.registry.get_vertical_handlers("test")

        assert len(handlers) == 2
        assert "h1" in handlers
        assert "h2" in handlers


class TestHandlerListing:
    """Tests for handler listing."""

    def setup_method(self):
        """Reset registry before each test."""
        HandlerRegistry._instance = None
        self.registry = HandlerRegistry()

        self.registry.register_vertical(
            "coding",
            {
                "code_validation": MagicMock(),
                "test_runner": MagicMock(),
            },
        )
        self.registry.register_vertical(
            "research",
            {
                "web_scraper": MagicMock(),
            },
        )

    def teardown_method(self):
        """Reset registry after each test."""
        HandlerRegistry._instance = None

    def test_list_all_handlers(self):
        """List all handlers across all verticals."""
        handlers = self.registry.list_handlers()

        assert "coding" in handlers
        assert set(handlers["coding"]) == {"code_validation", "test_runner"}
        assert "research" in handlers
        assert set(handlers["research"]) == {"web_scraper"}

    def test_list_filtered_by_vertical(self):
        """List handlers filtered by vertical."""
        handlers = self.registry.list_handlers("coding")

        assert "coding" in handlers
        assert "research" not in handlers


class TestHandlerSpecs:
    """Tests for handler specifications."""

    def setup_method(self):
        """Reset registry before each test."""
        HandlerRegistry._instance = None

    def teardown_method(self):
        """Reset registry after each test."""
        HandlerRegistry._instance = None

    def test_get_spec(self):
        """Get handler specification."""
        registry = HandlerRegistry()
        handler = MagicMock()
        handler.__class__.__name__ = "TestHandler"

        registry.register_vertical("test", {"h": handler}, category="validation")

        spec = registry.get_spec("test", "h")
        assert spec.name == "h"
        assert spec.category == "validation"
        assert spec.handler_class.__name__ == "TestHandler"

    def test_list_specs_by_category(self):
        """List specs filtered by category."""
        registry = HandlerRegistry()

        handler1 = MagicMock()
        handler1.__class__.__name__ = "ValidationHandler"
        handler2 = MagicMock()
        handler2.__class__.__name__ = "ExecutionHandler"

        registry.register_vertical("test", {"h1": handler1}, category="validation")
        registry.register_vertical("test", {"h2": handler2}, category="execution")

        specs = registry.list_specs(category="validation")
        assert len(specs) == 1
        assert specs[0].name == "h1"
        assert specs[0].category == "validation"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Reset registry before each test."""
        HandlerRegistry._instance = None

    def teardown_method(self):
        """Reset registry after each test."""
        HandlerRegistry._instance = None

    def test_get_handler_registry(self):
        """get_handler_registry returns singleton."""
        registry1 = get_handler_registry()
        registry2 = get_handler_registry()

        assert registry1 is registry2

    def test_register_vertical_handlers(self):
        """register_vertical_handlers convenience function."""
        handler = MagicMock()

        register_vertical_handlers("test", {"h": handler}, category="test")

        registry = get_handler_registry()
        assert registry.get_handler("test", "h") is handler

    def test_register_global_handler(self):
        """register_global_handler convenience function."""
        handler = MagicMock()

        register_global_handler("gh", handler, category="global")

        registry = get_handler_registry()
        assert registry.get_handler("any", "gh") is handler


class TestHandlerSpecValidation:
    """Tests for HandlerSpec validation."""

    def test_spec_validation_empty_name(self):
        """HandlerSpec with empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            HandlerSpec(
                name="",
                description="desc",
                category="cat",
                handler_class=str,
            )

    def test_spec_validation_empty_category(self):
        """HandlerSpec with empty category raises error."""
        with pytest.raises(ValueError, match="category cannot be empty"):
            HandlerSpec(
                name="test",
                description="desc",
                category="",
                handler_class=str,
            )

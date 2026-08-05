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

"""Tests for ToolCategoryRegistry (OCP-compliant tool category system)."""

import pytest

from victor.framework.tools import (
    ToolCategory,
    ToolCategoryRegistry,
    ToolSet,
    get_category_registry,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton before each test."""
    ToolCategoryRegistry.reset_instance()
    yield
    ToolCategoryRegistry.reset_instance()


class TestToolCategoryRegistrySingleton:
    """Tests for registry singleton behavior."""

    def test_get_instance_returns_singleton(self):
        """Should return same instance on repeated calls."""
        instance1 = ToolCategoryRegistry.get_instance()
        instance2 = ToolCategoryRegistry.get_instance()
        assert instance1 is instance2

    def test_reset_instance_creates_new(self):
        """Should create new instance after reset."""
        instance1 = ToolCategoryRegistry.get_instance()
        ToolCategoryRegistry.reset_instance()
        instance2 = ToolCategoryRegistry.get_instance()
        assert instance1 is not instance2

    def test_convenience_function(self):
        """get_category_registry should return singleton."""
        registry1 = get_category_registry()
        registry2 = ToolCategoryRegistry.get_instance()
        assert registry1 is registry2


class TestBuiltinCategories:
    """Tests for built-in category resolution."""

    def test_get_core_tools(self):
        """Should return built-in core tools."""
        registry = get_category_registry()
        tools = registry.get_tools("core")
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools

    def test_get_filesystem_tools(self):
        """Should return built-in filesystem tools."""
        registry = get_category_registry()
        tools = registry.get_tools("filesystem")
        assert "ls" in tools
        assert "glob" in tools

    def test_get_git_tools(self):
        """Should return built-in git tools."""
        registry = get_category_registry()
        tools = registry.get_tools("git")
        assert "git" in tools
        assert "git_status" in tools

    def test_is_builtin_category(self):
        """Should identify built-in categories."""
        registry = get_category_registry()
        assert registry.is_builtin_category("core") is True
        assert registry.is_builtin_category("git") is True
        assert registry.is_builtin_category("custom_rag") is False


class TestCustomCategoryRegistration:
    """Tests for custom category registration."""

    def test_register_new_category(self):
        """Should register custom category."""
        registry = get_category_registry()
        registry.register_category("rag", {"rag_search", "rag_query"})

        tools = registry.get_tools("rag")
        assert tools == {"rag_search", "rag_query"}

    def test_custom_category_in_all_categories(self):
        """Custom category should appear in get_all_categories."""
        registry = get_category_registry()
        registry.register_category("my_vertical", {"tool1", "tool2"})

        categories = registry.get_all_categories()
        assert "my_vertical" in categories

    def test_cannot_register_builtin_name(self):
        """Should reject registration with built-in category name."""
        registry = get_category_registry()
        with pytest.raises(ValueError, match="conflicts with built-in"):
            registry.register_category("core", {"my_tool"})


class TestCategoryExtension:
    """Tests for extending existing categories."""

    def test_extend_builtin_category(self):
        """Should extend built-in category with new tools."""
        registry = get_category_registry()

        # Get original tools
        original = registry.get_tools("search")
        assert "semantic_rag_search" not in original

        # Extend
        registry.extend_category("search", {"semantic_rag_search"})

        # Check extension
        extended = registry.get_tools("search")
        assert "semantic_rag_search" in extended
        # Original tools should still be present
        assert "grep" in extended

    def test_extend_custom_category(self):
        """Should extend custom category."""
        registry = get_category_registry()

        # Register custom category
        registry.register_category("rag", {"rag_search"})

        # Extend it
        registry.extend_category("rag", {"rag_query", "rag_ingest"})

        tools = registry.get_tools("rag")
        assert tools == {"rag_search", "rag_query", "rag_ingest"}

    def test_multiple_extensions(self):
        """Should accumulate multiple extensions."""
        registry = get_category_registry()

        registry.extend_category("testing", {"custom_test1"})
        registry.extend_category("testing", {"custom_test2"})

        tools = registry.get_tools("testing")
        assert "custom_test1" in tools
        assert "custom_test2" in tools


class TestCategoryLookup:
    """Tests for category lookup behavior."""

    def test_unknown_category_returns_empty(self):
        """Unknown category should return empty set."""
        registry = get_category_registry()
        tools = registry.get_tools("nonexistent_category")
        assert tools == set()

    def test_case_insensitive_lookup(self):
        """Lookup should be case-insensitive."""
        registry = get_category_registry()

        tools_lower = registry.get_tools("core")
        tools_upper = registry.get_tools("CORE")
        tools_mixed = registry.get_tools("Core")

        assert tools_lower == tools_upper == tools_mixed

    def test_get_all_categories_includes_builtins(self):
        """get_all_categories should include all built-in categories."""
        registry = get_category_registry()
        categories = registry.get_all_categories()

        for cat in ToolCategory:
            assert cat.value in categories


class TestToolSetIntegration:
    """Tests for ToolSet integration with registry."""

    def test_toolset_uses_registry(self):
        """ToolSet should resolve categories through registry."""
        # Extend a category
        registry = get_category_registry()
        registry.extend_category("core", {"my_custom_tool"})

        # Create ToolSet with that category
        toolset = ToolSet(categories={"core"})
        tools = toolset.get_tool_names()

        # Should include the extension
        assert "my_custom_tool" in tools

    def test_toolset_default_works(self):
        """ToolSet.default() should work with registry."""
        toolset = ToolSet.default()
        tools = toolset.get_tool_names()

        # Should have core tools
        assert "read" in tools
        # Should have filesystem tools
        assert "ls" in tools

    def test_toolset_custom_category(self):
        """ToolSet should work with custom categories."""
        registry = get_category_registry()
        registry.register_category("rag", {"rag_search", "rag_query"})

        toolset = ToolSet(categories={"rag"})
        tools = toolset.get_tool_names()

        assert tools == {"rag_search", "rag_query"}


class TestCacheInvalidation:
    """Tests for cache invalidation behavior."""

    def test_registration_invalidates_cache(self):
        """Registering category should invalidate cache."""
        registry = get_category_registry()

        # First lookup (populates cache)
        _ = registry.get_tools("core")

        # Register new category
        registry.register_category("new_cat", {"tool1"})

        # Should get new category (cache invalidated)
        tools = registry.get_tools("new_cat")
        assert tools == {"tool1"}

    def test_extension_invalidates_cache(self):
        """Extending category should invalidate cache."""
        registry = get_category_registry()

        # First lookup (populates cache)
        tools1 = registry.get_tools("search")

        # Extend
        registry.extend_category("search", {"new_search_tool"})

        # Should get updated result
        tools2 = registry.get_tools("search")
        assert "new_search_tool" in tools2
        assert "new_search_tool" not in tools1

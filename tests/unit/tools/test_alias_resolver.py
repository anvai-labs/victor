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

"""Tests for victor.tools.alias_resolver module."""

import pytest

from victor.tools.alias_resolver import (
    ToolAlias,
    ToolAliasResolver,
    get_alias_resolver,
)


@pytest.fixture
def resolver():
    """Create a fresh ToolAliasResolver instance for each test."""
    # Reset singleton to ensure clean state
    ToolAliasResolver.reset_instance()
    yield ToolAliasResolver.get_instance()
    # Clean up after test
    ToolAliasResolver.reset_instance()


class TestToolAlias:
    """Tests for ToolAlias dataclass."""

    def test_create_tool_alias(self):
        """Test creating a ToolAlias with default values."""
        alias = ToolAlias(canonical_name="shell")
        assert alias.canonical_name == "shell"
        assert alias.aliases == []
        assert alias.resolver is None

    def test_create_tool_alias_with_aliases(self):
        """Test creating a ToolAlias with aliases."""
        alias = ToolAlias(canonical_name="shell", aliases=["bash", "zsh", "sh"])
        assert alias.canonical_name == "shell"
        assert alias.aliases == ["bash", "zsh", "sh"]
        assert alias.resolver is None

    def test_create_tool_alias_with_resolver(self):
        """Test creating a ToolAlias with a custom resolver."""

        def custom_resolver(name: str) -> str:
            return "resolved_" + name

        alias = ToolAlias(canonical_name="custom", aliases=["alt"], resolver=custom_resolver)
        assert alias.canonical_name == "custom"
        assert alias.aliases == ["alt"]
        assert alias.resolver is custom_resolver
        assert alias.resolver("test") == "resolved_test"


class TestToolAliasResolverSingleton:
    """Tests for ToolAliasResolver singleton pattern."""

    def test_get_instance_returns_same_instance(self):
        """Test that get_instance returns the same instance."""
        ToolAliasResolver.reset_instance()
        instance1 = ToolAliasResolver.get_instance()
        instance2 = ToolAliasResolver.get_instance()
        assert instance1 is instance2
        ToolAliasResolver.reset_instance()

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance clears the singleton."""
        instance1 = ToolAliasResolver.get_instance()
        ToolAliasResolver.reset_instance()
        instance2 = ToolAliasResolver.get_instance()
        assert instance1 is not instance2
        ToolAliasResolver.reset_instance()

    def test_get_alias_resolver_function(self):
        """Test that get_alias_resolver returns the singleton."""
        ToolAliasResolver.reset_instance()
        instance1 = get_alias_resolver()
        instance2 = ToolAliasResolver.get_instance()
        assert instance1 is instance2
        ToolAliasResolver.reset_instance()


class TestToolAliasResolverRegister:
    """Tests for ToolAliasResolver.register method."""

    def test_register_with_aliases(self, resolver):
        """Test registering a tool with aliases."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        assert resolver.is_registered("shell")
        assert resolver.is_registered("bash")
        assert resolver.is_registered("zsh")
        assert resolver.is_registered("sh")

    def test_register_without_aliases(self, resolver):
        """Test registering a tool without aliases."""
        resolver.register("standalone", [])

        assert resolver.is_registered("standalone")

    def test_register_with_custom_resolver(self, resolver):
        """Test registering a tool with a custom resolver."""

        def custom_resolver(name: str) -> str:
            return "custom_" + name

        resolver.register("custom_tool", ["alt1", "alt2"], resolver=custom_resolver)

        assert resolver.is_registered("custom_tool")
        assert resolver.is_registered("alt1")
        assert resolver.is_registered("alt2")

    def test_register_multiple_tool_groups(self, resolver):
        """Test registering multiple tool groups."""
        resolver.register("shell", ["bash", "zsh"])
        resolver.register("grep", ["ripgrep", "rg"])

        assert resolver.is_registered("shell")
        assert resolver.is_registered("bash")
        assert resolver.is_registered("grep")
        assert resolver.is_registered("ripgrep")


class TestToolAliasResolverResolve:
    """Tests for ToolAliasResolver.resolve method."""

    def test_resolve_returns_canonical_when_enabled(self, resolver):
        """Test resolve returns canonical name when it's enabled."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        result = resolver.resolve("shell", enabled_tools=["shell", "bash"])
        assert result == "shell"

    def test_resolve_returns_first_enabled_alias(self, resolver):
        """Test resolve returns first enabled alias when canonical is not enabled."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        result = resolver.resolve("shell", enabled_tools=["zsh", "sh"])
        assert result == "zsh"

    def test_resolve_alias_to_enabled_canonical(self, resolver):
        """Test resolving an alias to an enabled canonical name."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        result = resolver.resolve("bash", enabled_tools=["shell"])
        assert result == "shell"

    def test_resolve_alias_to_enabled_sibling_alias(self, resolver):
        """Test resolving an alias to another enabled alias."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        result = resolver.resolve("bash", enabled_tools=["zsh"])
        assert result == "zsh"

    def test_resolve_returns_original_when_no_enabled_variant(self, resolver):
        """Test resolve returns original name when no variant is enabled."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        result = resolver.resolve("shell", enabled_tools=["grep", "find"])
        assert result == "shell"

    def test_resolve_unregistered_name(self, resolver):
        """Test resolve returns original name for unregistered tools."""
        result = resolver.resolve("unknown", enabled_tools=["shell", "bash"])
        assert result == "unknown"

    def test_resolve_empty_enabled_tools(self, resolver):
        """Test resolve with empty enabled_tools list."""
        resolver.register("shell", ["bash", "zsh"])

        result = resolver.resolve("shell", enabled_tools=[])
        assert result == "shell"


class TestToolAliasResolverCustomResolver:
    """Tests for custom resolver callback functionality."""

    def test_custom_resolver_is_called(self, resolver):
        """Test that custom resolver is called when provided."""
        calls = []

        def tracking_resolver(name: str) -> str:
            calls.append(name)
            return "custom_result"

        resolver.register("custom", ["alt"], resolver=tracking_resolver)

        result = resolver.resolve("custom", enabled_tools=["custom", "alt"])
        assert result == "custom_result"
        assert calls == ["custom"]

    def test_custom_resolver_overrides_default_logic(self, resolver):
        """Test that custom resolver completely overrides default resolution."""

        def always_return_foo(name: str) -> str:
            return "foo"

        resolver.register("bar", ["baz", "qux"], resolver=always_return_foo)

        # Even though "bar" is enabled, custom resolver returns "foo"
        result = resolver.resolve("bar", enabled_tools=["bar", "baz", "qux"])
        assert result == "foo"

    def test_custom_resolver_receives_requested_name(self, resolver):
        """Test that custom resolver receives the originally requested name."""
        received_names = []

        def capture_resolver(name: str) -> str:
            received_names.append(name)
            return name

        resolver.register("main", ["alias1", "alias2"], resolver=capture_resolver)

        resolver.resolve("alias1", enabled_tools=["main"])
        resolver.resolve("main", enabled_tools=["main"])

        assert received_names == ["alias1", "main"]

    def test_custom_resolver_with_dynamic_selection(self, resolver):
        """Test custom resolver that makes dynamic decisions."""
        import os

        def env_based_resolver(name: str) -> str:
            # Simulate choosing shell based on environment
            preferred = os.environ.get("PREFERRED_SHELL", "bash")
            return preferred

        resolver.register("shell", ["bash", "zsh", "fish"], resolver=env_based_resolver)

        # Without env var, should return default "bash"
        result = resolver.resolve("shell", enabled_tools=["bash", "zsh"])
        assert result == "bash"


class TestToolAliasResolverGetCanonical:
    """Tests for ToolAliasResolver.get_canonical method."""

    def test_get_canonical_for_canonical_name(self, resolver):
        """Test get_canonical returns canonical name for canonical name."""
        resolver.register("shell", ["bash", "zsh"])

        result = resolver.get_canonical("shell")
        assert result == "shell"

    def test_get_canonical_for_alias(self, resolver):
        """Test get_canonical returns canonical name for alias."""
        resolver.register("shell", ["bash", "zsh"])

        result = resolver.get_canonical("bash")
        assert result == "shell"

        result = resolver.get_canonical("zsh")
        assert result == "shell"

    def test_get_canonical_for_unregistered(self, resolver):
        """Test get_canonical returns original name for unregistered name."""
        result = resolver.get_canonical("unknown")
        assert result == "unknown"


class TestToolAliasResolverGetAliases:
    """Tests for ToolAliasResolver.get_aliases method."""

    def test_get_aliases_returns_aliases(self, resolver):
        """Test get_aliases returns the registered aliases."""
        resolver.register("shell", ["bash", "zsh", "sh"])

        aliases = resolver.get_aliases("shell")
        assert aliases == ["bash", "zsh", "sh"]

    def test_get_aliases_returns_empty_for_unregistered(self, resolver):
        """Test get_aliases returns empty list for unregistered name."""
        aliases = resolver.get_aliases("unknown")
        assert aliases == []

    def test_get_aliases_returns_copy(self, resolver):
        """Test get_aliases returns a copy, not the original list."""
        resolver.register("shell", ["bash", "zsh"])

        aliases = resolver.get_aliases("shell")
        aliases.append("sh")

        # Original should be unchanged
        assert resolver.get_aliases("shell") == ["bash", "zsh"]


class TestToolAliasResolverIsRegistered:
    """Tests for ToolAliasResolver.is_registered method."""

    def test_is_registered_for_canonical(self, resolver):
        """Test is_registered returns True for canonical name."""
        resolver.register("shell", ["bash", "zsh"])

        assert resolver.is_registered("shell") is True

    def test_is_registered_for_alias(self, resolver):
        """Test is_registered returns True for alias."""
        resolver.register("shell", ["bash", "zsh"])

        assert resolver.is_registered("bash") is True
        assert resolver.is_registered("zsh") is True

    def test_is_registered_for_unregistered(self, resolver):
        """Test is_registered returns False for unregistered name."""
        assert resolver.is_registered("unknown") is False


class TestToolAliasResolverIntegration:
    """Integration tests for common use cases."""

    def test_shell_variant_resolution(self, resolver):
        """Test typical shell variant resolution scenario."""
        # Register shell variants
        resolver.register("shell", ["bash", "zsh", "sh", "fish"])

        # System only has zsh available
        enabled = ["zsh", "grep", "find"]

        # Request for "shell" resolves to "zsh"
        assert resolver.resolve("shell", enabled) == "zsh"

        # Request for "bash" also resolves to "zsh" (first enabled)
        assert resolver.resolve("bash", enabled) == "zsh"

    def test_grep_tool_alternatives(self, resolver):
        """Test grep/ripgrep alternative resolution."""
        resolver.register("grep", ["ripgrep", "rg", "ag"])

        # Modern system with ripgrep
        enabled_modern = ["ripgrep", "fd", "zsh"]
        assert resolver.resolve("grep", enabled_modern) == "ripgrep"
        assert resolver.resolve("rg", enabled_modern) == "ripgrep"

        # Legacy system with standard grep
        enabled_legacy = ["grep", "find", "bash"]
        assert resolver.resolve("ripgrep", enabled_legacy) == "grep"

    def test_multiple_tool_groups_independence(self, resolver):
        """Test that multiple tool groups work independently."""
        resolver.register("shell", ["bash", "zsh"])
        resolver.register("grep", ["ripgrep", "rg"])
        resolver.register("find", ["fd", "fdfind"])

        enabled = ["zsh", "ripgrep", "fd"]

        assert resolver.resolve("shell", enabled) == "zsh"
        assert resolver.resolve("grep", enabled) == "ripgrep"
        assert resolver.resolve("find", enabled) == "fd"

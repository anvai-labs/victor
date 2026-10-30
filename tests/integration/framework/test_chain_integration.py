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

"""Integration tests for chain registry integration across verticals.

Vertical-agnostic: the coding vertical's activation tests live in
tests/integration/verticals/test_coding_chain_activation.py, whose CI job
installs the verticals.

Tests verify that:
- Chain factory registration works
- Chain discovery and access APIs work
- Vertical namespace isolation works
"""

import pytest

from victor.core.verticals.import_resolver import import_module_with_fallback


def _import_vertical_module(module_path: str):
    """Import vertical module using external-first fallback resolution."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None:
        raise ImportError(f"Vertical module not available: {module_path}")
    return module


def _import_vertical_attr(module_path: str, attr_name: str):
    """Import an attribute from a vertical module with fallback resolution."""
    module = _import_vertical_module(module_path)
    if not hasattr(module, attr_name):
        raise ImportError(f"Missing attribute {attr_name} in module {module_path}")
    return getattr(module, attr_name)


class TestChainFactoryRegistration:
    """Tests for chain factory registration pattern."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_chain_factory_registration(self):
        """Chain factories can be registered."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        # Register a factory
        def test_factory():
            return {"test": "chain"}

        registry.register_factory(
            "test_factory_chain",
            test_factory,
            vertical="coding",
            description="Test factory chain",
        )

        # Verify factory is listed
        factories = registry.list_factories(vertical="coding")
        assert "coding:test_factory_chain" in factories

    def test_chain_factory_execution(self):
        """Chain factories can be executed to create chains."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        # Register a factory
        def create_test_chain():
            return {"chain": "instance", "data": "test"}

        registry.register_factory(
            "dynamic_chain",
            create_test_chain,
            vertical="coding",
            description="Dynamic chain factory",
        )

        # Create from factory
        chain = registry.create("dynamic_chain", vertical="coding")

        assert chain is not None
        assert chain["chain"] == "instance"

    def test_chain_factory_multiple_creations(self):
        """Chain factories can be called multiple times."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        # Register a factory that returns new instances
        call_count = 0

        def counter_factory():
            nonlocal call_count
            call_count += 1
            return {"instance": call_count}

        registry.register_factory("counter_chain", counter_factory, vertical="coding")

        # Create multiple times
        chain1 = registry.create("counter_chain", vertical="coding")
        chain2 = registry.create("counter_chain", vertical="coding")

        # Should get different instances
        assert chain1["instance"] == 1
        assert chain2["instance"] == 2

    def test_chain_factory_error_handling(self):
        """Chain factory errors are properly raised."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        # Register a failing factory
        def failing_factory():
            raise ValueError("Factory error")

        registry.register_factory("failing_chain", failing_factory, vertical="coding")

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Factory execution failed"):
            registry.create("failing_chain", vertical="coding")


class TestChainNamespaceIsolation:
    """Tests for vertical namespace isolation in chain registry."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_vertical_namespace_prefix(self):
        """Chains from different verticals have different namespace prefixes."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        # Register chains from different verticals
        registry.register("test_chain", {"v": 1}, vertical="coding")
        registry.register("test_chain", {"v": 2}, vertical="research")

        # Should be able to get both
        coding_chain = registry.get("test_chain", vertical="coding")
        research_chain = registry.get("test_chain", vertical="research")

        assert coding_chain["v"] == 1
        assert research_chain["v"] == 2

    def test_chain_full_name_property(self):
        """Chain metadata has correct full_name with namespace."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        registry.register("my_chain", {"data": "test"}, vertical="coding")

        meta = registry.get_metadata("my_chain", vertical="coding")

        assert meta.full_name == "coding:my_chain"
        assert meta.name == "my_chain"
        assert meta.vertical == "coding"

    def test_list_chains_by_vertical(self):
        """Chains can be filtered by vertical."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        # Register chains across verticals
        registry.register("chain1", {"v": "c1"}, vertical="coding")
        registry.register("chain2", {"v": "c2"}, vertical="coding")
        registry.register("chain1", {"v": "r1"}, vertical="research")

        # List by vertical
        coding_chains = registry.list_chains(vertical="coding")
        research_chains = registry.list_chains(vertical="research")

        assert len(coding_chains) == 2
        assert len(research_chains) == 1

    def test_find_by_vertical_returns_dict(self):
        """find_by_vertical returns dict of chains."""
        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()

        registry.register("chain_a", {"data": "a"}, vertical="coding")
        registry.register("chain_b", {"data": "b"}, vertical="coding")

        coding_chains = registry.find_by_vertical("coding")

        assert isinstance(coding_chains, dict)
        assert len(coding_chains) == 2
        assert "coding:chain_a" in coding_chains
        assert "coding:chain_b" in coding_chains


class TestChainDecoratorRegistration:
    """Tests for @chain decorator registration pattern."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_chain_decorator_registration(self):
        """@chain decorator registers factory functions."""
        from victor.framework.chain_registry import get_chain_registry, chain

        # Use decorator to register
        @chain("coding:decorator_test", description="Decorator test chain")
        def my_test_chain():
            return {"decorated": "chain"}

        registry = get_chain_registry()

        # Should be registered as factory
        factories = registry.list_factories(vertical="coding")
        assert "coding:decorator_test" in factories

        # Should be creatable
        created = registry.create("decorator_test", vertical="coding")
        assert created["decorated"] == "chain"

    def test_chain_decorator_with_vertical_in_name(self):
        """@chain decorator supports "vertical:name" format."""
        from victor.framework.chain_registry import get_chain_registry, chain

        @chain("research:auto_chain", description="Auto chain")
        def auto_chain():
            return {"auto": True}

        registry = get_chain_registry()

        # Should be registered under research vertical
        assert registry.has("auto_chain", vertical="research")

    def test_chain_decorator_metadata(self):
        """@chain decorator registers complete metadata."""
        from victor.framework.chain_registry import get_chain_registry, chain

        @chain(
            "coding:meta_test",
            description="Metadata test chain",
            input_type="Dict[str, Any]",
            output_type="Dict[str, str]",
            tags=["test", "metadata"],
            version="2.0.0",
        )
        def meta_chain():
            return {"test": "metadata"}

        registry = get_chain_registry()
        meta = registry.get_metadata("meta_test", vertical="coding")

        assert meta.description == "Metadata test chain"
        assert meta.input_type == "Dict[str, Any]"
        assert meta.output_type == "Dict[str, str]"
        assert "test" in meta.tags
        assert "metadata" in meta.tags
        assert meta.version == "2.0.0"


class TestChainUtilityFunctions:
    """Tests for chain registry utility functions."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_register_chain_utility(self):
        """register_chain() utility function works."""
        from victor.framework.chain_registry import register_chain, get_chain_registry

        register_chain(
            "util_chain",
            {"util": "chain"},
            vertical="coding",
            description="Utility test chain",
            tags=["util"],
        )

        registry = get_chain_registry()
        assert registry.has("util_chain", vertical="coding")

    def test_get_chain_utility(self):
        """get_chain() utility function retrieves chains."""
        from victor.framework.chain_registry import register_chain, get_chain

        register_chain("get_test", {"get": "test"}, vertical="coding")

        chain = get_chain("get_test", vertical="coding")
        assert chain is not None
        assert chain["get"] == "test"

    def test_create_chain_utility(self):
        """create_chain() utility function creates from factories."""
        from victor.framework.chain_registry import chain, create_chain

        @chain("coding:create_test")
        def factory_chain():
            return {"created": True}

        result = create_chain("create_test", vertical="coding")
        assert result is not None
        assert result["created"] is True


class TestChainIntegrationScenarios:
    """End-to-end integration scenarios for chain registry."""

    def test_cross_vertical_chain_discovery(self):
        """Chains can be discovered across multiple verticals."""
        from victor.framework.chain_registry import get_chain_registry, register_chain

        registry = get_chain_registry()

        # Register chains from different verticals
        register_chain("chain1", {"v": 1}, vertical="coding", tags=["analysis"])
        register_chain("chain2", {"v": 2}, vertical="research", tags=["analysis"])
        register_chain("chain3", {"v": 3}, vertical="devops", tags=["deploy"])

        # Find by tag across verticals
        analysis_chains = registry.find_by_tag("analysis")
        assert len(analysis_chains) == 2

        # List all chains
        all_chains = registry.list_chains()
        assert len(all_chains) >= 3

    def test_chain_registry_serialization(self):
        """Chain registry can be serialized to dict."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # Import chains to populate registry
            _import_vertical_module("victor.coding.composed_chains")

            # Serialize
            serialized = registry.to_dict()

            # Should have all chains
            assert len(serialized) > 0

            # Check structure
            for key, meta_dict in serialized.items():
                assert "name" in meta_dict
                assert "vertical" in meta_dict
                assert "description" in meta_dict
                assert "version" in meta_dict
        except (ImportError, ModuleNotFoundError):
            # If composed_chains doesn't exist, test with empty registry
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()
            serialized = registry.to_dict()
            # Should work even if empty
            assert isinstance(serialized, dict)

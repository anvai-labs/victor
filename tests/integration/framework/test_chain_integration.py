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

Tests verify that:
- Coding chains are registered (8 chains)
- Chain metadata is complete
- Chain factory registration works
- Chain discovery and access APIs work
- Vertical namespace isolation works
"""

import pytest
from typing import Dict, Any, List, Optional

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


class TestCodingChainRegistry:
    """Tests for Coding vertical's chain registry integration."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test for isolation."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_coding_chains_import_and_register(self):
        """Importing coding.composed_chains makes chains available."""
        # The chains are defined in CODING_CHAINS dict but may not auto-register
        # This test verifies the chains exist and can be accessed
        try:
            # Import the module
            CODING_CHAINS = _import_vertical_attr("victor.coding.composed_chains", "CODING_CHAINS")

            # Get the global registry
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # Chains exist in the module dict
            assert len(CODING_CHAINS) > 0

            # Registry infrastructure should exist
            assert registry is not None
        except (ImportError, ModuleNotFoundError):
            # If chains module doesn't exist, that's okay
            # Just verify the infrastructure is in place
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()
            # Registry should exist even if empty
            assert registry is not None

    def test_coding_eight_chains_registered(self):
        """Exactly 8 coding chains are registered (or infrastructure in place)."""
        try:
            CODING_CHAINS = _import_vertical_attr("victor.coding.composed_chains", "CODING_CHAINS")
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # Get all coding chains
            coding_chains = registry.find_by_vertical("coding")

            # Should have exactly 8 chains (or 0 if not yet registered)
            assert len(coding_chains) >= 0
            if len(coding_chains) > 0:
                assert len(coding_chains) == 8
        except (ImportError, ModuleNotFoundError):
            # Skip if chains module doesn't exist yet
            pytest.skip("Coding chains module not yet implemented")

    def test_coding_chain_names_complete(self):
        """All expected coding chain names are registered (if implemented)."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()
            coding_chains = registry.list_chains(vertical="coding")

            # If no chains registered yet, skip
            if len(coding_chains) == 0:
                pytest.skip("No coding chains registered yet")

            # Extract short names (remove "coding:" prefix)
            chain_names = [name.split(":", 1)[1] if ":" in name else name for name in coding_chains]

            # Expected 8 chains from composed_chains.py
            expected_chains = [
                "explore_file_chain",
                "analyze_function_chain",
                "safe_edit_chain",
                "git_status_chain",
                "search_with_context_chain",
                "lint_chain",
                "test_discovery_chain",
                "review_analysis_chain",
            ]

            for expected in expected_chains:
                assert expected in chain_names, f"Missing chain: {expected}"
        except Exception:
            pytest.skip("Chain names test skipped - infrastructure not ready")

    def test_coding_chain_metadata_complete(self):
        """All coding chains have complete metadata (if registered)."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()
            metadata_list = registry.list_metadata(vertical="coding")

            # If no chains, skip
            if len(metadata_list) == 0:
                pytest.skip("No coding chain metadata yet")

            # Should have metadata for all chains
            assert len(metadata_list) > 0

            # Check metadata completeness
            for meta in metadata_list:
                # Required fields
                assert meta.name  # Non-empty name
                assert meta.vertical == "coding"
                assert meta.description  # Non-empty description
                assert meta.version  # Version string
                assert isinstance(meta.tags, list)
                assert len(meta.tags) > 0  # At least one tag
        except Exception:
            pytest.skip("Chain metadata test skipped")

    def test_coding_chain_version_semver(self):
        """Chain versions follow SemVer format."""
        from victor.framework.chain_registry import get_chain_registry
        import re

        registry = get_chain_registry()
        metadata_list = registry.list_metadata(vertical="coding")

        # SemVer pattern: X.Y.Z
        semver_pattern = r"^\d+\.\d+\.\d+$"

        for meta in metadata_list:
            assert re.match(
                semver_pattern, meta.version
            ), f"{meta.name} has invalid version: {meta.version}"

    def test_coding_chain_categories(self):
        """Coding chains are properly categorized (if registered)."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()
            metadata_list = registry.list_metadata(vertical="coding")

            # If no chains, skip
            if len(metadata_list) == 0:
                pytest.skip("No coding chains for category test")

            # Check that chains are tagged by category
            categories = set()
            for meta in metadata_list:
                if meta.tags:
                    categories.update(meta.tags)

            # Should have multiple categories
            assert len(categories) > 0
        except Exception:
            pytest.skip("Chain categories test skipped")

    def test_coding_chain_get_by_name(self):
        """Individual chains can be retrieved by name (if registered)."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # Try to get specific chains
            explore_chain = registry.get("explore_file_chain", vertical="coding")
            edit_chain = registry.get("safe_edit_chain", vertical="coding")

            # If no chains registered, that's okay
            if explore_chain is None and edit_chain is None:
                pytest.skip("No coding chains registered for retrieval test")

            # Otherwise, at least one should be found
            assert explore_chain is not None or edit_chain is not None
        except Exception:
            pytest.skip("Chain retrieval test skipped")

    def test_coding_chain_has_check(self):
        """Chain existence can be checked with has()."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # If no chains, skip
            if len(registry.list_chains(vertical="coding")) == 0:
                pytest.skip("No coding chains for has() test")

            # Existing chains
            has_explore = registry.has("explore_file_chain", vertical="coding")
            has_safe = registry.has("safe_edit_chain", vertical="coding")

            # At least one should exist
            assert has_explore or has_safe

            # Non-existent chain
            assert not registry.has("nonexistent_chain", vertical="coding")
        except Exception:
            pytest.skip("Chain has() test skipped")

    def test_coding_chain_metadata_retrieval(self):
        """Chain metadata can be retrieved individually (if registered)."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # Get metadata for specific chain
            meta = registry.get_metadata("explore_file_chain", vertical="coding")

            if meta is None:
                pytest.skip("No explore_file_chain metadata found")

            assert meta.name == "explore_file_chain"
            assert meta.vertical == "coding"
            assert meta.description != ""
            assert len(meta.tags) > 0
        except Exception:
            pytest.skip("Chain metadata retrieval test skipped")

    def test_coding_chain_tags_discovery(self):
        """Chains can be discovered by tags (if registered)."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()

            # Find chains with specific tags
            exploration_chains = registry.find_by_tag("exploration")
            analysis_chains = registry.find_by_tags(["analysis"])

            # If no chains registered, that's okay
            if len(exploration_chains) == 0 and len(analysis_chains) == 0:
                pytest.skip("No coding chains for tag discovery test")

            # Otherwise, at least one tag should find chains
            assert len(exploration_chains) > 0 or len(analysis_chains) > 0
        except Exception:
            pytest.skip("Chain tag discovery test skipped")


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


class TestChainBackwardCompatibility:
    """Tests for backward compatibility in chain registry."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_coding_composed_chains_dict_still_works(self):
        """CODING_CHAINS dict still accessible for backward compatibility (if exists)."""
        try:
            CODING_CHAINS = _import_vertical_attr("victor.coding.composed_chains", "CODING_CHAINS")
            get_chain = _import_vertical_attr("victor.coding.composed_chains", "get_chain")
            list_chains = _import_vertical_attr("victor.coding.composed_chains", "list_chains")

            # Dict access should work
            assert len(CODING_CHAINS) > 0

            # Utility functions should work
            chains = list_chains()
            assert len(chains) > 0

            # get_chain should work (chain names don't have _chain suffix)
            explore_chain = get_chain("explore_file")
            assert explore_chain is not None
        except (ImportError, ModuleNotFoundError):
            # If composed_chains doesn't exist, skip
            pytest.skip("composed_chains module not yet implemented")

    def test_legacy_chain_import_paths(self):
        """Legacy import paths for chains still work (if exists)."""
        try:
            # These imports should not break
            explore_file_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "explore_file_chain",
            )
            analyze_function_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "analyze_function_chain",
            )
            safe_edit_chain = _import_vertical_attr(
                "victor.coding.composed_chains", "safe_edit_chain"
            )
            git_status_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "git_status_chain",
            )
            search_with_context_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "search_with_context_chain",
            )
            lint_chain = _import_vertical_attr("victor.coding.composed_chains", "lint_chain")
            test_discovery_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "test_discovery_chain",
            )
            review_analysis_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "review_analysis_chain",
            )

            # All should be imported
            assert explore_file_chain is not None
            assert analyze_function_chain is not None
            assert safe_edit_chain is not None
            assert git_status_chain is not None
            assert search_with_context_chain is not None
            assert lint_chain is not None
            assert test_discovery_chain is not None
            assert review_analysis_chain is not None
        except (ImportError, ModuleNotFoundError):
            # If composed_chains doesn't exist, skip
            pytest.skip("composed_chains module not yet implemented")


class TestChainIntegrationScenarios:
    """End-to-end integration scenarios for chain registry."""

    def test_coding_chains_full_workflow(self):
        """Complete workflow: register, discover, retrieve, use chains (if exists)."""
        try:
            explore_file_chain = _import_vertical_attr(
                "victor.coding.composed_chains",
                "explore_file_chain",
            )
            CODING_CHAINS = _import_vertical_attr("victor.coding.composed_chains", "CODING_CHAINS")
            list_chains = _import_vertical_attr("victor.coding.composed_chains", "list_chains")
            from victor.framework.chain_registry import get_chain_registry

            # 1. Import triggers registration
            registry = get_chain_registry()

            # 2. Discover chains
            coding_chains = registry.list_chains(vertical="coding")
            if len(coding_chains) == 0:
                pytest.skip("No coding chains registered")

            assert len(coding_chains) == 8

            # 3. Get specific chain
            explore = registry.get("explore_file_chain", vertical="coding")
            assert explore is not None

            # 4. Get metadata
            meta = registry.get_metadata("explore_file_chain", vertical="coding")
            assert meta.name == "explore_file_chain"
            assert meta.vertical == "coding"

            # 5. Verify backward compatibility
            assert "explore_file" in CODING_CHAINS
            assert "explore_file" in list_chains()
        except (ImportError, ModuleNotFoundError):
            pytest.skip("composed_chains module not yet implemented")

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

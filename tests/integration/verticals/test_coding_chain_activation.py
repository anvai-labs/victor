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

"""Coding-vertical chain activation against the flat ChainRegistry.

Lives under tests/integration/verticals/ because it needs the coding vertical
installed; the verticals-integration CI job installs every first-party
vertical and runs this directory.
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


EXPECTED_CODING_CHAINS = (
    "explore_file",
    "analyze_function",
    "safe_edit",
    "git_status",
    "search_with_context",
    "lint",
    "test_discovery",
    "review_analysis",
)


def _activate_coding_chains() -> dict:
    """Register the coding vertical's chains the way vertical activation does.

    Activation (``step_handlers.apply_chains``) passes
    ``CodingAssistant.get_composed_chains()`` to the framework integration
    registry service, which calls ``ChainRegistry.register_from_vertical``.
    A fresh service instance is used so its fingerprint dedup cannot leak
    between tests.
    """
    from victor.framework.framework_integration_registry_service import (
        FrameworkIntegrationRegistryService,
    )

    coding_assistant = _import_vertical_attr("victor.coding.assistant", "CodingAssistant")
    chains = coding_assistant.get_composed_chains()
    count = FrameworkIntegrationRegistryService().register_chains("coding", chains)
    assert count == len(chains)
    return chains


class TestCodingChainRegistry:
    """Coding chains are discoverable in the flat ChainRegistry after activation."""

    @pytest.fixture(autouse=True)
    def reset_registry_before_tests(self):
        """Reset chain registry before each test for isolation."""
        from victor.framework.chain_registry import reset_chain_registry

        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_registry_is_empty_before_activation(self):
        """Importing the chains module alone does not populate the flat registry."""
        from victor.framework.chain_registry import get_chain_registry

        _import_vertical_attr("victor.coding.composed_chains", "CODING_CHAINS")

        assert get_chain_registry().list_chains(vertical="coding") == []

    def test_coding_eight_chains_registered(self):
        """Activation registers exactly the eight composed coding chains."""
        from victor.framework.chain_registry import get_chain_registry

        _activate_coding_chains()

        coding_chains = get_chain_registry().find_by_vertical("coding")
        assert len(coding_chains) == 8

    def test_coding_chain_names_complete(self):
        """All expected coding chains are listed under the coding namespace."""
        from victor.framework.chain_registry import get_chain_registry

        _activate_coding_chains()

        listed = set(get_chain_registry().list_chains(vertical="coding"))
        assert listed == {f"coding:{name}" for name in EXPECTED_CODING_CHAINS}

    def test_coding_chain_metadata(self):
        """Each activated chain carries direct-registration metadata."""
        from victor.framework.chain_registry import get_chain_registry

        _activate_coding_chains()

        metadata_list = get_chain_registry().list_metadata(vertical="coding")
        assert sorted(m.name for m in metadata_list) == sorted(EXPECTED_CODING_CHAINS)
        for meta in metadata_list:
            assert meta.vertical == "coding"
            assert meta.is_factory is False
            assert meta.full_name == f"coding:{meta.name}"

    def test_coding_chain_version_semver(self):
        """Chain versions follow SemVer format."""
        import re

        from victor.framework.chain_registry import get_chain_registry

        _activate_coding_chains()

        metadata_list = get_chain_registry().list_metadata(vertical="coding")
        assert metadata_list
        for meta in metadata_list:
            assert re.match(r"^\d+\.\d+\.\d+$", meta.version), meta

    def test_coding_chain_get_by_name(self):
        """Individual chains are retrievable and are the vertical's objects."""
        from victor.framework.chain_registry import get_chain, get_chain_registry

        chains = _activate_coding_chains()
        registry = get_chain_registry()

        assert registry.get("explore_file", vertical="coding") is chains["explore_file"]
        assert get_chain("safe_edit", vertical="coding") is chains["safe_edit"]

    def test_coding_chain_has_check(self):
        """has() reports activated chains and rejects unknown ones."""
        from victor.framework.chain_registry import get_chain_registry

        _activate_coding_chains()
        registry = get_chain_registry()

        assert registry.has("explore_file", vertical="coding")
        assert registry.has("safe_edit", vertical="coding")
        assert not registry.has("nonexistent_chain", vertical="coding")
        assert not registry.has("explore_file", vertical="devops")

    def test_coding_chain_metadata_retrieval(self):
        """Chain metadata can be retrieved individually."""
        from victor.framework.chain_registry import get_chain_registry

        _activate_coding_chains()

        meta = get_chain_registry().get_metadata("explore_file", vertical="coding")
        assert meta is not None
        assert meta.name == "explore_file"
        assert meta.vertical == "coding"

    def test_activated_chain_resolves_as_compute_handler(self):
        """A workflow ``chain:coding:<name>`` handler resolves after activation."""
        from victor.workflows.executors.compute import ComputeNodeExecutor

        _activate_coding_chains()

        handler = ComputeNodeExecutor()._resolve_chain_handler("chain:coding:explore_file")
        assert handler is not None
        assert callable(handler)


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
        """CODING_CHAINS dict still accessible for backward compatibility."""
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

    def test_legacy_chain_import_paths(self):
        """Legacy import paths for chains still work."""
        # These imports should not break
        explore_file_chain = _import_vertical_attr(
            "victor.coding.composed_chains",
            "explore_file_chain",
        )
        analyze_function_chain = _import_vertical_attr(
            "victor.coding.composed_chains",
            "analyze_function_chain",
        )
        safe_edit_chain = _import_vertical_attr("victor.coding.composed_chains", "safe_edit_chain")
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


class TestCodingChainScenarios:
    """End-to-end coding chain activation scenario."""

    def test_coding_chains_full_workflow(self):
        """Complete workflow: activate, discover, retrieve and inspect coding chains."""
        from victor.framework.chain_registry import get_chain_registry, reset_chain_registry

        reset_chain_registry()
        try:
            CODING_CHAINS = _import_vertical_attr("victor.coding.composed_chains", "CODING_CHAINS")
            list_chains = _import_vertical_attr("victor.coding.composed_chains", "list_chains")

            # 1. Activation registers the vertical's chains
            _activate_coding_chains()
            registry = get_chain_registry()

            # 2. Discover chains
            coding_chains = registry.list_chains(vertical="coding")
            assert len(coding_chains) == 8

            # 3. Get specific chain
            explore = registry.get("explore_file", vertical="coding")
            assert explore is CODING_CHAINS["explore_file"]

            # 4. Get metadata
            meta = registry.get_metadata("explore_file", vertical="coding")
            assert meta.name == "explore_file"
            assert meta.vertical == "coding"

            # 5. Module-level helpers stay consistent with the registry
            assert "explore_file" in CODING_CHAINS
            assert "explore_file" in list_chains()
        finally:
            reset_chain_registry()

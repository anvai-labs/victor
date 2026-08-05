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

"""Unit tests for product bundle system."""

from __future__ import annotations

import pytest

from victor_contracts.verticals.protocols.base import VerticalBase as SdkVerticalBase
from victor.verticals.product_bundle import (
    ProductBundle,
    BundleTier,
    get_bundle,
    list_bundles,
    get_bundles_for_vertical,
    resolve_bundle_dependencies,
    get_bundle_tier,
    get_install_command,
    FOUNDATION_BUNDLE,
    ENGINEERING_BUNDLE,
    DATA_BUNDLE,
    PROFESSIONAL_BUNDLE,
    ENTERPRISE_BUNDLE,
)
from victor.verticals.unified_registry import (
    UnifiedVerticalRegistry,
    VerticalStatus,
    VerticalInfo,
    get_registry,
    reset_registry,
)


class TestProductBundle:
    """Test suite for ProductBundle."""

    def test_bundle_creation(self):
        """Test creating a product bundle."""
        bundle = ProductBundle(
            name="test",
            display_name="Test Bundle",
            description="A test bundle",
            verticals=["victor-coding"],
            tier=BundleTier.ENGINEERING,
        )

        assert bundle.name == "test"
        assert bundle.display_name == "Test Bundle"
        assert bundle.verticals == ["victor-coding"]
        assert bundle.tier == BundleTier.ENGINEERING

    def test_bundle_get_all_verticals_no_deps(self):
        """Test getting all verticals for bundle without dependencies."""
        bundle = ProductBundle(
            name="test",
            display_name="Test Bundle",
            description="A test bundle",
            verticals=["victor-coding", "victor-devops"],
            tier=BundleTier.ENGINEERING,
        )

        all_verticals = bundle.get_all_verticals()
        assert all_verticals == {"victor-coding", "victor-devops"}

    def test_bundle_get_all_verticals_with_deps(self):
        """Test getting all verticals for bundle with dependencies."""
        # Use predefined bundles that have dependencies
        all_verticals = PROFESSIONAL_BUNDLE.get_all_verticals()

        # Should include engineering + data verticals
        assert "victor-coding" in all_verticals
        assert "victor-devops" in all_verticals
        assert "victor-rag" in all_verticals
        assert "victor-dataanalysis" in all_verticals
        assert "victor-research" in all_verticals

    def test_bundle_get_install_command(self):
        """Test getting install command for bundle."""
        bundle = ProductBundle(
            name="test",
            display_name="Test Bundle",
            description="A test bundle",
            verticals=[],
            tier=BundleTier.FOUNDATION,
        )

        command = bundle.get_install_command()
        assert command == "pip install victor-ai[test]"


class TestStandardBundles:
    """Test suite for standard product bundles."""

    def test_foundation_bundle(self):
        """Test foundation bundle definition."""
        assert FOUNDATION_BUNDLE.name == "foundation"
        assert FOUNDATION_BUNDLE.tier == BundleTier.FOUNDATION
        assert FOUNDATION_BUNDLE.verticals == []
        assert FOUNDATION_BUNDLE.dependencies == []

    def test_engineering_bundle(self):
        """Test engineering bundle definition."""
        assert ENGINEERING_BUNDLE.name == "engineering"
        assert ENGINEERING_BUNDLE.tier == BundleTier.ENGINEERING
        assert "victor-coding" in ENGINEERING_BUNDLE.verticals
        assert "victor-devops" in ENGINEERING_BUNDLE.verticals
        assert "security" in ENGINEERING_BUNDLE.optional_verticals

    def test_data_bundle(self):
        """Test data bundle definition."""
        assert DATA_BUNDLE.name == "data"
        assert DATA_BUNDLE.tier == BundleTier.DATA
        assert "victor-rag" in DATA_BUNDLE.verticals
        assert "victor-dataanalysis" in DATA_BUNDLE.verticals
        assert "victor-research" in DATA_BUNDLE.verticals

    def test_professional_bundle(self):
        """Test professional bundle definition."""
        assert PROFESSIONAL_BUNDLE.name == "professional"
        assert PROFESSIONAL_BUNDLE.tier == BundleTier.PROFESSIONAL
        assert "engineering" in PROFESSIONAL_BUNDLE.dependencies
        assert "data" in PROFESSIONAL_BUNDLE.dependencies

    def test_enterprise_bundle(self):
        """Test enterprise bundle definition."""
        assert ENTERPRISE_BUNDLE.name == "enterprise"
        assert ENTERPRISE_BUNDLE.tier == BundleTier.ENTERPRISE


class TestBundleRegistry:
    """Test suite for bundle registry functions."""

    def test_get_bundle(self):
        """Test getting bundle by name."""
        bundle = get_bundle("engineering")
        assert bundle is not None
        assert bundle.name == "engineering"

    def test_get_bundle_not_found(self):
        """Test getting non-existent bundle."""
        bundle = get_bundle("nonexistent")
        assert bundle is None

    def test_list_bundles(self):
        """Test listing all bundles."""
        bundles = list_bundles()
        assert len(bundles) == 5  # foundation, engineering, data, professional, enterprise

        bundle_names = {b.name for b in bundles}
        assert bundle_names == {
            "foundation",
            "engineering",
            "data",
            "professional",
            "enterprise",
        }

    def test_get_bundles_for_vertical(self):
        """Test getting bundles that include a vertical."""
        bundles = get_bundles_for_vertical("victor-coding")

        # victor-coding is in engineering, professional, and enterprise
        assert len(bundles) == 3
        bundle_names = {b.name for b in bundles}
        assert bundle_names == {"engineering", "professional", "enterprise"}

    def test_resolve_bundle_dependencies_no_deps(self):
        """Test resolving dependencies for bundle without dependencies."""
        verticals = resolve_bundle_dependencies("engineering")
        assert verticals == {"victor-coding", "victor-devops"}

    def test_resolve_bundle_dependencies_with_deps(self):
        """Test resolving dependencies for bundle with dependencies."""
        verticals = resolve_bundle_dependencies("professional")

        # Should include all verticals from engineering + data
        assert "victor-coding" in verticals
        assert "victor-devops" in verticals
        assert "victor-rag" in verticals
        assert "victor-dataanalysis" in verticals
        assert "victor-research" in verticals

    def test_resolve_bundle_dependencies_not_found(self):
        """Test resolving dependencies for non-existent bundle."""
        verticals = resolve_bundle_dependencies("nonexistent")
        assert verticals == set()

    def test_get_bundle_tier(self):
        """Test getting bundle by tier."""
        bundle = get_bundle_tier(BundleTier.ENGINEERING)
        assert bundle is not None
        assert bundle.tier == BundleTier.ENGINEERING
        assert bundle.name == "engineering"

    def test_get_install_command(self):
        """Test getting install command."""
        command = get_install_command("engineering")
        assert command == "pip install victor-ai[engineering]"

    def test_get_install_command_not_found(self):
        """Test getting install command for non-existent bundle."""
        command = get_install_command("nonexistent")
        assert command is None


class TestUnifiedVerticalRegistry:
    """Test suite for UnifiedVerticalRegistry."""

    @pytest.fixture
    def registry(self, monkeypatch):
        """Create a fresh registry for each test."""
        reset_registry()
        registry = UnifiedVerticalRegistry()

        async def _no_discovery() -> None:
            return None

        monkeypatch.setattr(registry, "_discover_from_entry_points", _no_discovery)
        return registry

    @pytest.mark.asyncio
    async def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert not registry._initialized

        await registry.initialize()

        assert registry._initialized

    @pytest.mark.asyncio
    async def test_register_vertical(self, registry):
        """Test registering a vertical."""
        await registry.initialize()

        registry.register_vertical(
            name="test-vertical",
            version="1.0.0",
            capabilities={"coding", "analysis"},
        )

        info = registry.get_vertical("test-vertical")
        assert info is not None
        assert info.name == "test-vertical"
        assert info.version == "1.0.0"
        assert info.capabilities == {"coding", "analysis"}

    @pytest.mark.asyncio
    async def test_register_vertical_with_metadata(self, registry):
        """Test registering a vertical with metadata."""
        await registry.initialize()

        registry.register_vertical(
            name="test-vertical",
            version="1.0.0",
            capabilities={"coding"},
            entry_point="test_vertical:TestVertical",
            metadata={"description": "A test vertical"},
        )

        info = registry.get_vertical("test-vertical")
        assert info.entry_point == "test_vertical:TestVertical"
        assert info.metadata == {"description": "A test vertical"}

    @pytest.mark.asyncio
    async def test_get_vertical_not_found(self, registry):
        """Test getting non-existent vertical."""
        await registry.initialize()

        info = registry.get_vertical("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_list_verticals(self, registry):
        """Test listing all verticals."""
        await registry.initialize()

        registry.register_vertical("vertical1", "1.0.0", {"cap1"})
        registry.register_vertical("vertical2", "2.0.0", {"cap2"})

        verticals = registry.list_verticals()
        assert len(verticals) == 2

        names = {v.name for v in verticals}
        assert names == {"vertical1", "vertical2"}

    @pytest.mark.asyncio
    async def test_list_verticals_with_status_filter(self, registry):
        """Test listing verticals with status filter."""
        await registry.initialize()

        registry.register_vertical("vertical1", "1.0.0", {"cap1"})
        registry.register_vertical("vertical2", "2.0.0", {"cap2"})

        # Mock one as installed, one as missing
        registry._verticals["vertical1"].status = VerticalStatus.INSTALLED
        registry._verticals["vertical2"].status = VerticalStatus.MISSING

        installed = registry.list_verticals(status=VerticalStatus.INSTALLED)
        assert len(installed) == 1
        assert installed[0].name == "vertical1"

    @pytest.mark.asyncio
    async def test_get_verticals_with_capability(self, registry):
        """Test getting verticals by capability."""
        await registry.initialize()

        registry.register_vertical("vertical1", "1.0.0", {"coding", "analysis"})
        registry.register_vertical("vertical2", "2.0.0", {"coding"})
        registry.register_vertical("vertical3", "3.0.0", {"analysis"})

        coding_verticals = registry.get_verticals_with_capability("coding")
        assert len(coding_verticals) == 2
        names = {v.name for v in coding_verticals}
        assert names == {"vertical1", "vertical2"}

    @pytest.mark.asyncio
    async def test_register_vertical_applies_shared_compatibility_gate(self, registry, monkeypatch):
        """Installed verticals should be marked incompatible when the shared gate fails."""

        await registry.initialize()
        monkeypatch.setattr(registry, "_check_installed_version", lambda _: "1.0.0")
        monkeypatch.setattr(
            "victor.core.verticals.compatibility_gate.get_framework_version",
            lambda: "1.2.3",
        )

        registry.register_vertical(
            "future-vertical",
            "1.0.0",
            {"cap1"},
            metadata={"min_framework_version": ">=9.9.9"},
        )

        info = registry.get_vertical("future-vertical")
        assert info is not None
        assert info.status == VerticalStatus.INCOMPATIBLE
        assert info.metadata["compatibility"]["compatible"] is False

    @pytest.mark.asyncio
    async def test_load_entry_point_registers_plugin_vertical(self, registry, monkeypatch):
        """Plugin entry points should register verticals through the unified registry context."""

        await registry.initialize()
        monkeypatch.setattr(registry, "_check_installed_version", lambda _: "1.2.3")

        class _TestVertical(SdkVerticalBase):
            name = "test-plugin-vertical"
            description = "Test plugin vertical"
            version = "1.2.3"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return cls.description

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "test"

        class _Plugin:
            def register(self, context):
                context.register_vertical(_TestVertical)

        class _EntryPoint:
            name = "test-plugin"
            value = "test_plugin:plugin"

            @staticmethod
            def load():
                return _Plugin()

        await registry._load_entry_point(_EntryPoint())

        info = registry.get_vertical("test-plugin-vertical")
        assert info is not None
        assert info.status == VerticalStatus.INSTALLED
        assert info.version == "1.2.3"
        assert info.entry_point == "test_plugin:plugin"
        assert info.metadata["plugin_name"] == "test-plugin"

    @pytest.mark.asyncio
    async def test_load_from_entry_point_group_uses_shared_registry(self, registry, monkeypatch):
        """Unified registry discovery should reuse the shared entry-point registry."""

        class _EntryPoint:
            def __init__(self, name: str, value: str) -> None:
                self.name = name
                self.value = value

        class _Group:
            def __init__(self) -> None:
                self.entry_points = {
                    "first": (_EntryPoint("first", "pkg:first"), False),
                    "second": (_EntryPoint("second", "pkg:second"), False),
                }

        class _Registry:
            @staticmethod
            def get_group(group_name: str):
                assert group_name == "victor.plugins"
                return _Group()

        seen: list[str] = []

        async def _load_entry_point(entry_point):
            seen.append(entry_point.name)

        monkeypatch.setattr(
            "victor.verticals.unified_registry.get_entry_point_registry",
            lambda: _Registry(),
        )
        monkeypatch.setattr(registry, "_load_entry_point", _load_entry_point)

        await registry._load_from_entry_point_group("victor.plugins")

        assert seen == ["first", "second"]

    @pytest.mark.asyncio
    async def test_get_capabilities(self, registry):
        """Test getting all capabilities."""
        await registry.initialize()

        registry.register_vertical("vertical1", "1.0.0", {"coding", "analysis"})
        registry.register_vertical("vertical2", "2.0.0", {"testing"})

        capabilities = registry.get_capabilities()
        assert capabilities == {"coding", "analysis", "testing"}

    @pytest.mark.asyncio
    async def test_is_bundle_available(self, registry):
        """Test checking if bundle is available."""
        await registry.initialize()

        # Register engineering bundle verticals as installed
        registry.register_vertical("victor-coding", "1.0.0", {"coding"})
        registry.register_vertical("victor-devops", "1.0.0", {"devops"})

        # Mark as installed
        registry._verticals["victor-coding"].status = VerticalStatus.INSTALLED
        registry._verticals["victor-devops"].status = VerticalStatus.INSTALLED

        assert registry.is_bundle_available("engineering")

    @pytest.mark.asyncio
    async def test_is_bundle_available_missing(self, registry):
        """Test checking bundle availability with missing verticals."""
        await registry.initialize()

        # Register only one of two required verticals
        registry.register_vertical("victor-coding", "1.0.0", {"coding"})
        registry._verticals["victor-coding"].status = VerticalStatus.INSTALLED

        assert not registry.is_bundle_available("engineering")

    @pytest.mark.asyncio
    async def test_get_missing_verticals_for_bundle(self, registry):
        """Test getting missing verticals for bundle."""
        await registry.initialize()

        # Register only one of two required verticals
        registry.register_vertical("victor-coding", "1.0.0", {"coding"})
        registry._verticals["victor-coding"].status = VerticalStatus.INSTALLED

        missing = registry.get_missing_verticals_for_bundle("engineering")
        assert missing == ["victor-devops"]

    @pytest.mark.asyncio
    async def test_reset(self, registry):
        """Test resetting registry."""
        await registry.initialize()

        registry.register_vertical("test", "1.0.0", {"cap"})
        assert len(registry._verticals) > 0

        registry.reset()

        assert len(registry._verticals) == 0
        assert not registry._initialized


class TestVerticalInfo:
    """Test suite for VerticalInfo."""

    def test_vertical_info_creation(self):
        """Test creating vertical info."""
        info = VerticalInfo(
            name="test",
            version="1.0.0",
            installed_version="1.0.0",
            status=VerticalStatus.INSTALLED,
            capabilities={"coding"},
            entry_point="test:Test",
            metadata={"description": "Test"},
        )

        assert info.name == "test"
        assert info.version == "1.0.0"
        assert info.status == VerticalStatus.INSTALLED

    def test_is_available(self):
        """Test is_available method."""
        info_installed = VerticalInfo(
            name="test",
            version="1.0.0",
            installed_version="1.0.0",
            status=VerticalStatus.INSTALLED,
            capabilities=set(),
            entry_point=None,
            metadata={},
        )

        info_missing = VerticalInfo(
            name="test",
            version="1.0.0",
            installed_version=None,
            status=VerticalStatus.MISSING,
            capabilities=set(),
            entry_point=None,
            metadata={},
        )

        assert info_installed.is_available()
        assert not info_missing.is_available()


class TestGlobalRegistry:
    """Test suite for global registry singleton."""

    @pytest.mark.asyncio
    async def test_get_registry_singleton(self):
        """Test that get_registry returns singleton."""
        reset_registry()

        registry1 = await get_registry()
        registry2 = await get_registry()

        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_reset_registry(self):
        """Test resetting global registry."""
        registry1 = await get_registry()
        reset_registry()
        registry2 = await get_registry()

        assert registry1 is not registry2

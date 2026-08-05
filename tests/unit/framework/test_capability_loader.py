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

"""Tests for dynamic capability loading (Phase 4.4)."""

import pytest
import tempfile
from pathlib import Path
from typing import List

from victor.framework.capability_loader import (
    CapabilityLoader,
    CapabilityEntry,
    CapabilityLoadError,
    capability,
    create_capability_loader,
    get_default_capability_loader,
)
from victor.framework.protocols import (
    CapabilityType,
    OrchestratorCapability,
)
from victor.agent.capability_registry import CapabilityRegistryMixin


class TestCapabilityLoader:
    """Tests for CapabilityLoader class."""

    def test_create_loader(self):
        """Test basic loader creation."""
        loader = CapabilityLoader()
        assert loader is not None
        assert loader.list_capabilities() == []

    def test_register_capability_manual(self):
        """Test manual capability registration."""
        loader = CapabilityLoader()

        def my_handler(data):
            return f"Processed: {data}"

        loader.register_capability(
            name="test_cap",
            handler=my_handler,
            capability_type=CapabilityType.TOOL,
            version="1.0",
            description="A test capability",
        )

        assert loader.has_capability("test_cap")
        entry = loader.get_capability("test_cap")
        assert entry is not None
        assert entry.name == "test_cap"
        assert entry.version == "1.0"
        assert entry.capability_type == CapabilityType.TOOL

    def test_unregister_capability(self):
        """Test capability unregistration."""
        loader = CapabilityLoader()

        loader.register_capability(
            name="to_remove",
            handler=lambda x: x,
            capability_type=CapabilityType.SAFETY,
        )

        assert loader.has_capability("to_remove")
        assert loader.unregister_capability("to_remove") is True
        assert not loader.has_capability("to_remove")

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent capability."""
        loader = CapabilityLoader()
        assert loader.unregister_capability("nonexistent") is False

    def test_list_capabilities(self):
        """Test listing capabilities."""
        loader = CapabilityLoader()

        loader.register_capability("cap1", lambda x: x)
        loader.register_capability("cap2", lambda x: x)
        loader.register_capability("cap3", lambda x: x)

        caps = loader.list_capabilities()
        assert len(caps) == 3
        assert "cap1" in caps
        assert "cap2" in caps
        assert "cap3" in caps

    def test_get_capabilities_by_type(self):
        """Test filtering capabilities by type."""
        loader = CapabilityLoader()

        loader.register_capability("tool1", lambda x: x, capability_type=CapabilityType.TOOL)
        loader.register_capability("safety1", lambda x: x, capability_type=CapabilityType.SAFETY)
        loader.register_capability("tool2", lambda x: x, capability_type=CapabilityType.TOOL)

        tools = loader.get_capabilities_by_type(CapabilityType.TOOL)
        assert len(tools) == 2

        safety = loader.get_capabilities_by_type(CapabilityType.SAFETY)
        assert len(safety) == 1


class TestCapabilityDecorator:
    """Tests for the @capability decorator."""

    def test_decorator_adds_metadata(self):
        """Test that decorator adds metadata to function."""

        @capability(
            name="decorated_cap",
            capability_type=CapabilityType.SAFETY,
            version="2.0",
            description="A decorated capability",
        )
        def my_func(patterns):
            return patterns

        assert hasattr(my_func, "_capability_meta")
        meta = my_func._capability_meta
        assert meta["name"] == "decorated_cap"
        assert meta["capability_type"] == CapabilityType.SAFETY
        assert meta["version"] == "2.0"

    def test_decorator_preserves_function(self):
        """Test that decorator preserves function behavior."""

        @capability(name="test_cap")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10


class TestCapabilityEntry:
    """Tests for CapabilityEntry dataclass."""

    def test_entry_from_capability(self):
        """Test creating entry from OrchestratorCapability."""
        cap = OrchestratorCapability(
            name="test",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="1.1",
        )
        entry = CapabilityEntry(
            capability=cap,
            handler=lambda x: x,
            source_module="test_module",
        )

        assert entry.name == "test"
        assert entry.version == "1.1"
        assert entry.capability_type == CapabilityType.TOOL
        assert entry.source_module == "test_module"


class TestModuleLoading:
    """Tests for loading capabilities from modules."""

    def test_load_from_path(self):
        """Test loading capabilities from a file path."""
        # Create a temporary capability module
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry

def my_handler(data):
    return f"Handler: {data}"

CAPABILITIES = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="file_loaded_cap",
            capability_type=CapabilityType.TOOL,
            setter="apply_file_loaded",
            version="1.0",
        ),
        handler=my_handler,
    )
]
""")
            temp_path = f.name

        try:
            loader = CapabilityLoader()
            loaded = loader.load_from_path(temp_path)

            assert "file_loaded_cap" in loaded
            assert loader.has_capability("file_loaded_cap")
            entry = loader.get_capability("file_loaded_cap")
            assert entry.handler is not None
        finally:
            Path(temp_path).unlink()

    def test_load_from_path_nonexistent(self):
        """Test loading from non-existent path raises error."""
        loader = CapabilityLoader()
        with pytest.raises(CapabilityLoadError):
            loader.load_from_path("/nonexistent/path/to/module.py")


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_capability_loader(self):
        """Test create_capability_loader factory."""
        loader = create_capability_loader()
        assert isinstance(loader, CapabilityLoader)

    def test_get_default_capability_loader(self):
        """Test get_default_capability_loader singleton."""
        loader1 = get_default_capability_loader()
        loader2 = get_default_capability_loader()
        assert loader1 is loader2


class TestDynamicRegistration:
    """Tests for dynamic capability registration in CapabilityRegistryMixin."""

    def test_register_dynamic_capability(self):
        """Test registering a dynamic capability on an orchestrator."""

        class MockOrchestrator(CapabilityRegistryMixin):
            def __init__(self):
                self.__init_capability_registry__()

        orch = MockOrchestrator()

        def custom_handler(data):
            return f"Custom: {data}"

        cap = OrchestratorCapability(
            name="dynamic_cap",
            capability_type=CapabilityType.SAFETY,
            setter="apply_dynamic",
            version="1.0",
        )

        result = orch.register_dynamic_capability(cap, setter_method=custom_handler)
        assert result is True
        assert orch.has_capability("dynamic_cap")
        assert orch.is_dynamic_capability("dynamic_cap")

    def test_unregister_dynamic_capability(self):
        """Test unregistering a dynamic capability."""

        class MockOrchestrator(CapabilityRegistryMixin):
            def __init__(self):
                self.__init_capability_registry__()

        orch = MockOrchestrator()

        cap = OrchestratorCapability(
            name="to_unregister",
            capability_type=CapabilityType.TOOL,
            setter="apply_it",
        )
        orch.register_dynamic_capability(cap, setter_method=lambda x: x)

        assert orch.unregister_dynamic_capability("to_unregister") is True
        assert not orch.has_capability("to_unregister")
        assert not orch.is_dynamic_capability("to_unregister")

    def test_cannot_unregister_builtin(self):
        """Test that built-in capabilities cannot be unregistered."""

        class MockOrchestrator(CapabilityRegistryMixin):
            def __init__(self):
                self.__init_capability_registry__()

        orch = MockOrchestrator()

        # enabled_tools is a built-in capability
        with pytest.raises(ValueError):
            orch.unregister_dynamic_capability("enabled_tools")

    def test_get_dynamic_capabilities(self):
        """Test getting list of dynamic capabilities."""

        class MockOrchestrator(CapabilityRegistryMixin):
            def __init__(self):
                self.__init_capability_registry__()

        orch = MockOrchestrator()

        # Register multiple dynamic capabilities
        for i in range(3):
            cap = OrchestratorCapability(
                name=f"dynamic_{i}",
                capability_type=CapabilityType.TOOL,
                setter=f"apply_{i}",
            )
            orch.register_dynamic_capability(cap, setter_method=lambda x: x)

        dynamic_caps = orch.get_dynamic_capabilities()
        assert len(dynamic_caps) == 3
        assert "dynamic_0" in dynamic_caps
        assert "dynamic_1" in dynamic_caps
        assert "dynamic_2" in dynamic_caps


class TestLoaderOrchestratorIntegration:
    """Tests for CapabilityLoader + Orchestrator integration."""

    def test_apply_loaded_capabilities_to_orchestrator(self):
        """Test applying loaded capabilities to an orchestrator."""

        class MockOrchestrator(CapabilityRegistryMixin):
            def __init__(self):
                self.__init_capability_registry__()

        orch = MockOrchestrator()
        loader = CapabilityLoader()

        # Register capabilities in loader
        loader.register_capability(
            name="plugin_cap",
            handler=lambda x: x,
            capability_type=CapabilityType.TOOL,
        )

        # Apply to orchestrator
        applied = loader.apply_to(orch)

        assert "plugin_cap" in applied
        assert orch.has_capability("plugin_cap")

    def test_load_capabilities_from_loader_method(self):
        """Test using orchestrator's load_capabilities_from_loader method."""

        class MockOrchestrator(CapabilityRegistryMixin):
            def __init__(self):
                self.__init_capability_registry__()

        orch = MockOrchestrator()
        loader = CapabilityLoader()

        loader.register_capability(
            name="via_method_cap",
            handler=lambda x: x,
        )

        applied = orch.load_capabilities_from_loader(loader)
        assert "via_method_cap" in applied

    def test_apply_to_prefers_public_register_dynamic_capability(self):
        """Loader should use public register_dynamic_capability when available."""

        class MockOrchestrator:
            def __init__(self):
                self.calls = []

            def register_dynamic_capability(
                self,
                capability,
                setter_method=None,
                getter_method=None,
            ):
                self.calls.append((capability.name, setter_method, getter_method))
                return True

            def _register_capability(self, *args, **kwargs):  # pragma: no cover - guard rail
                raise AssertionError("_register_capability should not be used")

        orch = MockOrchestrator()
        loader = CapabilityLoader()

        loader.register_capability(
            name="public_cap",
            handler=lambda x: x,
            capability_type=CapabilityType.TOOL,
        )

        applied = loader.apply_to(orch)

        assert applied == ["public_cap"]
        assert len(orch.calls) == 1
        assert orch.calls[0][0] == "public_cap"

    def test_apply_to_blocks_private_registration_fallback_in_strict_mode(self, monkeypatch):
        """Strict mode should block private registration fallback paths."""
        monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", "1")

        class LegacyOrchestrator:
            def _register_capability(self, *args, **kwargs):
                return None

        loader = CapabilityLoader()
        loader.register_capability(
            name="legacy_cap",
            handler=lambda x: x,
            capability_type=CapabilityType.TOOL,
        )

        with pytest.raises(RuntimeError, match="Private attribute fallback blocked"):
            loader.apply_to(LegacyOrchestrator())


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager(self):
        """Test using loader as context manager."""
        with CapabilityLoader() as loader:
            loader.register_capability("test", lambda x: x)
            assert loader.has_capability("test")
        # After exit, loader should be cleaned up (watcher stopped)


class TestFrameworkExports:
    """Tests for framework module exports."""

    def test_capability_loader_exported(self):
        """Test that CapabilityLoader is exported from framework."""
        from victor.framework import CapabilityLoader as ExportedLoader

        assert ExportedLoader is CapabilityLoader

    def test_capability_decorator_exported(self):
        """Test that capability decorator is exported from framework."""
        from victor.framework import capability as exported_capability

        assert exported_capability is capability

    def test_capability_entry_exported(self):
        """Test that CapabilityEntry is exported from framework."""
        from victor.framework import CapabilityEntry as ExportedEntry

        assert ExportedEntry is CapabilityEntry


class TestCapabilityPathKeys:
    """Tests for Path object keys in discover_capabilities — Wave M."""

    def test_discover_capabilities_returns_path_keys(self, tmp_path):
        """discover_capabilities should return Path objects as dict keys."""
        from victor.framework.capability_loader import CapabilityLoader

        # Create a temporary capability file
        caps_dir = tmp_path / "caps"
        caps_dir.mkdir()
        (caps_dir / "__init__.py").write_text("# test caps")
        (caps_dir / "test_capabilities.py").write_text("""
from victor.framework.protocols import OrchestratorCapability, CapabilityType

test_cap = OrchestratorCapability(
    name="test_capability",
    capability_type=CapabilityType.TOOL,
    setter="apply_test",
)
""")

        loader = CapabilityLoader()
        loader.add_plugin_dir(caps_dir)

        results = loader.discover_capabilities()

        # Should have Path keys, not string keys
        assert len(results) > 0
        for key in results.keys():
            assert isinstance(key, Path), f"Expected Path key, got {type(key)}"

    def test_capability_lookup_by_path_object(self, tmp_path):
        """Should be able to lookup capabilities using Path objects."""
        from victor.framework.capability_loader import CapabilityLoader

        # Create a temporary capability file
        caps_dir = tmp_path / "caps"
        caps_dir.mkdir()
        (caps_dir / "__init__.py").write_text("# test caps")
        caps_file = caps_dir / "test_capabilities.py"
        caps_file.write_text("""
from victor.framework.protocols import OrchestratorCapability, CapabilityType

test_cap = OrchestratorCapability(
    name="test_capability",
    capability_type=CapabilityType.TOOL,
    setter="apply_test",
)
""")

        loader = CapabilityLoader()
        loader.add_plugin_dir(caps_dir)

        results = loader.discover_capabilities()

        # Lookup using the Path object should work
        assert caps_file in results
        capability = results[caps_file]
        assert capability is not None

    def test_capability_lookup_by_string_fails(self, tmp_path):
        """String keys should not work for Path-based dict (if Path keys are used)."""
        from victor.framework.capability_loader import CapabilityLoader

        # Create a temporary capability file
        caps_dir = tmp_path / "caps"
        caps_dir.mkdir()
        (caps_dir / "__init__.py").write_text("# test caps")
        caps_file = caps_dir / "test_capabilities.py"
        caps_file.write_text("""
from victor.framework.protocols import OrchestratorCapability, CapabilityType

test_cap = OrchestratorCapability(
    name="test_capability",
    capability_type=CapabilityType.TOOL,
    setter="apply_test",
)
""")

        loader = CapabilityLoader()
        loader.add_plugin_dir(caps_dir)

        results = loader.discover_capabilities()

        # String key should not find the capability (keys are Path objects)
        str_key = str(caps_file)
        assert str_key not in results

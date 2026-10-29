"""TDD tests for decoupling Core VerticalBase from Framework imports.

Verifies that importing victor.core.verticals.base does NOT eagerly
load victor.framework.tools or victor.framework.capabilities.
"""

import importlib
import sys


class TestCoreFrameworkDecoupling:
    """Core VerticalBase must not eagerly import Framework modules."""

    def test_import_does_not_load_framework_tools(self):
        """Importing base should not pull in victor.framework.tools."""
        # Save original modules so we can restore them (avoid split-brain
        # where other test files hold references to the old VerticalRegistry)
        saved_mods = {}
        mods_to_clear = [k for k in sys.modules if "victor.core.verticals.base" in k]
        for m in mods_to_clear:
            saved_mods[m] = sys.modules.pop(m)

        # Also clear the framework module if loaded
        fwk_key = "victor.framework.tools"
        was_loaded = fwk_key in sys.modules
        if was_loaded:
            saved_mods[fwk_key] = sys.modules.pop(fwk_key)

        try:
            importlib.import_module("victor.core.verticals.base")
            assert fwk_key not in sys.modules, (
                "victor.framework.tools was eagerly loaded by importing "
                "victor.core.verticals.base — should use lazy imports"
            )
        finally:
            # Restore ALL original modules to prevent split-brain VerticalRegistry
            # (new module would have a fresh class with empty _registry)
            for key, mod in saved_mods.items():
                sys.modules[key] = mod

    def test_import_does_not_load_framework_capabilities(self):
        """Importing base should not pull in victor.framework.capabilities."""
        saved_mods = {}
        mods_to_clear = [k for k in sys.modules if "victor.core.verticals.base" in k]
        for m in mods_to_clear:
            saved_mods[m] = sys.modules.pop(m)

        cap_key = "victor.framework.capabilities"
        was_loaded = cap_key in sys.modules
        if was_loaded:
            saved_mods[cap_key] = sys.modules.pop(cap_key)

        try:
            importlib.import_module("victor.core.verticals.base")
            assert cap_key not in sys.modules, (
                "victor.framework.capabilities was eagerly loaded by importing "
                "victor.core.verticals.base — should use lazy imports"
            )
        finally:
            # Restore ALL original modules to prevent split-brain VerticalRegistry
            for key, mod in saved_mods.items():
                sys.modules[key] = mod

    def test_get_config_still_works(self):
        """get_config() should still produce valid VerticalConfig."""
        from victor.core.verticals.base import VerticalBase

        class TestVertical(VerticalBase):
            name = "test-decoupled"
            description = "test"

            @classmethod
            def get_name(cls):
                return "test-decoupled"

            @classmethod
            def get_description(cls):
                return "test"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "You are a test."

        config = TestVertical.get_config(use_cache=False)
        assert config is not None
        assert "read" in str(config)

    def test_get_stage_capability_lazy_loads(self):
        """_get_stage_capability() should lazy-load framework capabilities."""
        from victor.core.verticals.base import VerticalBase

        # The capability should be loadable via the lazy path
        cap = VerticalBase._get_stage_capability()
        assert cap is not None
        assert hasattr(cap, "get_stages")

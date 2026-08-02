# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Guard tests for FEP-0015: internal step_handlers symbols are unexported.

`CapabilityConfigStepHandler` and `ExtensionHandler` are internal-only and must
not be advertised as framework public API. The Phase 1 `DeprecationWarning`
shim has been removed (Phase 2): the old `ExtensionHandler` name now raises
`ImportError`/`AttributeError`; the internal name is `_ExtensionHandler`.
"""

import pytest

import victor.framework as framework
import victor.framework.step_handlers as step_handlers


class TestStepHandlersExportsFEP0015:
    """Verify FEP-0015 public-surface trimming (Phase 2: shim removed)."""

    @pytest.mark.parametrize("name", ["ExtensionHandler", "CapabilityConfigStepHandler"])
    def test_not_in_step_handlers_all(self, name: str) -> None:
        """Neither symbol appears in step_handlers.__all__."""
        assert name not in step_handlers.__all__

    @pytest.mark.parametrize("name", ["ExtensionHandler", "CapabilityConfigStepHandler"])
    def test_not_reexported_by_framework(self, name: str) -> None:
        """Neither symbol is re-exported by victor.framework."""
        assert name not in dir(framework)
        framework_all = getattr(framework, "__all__", None)
        if framework_all is not None:
            assert name not in framework_all

    def test_extension_handler_old_name_raises(self) -> None:
        """The old ExtensionHandler name is gone (shim removed in Phase 2)."""
        with pytest.raises(ImportError):
            from victor.framework.step_handlers import ExtensionHandler  # noqa: F401

        with pytest.raises(AttributeError):
            step_handlers.ExtensionHandler  # noqa: B018

    def test_capability_config_still_importable(self) -> None:
        """CapabilityConfigStepHandler stays importable (unrenamed, just unexported).

        It is not renamed (per FEP-0015 scope), so it remains a real module
        attribute; in-tree importers keep working with no warning noise.
        """
        from victor.framework.step_handlers import CapabilityConfigStepHandler

        assert CapabilityConfigStepHandler is step_handlers.__dict__["CapabilityConfigStepHandler"]

    def test_unknown_attribute_still_raises(self) -> None:
        """Genuine attribute errors are raised as usual."""
        with pytest.raises(AttributeError):
            step_handlers.DefinitelyNotARealSymbol  # noqa: B018

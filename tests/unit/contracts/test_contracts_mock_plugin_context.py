# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract shape tests for MockPluginContext — ensures SDK testing exports are stable."""

from __future__ import annotations


class TestMockPluginContextContractShape:
    def test_exported_from_victor_contracts_testing(self):
        from victor_contracts.testing import MockPluginContext

        assert MockPluginContext is not None

    def test_has_registered_tools_attribute(self):
        from victor_contracts.testing import MockPluginContext

        ctx = MockPluginContext()
        assert hasattr(ctx, "registered_tools")
        assert isinstance(ctx.registered_tools, list)

    def test_has_set_service_method(self):
        from victor_contracts.testing import MockPluginContext

        ctx = MockPluginContext()
        assert callable(getattr(ctx, "set_service", None))

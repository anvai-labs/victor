# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Tests for step handler contract validation during registration.

StepHandlerRegistry.add_handler() should validate capability_contracts
and warn about unsatisfiable contracts without breaking registration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from victor.framework.step_handlers import (
    BaseStepHandler,
    StepHandlerRegistry,
    ToolStepHandler,
)


class _MockHandler(BaseStepHandler):
    """Test handler with configurable contracts."""

    depends_on = ()
    side_effects = False

    def __init__(self, name: str, order: int, contracts: tuple = ()):
        self._name = name
        self._order = order
        self.capability_contracts = contracts

    @property
    def name(self) -> str:
        return self._name

    @property
    def order(self) -> int:
        return self._order

    def _do_apply(self, orchestrator, vertical, context, result):
        pass


class TestStepHandlerContractValidation:
    """Validate capability_contracts on handler registration."""

    def test_valid_handler_contracts_pass(self, caplog):
        """Handler with reasonable contracts registers without warnings."""
        registry = StepHandlerRegistry(handlers=[])
        handler = _MockHandler("test", order=50, contracts=(("tools", 2, ">=0.5.0"),))

        with caplog.at_level(logging.WARNING):
            registry.add_handler(handler)

        assert registry.get_handler("test") is not None
        # No contract warnings for reasonable version
        contract_warnings = [r for r in caplog.records if "contract" in r.message.lower()]
        assert len(contract_warnings) == 0

    def test_invalid_version_contract_warns(self, caplog):
        """Handler declaring impossible version emits warning."""
        registry = StepHandlerRegistry(handlers=[])
        handler = _MockHandler("bad_version", order=50, contracts=(("tools", 999, ">=999.0.0"),))

        with caplog.at_level(logging.WARNING):
            registry.add_handler(handler)

        # Handler still registered (non-blocking)
        assert registry.get_handler("bad_version") is not None
        # But warning emitted
        contract_warnings = [r for r in caplog.records if "contract" in r.message.lower()]
        assert len(contract_warnings) >= 1

    def test_empty_contracts_always_pass(self, caplog):
        """BaseStepHandler with empty contracts registers cleanly."""
        registry = StepHandlerRegistry(handlers=[])
        handler = _MockHandler("no_contracts", order=50, contracts=())

        with caplog.at_level(logging.WARNING):
            registry.add_handler(handler)

        assert registry.get_handler("no_contracts") is not None
        contract_warnings = [r for r in caplog.records if "contract" in r.message.lower()]
        assert len(contract_warnings) == 0

    def test_default_registry_validates(self):
        """StepHandlerRegistry.default() creates without contract errors."""
        # This should not raise — all built-in handlers have valid contracts
        registry = StepHandlerRegistry.default()
        assert len(registry.handlers) > 0

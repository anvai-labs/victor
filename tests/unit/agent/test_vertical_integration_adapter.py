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

"""Tests for VerticalIntegrationAdapter — DIP compliance and capability protocol paths.

Covers:
- apply_middleware routes through _invoke_capability (DIP write path)
- apply_safety_patterns routes through _invoke_capability (DIP write path)
- Graceful degradation when orchestrator has no capability registry
- ISP role-protocol structural subtype checks
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, call

import pytest

from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter
from victor.framework.protocols import (
    CapabilityDiscoveryProtocol,
    CapabilityMutationProtocol,
    CapabilityReadProtocol,
    CapabilityRegistryProtocol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    *,
    has_capability_registry: bool = True,
    capability_values: Optional[Dict[str, Any]] = None,
):
    """Build a mock orchestrator with a configurable capability registry."""
    capability_values = capability_values or {}
    orch = MagicMock()

    if has_capability_registry:
        orch.has_capability.side_effect = lambda name: name in capability_values
        orch.get_capability_value.side_effect = lambda name: capability_values.get(name)
        orch.invoke_capability.return_value = None  # success, no meaningful return
    else:
        del orch.has_capability
        del orch.get_capability_value
        del orch.invoke_capability

    return orch


def _make_middleware():
    mw = MagicMock()
    mw.__class__.__name__ = "MockMiddleware"
    return mw


def _make_pattern(pattern: str = r"rm -rf", risk_level: str = "high"):
    p = MagicMock()
    p.pattern = pattern
    p.description = f"test pattern: {pattern}"
    p.risk_level = risk_level
    p.category = "custom"
    return p


# ---------------------------------------------------------------------------
# ISP Role-Protocol structural subtype checks
# ---------------------------------------------------------------------------


class TestRoleProtocols:
    """Verify that the new role protocols are proper structural subtypes
    and that OrchestratorCapabilityMixin satisfies all of them."""

    def test_capability_discovery_protocol_is_runtime_checkable(self):
        from victor.agent.capability_registry import OrchestratorCapabilityMixin

        class _Stub(OrchestratorCapabilityMixin):
            # Satisfy the abstract attribute initialiser
            def __init__(self):
                self.__init_capability_registry__()

        stub = _Stub()
        assert isinstance(stub, CapabilityDiscoveryProtocol)

    def test_capability_read_protocol_is_runtime_checkable(self):
        from victor.agent.capability_registry import OrchestratorCapabilityMixin

        class _Stub(OrchestratorCapabilityMixin):
            def __init__(self):
                self.__init_capability_registry__()

        assert isinstance(_Stub(), CapabilityReadProtocol)

    def test_capability_mutation_protocol_is_runtime_checkable(self):
        from victor.agent.capability_registry import OrchestratorCapabilityMixin

        class _Stub(OrchestratorCapabilityMixin):
            def __init__(self):
                self.__init_capability_registry__()

        assert isinstance(_Stub(), CapabilityMutationProtocol)

    def test_capability_registry_protocol_still_valid(self):
        """Composite CapabilityRegistryProtocol unchanged — backward compat."""
        from victor.agent.capability_registry import OrchestratorCapabilityMixin

        class _Stub(OrchestratorCapabilityMixin):
            def __init__(self):
                self.__init_capability_registry__()

        assert isinstance(_Stub(), CapabilityRegistryProtocol)

    def test_object_without_methods_fails_discovery_protocol(self):
        assert not isinstance(object(), CapabilityDiscoveryProtocol)

    def test_object_without_methods_fails_read_protocol(self):
        assert not isinstance(object(), CapabilityReadProtocol)

    def test_object_without_methods_fails_mutation_protocol(self):
        assert not isinstance(object(), CapabilityMutationProtocol)


# ---------------------------------------------------------------------------
# apply_middleware — DIP write path
# ---------------------------------------------------------------------------


class TestApplyMiddlewareDIP:
    def test_empty_middleware_returns_early(self):
        orch = _make_orchestrator()
        adapter = VerticalIntegrationAdapter(orch)
        adapter.apply_middleware([])
        orch.invoke_capability.assert_not_called()

    def test_middleware_stored_via_invoke_capability(self):
        """Middleware persistence must go through invoke_capability, not getattr."""
        mock_chain = MagicMock()
        orch = _make_orchestrator(capability_values={"middleware_chain": mock_chain})
        # invoke_capability should succeed (returns without raising)
        orch.invoke_capability.return_value = None

        adapter = VerticalIntegrationAdapter(orch)
        mw = _make_middleware()
        adapter.apply_middleware([mw])

        # invoke_capability must have been called for middleware storage
        capability_names = [c.args[0] for c in orch.invoke_capability.call_args_list]
        assert "middleware" in capability_names

    def test_middleware_added_to_existing_chain(self):
        """When a chain already exists, middleware is appended via chain.add()."""
        mock_chain = MagicMock()
        mock_chain.add = MagicMock()
        orch = _make_orchestrator(capability_values={"middleware_chain": mock_chain})

        adapter = VerticalIntegrationAdapter(orch)
        mw1, mw2 = _make_middleware(), _make_middleware()
        adapter.apply_middleware([mw1, mw2])

        assert mock_chain.add.call_count == 2

    def test_graceful_degradation_without_capability_registry(self):
        """When orchestrator has no capability registry, adapter must not raise."""
        orch = _make_orchestrator(has_capability_registry=False)
        # Ensure the fallback path (set_middleware via getattr) doesn't blow up
        orch.set_middleware = MagicMock()

        adapter = VerticalIntegrationAdapter(orch)
        mw = _make_middleware()
        # Should not raise regardless of chain/middleware_chain availability
        try:
            adapter.apply_middleware([mw])
        except Exception as exc:
            pytest.fail(f"apply_middleware raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# apply_safety_patterns — DIP write path
# ---------------------------------------------------------------------------


class TestApplySafetyPatternsDIP:
    def test_empty_patterns_returns_early(self):
        orch = _make_orchestrator()
        adapter = VerticalIntegrationAdapter(orch)
        adapter.apply_safety_patterns([])
        orch.invoke_capability.assert_not_called()

    def test_patterns_stored_via_invoke_capability(self):
        """Safety-pattern persistence must go through invoke_capability."""
        mock_checker = MagicMock()
        orch = _make_orchestrator(
            capability_values={
                "safety_checker": mock_checker,
                "safety_patterns": mock_checker,  # capability read path
            }
        )
        orch.invoke_capability.return_value = None

        adapter = VerticalIntegrationAdapter(orch)
        pattern = _make_pattern()
        adapter.apply_safety_patterns([pattern])

        capability_names = [c.args[0] for c in orch.invoke_capability.call_args_list]
        assert "safety_patterns" in capability_names

    def test_patterns_applied_to_checker_via_add_custom_pattern(self):
        """Patterns applied to safety checker when checker supports add_custom_pattern."""
        mock_checker = MagicMock()
        mock_checker.add_custom_pattern = MagicMock()
        orch = _make_orchestrator(capability_values={"safety_checker": mock_checker})

        adapter = VerticalIntegrationAdapter(orch)
        pattern = _make_pattern(r"sudo .*", risk_level="critical")
        adapter.apply_safety_patterns([pattern])

        mock_checker.add_custom_pattern.assert_called_once_with(
            pattern=r"sudo .*",
            description=pattern.description,
            risk_level="critical",
            category="custom",
        )

    def test_patterns_applied_via_add_patterns_fallback(self):
        """Batch add_patterns used when add_custom_pattern is not available."""
        mock_checker = MagicMock(spec=["add_patterns"])
        mock_checker.add_patterns = MagicMock()
        orch = _make_orchestrator(capability_values={"safety_checker": mock_checker})

        adapter = VerticalIntegrationAdapter(orch)
        patterns = [_make_pattern(), _make_pattern()]
        adapter.apply_safety_patterns(patterns)

        mock_checker.add_patterns.assert_called_once_with(patterns)

    def test_graceful_degradation_without_capability_registry(self):
        """No capability registry must not raise — safety is non-critical."""
        orch = _make_orchestrator(has_capability_registry=False)
        orch.set_safety_patterns = MagicMock()

        adapter = VerticalIntegrationAdapter(orch)
        pattern = _make_pattern()
        try:
            adapter.apply_safety_patterns([pattern])
        except Exception as exc:
            pytest.fail(f"apply_safety_patterns raised unexpectedly: {exc}")

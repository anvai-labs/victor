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

"""Tests for capability registry mixin.

Tests the CapabilityRegistryMixin which provides explicit capability
discovery and invocation, replacing hasattr duck-typing.
"""

import pytest
from typing import Any, Dict, Optional, Set

from victor.framework.protocols import (
    CapabilityType,
    OrchestratorCapability,
    CapabilityRegistryProtocol,
    IncompatibleVersionError,
)
from victor.agent.capability_registry import CapabilityRegistryMixin

# =============================================================================
# Test Fixtures
# =============================================================================


class MockOrchestrator(CapabilityRegistryMixin):
    """Mock orchestrator for testing capability registry."""

    def __init__(self):
        # Simulate orchestrator attributes
        self.prompt_builder = MockPromptBuilder()
        self._middleware_chain = MockMiddlewareChain()
        self._safety_checker = MockSafetyChecker()
        self._sequence_tracker = MockSequenceTracker()
        self._runtime_intelligence_integration = {"name": "runtime-intelligence"}
        self._vertical_context = None
        self._rl_hooks = None
        self._team_specs = None
        self._enabled_tools: Set[str] = set()

        # Initialize capability registry
        self.__init_capability_registry__()

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set enabled tools."""
        self._enabled_tools = tools

    def get_enabled_tools(self) -> Set[str]:
        """Get enabled tools."""
        return set(self._enabled_tools)

    def set_vertical_context(self, context: Any) -> None:
        """Set vertical context."""
        self._vertical_context = context

    def get_vertical_context(self) -> Any:
        """Get vertical context."""
        return self._vertical_context

    def apply_vertical_middleware(self, middleware: list) -> None:
        """Apply vertical middleware."""
        for mw in middleware:
            self._middleware_chain.add(mw)

    def apply_vertical_safety_patterns(self, patterns: list) -> None:
        """Apply safety patterns."""
        self._safety_checker.add_patterns(patterns)

    @property
    def runtime_intelligence_integration(self) -> Any:
        """Expose the canonical runtime-intelligence integration surface."""
        return self._runtime_intelligence_integration


class MockPromptBuilder:
    """Mock prompt builder."""

    def __init__(self):
        self._custom_prompt = None
        self._task_type_hints = {}
        self._sections = []

    def set_custom_prompt(self, prompt: str) -> None:
        self._custom_prompt = prompt

    def get_custom_prompt(self) -> Optional[str]:
        return self._custom_prompt

    def set_task_type_hints(self, hints: dict) -> None:
        self._task_type_hints = hints

    def add_prompt_section(self, section: str) -> None:
        self._sections.append(section)


class MockMiddlewareChain:
    """Mock middleware chain."""

    def __init__(self):
        self._middleware = []

    def add(self, mw: Any) -> None:
        self._middleware.append(mw)


class MockSafetyChecker:
    """Mock safety checker."""

    def __init__(self):
        self._custom_patterns = []

    def add_patterns(self, patterns: list) -> None:
        self._custom_patterns.extend(patterns)


class MockSequenceTracker:
    """Mock sequence tracker."""

    def __init__(self):
        self._dependencies = {}
        self._sequences = []

    def set_dependencies(self, deps: dict) -> None:
        self._dependencies = deps

    def set_sequences(self, seqs: list) -> None:
        self._sequences = seqs


# =============================================================================
# Test OrchestratorCapability
# =============================================================================


class TestOrchestratorCapability:
    """Tests for OrchestratorCapability dataclass."""

    def test_create_capability_with_setter(self):
        """Test creating capability with setter method."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            description="Test capability",
        )
        assert cap.name == "test_cap"
        assert cap.capability_type == CapabilityType.TOOL
        assert cap.setter == "set_test"
        assert cap.getter is None
        assert cap.attribute is None

    def test_create_capability_with_getter(self):
        """Test creating capability with getter method."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.PROMPT,
            getter="get_test",
        )
        assert cap.getter == "get_test"
        assert cap.setter is None

    def test_create_capability_with_attribute(self):
        """Test creating capability with attribute access."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.MODE,
            attribute="test_attr",
        )
        assert cap.attribute == "test_attr"

    def test_capability_requires_access_method(self):
        """Test that capability must have at least one access method."""
        with pytest.raises(ValueError, match="must specify at least one of"):
            OrchestratorCapability(
                name="invalid",
                capability_type=CapabilityType.TOOL,
            )

    def test_all_capability_types(self):
        """Test all capability types exist."""
        expected_types = [
            "TOOL",
            "PROMPT",
            "MODE",
            "SAFETY",
            "RL",
            "TEAM",
            "WORKFLOW",
            "VERTICAL",
        ]
        for type_name in expected_types:
            assert hasattr(CapabilityType, type_name)


# =============================================================================
# Test Capability Versioning
# =============================================================================


class TestCapabilityVersioning:
    """Tests for capability version functionality."""

    def test_default_version_is_1_0(self):
        """Test capabilities default to version 1.0."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
        )
        assert cap.version == "1.0"

    def test_create_capability_with_custom_version(self):
        """Test creating capability with custom version."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="2.1",
        )
        assert cap.version == "2.1"

    def test_invalid_version_format_raises_error(self):
        """Test that invalid version format raises ValueError."""
        with pytest.raises(ValueError, match="invalid version"):
            OrchestratorCapability(
                name="test_cap",
                capability_type=CapabilityType.TOOL,
                setter="set_test",
                version="invalid",
            )

    def test_invalid_version_single_number(self):
        """Test that single number version is invalid."""
        with pytest.raises(ValueError, match="invalid version"):
            OrchestratorCapability(
                name="test_cap",
                capability_type=CapabilityType.TOOL,
                setter="set_test",
                version="1",
            )

    def test_invalid_version_three_parts(self):
        """Test that three-part version is invalid."""
        with pytest.raises(ValueError, match="invalid version"):
            OrchestratorCapability(
                name="test_cap",
                capability_type=CapabilityType.TOOL,
                setter="set_test",
                version="1.2.3",
            )

    def test_is_compatible_with_same_version(self):
        """Test capability is compatible with same version."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="1.5",
        )
        assert cap.is_compatible_with("1.5")

    def test_is_compatible_with_lower_version(self):
        """Test capability is compatible with lower required version."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="2.0",
        )
        assert cap.is_compatible_with("1.0")
        assert cap.is_compatible_with("1.5")

    def test_is_compatible_with_lower_minor(self):
        """Test capability is compatible with lower minor version."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="1.5",
        )
        assert cap.is_compatible_with("1.0")
        assert cap.is_compatible_with("1.3")

    def test_is_not_compatible_with_higher_version(self):
        """Test capability is not compatible with higher required version."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="1.0",
        )
        assert not cap.is_compatible_with("2.0")
        assert not cap.is_compatible_with("1.5")

    def test_version_comparison_semantic(self):
        """Test version comparison is semantic (1.10 > 1.9)."""
        cap = OrchestratorCapability(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_test",
            version="1.10",
        )
        assert cap.is_compatible_with("1.9")
        assert cap.is_compatible_with("1.10")
        assert not cap.is_compatible_with("1.11")

    def test_deprecated_capability(self):
        """Test creating deprecated capability."""
        cap = OrchestratorCapability(
            name="old_cap",
            capability_type=CapabilityType.TOOL,
            setter="set_old",
            version="1.0",
            deprecated=True,
            deprecated_message="Use 'new_cap' instead",
        )
        assert cap.deprecated is True
        assert "new_cap" in cap.deprecated_message


# =============================================================================
# Test CapabilityRegistryMixin
# =============================================================================


class TestCapabilityRegistryMixin:
    """Tests for CapabilityRegistryMixin."""

    @pytest.fixture
    def orchestrator(self):
        """Create mock orchestrator with capability registry."""
        return MockOrchestrator()

    def test_has_capability_returns_true_for_registered(self, orchestrator):
        """Test has_capability returns True for registered capabilities."""
        assert orchestrator.has_capability("enabled_tools")
        assert orchestrator.has_capability("prompt_builder")
        assert orchestrator.has_capability("vertical_context")
        assert orchestrator.has_capability("runtime_intelligence_integration")
        assert orchestrator.has_capability("intelligent_pipeline")

    def test_has_capability_returns_false_for_unknown(self, orchestrator):
        """Test has_capability returns False for unknown capabilities."""
        assert not orchestrator.has_capability("unknown_capability")

    def test_get_capability_returns_capability(self, orchestrator):
        """Test get_capability returns the capability declaration."""
        cap = orchestrator.get_capability("enabled_tools")
        assert cap is not None
        assert cap.name == "enabled_tools"
        assert cap.capability_type == CapabilityType.TOOL

    def test_get_capability_returns_none_for_unknown(self, orchestrator):
        """Test get_capability returns None for unknown capability."""
        assert orchestrator.get_capability("unknown") is None

    def test_get_capabilities_returns_all(self, orchestrator):
        """Test get_capabilities returns all registered capabilities."""
        caps = orchestrator.get_capabilities()
        assert isinstance(caps, dict)
        assert len(caps) > 0
        assert "enabled_tools" in caps
        assert "prompt_builder" in caps

    def test_get_capabilities_by_type(self, orchestrator):
        """Test get_capabilities_by_type filters correctly."""
        tool_caps = orchestrator.get_capabilities_by_type(CapabilityType.TOOL)
        assert "enabled_tools" in tool_caps

        prompt_caps = orchestrator.get_capabilities_by_type(CapabilityType.PROMPT)
        assert "prompt_builder" in prompt_caps
        assert "custom_prompt" in prompt_caps

    def test_invoke_capability_calls_setter(self, orchestrator):
        """Test invoke_capability calls the setter method."""
        test_tools = {"read", "write", "grep"}
        orchestrator.invoke_capability("enabled_tools", test_tools)
        assert orchestrator._enabled_tools == test_tools

    def test_invoke_capability_raises_for_unknown(self, orchestrator):
        """Test invoke_capability raises KeyError for unknown capability."""
        with pytest.raises(KeyError, match="not found"):
            orchestrator.invoke_capability("unknown_capability", "value")

    def test_invoke_capability_raises_for_no_setter(self, orchestrator):
        """Test invoke_capability raises TypeError if no setter."""
        # prompt_builder capability has attribute only, no setter
        with pytest.raises(TypeError, match="has no setter"):
            orchestrator.invoke_capability("prompt_builder", "value")

    def test_get_capability_value_returns_value(self, orchestrator):
        """Test get_capability_value returns the current value."""
        # Set up the prompt builder
        orchestrator.prompt_builder._custom_prompt = "Test prompt"

        # Get via attribute capability
        builder = orchestrator.get_capability_value("prompt_builder")
        assert builder is not None
        assert builder._custom_prompt == "Test prompt"

    def test_get_capability_value_returns_enabled_tools(self, orchestrator):
        """enabled_tools capability should expose getter-backed value."""
        orchestrator._enabled_tools = {"read", "write"}

        value = orchestrator.get_capability_value("enabled_tools")

        assert value == {"read", "write"}

    def test_get_capability_value_returns_vertical_context(self, orchestrator):
        """vertical_context capability should expose getter-backed value."""
        orchestrator._vertical_context = {"name": "coding"}

        value = orchestrator.get_capability_value("vertical_context")

        assert value == {"name": "coding"}

    def test_get_capability_value_returns_runtime_intelligence_integration(self, orchestrator):
        """Canonical runtime-intelligence capability should return the shared surface."""
        value = orchestrator.get_capability_value("runtime_intelligence_integration")

        assert value == {"name": "runtime-intelligence"}

    def test_get_capability_value_warns_for_deprecated_runtime_intelligence_alias(
        self, orchestrator
    ):
        """Deprecated capability alias should still work but emit a warning."""
        with pytest.warns(
            DeprecationWarning,
            match="Use 'runtime_intelligence_integration' instead.",
        ):
            value = orchestrator.get_capability_value("intelligent_pipeline")

        assert value == {"name": "runtime-intelligence"}

    def test_init_does_not_log_warning_for_builtin_deprecated_alias(self, caplog):
        """Compatibility alias registration should not spam interactive startup logs."""
        caplog.set_level("WARNING", logger="victor.agent.capability_registry")

        MockOrchestrator()

        assert "Registering deprecated capability" not in caplog.text

    def test_get_capability_value_raises_for_unknown(self, orchestrator):
        """Test get_capability_value raises for unknown capability."""
        with pytest.raises(KeyError, match="not found"):
            orchestrator.get_capability_value("unknown")

    def test_get_capability_version_returns_version(self, orchestrator):
        """Test get_capability_version returns correct version."""
        version = orchestrator.get_capability_version("enabled_tools")
        assert version == "1.0"

    def test_get_capability_version_returns_none_for_unknown(self, orchestrator):
        """Test get_capability_version returns None for unknown capability."""
        assert orchestrator.get_capability_version("unknown") is None

    def test_has_capability_with_min_version_check(self, orchestrator):
        """Test has_capability with version requirements."""
        # Capability is v1.0, should pass for 1.0 requirement
        assert orchestrator.has_capability("enabled_tools", min_version="1.0")
        # Should fail for higher version requirement
        assert not orchestrator.has_capability("enabled_tools", min_version="2.0")

    def test_invoke_capability_with_compatible_version(self, orchestrator):
        """Test invoke_capability with compatible version requirement."""
        tools = {"read", "write"}
        # Should work with compatible version
        orchestrator.invoke_capability("enabled_tools", tools, min_version="1.0")
        assert orchestrator._enabled_tools == tools

    def test_invoke_capability_with_incompatible_version_raises(self, orchestrator):
        """Test invoke_capability raises IncompatibleVersionError for incompatible version."""
        tools = {"read", "write"}
        with pytest.raises(IncompatibleVersionError) as exc_info:
            orchestrator.invoke_capability("enabled_tools", tools, min_version="2.0")
        assert exc_info.value.capability_name == "enabled_tools"
        assert exc_info.value.required_version == "2.0"
        assert exc_info.value.actual_version == "1.0"


# =============================================================================
# Test Capability Integration
# =============================================================================


class TestCapabilityIntegration:
    """Integration tests for capability registry with orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create mock orchestrator."""
        return MockOrchestrator()

    def test_enabled_tools_capability_end_to_end(self, orchestrator):
        """Test enabled_tools capability works end-to-end."""
        # Check capability exists
        assert orchestrator.has_capability("enabled_tools")

        # Invoke capability
        tools = {"read", "write"}
        orchestrator.invoke_capability("enabled_tools", tools)

        # Verify result
        assert orchestrator._enabled_tools == tools

    def test_vertical_context_capability_end_to_end(self, orchestrator):
        """Test vertical_context capability works end-to-end."""
        assert orchestrator.has_capability("vertical_context")

        # Create mock context
        mock_context = {"vertical_name": "coding", "version": "1.0"}
        orchestrator.invoke_capability("vertical_context", mock_context)

        assert orchestrator._vertical_context == mock_context

    def test_custom_prompt_capability_end_to_end(self, orchestrator):
        """Test custom_prompt capability works end-to-end."""
        assert orchestrator.has_capability("custom_prompt")

        # Set custom prompt
        orchestrator.invoke_capability("custom_prompt", "Custom system prompt")

        # Verify it was set
        assert orchestrator.prompt_builder._custom_prompt == "Custom system prompt"

    def test_rl_hooks_capability(self, orchestrator):
        """Test rl_hooks capability."""
        assert orchestrator.has_capability("rl_hooks")

        mock_hooks = {"on_success": lambda: None}
        orchestrator.invoke_capability("rl_hooks", mock_hooks)

        assert orchestrator._rl_hooks == mock_hooks

    def test_team_specs_capability(self, orchestrator):
        """Test team_specs capability."""
        assert orchestrator.has_capability("team_specs")

        mock_specs = {"feature_team": {"members": 3}}
        orchestrator.invoke_capability("team_specs", mock_specs)

        assert orchestrator._team_specs == mock_specs


# =============================================================================
# Test Protocol Conformance
# =============================================================================


class TestProtocolConformance:
    """Tests for CapabilityRegistryProtocol conformance."""

    def test_mock_orchestrator_conforms_to_protocol(self):
        """Test that MockOrchestrator conforms to CapabilityRegistryProtocol."""
        orch = MockOrchestrator()

        # Check all protocol methods exist
        assert hasattr(orch, "get_capabilities")
        assert hasattr(orch, "has_capability")
        assert hasattr(orch, "get_capability")
        assert hasattr(orch, "invoke_capability")
        assert hasattr(orch, "get_capability_value")
        assert hasattr(orch, "get_capabilities_by_type")

        # Check methods are callable
        assert callable(orch.get_capabilities)
        assert callable(orch.has_capability)
        assert callable(orch.get_capability)
        assert callable(orch.invoke_capability)
        assert callable(orch.get_capability_value)
        assert callable(orch.get_capabilities_by_type)

    def test_isinstance_check(self):
        """Test isinstance check with protocol."""
        orch = MockOrchestrator()
        assert isinstance(orch, CapabilityRegistryProtocol)


# =============================================================================
# Test Vertical Integration Helper Functions
# =============================================================================


class TestVerticalIntegrationHelpers:
    """Tests for _check_capability and _invoke_capability helpers."""

    def test_check_capability_with_protocol(self):
        """Test _check_capability uses protocol when available."""
        from victor.framework.vertical_integration import _check_capability

        orch = MockOrchestrator()
        assert _check_capability(orch, "enabled_tools")
        assert not _check_capability(orch, "unknown")

    def test_check_capability_with_fallback(self):
        """Test _check_capability falls back to hasattr."""
        from victor.framework.vertical_integration import _check_capability

        # Object without capability registry
        class PlainObject:
            def set_enabled_tools(self, tools):
                pass

        obj = PlainObject()
        assert _check_capability(obj, "enabled_tools")
        assert not _check_capability(obj, "unknown")

    def test_check_capability_fallback_blocked_in_protocol_strict_mode(self, monkeypatch):
        """Protocol strict mode should block duck-typed fallback checks."""
        from victor.framework.vertical_integration import _check_capability

        class PlainObject:
            def set_enabled_tools(self, tools):
                pass

        monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PROTOCOL_FALLBACKS", "1")

        with pytest.raises(RuntimeError, match="Protocol fallback blocked"):
            _check_capability(PlainObject(), "enabled_tools")

    def test_invoke_capability_with_protocol(self):
        """Test _invoke_capability uses protocol when available."""
        from victor.framework.vertical_integration import _invoke_capability

        orch = MockOrchestrator()
        tools = {"read", "write"}
        _invoke_capability(orch, "enabled_tools", tools)
        assert orch._enabled_tools == tools

    def test_invoke_capability_with_fallback(self):
        """Test _invoke_capability falls back to method call."""
        from victor.framework.vertical_integration import _invoke_capability

        class PlainObject:
            def __init__(self):
                self.tools = set()

            def set_enabled_tools(self, tools):
                self.tools = tools

        obj = PlainObject()
        tools = {"read", "write"}
        _invoke_capability(obj, "enabled_tools", tools)
        assert obj.tools == tools

    def test_invoke_capability_fallback_blocked_in_protocol_strict_mode(self, monkeypatch):
        """Protocol strict mode should block duck-typed fallback invocation."""
        from victor.framework.vertical_integration import _invoke_capability

        class PlainObject:
            def __init__(self):
                self.tools = set()

            def set_enabled_tools(self, tools):
                self.tools = tools

        monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PROTOCOL_FALLBACKS", "1")

        with pytest.raises(RuntimeError, match="Protocol fallback blocked"):
            _invoke_capability(PlainObject(), "enabled_tools", {"read"})

    def test_check_capability_with_version_requirement(self):
        """Test _check_capability with min_version requirement."""
        from victor.framework.vertical_integration import _check_capability

        orch = MockOrchestrator()
        # Capability is v1.0
        assert _check_capability(orch, "enabled_tools", min_version="1.0")
        assert not _check_capability(orch, "enabled_tools", min_version="2.0")

    def test_invoke_capability_with_version_requirement(self):
        """Test _invoke_capability with min_version requirement."""
        from victor.framework.vertical_integration import _invoke_capability

        orch = MockOrchestrator()
        tools = {"read", "write"}

        # Should work with compatible version
        _invoke_capability(orch, "enabled_tools", tools, min_version="1.0")
        assert orch._enabled_tools == tools

    def test_invoke_capability_with_incompatible_version(self):
        """Test _invoke_capability raises for incompatible version."""
        from victor.framework.vertical_integration import _invoke_capability

        orch = MockOrchestrator()
        tools = {"read", "write"}

        with pytest.raises(IncompatibleVersionError):
            _invoke_capability(orch, "enabled_tools", tools, min_version="2.0")

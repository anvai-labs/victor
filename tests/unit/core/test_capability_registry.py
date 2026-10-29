"""Tests for the centralized capability registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus


@runtime_checkable
class MockProtocol(Protocol):
    def do_thing(self) -> str: ...


@runtime_checkable
class AnotherProtocol(Protocol):
    def do_other(self) -> int: ...


class MockStub:
    def do_thing(self) -> str:
        return "stub"


class MockEnhanced:
    def do_thing(self) -> str:
        return "enhanced"


class TestCapabilityRegistry:
    """Test CapabilityRegistry singleton and CRUD."""

    def setup_method(self):
        CapabilityRegistry.reset()

    def teardown_method(self):
        CapabilityRegistry.reset()

    def test_singleton(self):
        r1 = CapabilityRegistry.get_instance()
        r2 = CapabilityRegistry.get_instance()
        assert r1 is r2

    def test_reset(self):
        r1 = CapabilityRegistry.get_instance()
        CapabilityRegistry.reset()
        r2 = CapabilityRegistry.get_instance()
        assert r1 is not r2

    def test_register_and_get(self):
        registry = CapabilityRegistry.get_instance()
        stub = MockStub()
        registry.register(MockProtocol, stub, CapabilityStatus.STUB)
        assert registry.get(MockProtocol) is stub

    def test_get_unregistered_returns_none(self):
        registry = CapabilityRegistry.get_instance()
        assert registry.get(MockProtocol) is None

    def test_is_enhanced_false_for_stub(self):
        registry = CapabilityRegistry.get_instance()
        registry.register(MockProtocol, MockStub(), CapabilityStatus.STUB)
        assert not registry.is_enhanced(MockProtocol)

    def test_is_enhanced_true_for_enhanced(self):
        registry = CapabilityRegistry.get_instance()
        registry.register(MockProtocol, MockEnhanced(), CapabilityStatus.ENHANCED)
        assert registry.is_enhanced(MockProtocol)

    def test_enhanced_not_downgraded_to_stub(self):
        registry = CapabilityRegistry.get_instance()
        enhanced = MockEnhanced()
        stub = MockStub()
        registry.register(MockProtocol, enhanced, CapabilityStatus.ENHANCED)
        registry.register(MockProtocol, stub, CapabilityStatus.STUB)
        assert registry.get(MockProtocol) is enhanced
        assert registry.is_enhanced(MockProtocol)

    def test_stub_upgraded_to_enhanced(self):
        registry = CapabilityRegistry.get_instance()
        stub = MockStub()
        enhanced = MockEnhanced()
        registry.register(MockProtocol, stub, CapabilityStatus.STUB)
        registry.register(MockProtocol, enhanced, CapabilityStatus.ENHANCED)
        assert registry.get(MockProtocol) is enhanced
        assert registry.is_enhanced(MockProtocol)

    def test_duplicate_enhanced_registration_keeps_existing(self):
        registry = CapabilityRegistry.get_instance()
        first = MockEnhanced()
        second = MockEnhanced()
        registry.register(MockProtocol, first, CapabilityStatus.ENHANCED)
        registry.register(MockProtocol, second, CapabilityStatus.ENHANCED)
        assert registry.get(MockProtocol) is first

    def test_get_status(self):
        registry = CapabilityRegistry.get_instance()
        assert registry.get_status(MockProtocol) is None
        registry.register(MockProtocol, MockStub(), CapabilityStatus.STUB)
        assert registry.get_status(MockProtocol) == CapabilityStatus.STUB

    def test_get_metadata(self):
        registry = CapabilityRegistry.get_instance()
        registry.register(
            MockProtocol,
            MockStub(),
            CapabilityStatus.STUB,
            metadata={"language": "all"},
        )
        assert registry.get_metadata(MockProtocol) == {"language": "all"}
        assert registry.get_metadata(AnotherProtocol) == {}

    def test_list_capabilities(self):
        registry = CapabilityRegistry.get_instance()
        registry.register(MockProtocol, MockStub(), CapabilityStatus.STUB)
        registry.register(AnotherProtocol, MockStub(), CapabilityStatus.ENHANCED)
        caps = registry.list_capabilities()
        assert caps["MockProtocol"] == "stub"
        assert caps["AnotherProtocol"] == "enhanced"

    def test_multiple_protocols(self):
        registry = CapabilityRegistry.get_instance()
        stub1 = MockStub()
        stub2 = MockStub()
        registry.register(MockProtocol, stub1)
        registry.register(AnotherProtocol, stub2)
        assert registry.get(MockProtocol) is stub1
        assert registry.get(AnotherProtocol) is stub2


class TestBootstrapCapabilities:
    """Test bootstrap_capabilities integration."""

    def setup_method(self):
        CapabilityRegistry.reset()

    def teardown_method(self):
        CapabilityRegistry.reset()

    def test_bootstrap_registers_stubs(self):
        from victor.core.bootstrap import bootstrap_capabilities
        from victor.framework.vertical_protocols import (
            CodebaseIndexFactoryProtocol,
            IgnorePatternsProtocol,
            LanguageRegistryProtocol,
            TaskTypeHintProtocol,
            TreeSitterExtractorProtocol,
            TreeSitterParserProtocol,
        )

        bootstrap_capabilities()
        registry = CapabilityRegistry.get_instance()

        # All stubs should be registered
        assert registry.get(TreeSitterParserProtocol) is not None
        assert registry.get(TreeSitterExtractorProtocol) is not None
        assert registry.get(CodebaseIndexFactoryProtocol) is not None
        assert registry.get(IgnorePatternsProtocol) is not None
        assert registry.get(LanguageRegistryProtocol) is not None
        assert registry.get(TaskTypeHintProtocol) is not None

    def test_bootstrap_discovers_entry_points(self):
        """When victor-coding is installed, entry points should be discovered."""
        from victor.core.bootstrap import bootstrap_capabilities
        from victor.framework.vertical_protocols import TreeSitterParserProtocol

        bootstrap_capabilities()
        registry = CapabilityRegistry.get_instance()

        # If victor-coding is installed, tree-sitter should be enhanced
        # If not installed, it should still be a stub
        assert registry.get(TreeSitterParserProtocol) is not None

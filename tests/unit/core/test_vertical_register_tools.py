"""Tests for VerticalBase.register_tools() hook."""

import pytest

from victor.core.verticals.base import VerticalBase


def _make_vertical(name: str, register_tools_fn=None):
    """Create a concrete vertical with optional register_tools override."""
    attrs = {
        "name": name,
        "description": f"Test vertical {name}",
        "get_tools": classmethod(lambda cls: ["read"]),
        "get_system_prompt": classmethod(lambda cls: "test prompt"),
    }
    if register_tools_fn is not None:
        attrs["register_tools"] = classmethod(register_tools_fn)

    return type(f"TestVertical_{name}", (VerticalBase,), attrs)


class TestVerticalRegisterTools:
    """Verify the register_tools() hook on VerticalBase."""

    def test_default_is_noop(self):
        """Default register_tools() should be a no-op that doesn't raise."""
        v = _make_vertical("noop_tools")
        # Should not raise
        v.register_tools(None)
        v.register_tools(object())

    def test_override_is_called(self):
        """An overridden register_tools() should be called with the registry."""
        calls = []

        def custom_register(cls, registry):
            calls.append(registry)

        v = _make_vertical("custom_tools", register_tools_fn=custom_register)
        sentinel = object()
        v.register_tools(sentinel)

        assert len(calls) == 1
        assert calls[0] is sentinel

    def test_exception_in_hook_does_not_propagate_in_step_handler(self):
        """Exception in register_tools() should be catchable without crash."""

        def failing_register(cls, registry):
            raise RuntimeError("tool registration failed")

        v = _make_vertical("failing_tools", register_tools_fn=failing_register)

        # The step handler wraps this in try/except, but the method itself raises
        with pytest.raises(RuntimeError, match="tool registration failed"):
            v.register_tools(object())

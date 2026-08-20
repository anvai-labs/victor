# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""TDD tests for extension handler infrastructure and first 3 handlers."""

from unittest.mock import MagicMock, patch
from typing import Any, ClassVar, List, Optional

import pytest

# ── BaseExtensionHandler Tests ──


class TestBaseExtensionHandler:
    def test_load_raises_not_implemented(self):
        from victor.core.verticals.extension_handlers.base import BaseExtensionHandler

        with pytest.raises(NotImplementedError):
            BaseExtensionHandler.load(MagicMock())

    def test_subclass_must_define_extension_type(self):
        from victor.core.verticals.extension_handlers.base import BaseExtensionHandler

        class BadHandler(BaseExtensionHandler):
            pass

        assert (
            not hasattr(BadHandler, "extension_type")
            or BadHandler.extension_type is BaseExtensionHandler.extension_type
        )


# ── ExtensionHandlerRegistry Tests ──


class TestExtensionHandlerRegistry:
    def setup_method(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        self._saved = dict(ExtensionHandlerRegistry._handlers)

    def teardown_method(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        ExtensionHandlerRegistry._handlers = self._saved

    def test_register_and_retrieve(self):
        from victor.core.verticals.extension_handlers.base import BaseExtensionHandler
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        class TestHandler(BaseExtensionHandler):
            extension_type = "test_ext"

            @classmethod
            def load(cls, ctx):
                return "loaded"

        ExtensionHandlerRegistry.register(TestHandler)
        assert ExtensionHandlerRegistry.get("test_ext") is TestHandler

    def test_get_returns_none_for_unknown(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        assert ExtensionHandlerRegistry.get("nonexistent_type_xyz") is None

    def test_all_handlers_returns_copy(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        handlers = ExtensionHandlerRegistry.all_handlers()
        assert isinstance(handlers, dict)
        # Mutating the copy should not affect registry
        handlers["fake"] = None
        assert "fake" not in ExtensionHandlerRegistry._handlers

    def test_register_as_decorator(self):
        from victor.core.verticals.extension_handlers.base import BaseExtensionHandler
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        @ExtensionHandlerRegistry.register
        class DecoratedHandler(BaseExtensionHandler):
            extension_type = "decorated_test"

            @classmethod
            def load(cls, ctx):
                return None

        assert ExtensionHandlerRegistry.get("decorated_test") is DecoratedHandler


# ── MiddlewareHandler Tests ──


class TestMiddlewareHandler:
    def test_returns_empty_when_no_candidates(self):
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )

        ctx = MagicMock()
        ctx._find_available_candidates.return_value = []
        result = MiddlewareHandler.load(ctx)
        assert result == []

    def test_returns_empty_on_import_error(self):
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )

        ctx = MagicMock()
        ctx._find_available_candidates.return_value = ["some.module.middleware"]
        ctx._module_resolver.try_load_from_candidates.side_effect = ImportError("nope")
        result = MiddlewareHandler.load(ctx)
        assert result == []

    def test_returns_empty_when_factory_is_none(self):
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )

        ctx = MagicMock()
        ctx._find_available_candidates.return_value = ["some.module.middleware"]
        ctx._module_resolver.try_load_from_candidates.return_value = None
        result = MiddlewareHandler.load(ctx)
        assert result == []

    def test_loads_via_factory_and_caches(self):
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )

        middleware_list = [MagicMock(), MagicMock()]
        factory = MagicMock(return_value=middleware_list)

        ctx = MagicMock()
        ctx._find_available_candidates.return_value = ["mod.middleware"]
        ctx._module_resolver.try_load_from_candidates.return_value = factory
        # _get_cached_extension calls the factory lambda
        ctx._get_cached_extension.side_effect = lambda key, fn: fn()

        result = MiddlewareHandler.load(ctx)
        assert len(result) == 2
        ctx._get_cached_extension.assert_called_once()

    def test_extension_type(self):
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )

        assert MiddlewareHandler.extension_type == "middleware"

    def test_registered_in_registry(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        # Import triggers @register decorator
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )  # noqa: F401

        assert ExtensionHandlerRegistry.get("middleware") is MiddlewareHandler


# ── SafetyHandler Tests ──


class TestSafetyHandler:
    def test_delegates_to_resolve_factory(self):
        from victor.core.verticals.extension_handlers.safety import SafetyHandler

        mock_extension = MagicMock()
        ctx = MagicMock()
        ctx._resolve_factory_extension.return_value = mock_extension

        result = SafetyHandler.load(ctx)
        assert result is mock_extension
        ctx._resolve_factory_extension.assert_called_once_with("safety_extension", "safety")

    def test_returns_none_when_no_safety(self):
        from victor.core.verticals.extension_handlers.safety import SafetyHandler

        ctx = MagicMock()
        ctx._resolve_factory_extension.return_value = None
        assert SafetyHandler.load(ctx) is None

    def test_extension_type(self):
        from victor.core.verticals.extension_handlers.safety import SafetyHandler

        assert SafetyHandler.extension_type == "safety"

    def test_registered_in_registry(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )
        from victor.core.verticals.extension_handlers.safety import (
            SafetyHandler,
        )  # noqa: F401

        assert ExtensionHandlerRegistry.get("safety") is SafetyHandler


# ── PromptHandler Tests ──


class TestPromptHandler:
    def test_entry_point_takes_precedence(self):
        from victor.core.verticals.extension_handlers.prompt import PromptHandler

        ep_contributor = MagicMock()
        ctx = MagicMock()
        ctx._load_named_entry_point_extension.return_value = ep_contributor

        result = PromptHandler.load(ctx)
        assert result is ep_contributor
        # Factory should NOT be called when entry point succeeds
        ctx._resolve_factory_extension.assert_not_called()

    def test_falls_back_to_factory(self):
        from victor.core.verticals.extension_handlers.prompt import PromptHandler

        factory_contributor = MagicMock()
        ctx = MagicMock()
        ctx._load_named_entry_point_extension.return_value = None
        ctx._resolve_factory_extension.return_value = factory_contributor

        result = PromptHandler.load(ctx)
        assert result is factory_contributor
        ctx._resolve_factory_extension.assert_called_once_with("prompt_contributor", "prompts")

    def test_returns_none_when_nothing_found(self):
        from victor.core.verticals.extension_handlers.prompt import PromptHandler

        ctx = MagicMock()
        ctx._load_named_entry_point_extension.return_value = None
        ctx._resolve_factory_extension.return_value = None
        assert PromptHandler.load(ctx) is None

    def test_extension_type(self):
        from victor.core.verticals.extension_handlers.prompt import PromptHandler

        assert PromptHandler.extension_type == "prompt"

    def test_registered_in_registry(self):
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )
        from victor.core.verticals.extension_handlers.prompt import (
            PromptHandler,
        )  # noqa: F401

        assert ExtensionHandlerRegistry.get("prompt") is PromptHandler

"""Formatter test fixtures.

Provides autouse fixtures for test isolation in formatter tests.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_formatter_registry_state():
    """Restore formatter registry and cache state between formatter tests.

    Formatter tests intentionally register mock formatters for real tool names
    such as ``code_search``. The registry is a process-wide singleton, so those
    mutations must not leak into later tool tests.
    """
    from victor.tools.formatters.registry import (
        _format_cache,
        get_formatter_registry,
    )

    registry = get_formatter_registry()
    original_formatters = dict(registry._formatters)
    _format_cache.invalidate()

    yield

    registry._formatters = original_formatters
    _format_cache.invalidate()

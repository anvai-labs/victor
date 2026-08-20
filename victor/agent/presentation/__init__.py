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

"""Presentation abstraction layer for the agent module.

This package provides a clean abstraction for presentation/formatting concerns,
decoupling the agent layer from direct UI dependencies.

Components:
    - PresentationProtocol: Protocol defining the presentation interface
    - EmojiPresentationAdapter: Settings-aware adapter using core symbols
    - NullPresentationAdapter: Plain text adapter for testing/headless use
    - create_presentation_adapter: Factory function for default adapter

Usage:
    from victor.agent.presentation import (
        PresentationProtocol,
        EmojiPresentationAdapter,
        create_presentation_adapter,
    )

    # Create default adapter (EmojiPresentationAdapter)
    adapter = create_presentation_adapter()

    # Use in component
    class MyComponent:
        def __init__(self, presentation: PresentationProtocol):
            self._presentation = presentation

        def show_success(self, message: str):
            icon = self._presentation.icon("success")
            print(f"{icon} {message}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor.agent.presentation.null_adapter import NullPresentationAdapter
from victor.agent.presentation.protocols import PresentationProtocol

if TYPE_CHECKING:
    from victor.agent.presentation.emoji_adapter import EmojiPresentationAdapter


def _get_emoji_adapter_class() -> type:
    from victor.agent.presentation.emoji_adapter import EmojiPresentationAdapter

    return EmojiPresentationAdapter


def create_presentation_adapter(*, use_null: bool = False) -> PresentationProtocol:
    """Create a presentation adapter.

    Factory function to create the appropriate presentation adapter.

    Args:
        use_null: If True, create NullPresentationAdapter for testing/headless.
                  If False (default), create EmojiPresentationAdapter.

    Returns:
        A PresentationProtocol implementation.

    Example:
        # Default adapter (respects settings)
        adapter = create_presentation_adapter()

        # Null adapter for testing
        adapter = create_presentation_adapter(use_null=True)
    """
    if use_null:
        return NullPresentationAdapter()
    cls = _get_emoji_adapter_class()
    return cls()


def __getattr__(name: str):
    if name == "EmojiPresentationAdapter":
        return _get_emoji_adapter_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PresentationProtocol",
    "EmojiPresentationAdapter",
    "NullPresentationAdapter",
    "create_presentation_adapter",
]

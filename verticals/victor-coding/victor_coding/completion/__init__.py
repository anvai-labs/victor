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

"""Inline completion API for IDE integration.

This module provides an extensible completion system for code editors
and IDE integrations. Supports multiple completion sources:
- LSP servers for language-aware completions
- AI models for intelligent suggestions
- Snippets for template-based completions

Example usage:
    from victor_coding.completion import get_completion_manager
    from pathlib import Path

    manager = get_completion_manager()

    # Get standard completions
    completions = await manager.complete(
        file_path=Path("main.py"),
        line=10,
        character=5,
    )

    # Get inline (ghost text) completions
    inline = await manager.complete_inline(
        file_path=Path("main.py"),
        line=10,
        character=5,
        file_content="def hello():\n    ",
    )

    # Stream inline completions
    async for token in manager.stream_inline_completion(
        file_path=Path("main.py"),
        line=10,
        character=5,
    ):
        print(token, end="", flush=True)
"""

from victor_coding.completion.protocol import (
    CompletionCapabilities,
    CompletionContext,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionMetrics,
    CompletionParams,
    CompletionTriggerKind,
    InlineCompletionItem,
    InlineCompletionList,
    InlineCompletionParams,
    InsertTextFormat,
    Position,
    Range,
    TextEdit,
)
from victor_coding.completion.provider import (
    BaseCompletionProvider,
    CachingCompletionProvider,
    CompositeCompletionProvider,
    CompletionProvider,
    StreamingCompletionProvider,
)
from victor_coding.completion.registry import (
    CompletionProviderRegistry,
    get_completion_registry,
    reset_completion_registry,
)
from victor_coding.completion.manager import (
    CompletionManager,
    get_completion_manager,
    reset_completion_manager,
)

__all__ = [
    # Protocol types
    "CompletionCapabilities",
    "CompletionContext",
    "CompletionItem",
    "CompletionItemKind",
    "CompletionList",
    "CompletionMetrics",
    "CompletionParams",
    "CompletionTriggerKind",
    "InlineCompletionItem",
    "InlineCompletionList",
    "InlineCompletionParams",
    "InsertTextFormat",
    "Position",
    "Range",
    "TextEdit",
    # Provider classes
    "BaseCompletionProvider",
    "CachingCompletionProvider",
    "CompositeCompletionProvider",
    "CompletionProvider",
    "StreamingCompletionProvider",
    # Registry
    "CompletionProviderRegistry",
    "get_completion_registry",
    "reset_completion_registry",
    # Manager
    "CompletionManager",
    "get_completion_manager",
    "reset_completion_manager",
]

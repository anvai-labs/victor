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

"""Code completion providers for IDE integration.

This module provides inline code completion capabilities using LSP servers,
AI models, and snippet templates. Supports both standard LSP completion
and Copilot-style ghost text.

Example:
    from victor.processing.completion import CompletionManager, CompletionParams

    manager = CompletionManager()
    manager.register(AICompletionProvider(client))

    params = CompletionParams(file_path=Path("test.py"), position=Position(10, 5))
    completions = await manager.get_completions(params)
"""

from victor.processing.completion.protocol import (
    CompletionCapabilities,
    CompletionContext,
    CompletionItem,
    CompletionItemKind,
    CompletionItemLabelDetails,
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
from victor.processing.completion.provider import (
    BaseCompletionProvider,
    CompletionProvider,
)
from victor.processing.completion.registry import CompletionProviderRegistry
from victor.processing.completion.manager import CompletionManager
from victor.processing.completion.providers import (
    AICompletionProvider,
    LSPCompletionProvider,
    SnippetCompletionProvider,
)

__all__ = [
    # Protocol types
    "CompletionCapabilities",
    "CompletionContext",
    "CompletionItem",
    "CompletionItemKind",
    "CompletionItemLabelDetails",
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
    # Provider interface
    "BaseCompletionProvider",
    "CompletionProvider",
    # Registry and manager
    "CompletionProviderRegistry",
    "CompletionManager",
    # Built-in providers
    "AICompletionProvider",
    "LSPCompletionProvider",
    "SnippetCompletionProvider",
]
